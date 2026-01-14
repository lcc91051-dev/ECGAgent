import os
import json
import numpy as np
import pickle
import time
import yaml
import logging
import wfdb
from openai import OpenAI

# --- 导入工具 ---
from tools.register import function_register
from tools.registerData import registerData
import tools.stress_tool  
import tools.arrhythmia   
import tools.clinical_reasoning # 新增：临床推理与 RAG 接口

from prompt import getSystemPrompt
from utils.parseCalling import extract_tool_calls, has_config_parameter

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ECGAgent")

class ECGAgent:
    def __init__(self, config_path: str, file_path: str, api_key: str, base_url: str):
        logger.info("Initializing ECGAgent for scientific analysis...")
        
        # 1. 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {}

        # 加载报告模板
        template_path = os.path.join(self.config.get('report_template_path', 'config'), 'report_template.yaml')
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as file:
                report_template = yaml.safe_load(file)
        else:
            report_template = "Standard ECG Analysis Report Template"

        self.file_path = file_path

        # 2. 加载数据并注册
        logger.info(f"Loading ECG data source: {file_path}")
        ecg_data, sampling_rate = self.load_ecg_data(self.file_path)
        
        # 注册数据
        registerData(ecg_data)
        
        # 总是优先使用数据探测到的采样率，除非加载失败
        self.config['fs'] = sampling_rate
        logger.info(f"Using sampling rate: {sampling_rate}Hz")

        # 3. 构建患者基本信息
        duration_seconds = len(ecg_data) / self.config['fs'] if ecg_data.ndim == 1 else ecg_data.shape[1] / self.config['fs']
        
        self.info = {
            "patient_id": "SUBJ_B72",
            "sex": "Unknown",
            "age": 25,
            "recording_duration_seconds": round(duration_seconds, 2),
            "sampling_rate_hz": self.config['fs'],
            "leads": "Single-lead" if ecg_data.ndim == 1 else f"{ecg_data.shape[0]}-leads",
            "data_source": os.path.basename(file_path)
        }
        logger.info(f"Data registered. Duration: {self.info['recording_duration_seconds']}s")

        # 4. 初始化 LLM
        self.tool_schemas = function_register.export_tool_schemas()
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 构造 System Prompt
        prior_knowledge = self.config.get('prior_knowledge', {})
        self.system_prompt = getSystemPrompt(self.info, self.tool_schemas, prior_knowledge, report_template)
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        
        self.max_rounds = self.config.get('max_rounds', 5)

    def load_ecg_data(self, file_path):
        """
        [多格式适配] 支持 .pkl(WESAD), .dat(WFDB), .npy, .csv
        """
        try:
            # A. WFDB 格式 (医学研究标准)
            if os.path.exists(file_path + ".hea") or file_path.endswith('.dat'):
                base_path = file_path.replace('.dat', '').replace('.hea', '')
                record = wfdb.rdsamp(base_path)
                data = record[0].T  # 转置为 [Channels, Time]
                fs = record[1]['fs']
                logger.info(f"Loaded WFDB data. Channels: {data.shape[0]}, FS: {fs}")
                return data, fs

            # B. WESAD / Pickle 格式
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    content = pickle.load(f, encoding='latin1')
                if 'signal' in content and 'chest' in content['signal']:
                    ecg = content['signal']['chest']['ECG'].flatten()
                else:
                    ecg = np.array(content).flatten()
                return ecg, self.config.get('fs', 700)

            # C. Numpy 格式
            elif file_path.endswith('.npy'):
                return np.load(file_path).flatten(), self.config.get('fs', 100)

            # D. 测试模式
            elif file_path == "dummy":
                fs = 100
                t = np.linspace(0, 100, 100 * 100)
                ecg = np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(len(t))
                return ecg, fs

            else:
                raise ValueError(f"Unknown file format: {file_path}")

        except Exception as e:
            logger.warning(f"Data loading failed: {e}. Falling back to clean dummy data.")
            return np.zeros(10000), 100

    def call_model(self):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.get("llm_model", "/maas/qwen/Qwen3-235B-A22B"),
                messages=self.messages,
                temperature=0.7
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            logger.error(f"LLM API Call Error: {e}")
            return "Connection error. Please try again."

    def handle_tool_calls(self, response):
        calls = extract_tool_calls(response)
        if not calls:
            return False

        for call in calls:
            func_name = call['name']
            args = call['args']

            logger.info(f">>> Agent executing: {func_name}({args})")

            func = function_register.get_function(func_name)
            if not func:
                result = f"Error: Tool {func_name} not found."
            else:
                if has_config_parameter(func) and 'config' not in args:
                    args['config'] = self.config
                try:
                    result = func(**args)
                except Exception as e:
                    logger.exception(f"Tool execution failed: {func_name}")
                    result = f"Runtime Error: {str(e)}"

            # 注入结果
            self.messages.append({
                "role": "user",
                "content": f"<RETURN> {json.dumps(result, ensure_ascii=False)} </RETURN>"
            })
        return True

    def run(self, query):
        logger.info(f"Human: {query}")
        self.messages.append({'role': 'user', 'content': query})

        for r in range(self.max_rounds):
            response = self.call_model()
            logger.info(f"Agent Round {r+1}: {response}")
            
            if not self.handle_tool_calls(response):
                return response
        
        return "Max interaction rounds reached."

if __name__ == "__main__":
    api_key = "sk-47drgpndfxlpk7yfmhuldqo4bumpfqohzrfzoct67uwi5usj"
    
    agent = ECGAgent(
        config_path="config/config.json",
        file_path="data/01048_lr",
        api_key=api_key,
        base_url="https://maas-api.lanyun.net/v1"
    )

    agent.run("Please check my heart rhythm for any abnormalities in this 10-second recording.")