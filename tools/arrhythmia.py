import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from .register import function_register
from .registerData import getRegisteredData
# 核心：直接从你的模型定义文件中导入 HRVNet (EEGNet 架构)
from eval.arrhythmia.model import HRVNet

LABELS = ['Normal', 'Myocardial Infarction', 'ST/T Change', 'Conduction Disturbance', 'Hypertrophy']
# 优先查找 eval/arrhythmia/checkpoints 下的最新模型，兼容旧路径
CHECKPOINT_PATH = os.path.join(project_root, "eval", "arrhythmia", "checkpoints", "best_model.pth")
LEGACY_MODEL_PATH = os.path.join(current_dir, "weights", "hrv_net_best.pth")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model_arrhythmia = None

def _load_arrhythmia_model():
    global _model_arrhythmia
    if _model_arrhythmia is None:
        _model_arrhythmia = HRVNet(n_channels=12, n_samples=1000, n_classes=5)
        
        # 权重加载逻辑：优先加载训练产生的新权重
        if os.path.exists(CHECKPOINT_PATH):
            load_path = CHECKPOINT_PATH
        elif os.path.exists(LEGACY_MODEL_PATH):
            load_path = LEGACY_MODEL_PATH
        else:
            print(f"[Arrhythmia] Warning: No model weights found at {CHECKPOINT_PATH} or {LEGACY_MODEL_PATH}")
            return _model_arrhythmia

        try:
            _model_arrhythmia.load_state_dict(torch.load(load_path, map_location=DEVICE))
            _model_arrhythmia.to(DEVICE)
            _model_arrhythmia.eval()
            print(f"[Arrhythmia] HRVNet loaded successfully from {load_path} on {DEVICE}")
        except Exception as e:
            print(f"[Arrhythmia] Error loading weights from {load_path}: {e}")
            
    return _model_arrhythmia

@function_register.register(
    name="arrhythmiaAnalysis",
    description="Analyze 12-lead ECG for arrhythmia using the robust EEGNet architecture.",
    parameters=[
        {"name": "start_time", "type": "integer", "description": "Start time in seconds", "required": True},
        {"name": "end_time", "type": "integer", "description": "End time in seconds", "required": True}
    ]
)
def arrhythmiaAnalysis(start_time: int, end_time: int, config: dict = None):
    TARGET_FS = 100
    TARGET_SAMPLES = 1000
    model = _load_arrhythmia_model()
    
    # 1. 获取基础采样率 (默认为 100Hz，优先从 config 获取)
    base_fs = config.get('fs', 100) if config else 100
    
    # 2. 获取数据切片 (使用原始采样率以保证时间跨度准确)
    data = getRegisteredData(start_time, end_time, {'fs': base_fs})
    if data is None or data.size == 0: return [{"error": "No data"}]
    
    # 确保 data 是 2D (channels, samples)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # 3. 如果采样率不是 100Hz，进行重采样
    if base_fs != TARGET_FS:
        from scipy import signal
        num_samples = int((end_time - start_time) * TARGET_FS)
        if num_samples <= 0: return [{"error": "Invalid time range"}]
        
        # 对每个导联进行重采样
        resampled_data = signal.resample(data, TARGET_SAMPLES, axis=1)
        data = resampled_data
        print(f"[Arrhythmia] Resampled data from {base_fs}Hz to {TARGET_FS}Hz")
    
    # 4. 长度对齐 (确保是 1000 点)
    if data.shape[1] > TARGET_SAMPLES: 
        data = data[:, :TARGET_SAMPLES]
    elif data.shape[1] < TARGET_SAMPLES:
        pad = np.zeros((data.shape[0], TARGET_SAMPLES - data.shape[1]))
        data = np.hstack([data, pad])
    
    # 5. 导联对齐 (确保是 12 导联)
    if data.shape[0] < 12:
        pad = np.zeros((12 - data.shape[0], TARGET_SAMPLES))
        data = np.vstack([data, pad])
    data = data[:12, :]

    # 预处理
    # 预处理：不再强制使用中值滤波（除非有强干扰），以保持与训练集 Z-Score 逻辑的一致性
    # 按照 eval/arrhythmia/dataset.py 的逻辑：对每个导联独立进行 Z-Score
    data = data.astype(np.float32)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data = (data - mean) / (std + 1e-6)

    # 推理 (注意 EEGNet 需要 4D 输入: B, C, H, W)
    x_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    findings = {LABELS[i]: round(float(probs[i]), 3) for i in range(len(LABELS))}
    top_label = LABELS[np.argmax(probs)]
    
    return [{
        "time_range": f"{start_time}-{end_time}s",
        "primary_finding": top_label if findings[top_label] > 0.5 else "Likely Normal",
        "findings": findings
    }]
