import torch
import numpy as np
import os
import pickle
import logging
from .register import function_register
from .registerData import getRegisteredData
from .ecg_features import compute_ecg_features, get_feature_names

# 配置日志
logger = logging.getLogger("StressTool")

# 资源路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'localModels', 'stress_mlp.pth')
SCALER_PATH = os.path.join(CURRENT_DIR, 'localModels', 'stress_scaler.pkl')

_model_instance = None
_scaler_instance = None

def _load_resources():
    """懒加载模型和标量器"""
    global _model_instance, _scaler_instance
    
    if _model_instance is None:
        from .localModels.stress_model import HRV_MLP  # 确保路径正确
        _model_instance = HRV_MLP(input_dim=7, num_classes=2)
        if os.path.exists(MODEL_PATH):
            _model_instance.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            _model_instance.eval()
            logger.info(f"Loaded stress model from {MODEL_PATH}")
        else:
            logger.error(f"Stress model not found at {MODEL_PATH}")

    if _scaler_instance is None:
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                _scaler_instance = pickle.load(f)
            logger.info(f"Loaded scaler from {SCALER_PATH}")
        else:
            logger.error(f"Scaler not found at {SCALER_PATH}")

@function_register.register(
    name="analyze_stress",
    description=(
        "Analyze mental stress levels based on Heart Rate Variability (HRV) metrics. "
        "Calculates time-domain (SDNN, RMSSD) and frequency-domain (LF/HF) features. "
        "Best results with segments >= 60 seconds."
    ),
    parameters=[
        {"name": "start_time", "type": "number", "description": "Start time in seconds", "required": True},
        {"name": "end_time", "type": "number", "description": "End time in seconds", "required": True},
        {"name": "config", "type": "object", "description": "Contains sampling rate 'fs'", "required": False}
    ],
    returns={"type": "object", "description": "Stress assessment and detailed HRV metrics."}
)
def analyze_stress(start_time, end_time, config=None):
    if config is None: config = {'fs': 100}
    fs = config.get('fs', 100)

    try:
        # 1. 获取数据
        raw_data = getRegisteredData(s=start_time, e=end_time, config=config)
        if raw_data is None or raw_data.size == 0:
            return {"error": "No data found for the selected time range."}
        
        # 统一处理为 1D 信号 (Stress 通常使用单导联)
        ecg_signal = raw_data.flatten() if raw_data.ndim == 1 else raw_data[0]

        # 2. 调用高阶特征提取流水线
        # 这包含了中值滤波、Pan-Tompkins 检测和 Task Force 标准计算
        features = compute_ecg_features(ecg_signal, fs=fs)
        
        if np.all(features == 0):
            return {"error": "Signal quality too low to extract valid HRV features (no R-peaks detected)."}

        # 3. 准备输出指标信息
        feat_names = get_feature_names()
        metrics = {feat_names[i]: round(float(features[i]), 3) for i in range(len(features))}

        # 4. 模型推理
        _load_resources()
        if _model_instance and _scaler_instance:
            feat_scaled = _scaler_instance.transform(features.reshape(1, -1))
            with torch.no_grad():
                logits = _model_instance(torch.tensor(feat_scaled, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1)[0]
                stress_idx = torch.argmax(probs).item()
                confidence = probs[stress_idx].item()
            
            states = ["Relaxed/Resting", "Stressed/Anxious"]
            assessment = states[stress_idx]
        else:
            assessment = "Baseline analysis only (Model unavailable)"
            confidence = 0.0

        return {
            "time_window": f"{start_time}s - {end_time}s",
            "stress_assessment": assessment,
            "confidence": round(confidence, 2),
            "hrv_metrics": metrics,
            "interpretation": (
                "High RMSSD and SDNN generally indicate good cardiac health and low stress. "
                "A significantly elevated LF/HF ratio may suggest sympathetic dominance (higher stress)."
            )
        }

    except Exception as e:
        logger.exception("Error in analyze_stress")
        return {"error": str(e)}
