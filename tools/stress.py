import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from .register import function_register
from .registerData import getRegisteredData
# 引用 preprocessing 中的特征计算函数 (稍后在下面更新该文件)
from .preprocessing import compute_ecg_features


# ---------------------------------------------------------
# 1. 定义模型结构 (需与 model.py 保持完全一致)
# ---------------------------------------------------------
class HRV_MLP(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_classes=2):
        super(HRV_MLP, self).__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output Layer
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 2. 初始化模型与加载权重
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 请确保文件名与此处一致
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "stress_mlp.pth")
SCALER_PATH = os.path.join(CURRENT_DIR, "localModels", "stress_scaler.pkl")

# 加载模型
model_stress = HRV_MLP()
if os.path.exists(MODEL_PATH):
    try:
        model_stress.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f"[Stress] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[Stress] Error loading model: {e}")
else:
    print(f"[Stress] Warning: Model file not found at {MODEL_PATH}")
model_stress.eval()

# 加载 Scaler (非常重要！否则预测结果全是错的)
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[Stress] Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"[Stress] Error loading scaler: {e}")
else:
    print(f"[Stress] Warning: Scaler file not found at {SCALER_PATH}")


# ---------------------------------------------------------
# 3. 注册工具
# ---------------------------------------------------------
@function_register.register(
    description=(
            "Analyze physiological stress levels using ECG Heart Rate Variability (HRV). "
            "Input window should be at least 60 seconds for reliable HRV analysis. "
            "Returns probabilities for 'Relaxed' vs 'Stress' state, along with key HRV features."
    ),
    parameters=[
        {"name": "start", "type": "int", "description": "Start time in seconds"},
        {"name": "end", "type": "int", "description": "End time in seconds"}
    ],
    returns={
        "type": "List[Dict]",
        "description": "Assessment result with probabilities and HRV metrics."
    }
)
def stressAnalysis(start: int, end: int, config):
    # HRV 分析通常需要较长的时间窗口，建议至少 30秒，最好 60秒以上
    duration = end - start
    if duration < 30:
        return [{"warning": f"Duration {duration}s is too short for reliable HRV analysis (min 30s recommended)."}]

    # 1. 获取数据 [Channels, Samples]
    data = getRegisteredData(start, end, config)
    fs = config.get('fs', 100)

    # 假设取第1个通道 (通常 ECG 信号在 Ch0)
    # 如果是多导联，这里默认只分析主导联
    ecg_signal = data[0, :]

    # 2. 计算特征 (调用 preprocessing 中的函数)
    features = compute_ecg_features(ecg_signal, fs=fs)

    # 检查特征有效性 (如果没检测到 R 波，features 会是全0)
    if np.sum(features) == 0:
        return [{"warning": "Signal quality too low or no R-peaks detected in this segment."}]

    # 3. 归一化 (Scaler)
    if scaler:
        # scaler expects (n_samples, n_features)
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    else:
        # 如果没找到 scaler，只能硬着头皮裸奔（结果可能不准）
        features_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)

    # 4. 推理
    with torch.no_grad():
        logits = model_stress(features_tensor)
        probs = torch.softmax(logits, dim=1)
        # 根据 WESAD 映射: 0=Baseline/Amusement, 1=Stress
        prob_relax = probs[0, 0].item()
        prob_stress = probs[0, 1].item()

    # 5. 生成结果
    label = "High Stress" if prob_stress > 0.5 else "Relaxed / Normal"

    return [{
        "duration": f"{start}s-{end}s",
        "Assessment": label,
        "Prob": {
            "Relaxed": round(prob_relax, 2),
            "Stress": round(prob_stress, 2)
        },
        "HRV_Metrics": {
            "Mean_RR": f"{features[0]:.1f} ms",
            "RMSSD": f"{features[2]:.1f} ms (Parasympathetic)",
            "LF/HF": f"{features[6]:.2f} (Sympathetic Balance)"
        }
    }]