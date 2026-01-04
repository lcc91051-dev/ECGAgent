import torch
import torch.nn as nn
import numpy as np
import os
import scipy.signal as signal
from .register import function_register
from .registerData import getRegisteredData


# ---------------------------------------------------------
# 1. 定义模型结构 (完全复刻 uploaded: model.py)
# ---------------------------------------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1),
                                   padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class HRVNet(nn.Module):
    def __init__(self, n_channels=12, n_samples=1000, n_classes=5, F1=8, D=2, F2=16, dropout=0.25):
        super(HRVNet, self).__init__()

        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout)

        self.sep = SeparableConv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, n_classes)

    def forward(self, x):
        # x shape: (Batch, Channels, Time) -> (Batch, 1, Channels, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.sep(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------
# 2. 初始化模型
# ---------------------------------------------------------
# 诊断标签 (PTB-XL 标准)
LABELS = ['Normal', 'Myocardial Infarction', 'ST/T Change', 'Conduction Disturbance', 'Hypertrophy']

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 请确保文件名是这个
MODEL_PATH = os.path.join(CURRENT_DIR, "localModels", "arrhythmia_cnn.pth")

model_arrhythmia = HRVNet(n_channels=12, n_classes=5)

if os.path.exists(MODEL_PATH):
    try:
        model_arrhythmia.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f"[Arrhythmia] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[Arrhythmia] Error loading model: {e}")
else:
    print(f"[Arrhythmia] Warning: Model file not found at {MODEL_PATH}")

model_arrhythmia.eval()


# ---------------------------------------------------------
# 3. 注册工具
# ---------------------------------------------------------
@function_register.register(
    description=(
            "Detect cardiac abnormalities and arrhythmias using 12-lead ECG analysis (based on PTB-XL standard). "
            "Can detect: Normal, Myocardial Infarction (MI), ST/T Changes, Conduction Disturbance, Hypertrophy. "
            "Input segment should be at least 10 seconds. "
            "If input is single-lead, it will be adapted for the model (though accuracy may decrease)."
    ),
    parameters=[
        {"name": "start", "type": "int", "description": "Start time in seconds"},
        {"name": "end", "type": "int", "description": "End time in seconds"}
    ],
    returns={
        "type": "List[Dict]",
        "description": "Diagnosis probabilities for each 10s segment."
    }
)
def arrhythmiaAnalysis(start: int, end: int, config):
    SEGMENT_LEN = 10  # 模型训练时是 10秒
    TARGET_FS = 100  # 模型训练时是 100Hz

    duration = end - start
    if duration < SEGMENT_LEN:
        return [{"warning": f"Duration {duration}s is too short. Minimum required is {SEGMENT_LEN}s."}]

    # 1. 获取数据 [Channels, Time]
    data = getRegisteredData(start, end, config)
    fs_original = config.get('fs', 100)

    # 2. 重采样到 100Hz
    num_samples_target = int(duration * TARGET_FS)
    if fs_original != TARGET_FS:
        data = signal.resample(data, num_samples_target, axis=1)

    # 3. 维度/导联适配
    # 模型期望输入: [Batch, 12, 1000]
    # 当前数据: [C, T]
    n_channels, n_points = data.shape

    # 如果不是 12 导联 (比如单导联)，则复制扩充以匹配模型输入
    if n_channels != 12:
        # print(f"[Arrhythmia] Note: Input has {n_channels} channels. Duplicating to match 12-lead model.")
        # 简单的策略：重复堆叠。如果是单导联，就堆12次。
        # 如果是 3导联，堆4次。
        if n_channels == 1:
            data = np.tile(data, (12, 1))
        elif n_channels < 12:
            # 补零或复制，这里选择用零填充剩余通道
            pad = np.zeros((12 - n_channels, n_points))
            data = np.vstack([data, pad])
        elif n_channels > 12:
            # 截取前12个
            data = data[:12, :]

    # 4. 切分片段 (10秒一段, 无重叠)
    points_per_seg = SEGMENT_LEN * TARGET_FS
    num_segs = n_points // points_per_seg

    results = []

    for i in range(num_segs):
        seg_start_idx = i * points_per_seg
        seg_end_idx = seg_start_idx + points_per_seg

        # [12, 1000]
        segment = data[:, seg_start_idx:seg_end_idx]

        # 转 Tensor: [1, 12, 1000]
        x_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model_arrhythmia(x_tensor)
            # 多标签分类用 Sigmoid
            probs = torch.sigmoid(logits)
            probs_np = probs[0].cpu().numpy()

        # 整理结果
        seg_result = {}
        for idx, label_name in enumerate(LABELS):
            seg_result[label_name] = round(float(probs_np[idx]), 2)

        start_t = start + i * SEGMENT_LEN
        end_t = start_t + SEGMENT_LEN

        # 找出概率最大的问题 (除了 Normal)
        abnormal_probs = {k: v for k, v in seg_result.items() if k != 'Normal'}
        top_condition = max(abnormal_probs, key=abnormal_probs.get)
        top_prob = abnormal_probs[top_condition]

        assessment = "Normal"
        if top_prob > 0.5:  # 阈值可调
            assessment = f"Risk of {top_condition}"

        results.append({
            "duration": f"{start_t:.1f}s-{end_t:.1f}s",
            "Assessment": assessment,
            "Probabilities": seg_result
        })

    return results