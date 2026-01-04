import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    [完全复刻] EEGAgent/eval/sleep/train.py 中的深度可分离卷积
    """

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
    """
    [架构复刻] EEGNet 模型 (适配 PTB-XL 心电版)
    """

    def __init__(self,
                 n_channels=12,  # <--- 适配点：PTB-XL 是 12 导联
                 n_samples=1000,  # <--- 适配点：10秒 x 100Hz = 1000点
                 n_classes=5,  # NORM, MI, STTC, CD, HYP
                 F1=8,  # 初始卷积核数 (EEGAgent Sleep 默认为8或32，这里用8轻量级)
                 D=2,  # 深度参数 (每个时间卷积核对应 D 个空间卷积核)
                 F2=None,
                 kernel_length=64,  # 时间卷积核长度 (约0.64秒)
                 dropout=0.25):  # 常用 Dropout
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.n_channels = n_channels
        self.n_samples = n_samples

        # --- Block 1: 时间卷积 (Temporal Conv) ---
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # --- Block 2: 空间卷积 (Spatial Conv / Depthwise) ---
        # 这一步将融合 12 个导联的信息，提取"空间"特征
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout)

        # --- Block 3: 可分离卷积 (Separable Conv) ---
        self.sep = SeparableConv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropout)

        # --- 分类头 ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, n_classes)

        # 权重初始化 (复刻原版)
        self._initialize_weights()

    def forward(self, x):
        # 适配输入: (Batch, 12, 1000) -> (Batch, 1, 12, 1000)
        # EEGNet 需要一个额外的维度来视为"单通道图像"
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
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)