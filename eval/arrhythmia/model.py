import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1d(nn.Module):
    """
    1D 残差块 (Residual Block)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HRVNet(nn.Module):
    """
    升级版 HRVNet: 采用 ResNet-1d 架构 (深度特征提取)
    专为 PTB-XL 12导联 ECG 设计，旨在提升 F1 分数。
    """

    def __init__(self,
                 n_channels=12,
                 n_samples=1000,
                 n_classes=5,
                 base_filters=64,
                 dropout=0.2):
        super(HRVNet, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # 1. 第一层：初始特征提取
        self.first_layer = nn.Sequential(
            nn.Conv1d(n_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 2. 残差阶段 (Residual Stages)
        # Stage 1: 64 filters
        self.stage1 = self._make_layer(base_filters, base_filters, blocks=2)
        # Stage 2: 128 filters
        self.stage2 = self._make_layer(base_filters, base_filters * 2, blocks=2, stride=2)
        # Stage 3: 256 filters
        self.stage3 = self._make_layer(base_filters * 2, base_filters * 4, blocks=2, stride=2)
        # Stage 4: 512 filters
        self.stage4 = self._make_layer(base_filters * 4, base_filters * 8, blocks=2, stride=2)

        # 3. 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 4. 分类头 (Classifier)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_filters * 8, n_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(ResBlock1d(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResBlock1d(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入适配：(Batch, 1, 12, 1000) -> (Batch, 12, 1000)
        # 如果从 tools/arrhythmia.py 传入 4D 数据，需 squeeze
        if x.dim() == 4:
            x = x.squeeze(1)

        x = self.first_layer(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)