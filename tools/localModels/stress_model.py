import torch
import torch.nn as nn


class HRV_MLP(nn.Module):
    """
    Stress Prediction MLP Model
    Input: 7 HRV Features [Mean_RR, SDNN, RMSSD, pNN50, LF_log, HF_log, LF/HF]
    Output: 2 Classes [Relaxed, Stress]
    """
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
            # 只有当 hidden_dim // 2 > 1 时加入 Dropout 才有意义，且为了保证 batchnorm 在 eval 时正常
            nn.Dropout(0.3),

            # Output Layer
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 7)
        return self.net(x)
