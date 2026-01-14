import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import sys
import random

# 添加项目根目录和当前目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

# 统一引用：使用 tools 目录下的模型定义（与推理一致）
from dataset import WESADFeatureDataset
from tools.localModels.stress_model import HRV_MLP

# --- 配置参数 ---
# 自动探测数据集路径 (相对于项目根目录)
ROOT_DIR = os.path.join(PROJECT_ROOT, 'eval/stress/data/WESAD')
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_PATH, 'checkpoints_mlp')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mlp_loso():
    set_seed(42)
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("StressTrain")
    
    logger.info("--- 启动基于特征的 MLP 训练 (Stress vs Non-Stress) ---")

    overall_f1 = []
    overall_acc = []

    for test_sub in ALL_SUBJECTS:
        train_subs = [s for s in ALL_SUBJECTS if s != test_sub]
        logger.info(f"Processing Fold: Test S{test_sub}")

        # 1. 加载数据
        logger.info("  Extracting features (this may take a while)...")
        train_ds = WESADFeatureDataset(ROOT_DIR, train_subs)
        test_ds = WESADFeatureDataset(ROOT_DIR, [test_sub])

        if len(train_ds) == 0: continue

        scaler = StandardScaler()
        X_train = np.array(train_ds.samples)
        X_test = np.array(test_ds.samples)

        scaler.fit(X_train)

        # 覆盖回 dataset
        train_ds.samples = [torch.tensor(x, dtype=torch.float32) for x in scaler.transform(X_train)]
        test_ds.samples = [torch.tensor(x, dtype=torch.float32) for x in scaler.transform(X_test)]

        # 3. DataLoader
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 4. 模型初始化
        model = HRV_MLP(input_dim=7, num_classes=2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(DEVICE))

        # 5. 训练循环
        best_f1 = 0.0

        for epoch in range(EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x)
                    preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                    labels.extend(y.cpu().numpy())

            f1 = f1_score(labels, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'mlp_s{test_sub}.pth'))

        logger.info(f"  --> S{test_sub} Best F1: {best_f1:.4f}")
        overall_f1.append(best_f1)

    logger.info(f"Final Avg F1: {np.mean(overall_f1):.4f}")


if __name__ == '__main__':
    train_mlp_loso()