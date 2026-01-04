import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import random

# 注意：请确保文件名和类名引用正确
from dataset import WESADFeatureDataset
from model import HRV_MLP

# --- 配置参数 ---
# 请确认路径是否正确
ROOT_DIR = r'D:\pycharm\EEG\EEGAgent-main\eval\stress\data\WESAD'
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
    print("--- 启动基于特征的 MLP 训练 (Stress vs Non-Stress) ---")

    overall_f1 = []
    overall_acc = []

    for test_sub in ALL_SUBJECTS:
        train_subs = [s for s in ALL_SUBJECTS if s != test_sub]
        print(f"\nProcessing Fold: Test S{test_sub}")

        # 1. 加载数据 (此时 Dataset 内部已经调用了 tools 计算好了特征)
        print("  Extracting features (this may take a while)...")
        train_ds = WESADFeatureDataset(ROOT_DIR, train_subs)
        test_ds = WESADFeatureDataset(ROOT_DIR, [test_sub])

        if len(train_ds) == 0: continue

        # 2. [关键修复] 特征标准化 (StandardScaler)
        scaler = StandardScaler()

        # --- [修改点] 这里的 samples 已经是 numpy array 了，不需要 .numpy() ---
        X_train = np.array(train_ds.samples)
        X_test = np.array(test_ds.samples)

        scaler.fit(X_train)  # 只在训练集上 fit

        # 覆盖回 dataset (这时再转成 Tensor)
        train_ds.samples = [torch.tensor(x, dtype=torch.float32) for x in scaler.transform(X_train)]
        test_ds.samples = [torch.tensor(x, dtype=torch.float32) for x in scaler.transform(X_test)]

        # 3. DataLoader
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 4. 模型初始化
        model = HRV_MLP(input_dim=7, num_classes=2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        # 类别权重
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

            f1 = f1_score(labels, preds, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'mlp_s{test_sub}.pth'))

        print(f"  --> S{test_sub} Best F1: {best_f1:.4f}")
        overall_f1.append(best_f1)

    print(f"\nFinal Avg F1: {np.mean(overall_f1):.4f}")


if __name__ == '__main__':
    train_mlp_loso()