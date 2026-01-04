import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import random
import pickle  # 用来保存 scaler

# 引入之前的 Dataset 和 Model
from dataset import WESADFeatureDataset
from model import HRV_MLP

# --- 配置 ---
ROOT_DIR = r'D:\pycharm\EEG\EEGAgent-main\eval\stress\data\WESAD'
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
BATCH_SIZE = 64
EPOCHS = 20  # 全量数据多，可以稍微少跑几轮，或者保持 20
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 最终模型保存位置 (建议放到 tools 目录下，方便 Agent 调用)
# 这里假设 tools 在 ../../../tools
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tools/localModels'))
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_final_production():
    set_seed(42)
    print("--- 启动最终全量训练 (Final Production Training) ---")
    print(f"数据将保存到: {SAVE_DIR}")

    # 1. 加载所有人的数据
    print("Loading ALL subjects...")
    full_ds = WESADFeatureDataset(ROOT_DIR, ALL_SUBJECTS)

    if len(full_ds) == 0:
        print("Error: No data loaded!")
        return

    # 2. 全局标准化 (Fit on ALL data)
    print("Fitting StandardScaler on entire dataset...")
    scaler = StandardScaler()
    X_all = np.array(full_ds.samples)
    scaler.fit(X_all)  # 计算所有数据的均值和方差

    # 应用标准化
    full_ds.samples = [torch.tensor(x, dtype=torch.float32) for x in scaler.transform(X_all)]

    # 3. 保存 Scaler (重要！！！)
    # 以后 Agent 推理时，必须加载这个文件来处理新数据
    scaler_path = os.path.join(SAVE_DIR, 'stress_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # 4. 准备 DataLoader
    train_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 初始化模型
    model = HRV_MLP(input_dim=7, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(DEVICE))

    # 6. 训练
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

    # 7. 保存最终模型
    model_path = os.path.join(SAVE_DIR, 'stress_mlp.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Final Model saved to: {model_path}")
    print("✅ 训练完成！现在你可以去写 Agent 的推理代码了。")


if __name__ == '__main__':
    train_final_production()