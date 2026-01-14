import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from sklearn.metrics import f1_score
import numpy as np

# 导入适配好的模型和之前的数据集
from model import HRVNet
from dataset import PTBXLDataset

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50  # ResNet 深，建议稍微多跑几轮
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 路径设置
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_PATH, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')


def calculate_metrics(outputs, targets):
    """
    计算多标签分类的指标
    """
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()

    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # 1. Accuracy (Label-wise)
    acc = (preds_np == targets_np).mean()

    # 2. Macro-F1 (关键指标)
    f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)

    return acc, f1


def train_model():
    print(f"--- 启动训练 (ResNet-1d 升级版) ---")
    print(f"设备: {DEVICE} | 任务: 5-Class Arrhythmia Classification")

    # 1. 加载数据
    train_dataset = PTBXLDataset(BASE_PATH, mode='train')
    val_dataset = PTBXLDataset(BASE_PATH, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # 2. 初始化 ResNet-1d 模型
    model = HRVNet(n_channels=12, n_samples=1000, n_classes=5).to(DEVICE)

    # 3. 优化器 (使用 CosineAnnealing 调度器可以进一步提升性能)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. 损失函数
    # 工业建议：如果 F1 还是低，可以传入 pos_weight 来平衡正负例
    criterion = nn.BCEWithLogitsLoss()

    # 5. 训练循环
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()

        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # --- 验证阶段 ---
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                all_preds.append(outputs)
                all_labels.append(y)

        val_outputs = torch.cat(all_preds)
        val_targets = torch.cat(all_labels)
        val_acc, val_f1 = calculate_metrics(val_outputs, val_targets)

        # --- 保存最佳模型 ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            save_msg = "--> 💾 Model Saved (New Best F1)"
        else:
            save_msg = ""

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{EPOCHS} [{epoch_time:.0f}s] | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} {save_msg}")

    print("-" * 30)
    print(f"训练结束! 最佳 Macro-F1: {best_f1:.4f}")


if __name__ == '__main__':
    train_model()
