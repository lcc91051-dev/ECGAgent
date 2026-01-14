import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import sys
import random
import pickle  # 用来保存 scaler

# 添加项目根目录和当前目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

# 统一引用：使用 tools 目录下的模型定义（与推理一致）
from dataset import WESADFeatureDataset
from tools.localModels.stress_model import HRV_MLP

# --- 配置 ---
# 自动探测数据集路径 (相对于项目根目录)
ROOT_DIR = os.path.join(PROJECT_ROOT, 'eval/stress/data/WESAD')
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 最终模型保存位置
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tools/localModels'))
os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_final_production():
    import logging
    from sklearn.metrics import f1_score, accuracy_score
    from torch.utils.data import random_split
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("StressFinal")
    
    set_seed(42)
    logger.info("--- 启动最终生产级训练 (Final Production Training) ---")
    logger.info(f"模型与标量器将保存至: {SAVE_DIR}")

    # 1. 加载所有受试者数据
    logger.info("正在加载 WESAD 全量特征数据 (此过程较慢)...")
    full_ds = WESADFeatureDataset(ROOT_DIR, ALL_SUBJECTS)

    if len(full_ds) == 0:
        logger.error("错误: 未加载到任何数据，请检查 ROOT_DIR 路径。")
        return

    # 2. 全局标准化 (Fit on ALL data)
    logger.info("对全量数据进行特征标准化...")
    scaler = StandardScaler()
    X_raw = np.array(full_ds.samples)
    scaler.fit(X_raw)

    # 3. 保存 Scaler (推理必备)
    scaler_path = os.path.join(SAVE_DIR, 'stress_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler 已保存至: {scaler_path}")

    # 4. 划分训练集和验证集 (9:1) 用于生产前的质量检查
    val_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_subset, val_subset = random_split(full_ds, [train_size, val_size])
    
    # 手动应用标准化 (由于 Dataset 内部是通过索引访问 samples 列表，
    # 我们直接修改 samples 列表虽然会影响 subset，但为了严谨，我们在 DataLoader 取出后处理或预处理)
    # 预处理全量数据
    full_ds.samples = [torch.tensor(x, dtype=torch.float32) for x in scaler.transform(X_raw)]

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"训练样本数: {train_size} | 验证样本数: {val_size}")

    # 5. 初始化模型
    model = HRV_MLP(input_dim=7, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 考虑类别不平衡：WESAD 中非压力状态通常较多
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(DEVICE))

    # 6. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 每 5 轮或最后一轮进行一次验证
        if (epoch + 1) % 5 == 0 or (epoch == EPOCHS - 1):
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x)
                    all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
            logger.info(f"Epoch {epoch+1:02d}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.4f} - Val F1: {f1:.4f}")
        else:
            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch {epoch+1:02d}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    # 7. 保存最终模型权重
    model_path = os.path.join(SAVE_DIR, 'stress_mlp.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"最终模型权重已保存至: {model_path}")
    logger.info("✅ 生产级训练流程运行成功！")


if __name__ == '__main__':
    train_final_production()