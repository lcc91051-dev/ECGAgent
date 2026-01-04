import os
import torch
from torch.utils.data import DataLoader
from dataset import PTBXLDataset

# 获取当前脚本所在目录 (.../eval/arrhythmia)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

print(">>> 开始测试 Dataset 加载...")
print(f"数据根目录: {BASE_PATH}")

try:
    # 1. 初始化 Dataset (测试模式)
    ds = PTBXLDataset(data_path=BASE_PATH, mode='test')
    print(f"[✅ 成功] Dataset 初始化完成")
    print(f"   - 样本数量: {len(ds)} (预期约 2200 条)")
    print(f"   - 分类类别: {ds.classes}")

    # 2. 读取单个样本
    x, y = ds[0]
    print(f"\n[✅ 成功] 读取单个样本")
    print(f"   - 输入形状: {x.shape} (应为 [12, 1000])")
    print(f"   - 标签内容: {y} (Multi-hot 向量)")

    # 3. 测试 DataLoader 批量读取
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"\n[✅ 成功] DataLoader 批量读取测试")
    print(f"   - Batch X: {batch_x.shape} (应为 [32, 12, 1000])")
    print(f"   - Batch Y: {batch_y.shape} (应为 [32, 5])")

    print("\n恭喜！数据流水线完全打通，随时可以开始训练模型！")

except Exception as e:
    print(f"\n[❌ 失败] 发生错误: {e}")
    # 打印详细报错以便排查
    import traceback

    traceback.print_exc()