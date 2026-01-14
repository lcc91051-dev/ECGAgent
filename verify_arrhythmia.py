import os
import sys
import numpy as np
import torch

# 1. 环境与路径检查
print("=== Level 1: Environment & Paths ===")
try:
    import pandas as pd
    import wfdb
    import scipy
    print("✅ Core dependencies (pandas, wfdb, scipy, torch) found.")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")

project_root = os.getcwd()
sys.path.append(project_root)

paths_to_check = {
    "Metadata CSV": "eval/arrhythmia/data/ptbxl_database.csv",
    "SCP Statements": "eval/arrhythmia/data/scp_statements.csv",
    "Model Weights": "eval/arrhythmia/checkpoints/best_model.pth",
    "Records Dir": "eval/arrhythmia/records100"
}

for name, rel_path in paths_to_check.items():
    full_path = os.path.join(project_root, rel_path)
    if os.path.exists(full_path):
        print(f"✅ {name} found at: {rel_path}")
    else:
        print(f"⚠️ {name} NOT found at: {rel_path}")

# 2. 逻辑与推理检查 (Mock Test)
print("\n=== Level 2: Logic & Inference Mock ===")
try:
    from tools.registerData import registerData
    from tools.arrhythmia import arrhythmiaAnalysis
    
    # 模拟 12 导联 500Hz 数据 (WESAD 或其他来源常见频率)
    # 10 秒数据 = 5000 个采样点
    fs_mock = 500
    duration = 10
    channels = 12
    dummy_data = np.random.randn(channels, fs_mock * duration)
    
    print(f"Registering dummy data: {dummy_data.shape} at {fs_mock}Hz")
    registerData(dummy_data)
    
    # 调用工具 (内部会自动触发重采样到 100Hz)
    print("Running arrhythmiaAnalysis(0, 10)...")
    # 模拟 Agent 注入 config
    config = {"fs": fs_mock}
    result = arrhythmiaAnalysis(0, 10, config=config)
    
    print("Result:")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if "findings" in result[0]:
        print("✅ Inference flow Successful!")
    else:
        print("❌ Inference failed or returned error.")

except Exception as e:
    print(f"❌ Mock Test failed with error: {e}")
    import traceback
    traceback.print_exc()

# 3. 训练数据集检查
print("\n=== Level 3: Dataset Training check ===")
try:
    from eval.arrhythmia.dataset import PTBXLDataset
    data_path = os.path.join(project_root, "eval/arrhythmia")
    ds = PTBXLDataset(data_path, mode='val')
    print(f"✅ PTBXLDataset initialized. Found {len(ds)} validation samples.")
except Exception as e:
    print(f"⚠️ Dataset check failed: {e} (This might be due to missing raw .dat files)")

print("\n=== Verification Complete ===")
