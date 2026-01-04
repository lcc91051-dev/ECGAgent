from .register import function_register

# --- 1. 基础工具导入 (Base Tools) ---
# 这些是 Agent 运行的基石，必须导入
from . import baseInfo
from . import dataloader
from . import registerData

# --- 2. ECG 核心业务导入 (ECG Core Tools) ---
# 这里只导入我们新写的两个核心分析模块
# 请确保你的 tools 文件夹里确实有 stress.py 和 arrhythmia.py
from . import stress
from . import arrhythmia

# --- 3. 导出注册器 (Export) ---
__all__ = ['function_register']