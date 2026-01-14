import numpy as np
import threading

# 线程安全的数据存储
_data_lock = threading.Lock()
_register_data = None


def registerData(data):
    """
    注册全局数据。
    data: numpy array, 支持 1D (单通道 WESAD) 或 2D (多通道 PTB-XL)
    线程安全。
    """
    global _register_data
    with _data_lock:
        _register_data = data
        if data is not None:
            # 打印一下形状，方便调试
            print(f"[System] Data registered. Shape: {data.shape}")


def getRegisteredData(s=None, e=None, config=None):
    """
    获取数据切片。
    s: start time (秒)
    e: end time (秒)
    config: 包含 'fs' 的字典
    
    线程安全，带边界保护。
    """
    global _register_data
    
    with _data_lock:
        if _register_data is None:
            raise RuntimeError("Data not registered. Call registerData() first.")
        
        # 如果没传时间参数，返回所有数据的副本
        if s is None or e is None or config is None:
            return _register_data.copy()

        fs = int(config.get('fs', 700))  # 默认 WESAD 700Hz

        # 将秒转换为索引
        start_idx = int(float(s) * fs)
        end_idx = int(float(e) * fs)

        # --- 边界保护 ---
        data_len = _register_data.shape[-1] if _register_data.ndim > 1 else _register_data.shape[0]
        
        # 检查负索引
        if start_idx < 0:
            print(f"[Warning] start_idx ({start_idx}) < 0, setting to 0")
            start_idx = 0
        
        # 检查起始索引超出范围
        if start_idx >= data_len:
            print(f"[Warning] start_idx ({start_idx}) >= data_len ({data_len})")
            return np.array([])
        
        # 检查结束索引
        if end_idx > data_len:
            print(f"[Warning] end_idx ({end_idx}) > data_len ({data_len}), clipping to {data_len}")
            end_idx = data_len
        
        # 检查索引顺序
        if start_idx >= end_idx:
            print(f"[Warning] start_idx ({start_idx}) >= end_idx ({end_idx})")
            return np.array([])

        # --- 关键修改：兼容单通道和多通道 ---
        if _register_data.ndim == 1:
            # 针对 WESAD (1D 数组)
            return _register_data[start_idx:end_idx].copy()
        else:
            # 针对 PTB-XL (2D 数组 [channels, samples])
            return _register_data[:, start_idx:end_idx].copy()


def clearRegisteredData():
    """
    清除注册的数据（用于测试或重新加载）。
    """
    global _register_data
    with _data_lock:
        _register_data = None
        print("[System] Registered data cleared.")