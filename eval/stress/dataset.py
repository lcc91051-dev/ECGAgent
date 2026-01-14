import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import sys
from scipy import signal as scipy_signal
from math import gcd

# ==========================================
# 路径设置：确保能导入 tools 模块
# ==========================================
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../../')
sys.path.insert(0, PROJECT_ROOT)

# 使用 tools/ecg_features.py (与推理时完全一致)
from tools.ecg_features import compute_ecg_features


# ==========================================
# 采样率转换函数 (与 stress_tool.py 保持一致)
# ==========================================
def resample_signal(signal: np.ndarray, fs_original: int, fs_target: int) -> np.ndarray:
    """
    抗锯齿降采样 (与 tools/stress_tool.py 中的 _resample_signal 保持一致)
    
    使用 scipy.signal.resample_poly 进行高质量重采样。
    """
    if fs_original == fs_target:
        return signal
    
    if fs_original > fs_target:
        g = gcd(fs_original, fs_target)
        up = fs_target // g
        down = fs_original // g
        return scipy_signal.resample_poly(signal, up, down)
    else:
        # 上采样不处理，直接返回
        return signal


class WESADFeatureDataset(Dataset):
    """
    WESAD 数据集的 HRV 特征版本
    
    处理流程:
        1. 加载 WESAD .pkl 文件 (原始 700Hz)
        2. 使用 resample_poly 降采样到 100Hz (抗锯齿)
        3. 滑动窗口提取片段
        4. 调用 tools/ecg_features.py 计算 7 维 HRV 特征
    
    输出:
        x: (7,) 特征向量 [Mean_RR, SDNN, RMSSD, pNN50, LF, HF, LF/HF]
        y: 0 (Relaxed) 或 1 (Stress)
    
    与推理完全一致:
        - 降采样方法: resample_poly (与 stress_tool.py 一致)
        - 特征提取: tools/ecg_features.py (与推理时相同)
    """
    
    # 类常量
    ORIGINAL_FS = 700    # WESAD 原始采样率
    TARGET_FS = 100      # 特征提取目标采样率 (与 stress_tool.py 一致)
    
    def __init__(self, root_dir, subjects, window_sec=60, step_sec=1):
        """
        Args:
            root_dir: WESAD 数据集根目录 (包含 S2, S3, ... 子文件夹)
            subjects: 受试者 ID 列表, 如 [2, 3, 4, ...]
            window_sec: 窗口大小 (秒), 默认 60秒
            step_sec: 滑动步长 (秒), 默认 1秒
        """
        self.samples = []
        self.labels = []
        
        # 二分类映射: Baseline(1)/Amusement(3) -> 0 (Relaxed), Stress(2) -> 1
        self.label_map = {1: 0, 2: 1, 3: 0}
        
        # 计算目标采样率下的窗口和步长
        window_samples = window_sec * self.TARGET_FS
        step_samples = step_sec * self.TARGET_FS
        
        print(f"[Dataset] Loading WESAD data from: {root_dir}")
        print(f"[Dataset] Window: {window_sec}s ({window_samples} samples @ {self.TARGET_FS}Hz)")
        print(f"[Dataset] Step: {step_sec}s ({step_samples} samples)")
        
        for sub_id in subjects:
            pkl_path = os.path.join(root_dir, f'S{sub_id}', f'S{sub_id}.pkl')
            if not os.path.exists(pkl_path):
                print(f"[Dataset] Warning: {pkl_path} not found, skipping.")
                continue
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # 1. 提取原始 ECG (700Hz)
            ecg_raw = data['signal']['chest']['ECG'].flatten()
            labels_raw = data['label']
            
            # 2. 使用 resample_poly 降采样到 100Hz (与推理一致！)
            ecg = resample_signal(ecg_raw, self.ORIGINAL_FS, self.TARGET_FS)
            
            # 标签也需要降采样 (但标签不需要抗锯齿，用简单抽取即可)
            downsample_factor = self.ORIGINAL_FS // self.TARGET_FS
            labels = labels_raw[::downsample_factor]
            
            # 确保 ECG 和 labels 长度匹配
            min_len = min(len(ecg), len(labels))
            ecg = ecg[:min_len]
            labels = labels[:min_len]
            
            # 3. 滑动窗口提取特征
            n_windows = 0
            for i in range(0, len(ecg) - window_samples, step_samples):
                # 取窗口中点的标签
                mid_idx = i + window_samples // 2
                label_val = labels[mid_idx]
                
                if label_val not in self.label_map:
                    continue  # 跳过未定义的标签 (如过渡段)
                
                segment = ecg[i: i + window_samples]
                
                # 4. 计算 HRV 特征 (使用 tools/ecg_features.py)
                feats = compute_ecg_features(segment, fs=self.TARGET_FS)
                
                # 过滤无效特征 (全零表示 R 波检测失败)
                if np.all(feats == 0):
                    continue
                
                self.samples.append(feats)
                self.labels.append(self.label_map[label_val])
                n_windows += 1
            
            print(f"[Dataset] S{sub_id}: {n_windows} valid windows extracted")
        
        print(f"[Dataset] Total samples: {len(self.samples)}")
        print(f"[Dataset] Class distribution: "
              f"Relaxed={self.labels.count(0)}, Stress={self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y