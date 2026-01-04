import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import sys

# 把项目根目录加进来，确保能 import tools
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ecg_features import compute_ecg_features


class WESADFeatureDataset(Dataset):
    def __init__(self, root_dir, subjects, window_size=700 * 60, step=700):
        """
        基于特征的 WESAD 数据集
        输出不再是 (1, 6000) 的波形，而是 (7,) 的特征向量
        """
        self.samples = []
        self.labels = []

        # 二分类映射: Baseline(1)/Amusement(3) -> 0, Stress(2) -> 1
        self.label_map = {1: 0, 2: 1, 3: 0}

        for sub_id in subjects:
            pkl_path = os.path.join(root_dir, f'S{sub_id}', f'S{sub_id}.pkl')
            if not os.path.exists(pkl_path):
                continue

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

            # WESAD 原始数据是 700Hz
            ecg = data['signal']['chest']['ECG'].flatten()
            labels = data['label']

            # 1. 可以在这里做下采样，加快特征计算速度
            # 700Hz -> 100Hz (downsample_factor=7)
            # compute_ecg_features 里的 fs 参数也要对应改为 100
            factor = 7
            ecg = ecg[::factor]
            labels = labels[::factor]
            real_fs = 100
            real_window = window_size // factor
            real_step = step // factor

            max_len = len(ecg)

            # 2. 滑动窗口 -> 计算特征
            for i in range(0, max_len - real_window, real_step):
                mid_idx = i + real_window // 2
                label_val = labels[mid_idx]

                if label_val in self.label_map:
                    segment = ecg[i: i + real_window]

                    # --- [关键] 调用工具计算特征 ---
                    feats = compute_ecg_features(segment, fs=real_fs)

                    # 过滤掉无效特征 (例如全0)
                    if np.sum(feats) == 0:
                        continue

                    self.samples.append(feats)
                    self.labels.append(self.label_map[label_val])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # x: (7,)
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y