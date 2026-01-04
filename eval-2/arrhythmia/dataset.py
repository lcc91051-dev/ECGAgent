import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.utils.data import Dataset
import os


class PTBXLDataset(Dataset):
    def __init__(self, data_path, mode='train', test_fold=10):
        """
        PTB-XL 数据集加载器 (增强版)
        包含了: 标签聚合、数据划分、质量过滤(去噪)、Z-Score归一化

        Args:
            data_path (str): 数据集根目录 (包含 records100 和 data 文件夹)
            mode (str): 'train' | 'val' | 'test'
            test_fold (int): 第10折作为测试集
        """
        self.data_path = data_path

        # --- 1. 读取元数据 ---
        csv_path = os.path.join(data_path, 'data', 'ptbxl_database.csv')
        self.df = pd.read_csv(csv_path, index_col='ecg_id')
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # --- 2. 质量过滤 (Quality Filtering) [新功能] ---
        # 根据论文，static_noise 和 burst_noise 不为空则代表有噪声 [cite: 561, 597]
        # 我们只保留那些"干净"的数据
        initial_len = len(self.df)
        self.df = self.df[self.df.static_noise.isna() & self.df.burst_noise.isna()]
        print(f"[{mode}] 质量过滤: 剔除了 {initial_len - len(self.df)} 条低质量数据 (剩余 {len(self.df)} 条)")

        # --- 3. 标签聚合 (71类 -> 5大类) ---
        scp_path = os.path.join(data_path, 'data', 'scp_statements.csv')
        agg_df = pd.read_csv(scp_path, index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]  # 只取诊断标签

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(aggregate_diagnostic)

        # --- 4. 数据集划分 ---
        if mode == 'test':
            self.df = self.df[self.df.strat_fold == test_fold]
        elif mode == 'val':
            self.df = self.df[self.df.strat_fold == (test_fold - 1)]
        else:
            self.df = self.df[self.df.strat_fold <= (test_fold - 2)]

        # 定义 5 大类
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        self.class_map = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- A. 路径修复 ---
        filename = row.filename_lr
        if filename.startswith("records100/"):
            filename = filename.replace("records100/", "")
        elif filename.startswith("records100\\"):
            filename = filename.replace("records100\\", "")

        wfdb_path = os.path.join(self.data_path, 'records100', filename)

        # --- B. 读取波形 ---
        # data shape: (Time=1000, Channels=12)
        data, meta = wfdb.rdsamp(wfdb_path)

        # 转为 Tensor 并转置: (Channels, Time) -> (12, 1000)
        x = torch.tensor(data, dtype=torch.float32).T

        # --- C. 归一化 (Z-Score Normalization) [新功能] ---
        # 对每个导联独立计算均值和标准差，使得数据符合 N(0, 1) 分布
        # 这对 CNN 模型的收敛至关重要
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-6)  # 加 1e-6 防止除以0

        # --- D. 处理标签 ---
        y = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in row.diagnostic_superclass:
            if label in self.class_map:
                y[self.class_map[label]] = 1.0

        return x, y