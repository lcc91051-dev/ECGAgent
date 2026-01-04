import mne
import numpy as np
import pandas as pd
import os
from .preprocessing import preprocessing

# 设置日志级别，减少刷屏
mne.set_log_level('WARNING')


def create_raw_from_data(data, sfreq, ch_names=None):
    """
    [辅助函数] 将 numpy array (n_channels, n_samples) 包装为 mne.Raw 对象
    以便利用 mne 的滤波和重采样功能
    """
    n_channels, n_samples = data.shape
    if ch_names is None:
        ch_names = [f'Ch{i + 1}' for i in range(n_channels)]

    # 创建 info, channel type 设为 ecg
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['ecg'] * n_channels)
    raw = mne.io.RawArray(data, info)
    return raw


def dataLoad(file_path, config):
    """
    [通用入口] 加载 ECG 数据 (支持 EDF, CSV, TXT)

    Args:
        file_path: 文件路径
        config: 配置字典，必须包含 'fs' (目标采样率)。
                如果是 CSV 文件，建议在 config 中包含 'src_fs' (源采样率)，否则默认等于 'fs'
    Returns:
        numpy.ndarray: shape [channels, samples]
    """
    ext = os.path.splitext(file_path)[-1].lower()
    raw = None
    target_fs = config.get('fs', 100)  # 默认目标采样率 100Hz

    # --- A. 加载数据 ---
    if ext in ['.edf', '.bdf']:
        # 读取 EDF, 预加载数据
        # exclude=[] 确保读取所有通道，不做剔除
        raw = mne.io.read_raw_edf(file_path, preload=True, exclude=[])

    elif ext in ['.csv', '.txt']:
        # 读取 CSV
        try:
            # 1. 尝试读取 (假设第一行是表头)
            df = pd.read_csv(file_path)

            # 简单的启发式检查：如果所有列都是数字，可能没有表头
            # 这里默认假设：Standard CSV (Rows=Samples, Cols=Channels)
            try:
                # 尝试转 float，如果第一行是列名这里会报错 (或者变成 NaN)
                data_check = df.values.astype(float)
                ch_names = list(df.columns)
                data = data_check.T  # 转置为 (Channels, Samples)
            except:
                # 读取失败，说明可能有非数字表头，上面的 read_csv 已经正确处理了表头
                # 如果是纯文本无表头：
                df = pd.read_csv(file_path, header=None)
                data = df.values.astype(float).T
                ch_names = [f"Ch{i}" for i in range(len(df.columns))]

            # CSV 这种文本格式通常不带采样率信息
            # 如果 config 里没写 'src_fs'，我们暂且认为它不需要重采样 (src=target)
            src_fs = config.get('src_fs', target_fs)

            raw = create_raw_from_data(data, src_fs, ch_names)

        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {str(e)}")

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # --- B. 预处理 (复用 preprocessing.py) ---
    # 这会执行: 滤波(Filter) -> 陷波(Notch) -> 重采样(Resample)
    if raw is not None:
        raw = preprocessing(raw, config)

    # --- C. 返回数据 ---
    # 返回 float32 格式的 numpy 数组
    return raw.get_data().astype(np.float32)


# 兼容性接口：保留旧函数名指向新逻辑，防止 Agent 其他地方报错
load_MDD_edf = dataLoad
load_Sleep_edf = dataLoad