import numpy as np
from scipy.signal import find_peaks, welch


def compute_ecg_features(ecg_signal, fs=700):
    """
    [核心工具] 输入一段 ECG 信号，计算生理特征向量。

    Args:
        ecg_signal (array): 原始 ECG 信号 (1D numpy array)
        fs (int): 采样率 (WESAD 默认为 700Hz)

    Returns:
        features (np.array): 7维特征向量
        [Mean_RR, Std_RR, RMSSD, pNN50, LF, HF, LF/HF]
    """
    # 1. 预处理：简单的归一化，防止幅度差异过大
    if len(ecg_signal) == 0:
        return np.zeros(7, dtype=np.float32)

    # 2. R波峰值检测
    # distance: 假设心率上限 200bpm -> R波间隔至少 0.3秒
    # height: 简单的自适应阈值
    min_dist = int(0.3 * fs)
    peaks, _ = find_peaks(ecg_signal, distance=min_dist, height=np.mean(ecg_signal))

    if len(peaks) < 2:
        # 如果找不到足够的波峰，返回零向量
        return np.zeros(7, dtype=np.float32)

    # 3. 计算 RR 间隔 (毫秒)
    rr_intervals = np.diff(peaks) / fs * 1000  # ms

    # --- 时域特征 (Time Domain) ---
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)

    # RMSSD: 均方根差 (反映副交感神经活性，压力大时降低)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else 0

    # pNN50: 相邻RR间隔差值超过 50ms 的占比
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(rr_intervals) if len(rr_intervals) > 0 else 0

    # --- 频域特征 (Frequency Domain) ---
    # 使用 Welch 法估计功率谱密度
    # 注意：直接对原始波形做 Welch 是为了简化工程。
    # 学术上通常应对 RR 间隔插值后做 FFT，但在深度学习特征工程中，
    # 原始波形的频域能量分布也能反映很多信息。
    f, Pxx = welch(ecg_signal, fs=fs, nperseg=fs * 10)  # 10秒一个窗

    # 定义频带: LF (0.04-0.15Hz), HF (0.15-0.4Hz)
    lf_band = (f >= 0.04) & (f < 0.15)
    hf_band = (f >= 0.15) & (f < 0.40)

    lf_power = np.trapz(Pxx[lf_band], f[lf_band])
    hf_power = np.trapz(Pxx[hf_band], f[hf_band])

    if hf_power == 0:
        lf_hf_ratio = 0
    else:
        lf_hf_ratio = lf_power / hf_power

    # 4. 组装特征
    features = np.array([
        mean_rr,
        std_rr,
        rmssd,
        pnn50,
        lf_power,
        hf_power,
        lf_hf_ratio
    ], dtype=np.float32)

    # 对部分数值范围大的特征做 Log 压缩，利于神经网络学习
    # 尤其是 Power 和 Ratio 这种可能很大的值
    features[4:] = np.log1p(features[4:])

    return features