import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks, welch


# --- 原有的预处理函数 (保持不变) ---
def preprocessing(raw, config):
    fs = config.get('fs', 100)
    l_freq = config.get('l_freq', 0.5)
    h_freq = config.get('h_freq', 50)
    notch_freq = config.get('notch_freq', None)  # 50 or 60

    # 1. 滤波
    # 注意：MNE 的 filter 函数默认使用 FIR，需要数据长度足够
    try:
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        if notch_freq is not None:
            raw.notch_filter(freqs=notch_freq, verbose=False)
    except Exception as e:
        print(f"Warning during filtering: {e}")

    # 2. 重采样
    if raw.info['sfreq'] != fs:
        raw = raw.resample(sfreq=fs, npad='auto', verbose=False)

    return raw


# --- 新增：ECG 特征提取函数 (来自 ecg_features.py) ---
def compute_ecg_features(ecg_signal, fs=100):
    """
    输入一段 1D ECG 信号，计算 7 维生理特征向量。
    Returns: [Mean_RR, Std_RR, RMSSD, pNN50, LF, HF, LF/HF]
    """
    if len(ecg_signal) == 0:
        return np.zeros(7, dtype=np.float32)

    # 1. R波峰值检测
    # distance: 假设最大心率 200bpm -> 最小间隔 0.3s
    min_dist = int(0.3 * fs)
    # height: 自适应阈值，大于均值即可 (简单的 R 波检测)
    peaks, _ = find_peaks(ecg_signal, distance=min_dist, height=np.mean(ecg_signal))

    if len(peaks) < 2:
        return np.zeros(7, dtype=np.float32)

    # 2. 计算 RR 间隔 (毫秒)
    rr_intervals = np.diff(peaks) / fs * 1000  # ms

    # --- 时域特征 ---
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)

    # RMSSD
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else 0

    # pNN50
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = nn50 / len(rr_intervals) if len(rr_intervals) > 0 else 0

    # --- 频域特征 (Welch) ---
    # 对原始信号做 PSD (简化版，严格学术应做 RR 间隔插值)
    f, Pxx = welch(ecg_signal, fs=fs, nperseg=min(len(ecg_signal), 256))

    # LF: 0.04 - 0.15 Hz, HF: 0.15 - 0.4 Hz
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f < 0.4)

    lf_pow = np.trapz(Pxx[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0
    hf_pow = np.trapz(Pxx[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0

    lf_hf_ratio = lf_pow / hf_pow if hf_pow > 0 else 0

    return np.array([mean_rr, std_rr, rmssd, pnn50, lf_pow, hf_pow, lf_hf_ratio], dtype=np.float32)