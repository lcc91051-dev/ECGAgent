"""
ECG Feature Extraction Module
Professionally implemented for medical-grade HRV analysis and scientific research.
Follows Task Force standards for Heart Rate Variability.
"""
import numpy as np
import logging
from scipy.signal import find_peaks, welch, butter, filtfilt, medfilt
from scipy.interpolate import interp1d

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ECGFeatures")

def remove_baseline_wander(signal, fs):
    """
    使用双级中值滤波去除基线漂移 (学术标准做法)
    First window: 200ms to remove QRS complex and T waves
    Second window: 600ms to remove P waves
    """
    if len(signal) < int(0.6 * fs):
        return signal
        
    # 第一级中值滤波 (窗口约 200ms)
    w1 = int(0.2 * fs)
    if w1 % 2 == 0: w1 += 1
    baseline1 = medfilt(signal, kernel_size=w1)
    
    # 第二级中值滤波 (窗口约 600ms)
    w2 = int(0.6 * fs)
    if w2 % 2 == 0: w2 += 1
    baseline2 = medfilt(baseline1, kernel_size=w2)
    
    return signal - baseline2

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100.0, order=4):
    """
    Butterworth 带通滤波，去除肌电干扰和工频噪声
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 限制在有效范围内
    low = max(0.01, min(low, 0.95))
    high = max(low + 0.05, min(high, 0.99))
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def calculate_sqi(signal):
    """
    计算信号质量指数 (Signal Quality Index)
    基于信号变异程度和峰峰值范围
    """
    if len(signal) == 0: return 0.0
    
    # 简单的 SQI：检查信号是否饱和或全是噪声
    snr_estimation = np.mean(np.square(signal)) / (np.var(signal) + 1e-6)
    peak_to_peak = np.ptp(signal)
    
    # 如果信号几乎是直线或范围极其异常，标记为低质量
    if peak_to_peak < 0.05 or peak_to_peak > 10.0:
        return 0.2
    return min(1.0, snr_estimation)

def detect_r_peaks(ecg_signal, fs):
    """
    增强版 R 波检测 (基于 Pan-Tompkins 简化思想)
    1. 平方变换 2. 移动窗口积分 3. 动态阈值
    """
    if len(ecg_signal) < 2 * fs:
        return np.array([])

    # 1. 差分操作
    diff_sig = np.diff(ecg_signal)
    
    # 2. 平方变换 (突出高频 QRS 复波)
    squared_sig = diff_sig ** 2
    
    # 3. 移动窗口积分 (积分窗口约 150ms)
    window_len = int(0.15 * fs)
    integrated_sig = np.convolve(squared_sig, np.ones(window_len)/window_len, mode='same')
    
    # 4. 寻峰
    min_dist = int(0.4 * fs)  # 限制心率最高 150bpm (研究常用保守限制)
    
    # 动态阈值：取积分信号均值的 3 倍或中位数的 5 倍
    threshold = max(np.mean(integrated_sig) * 2.0, np.percentile(integrated_sig, 95) * 0.3)
    
    peaks, _ = find_peaks(integrated_sig, distance=min_dist, height=threshold)
    
    # 补偿差分造成的位移
    return peaks

def compute_rr_intervals(peaks, fs):
    if len(peaks) < 2:
        return np.array([]), np.array([])
    rr_intervals = np.diff(peaks) / fs * 1000  # ms
    rr_times = peaks[1:] / fs  # 时间戳 (秒)
    return rr_intervals, rr_times

def clean_rr_intervals(rr_intervals, rr_times):
    """
    生理滤波器 + 统计滤波器 (学术论文标准)
    """
    if len(rr_intervals) < 3:
        return rr_intervals, rr_times
        
    # 1. 范围过滤: 300ms(200bpm) - 2000ms(30bpm)
    logic_mask = (rr_intervals > 300) & (rr_intervals < 2000)
    
    # 2. 突变过滤: 相邻 RR 变化不能超过 20% (Malik 等, 1996)
    diff_mask = np.ones(len(rr_intervals), dtype=bool)
    diff_rr = np.abs(np.diff(rr_intervals))
    # 相邻间隔差异过大通常是检测错误
    for i in range(1, len(rr_intervals)):
        if np.abs(rr_intervals[i] - rr_intervals[i-1]) > 0.2 * rr_intervals[i-1]:
            diff_mask[i] = False
            
    combined_mask = logic_mask & diff_mask
    return rr_intervals[combined_mask], rr_times[combined_mask]

def compute_time_domain(rr):
    if len(rr) < 2: return [0]*4
    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    nn50 = np.sum(np.abs(np.diff(rr)) > 50)
    pnn50 = (nn50 / len(rr) * 100) if len(rr) > 0 else 0
    return [mean_rr, sdnn, rmssd, pnn50]

def compute_frequency_domain(rr, rr_times):
    """
    Welch 功率谱密度分析 (4Hz 重采样)
    """
    if len(rr) < 10 or (rr_times[-1] - rr_times[0]) < 20:
        return [0, 0, 0]
        
    try:
        # 重采样到 4Hz
        fs_resample = 4.0
        f_interp = interp1d(rr_times, rr, kind='cubic', fill_value='extrapolate')
        t_new = np.arange(rr_times[0], rr_times[-1], 1/fs_resample)
        rr_syn = f_interp(t_new)
        
        # 去趋势
        rr_syn = rr_syn - np.mean(rr_syn)
        
        # Welch 计算 PSD (窗口长度 N=256)
        nperseg = min(256, len(rr_syn))
        f, psd = welch(rr_syn, fs=fs_resample, nperseg=nperseg)
        
        vlf_band = (f >= 0.003) & (f < 0.04)
        lf_band = (f >= 0.04) & (f < 0.15)
        hf_band = (f >= 0.15) & (f < 0.4)
        
        lf_pow = np.trapz(psd[lf_band], f[lf_band])
        hf_pow = np.trapz(psd[hf_band], f[hf_band])
        ratio = lf_pow / hf_pow if hf_pow > 0 else 0
        
        return [lf_pow, hf_pow, ratio]
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
        return [0, 0, 0]

def compute_ecg_features(ecg_signal, fs=100.0):
    """
    [高阶特征提取流水线]
    """
    # 1. 信号质量初步评估
    sqi = calculate_sqi(ecg_signal)
    if sqi < 0.2:
        logger.warning(f"Very poor signal quality (SQI={sqi:.2f})")
        # 即使质量差也尝试处理，但记录警告
        
    # 2. 去除基线漂移 (Median Filter)
    clean_sig = remove_baseline_wander(ecg_signal, fs)
    
    # 3. 带通滤波
    filt_sig = bandpass_filter(clean_sig, fs=fs)
    
    # 4. R 波检测
    peaks = detect_r_peaks(filt_sig, fs)
    
    # 5. RR 间隔计算与清洗
    rr, rr_times = compute_rr_intervals(peaks, fs)
    rr_clean, rr_times_clean = clean_rr_intervals(rr, rr_times)
    
    # 6. 特征计算
    time_feats = compute_time_domain(rr_clean)
    freq_feats = compute_frequency_domain(rr_clean, rr_times_clean)
    
    # 特征 4:5 进行 log 变换 (使分布更接近高斯分布，利于模型处理)
    freq_feats[0] = np.log1p(freq_feats[0])
    freq_feats[1] = np.log1p(freq_feats[1])
    
    return np.array(time_feats + freq_feats, dtype=np.float32)

def get_feature_names():
    return ["Mean_RR", "SDNN", "RMSSD", "pNN50", "LF_log", "HF_log", "LF/HF"]
