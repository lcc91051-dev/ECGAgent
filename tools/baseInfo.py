import os
import pandas as pd


def baseInfo(file_path):
    """
    获取文件的基本元数据 (支持 EDF header 和 CSV 基本信息)
    """
    ext = os.path.splitext(file_path)[-1].lower()
    minimal_info = {}

    if ext in ['.edf', '.bdf']:
        # --- EDF 处理逻辑 (保留原逻辑，增加容错) ---
        try:
            with open(file_path, 'rb') as f:
                header_bytes = f.read(256)

            if len(header_bytes) >= 256:
                # 解析 EDF Header
                # patient_id = header_bytes[8:88].decode('ascii').strip()
                startdate = header_bytes[168:176].decode('ascii').strip()
                starttime = header_bytes[176:184].decode('ascii').strip()
                num_data_records = float(header_bytes[236:244].decode('ascii').strip())
                duration_of_record = float(header_bytes[244:252].decode('ascii').strip())

                total_seconds = num_data_records * duration_of_record

                minimal_info = {
                    "type": "EDF",
                    "start_date": startdate,
                    "start_time": starttime,
                    "duration_sec": f"{total_seconds:.2f}",
                }
        except Exception as e:
            minimal_info = {"error": f"EDF parse error: {str(e)}"}

    elif ext in ['.csv', '.txt']:
        # --- CSV 处理逻辑 ---
        try:
            # 只读前几行和文件大小，避免大文件读取太慢
            df_head = pd.read_csv(file_path, nrows=5)
            file_stats = os.stat(file_path)

            # 粗略估计行数 (如果文件不大，可以直接读 len)
            # 这里为了性能，我们只返回文件大小和列数
            minimal_info = {
                "type": "CSV",
                "file_size_mb": f"{file_stats.st_size / (1024 * 1024):.2f} MB",
                "num_channels": df_head.shape[1],
                "columns": list(df_head.columns),
                "note": "Sampling rate depends on device configuration."
            }
        except Exception as e:
            minimal_info = {"error": f"CSV parse error: {str(e)}"}

    return minimal_info


def get_age_factor(age, age_factors):
    # 保留原逻辑
    for factor in age_factors:
        if factor["min_age"] <= age < factor["max_age"]:
            return factor
    return None