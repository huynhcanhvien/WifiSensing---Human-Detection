import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dpkt 
import pandas as pd
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ================= CẤU HÌNH TỐI ƯU M1 =================
FILE_TO_CHECK = "data/Data/empty/breath_20260315_181739.pcap" 

TARGET_MACS = [
    bytes.fromhex("d4e9f4a43c18"),  
    bytes.fromhex("d4e9f4a36480")
]

WINDOW_SEC = 10
TARGET_FS = 20.0
TRIM_SEC = 15.0
MAX_TIME_TO_COLLECT = TRIM_SEC + WINDOW_SEC + 5 
# =====================================================

def hampel_filter_unit(series_data):
    series = pd.Series(series_data)
    window_size = 5; n_sigmas = 3; k = 1.4826
    rolling_median = series.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = series.rolling(window=window_size, center=True, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    threshold = n_sigmas * k * rolling_mad
    outlier_idx = np.abs(series - rolling_median) > threshold
    series[outlier_idx] = rolling_median[outlier_idx]
    return series.values

def interpolate_dc_pilots(csi_matrix):
    # Null subcarriers indices theo chuẩn ESP32 (logic từ file plot.py của bạn)
    null_subcarriers = [0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34, 59, 60, 61, 62, 63]
    valid_subcarriers = [i for i in range(64) if i not in null_subcarriers]
    x = np.arange(64)
    output = np.zeros_like(csi_matrix)
    for i in range(csi_matrix.shape[0]):
        f = interpolate.interp1d(valid_subcarriers, csi_matrix[i, valid_subcarriers], kind='linear', fill_value="extrapolate")
        output[i, :] = f(x)
    return output

def parse_pcap_fast(file_path):
    print(f"🚀 M1 Pro đang giải mã: {os.path.basename(file_path)}")
    data_rows = {mac: [] for mac in TARGET_MACS}
    start_ts = None
    
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            if start_ts is None: start_ts = ts
            if ts - start_ts > MAX_TIME_TO_COLLECT: break
            
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                udp = ip.data
                if udp.dport == 5500:
                    payload = udp.data
                    mac_addr = payload[4:10]
                    if mac_addr in TARGET_MACS:
                        raw_np = np.frombuffer(payload[-256:], dtype=np.int16)
                        amp = np.sqrt(raw_np[0::2].astype(float)**2 + raw_np[1::2].astype(float)**2)
                        data_rows[mac_addr].append(np.insert(amp, 0, ts - start_ts))
            except: continue

    final_results = {}
    for mac, rows in data_rows.items():
        if len(rows) > 10:
            arr = np.array(rows)
            # 1. FFT SHIFT (Đưa DC về giữa) - QUAN TRỌNG ĐỂ HẾT XẤU
            csi_shifted = np.fft.fftshift(arr[:, 1:], axes=1)
            # 2. Nội suy DC (Lấp lỗ trống)
            csi_interp = interpolate_dc_pilots(csi_shifted)
            
            final_results[mac] = {'time': arr[:, 0], 'csi': csi_interp}
    return final_results

def process_and_plot_m1(file_path):
    raw_data = parse_pcap_fast(file_path)
    if not raw_data: return

    t_min = max(d['time'].min() for d in raw_data.values()) + TRIM_SEC
    t_max = t_min + WINDOW_SEC
    t_global = np.arange(t_min, t_max, 1.0 / TARGET_FS)
    
    processed_plots = []
    
    num_cores = multiprocessing.cpu_count()
    for mac, data in raw_data.items():
        # Lọc khung nhìn
        mask = (data['time'] >= t_min - 1) & (data['time'] <= t_max + 1)
        time_win = data['time'][mask]
        csi_win = data['csi'][mask]
        
        # 3. HAMPEL SONG SONG (Tận dụng M1)
        with ProcessPoolExecutor(max_workers=num_cores) as ex:
            clean_csi = np.array(list(ex.map(hampel_filter_unit, [csi_win[:, i] for i in range(64)]))).T
        
        # 4. Nội suy thời gian 20Hz
        f = interpolate.interp1d(time_win, clean_csi, axis=0, fill_value="extrapolate")
        processed_plots.append(f(t_global))

    # --- VẼ BIỂU ĐỒ CHUẨN ĐẸP ---
    fig = plt.figure(figsize=(16, 10))
    plt.suptitle(f"M1 PRO OPTIMIZED (DSP FIXED): {os.path.basename(file_path)}", fontsize=15, fontweight='bold')

    for i in range(2):
        ax = fig.add_subplot(2, 2, i+1)
        data = processed_plots[i].T
        # Dùng percentile 95 để heatmap không bị cháy sáng
        vmax = np.percentile(data, 95)
        sns.heatmap(data, cmap='viridis', ax=ax, cbar=True, vmin=0, vmax=vmax)
        ax.set_title(f"MAC: {TARGET_MACS[i].hex(':')}")

    # Vẽ 3D cho MAC 1
    ax3 = fig.add_subplot(2, 1, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(64))
    Z = processed_plots[0].T
    # Giới hạn trục Z để biểu đồ 3D không bị "gai" làm xấu
    z_limit = np.percentile(Z, 99)
    ax3.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax3.set_zlim(0, z_limit)
    ax3.set_title("3D View (Amplitude Capped at 99th Percentile)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_and_plot_m1(FILE_TO_CHECK)