import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dpkt # Thư viện parsing siêu nhanh
import pandas as pd
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ================= CẤU HÌNH TỐI ƯU M1 =================
FILE_TO_CHECK = "data/Data/empty/breath_20260315_181739.pcap" 

TARGET_MACS = [
    bytes.fromhex("d4e9f4a36480"), # Để dạng bytes để so sánh cực nhanh
    bytes.fromhex("d4e9f4a43c18")
]

WINDOW_SEC = 10
TARGET_FS = 20.0
TRIM_SEC = 15.0
# Chỉ đọc đủ dữ liệu để plot (Ví dụ: 15s trim + 10s window + 5s dư)
MAX_TIME_TO_COLLECT = TRIM_SEC + WINDOW_SEC + 5 
# =====================================================

def hampel_unit(series_data):
    """Xử lý Hampel cho 1 subcarrier đơn lẻ"""
    series = pd.Series(series_data)
    window_size = 5
    n_sigmas = 3
    rolling_median = series.rolling(window=2*window_size+1, center=True, min_periods=1).median()
    rolling_mad = series.rolling(window=2*window_size+1, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    threshold = n_sigmas * 1.4826 * rolling_mad
    outlier_idx = np.abs(series - rolling_median) > threshold
    series[outlier_idx] = rolling_median[outlier_idx]
    return series.values

def parallel_hampel(input_matrix):
    """Chia việc cho các nhân M1 xử lý Hampel song song"""
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map từng cột (subcarrier) vào các nhân
        results = list(executor.map(hampel_unit, [input_matrix[:, i] for i in range(input_matrix.shape[1])]))
    return np.array(results).T

def parse_pcap_fast(file_path):
    print(f"🚀 M1 Pro đang 'nuốt' file: {file_path}")
    data_rows = {mac: [] for mac in TARGET_MACS}
    start_ts = None
    
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            if start_ts is None: start_ts = ts
            
            # Thoát sớm nếu đã đủ dữ liệu để plot, tránh đọc hết 3GB
            if ts - start_ts > MAX_TIME_TO_COLLECT:
                print(f"✅ Đã thu đủ {MAX_TIME_TO_COLLECT}s dữ liệu mẫu. Dừng đọc để tiết kiệm thời gian.")
                break
            
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                # Kiểm tra nhanh gói UDP port 5500
                if isinstance(eth.data, dpkt.ip.IP):
                    ip = eth.data
                    if isinstance(ip.data, dpkt.udp.UDP):
                        udp = ip.data
                        if udp.dport == 5500:
                            payload = udp.data
                            if len(payload) >= 274:
                                mac_addr = payload[4:10]
                                if mac_addr in TARGET_MACS:
                                    # Giải mã CSI nhị phân
                                    raw_np = np.frombuffer(payload[-256:], dtype=np.int16)
                                    # Vector hóa tính biên độ
                                    real = raw_np[0::2].astype(float)
                                    imag = raw_np[1::2].astype(float)
                                    amp = np.sqrt(real**2 + imag**2)
                                    
                                    data_rows[mac_addr].append(np.insert(amp, 0, ts - start_ts))
            except: continue

    # Chuyển đổi sang DataFrame
    final_dfs = {}
    for mac, rows in data_rows.items():
        if rows:
            arr = np.array(rows)
            cols = ['timestamp'] + [f'sub_{i}' for i in range(64)]
            final_dfs[mac] = pd.DataFrame(arr, columns=cols)
    return final_dfs

def process_and_plot_m1(file_path):
    dfs = parse_pcap_fast(file_path)
    if not dfs or len(dfs) < 2:
        print("❌ Không thu thập đủ dữ liệu từ 2 MAC.")
        return

    t_min = max(df['timestamp'].min() for df in dfs.values()) + TRIM_SEC
    t_max = t_min + WINDOW_SEC
    t_global = np.arange(t_min, t_max, 1.0 / TARGET_FS)
    
    processed_macs = []
    mac_names = []

    for mac, df in dfs.items():
        # Lọc dữ liệu trong khung nhìn
        sub_df = df[(df['timestamp'] >= t_min - 1) & (df['timestamp'] <= t_max + 1)]
        if len(sub_df) < 10: continue

        # DSP song song trên M1
        csi_raw = sub_df[[f'sub_{i}' for i in range(6, 59)]].values
        print(f"⚙️ Đang lọc Hampel song song cho MAC {mac.hex(':')[-5:]}...")
        csi_denoised = parallel_hampel(csi_raw)
        
        # Nội suy
        f = interpolate.interp1d(sub_df['timestamp'].values, csi_denoised, axis=0, fill_value="extrapolate")
        csi_resampled = f(t_global)
        processed_macs.append(csi_resampled)
        mac_names.append(mac.hex(':'))

    # Vẽ biểu đồ
    fig = plt.figure(figsize=(16, 10))
    plt.suptitle(f"M1 PRO OPTIMIZED ANALYSIS: {os.path.basename(file_path)}", fontsize=16)

    for i in range(2):
        ax = fig.add_subplot(2, 2, i+1)
        sns.heatmap(processed_macs[i].T, cmap='viridis' if i==0 else 'magma', ax=ax, cbar=False)
        ax.set_title(f"MAC: {mac_names[i]}")

    ax3 = fig.add_subplot(2, 1, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(len(t_global)), np.arange(53))
    ax3.plot_surface(X, Y, processed_macs[0].T, cmap='viridis', edgecolor='none', alpha=0.8)
    ax3.set_title("3D View (10s Window)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_and_plot_m1(FILE_TO_CHECK)