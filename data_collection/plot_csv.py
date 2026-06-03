import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import interpolate
import sys
import os

FILE_PATH = "data/Data/breathing/breath_20260203_225036_realtime.csv"
WINDOW_DURATION = 10  # Xem 10 giây một lần

HAMPEL_WINDOW = 5
HAMPEL_SIGMA = 3
INTERPOLATE_DC = True

# --- 1. HÀM LỌC HAMPEL (Xử lý gai) ---
def hampel_filter(input_matrix):
    n_packets, n_subcarriers = input_matrix.shape
    output = input_matrix.copy()
    k = 1.4826 # Hệ số cho phân phối Gaussian
    
    print("   ... Đang chạy Hampel Filter (có thể mất 1-2s)...")
    for sc in range(n_subcarriers):
        series = input_matrix[:, sc]
        if np.all(np.isnan(series)): continue
        
        # Dùng Pandas rolling để tính toán nhanh hơn vòng lặp Python thuần
        s_series = pd.Series(series)
        rolling_median = s_series.rolling(window=2*HAMPEL_WINDOW+1, center=True, min_periods=1).median()
        rolling_mad = s_series.rolling(window=2*HAMPEL_WINDOW+1, center=True, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        
        threshold = HAMPEL_SIGMA * k * rolling_mad
        outlier_idx = np.abs(s_series - rolling_median) > threshold
        
        # Thay thế outlier bằng median
        output[outlier_idx, sc] = rolling_median[outlier_idx]
        
    return output

# --- 2. HÀM NỘI SUY DC (Lấp đầy lỗ trống) ---
def interpolate_dc(csi_matrix):
    n_packets, n_subcarriers = csi_matrix.shape
    x = np.arange(n_subcarriers)
    
    # Mask các vị trí không hợp lệ (Null subcarriers & DC)
    # Theo chuẩn ESP32 HT20:
    # 0-5: Guard intervals (trái)
    # 59-63: Guard intervals (phải)
    # 29-34: DC subcarriers (giữa - do FFT shift)
    mask_valid = np.ones(n_subcarriers, dtype=bool)
    mask_valid[0:6] = False
    mask_valid[59:64] = False
    mask_valid[29:35] = False
    
    output = csi_matrix.copy()
    # Xóa rìa
    output[:, 0:6] = 0
    output[:, 59:64] = 0
    
    # Nội suy vùng giữa (DC)
    for i in range(n_packets):
        y = csi_matrix[i, :]
        # Chỉ nội suy dựa trên các điểm hợp lệ
        f = interpolate.interp1d(x[mask_valid], y[mask_valid], kind='linear', fill_value="extrapolate")
        output[i, 29:35] = f(np.arange(29, 35))
        
    return output

def process_signal_chain(csi_raw):
    """Chuỗi xử lý tín hiệu: Shift -> Interp -> Hampel -> Normalize"""
    # 1. FFT Shift: Đưa tần số 0 vào giữa (quan trọng để khớp với logic subcarrier ở trên)
    shifted_matrix = np.fft.fftshift(csi_raw, axes=1)
    
    # 2. Nội suy DC
    if INTERPOLATE_DC:
        interp_matrix = interpolate_dc(shifted_matrix)
    else:
        interp_matrix = shifted_matrix.copy()
    
    # 3. Lọc nhiễu Hampel
    denoised_matrix = hampel_filter(interp_matrix)
    
    # 4. Chuẩn hóa L2 (L2 Normalization)
    # Giúp biên độ các gói tin đồng đều nhau, không bị ảnh hưởng bởi gain của bộ thu
    norms = np.linalg.norm(denoised_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1 # Tránh chia cho 0
    normalized_matrix = denoised_matrix / norms 
    
    # Gán NaN cho các vùng rìa để không vẽ lên biểu đồ
    normalized_matrix[:, 0:6] = np.nan
    normalized_matrix[:, 59:64] = np.nan
    
    return normalized_matrix


def load_data(file_path):
    print(f"🚀 Đang đọc file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Lỗi: {e}"); return None

    # Xử lý thời gian
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    
    mac_list = df['mac'].unique()
    processed_data = {}

    for mac in mac_list:
        print(f"⚙️ Đang xử lý tín hiệu cho MAC: {mac}...")
        sub_df = df[df['mac'] == mac].sort_values('timestamp')
        
        timestamps = sub_df['timestamp'].values
        # Lấy cột sub_0 -> sub_63
        csi_cols = [c for c in sub_df.columns if c.startswith('sub_')]
        csi_raw = sub_df[csi_cols].values
        
        # --- ÁP DỤNG QUY TRÌNH XỬ LÝ MỚI ---
        csi_processed = process_signal_chain(csi_raw)
        
        intervals = np.diff(timestamps, prepend=timestamps[0]) * 1000

        processed_data[mac] = {
            'time': timestamps,
            'csi': csi_processed, # Dữ liệu đã xử lý sạch sẽ
            'interval': intervals
        }
        
    return processed_data

def plot_dashboard(data_dict):
    if not data_dict: return

    num_devices = len(data_dict)
    fig, axes = plt.subplots(num_devices, 3, figsize=(18, 4 * num_devices))
    plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.25)
    
    if num_devices == 1: axes = np.expand_dims(axes, axis=0)

    plots = {}
    max_time = 0

    for i, (mac, d) in enumerate(data_dict.items()):
        t = d['time']
        csi = d['csi']
        interval = d['interval']
        max_time = max(max_time, t[-1])
        
        ax_row = axes[i]
        
        # --- CỘT 1: INTERVAL ---
        avg_freq = 1000 / np.mean(interval) if len(interval) > 0 else 0
        ax_row[0].plot(t, interval, color='#1f77b4', lw=1, alpha=0.8)
        ax_row[0].set_title(f"MAC: {mac}\nInterval (Freq: {avg_freq:.1f} Hz)", fontweight='bold', fontsize=10)
        ax_row[0].set_ylabel("ms")
        ax_row[0].set_ylim(0, 50)
        ax_row[0].grid(True, linestyle=':', alpha=0.5)

        # --- CỘT 2: HEATMAP (Đã chuẩn hóa) ---
        # Tính vmin/vmax động để heatmap có độ tương phản tốt nhất
        # Bỏ qua NaN khi tính toán
        vmin, vmax = np.nanpercentile(csi, 5), np.nanpercentile(csi, 95)
        
        im = ax_row[1].imshow(csi.T, aspect='auto', cmap='viridis', origin='lower',
                              extent=[t[0], t[-1], 0, 64], vmin=vmin, vmax=vmax)
        ax_row[1].set_title("CSI Heatmap", fontweight='bold', fontsize=10)
        ax_row[1].set_ylabel("Subcarrier Index")
        
        # --- CỘT 3: AMPLITUDE SHAPE ---
        # Vẽ hình dạng trung bình
        avg_shape = np.nanmean(csi, axis=0)
        
        # Vẽ "bóng" của toàn bộ dữ liệu (màu xanh nhạt) để thấy độ biến động
        x_axis = np.arange(64)
        # Chỉ vẽ 500 đường mẫu ngẫu nhiên để không bị lag
        step = max(1, len(csi) // 500)
        ax_row[2].plot(x_axis, csi[::step].T, color='blue', alpha=0.01)
        
        # Vẽ đường trung bình đậm
        line_shape, = ax_row[2].plot(x_axis, avg_shape, color='black', lw=2, linestyle='--')
        
        ax_row[2].set_title("Amplitude Shape", fontweight='bold', fontsize=10)
        ax_row[2].set_xlabel("Subcarrier Index")
        ax_row[2].set_xlim(0, 63)
        # Tự động scale trục Y
        y_max = np.nanmax(avg_shape) * 1.5 if not np.isnan(avg_shape).all() else 1
        ax_row[2].set_ylim(0, y_max)
        ax_row[2].grid(True, linestyle=':', alpha=0.5)

        plots[mac] = {
            'data': d,
            'ax_interval': ax_row[0],
            'ax_heatmap': ax_row[1],
            'ax_shape': ax_row[2],
            'line_shape': line_shape
        }

    # --- SLIDER ---
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='#f0f0f0')
    slider = Slider(ax_slider, 'Time Scroll', 0, max(0, max_time - WINDOW_DURATION), valinit=0)

    def update(val):
        start_t = slider.val
        end_t = start_t + WINDOW_DURATION
        
        for mac, p in plots.items():
            t = p['data']['time']
            csi = p['data']['csi']
            
            # Update trục X
            p['ax_interval'].set_xlim(start_t, end_t)
            p['ax_heatmap'].set_xlim(start_t, end_t)
            
            # Update Shape (Cột 3) theo khung nhìn
            idx_start = np.searchsorted(t, start_t)
            idx_end = np.searchsorted(t, end_t)
            
            if idx_end > idx_start:
                window_csi = csi[idx_start:idx_end]
                current_avg = np.nanmean(window_csi, axis=0)
                p['line_shape'].set_ydata(current_avg)
            
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    
    print("📊 Dashboard đã sẵn sàng!")
    plt.show()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    if os.path.exists(target):
        data = load_data(target)
        plot_dashboard(data)
    else:
        print(f"❌ Không tìm thấy file: {target}")