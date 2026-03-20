import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

# ================= 1. CẤU HÌNH ĐƯỜNG DẪN =================
# Điền đường dẫn đến 1 file CSV cụ thể mà bạn muốn test
TEST_FILE_PATH = "data/Data/breathing/breath_20260203_215850_realtime.csv"

# ================= 2. CẤU HÌNH DSP & AI (GIỮ NGUYÊN BẢN CHUẨN) =================
WINDOW_SEC = 10           # Cửa sổ 10 giây
TARGET_FS = 20.0          # Ép về 20Hz
MIN_FS_THRESHOLD = 15.0   # Ngưỡng rớt mạng
OVERLAP_PCT = 0.5         # Trượt 50%
TRIM_SEC = 15.0           # Cắt bỏ 15s đầu và 15s cuối

HAMPEL_WINDOW = 5
HAMPEL_SIGMA = 3
INTERPOLATE_DC = True
# ==============================================================================

# --- CÁC HÀM TIỀN XỬ LÝ LÕI ---
def hampel_filter(input_matrix):
    output = input_matrix.copy()
    for sc in range(input_matrix.shape[1]):
        series = input_matrix[:, sc]
        if np.all(np.isnan(series)): continue
        s_series = pd.Series(series)
        rolling_median = s_series.rolling(window=2*HAMPEL_WINDOW+1, center=True, min_periods=1).median()
        rolling_mad = s_series.rolling(window=2*HAMPEL_WINDOW+1, center=True, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        threshold = HAMPEL_SIGMA * 1.4826 * rolling_mad
        outlier_idx = np.abs(s_series - rolling_median) > threshold
        output[outlier_idx, sc] = rolling_median[outlier_idx]
    return output

def interpolate_dc(csi_matrix):
    output = csi_matrix.copy()
    mask_valid = np.ones(64, dtype=bool)
    mask_valid[0:6] = False; mask_valid[59:64] = False; mask_valid[29:35] = False
    output[:, 0:6] = 0; output[:, 59:64] = 0
    for i in range(csi_matrix.shape[0]):
        f = interpolate.interp1d(np.arange(64)[mask_valid], csi_matrix[i, mask_valid], kind='linear', fill_value="extrapolate")
        output[i, 29:35] = f(np.arange(29, 35))
    return output

def process_signal_chain(csi_raw):
    """FFT Shift -> Interp DC -> Hampel -> L2 Norm"""
    shifted_matrix = np.fft.fftshift(csi_raw, axes=1)
    interp_matrix = interpolate_dc(shifted_matrix) if INTERPOLATE_DC else shifted_matrix.copy()
    denoised_matrix = hampel_filter(interp_matrix)
    norms = np.linalg.norm(denoised_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    normalized_matrix = denoised_matrix / norms
    normalized_matrix[:, 0:6] = np.nan
    normalized_matrix[:, 59:64] = np.nan
    return normalized_matrix

def sliding_window(csi_matrix, fs, window_sec, overlap_pct):
    """Cắt cửa sổ trượt"""
    window_samples = int(fs * window_sec)
    step_samples = int(window_samples * (1.0 - overlap_pct))
    num_samples = csi_matrix.shape[0]
    windows = []
    for start in range(0, num_samples - window_samples + 1, step_samples):
        end = start + window_samples
        windows.append(csi_matrix[start:end, :])
    return np.array(windows)

# --- HÀM XỬ LÝ 1 FILE VÀ TRẢ VỀ TENSOR ---
def inspect_single_file(file_path):
    print(f"🚀 BẮT ĐẦU XỬ LÝ FILE: {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file tại: {file_path}")
        return None, None

    try: df = pd.read_csv(file_path)
    except Exception as e: 
        print(f"❌ Lỗi đọc file CSV: {e}")
        return None, None
        
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        
    t_max = df['timestamp'].max()
    print(f"   -> Thời lượng file gốc: {t_max:.2f} giây")
    
    min_required_time = (TRIM_SEC * 2) + WINDOW_SEC
    if t_max < min_required_time: 
        print(f"❌ File quá ngắn ({t_max:.1f}s). Cần tối thiểu {min_required_time}s (Cắt đầu đuôi {TRIM_SEC}s + 1 frame {WINDOW_SEC}s).")
        return None, None
        
    # Gọt 15s đầu/cuối
    df = df[(df['timestamp'] >= TRIM_SEC) & (df['timestamp'] <= (t_max - TRIM_SEC))]
    print(f"   -> Thời lượng sau khi gọt {TRIM_SEC}s đầu/cuối: {df['timestamp'].max() - df['timestamp'].min():.2f} giây")
    
    mac_list = df['mac'].unique()
    if len(mac_list) != 2: 
        print(f"❌ File không đủ chuẩn MIMO (Chỉ có {len(mac_list)} MAC).")
        return None, None
    
    # Kiểm định mạng
    duration = df['timestamp'].max() - df['timestamp'].min()
    for mac in mac_list:
        packet_count = len(df[df['mac'] == mac])
        actual_fs = packet_count / duration
        print(f"   -> Tần số thu thực tế (MAC {mac[-5:]}): {actual_fs:.2f} Hz")
        if actual_fs < MIN_FS_THRESHOLD:
            print(f"❌ LỖI: Tần số quá thấp (< {MIN_FS_THRESHOLD}Hz), mạng bị lag nặng!")
            return None, None
            
    t_starts = [df[df['mac'] == mac]['timestamp'].min() for mac in mac_list]
    t_ends   = [df[df['mac'] == mac]['timestamp'].max() for mac in mac_list]
    t_global = np.arange(max(t_starts), min(t_ends), 1.0 / TARGET_FS)
    
    list_of_mac_tensors = []
    
    for i, mac in enumerate(mac_list):
        print(f"⚙️ Đang xử lý tín hiệu cho MAC {i+1}...")
        sub_df = df[df['mac'] == mac].sort_values('timestamp')
        timestamps = sub_df['timestamp'].values
        csi_raw = sub_df[[c for c in sub_df.columns if c.startswith('sub_')]].values
        
        csi_clean = np.nan_to_num(process_signal_chain(csi_raw), nan=0.0)
        csi_effective = csi_clean[:, 6:59] 
        
        # Resample về 20Hz
        f_interp = interpolate.interp1d(timestamps, csi_effective, axis=0, kind='linear', fill_value="extrapolate")
        csi_resampled = f_interp(t_global)
        
        # Cắt frame
        x_windows_mac = sliding_window(csi_resampled, TARGET_FS, WINDOW_SEC, OVERLAP_PCT)
        list_of_mac_tensors.append(x_windows_mac)

    X_combined = np.stack(list_of_mac_tensors, axis=-1)
    
    print("\n" + "="*50)
    print(f"✅ XỬ LÝ HOÀN TẤT!")
    print(f"📊 Kích thước Tensor đầu ra : {X_combined.shape}")
    print(f"   - Số lượng Frame         : {X_combined.shape[0]}")
    print(f"   - Chiều dài Time-steps   : {X_combined.shape[1]} ({WINDOW_SEC}s @ {TARGET_FS}Hz)")
    print(f"   - Số lượng Subcarriers   : {X_combined.shape[2]}")
    print(f"   - Số lượng MACs          : {X_combined.shape[3]}")
    
    return X_combined, mac_list

# --- HÀM VẼ BIỂU ĐỒ (PLOT) CÁC MẪU FRAME ---
def plot_random_frames(X_tensor, mac_list, num_samples=2):
    """Bốc ngẫu nhiên vài Frame để vẽ Heatmap cho cả 2 kênh MAC"""
    if X_tensor is None or len(X_tensor) == 0:
        return
        
    num_total_frames = X_tensor.shape[0]
    # Lấy ngẫu nhiên vài index để vẽ
    sample_indices = random.sample(range(num_total_frames), min(num_samples, num_total_frames))
    
    print(f"\n🎨 Đang vẽ {len(sample_indices)} Frame ngẫu nhiên (Vui lòng đóng cửa sổ hình ảnh để xem hình tiếp theo)...")
    
    for idx in sample_indices:
        frame_data = X_tensor[idx] # Kích thước: (200, 53, 2)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"Mẫu kiểm tra - Frame #{idx} (Độ dài: {WINDOW_SEC}s)", fontsize=14, fontweight='bold')
        
        for mac_idx in range(2):
            mac_name = mac_list[mac_idx]
            # frame_data[:, :, mac_idx] có shape (200, 53).
            # Chuyển vị (.T) để trục X là Thời gian (200), trục Y là Subcarrier (53)
            data_to_plot = frame_data[:, :, mac_idx].T 
            
            # Tính giới hạn màu sắc bỏ qua NaN
            vmin = np.nanpercentile(data_to_plot, 2)
            vmax = np.nanpercentile(data_to_plot, 98)
            
            im = axes[mac_idx].imshow(data_to_plot, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axes[mac_idx].set_title(f"MAC: {mac_name}", fontsize=11)
            axes[mac_idx].set_ylabel("Subcarrier Index (53)")
            axes[mac_idx].set_xlabel("Time Steps (200 steps = 10 giây)")
            plt.colorbar(im, ax=axes[mac_idx], label="Amplitude (L2 Norm)")
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Thay TEST_FILE_PATH ở trên cùng bằng đường dẫn thực tế của bạn
    tensor_result, macs = inspect_single_file(TEST_FILE_PATH)
    
    if tensor_result is not None:
        # Vẽ 3 frame ngẫu nhiên
        plot_random_frames(tensor_result, macs, num_samples=3)