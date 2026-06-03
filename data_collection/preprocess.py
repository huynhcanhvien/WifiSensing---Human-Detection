"""
data_collection/preprocess.py — Tiền xử lý và kiểm tra file CSV realtime.

Script dùng để xử lý file CSV thu thập từ receiver.sh và kiểm tra chất lượng dữ liệu.
DSP logic được import từ src.dsp (không nhân bản code nữa).

Cách dùng:
    python data_collection/preprocess.py
    python data_collection/preprocess.py --file data/Data/breathing/breath_20260203_215850_realtime.csv

Cấu hình nhanh (tham số command line hoặc chỉnh trực tiếp DEFAULT_* bên dưới):
    WINDOW_SEC = 10    — độ dài cửa sổ (giây)
    TARGET_FS  = 20    — tần số resample (Hz)
    TRIM_SEC   = 15    — cắt bỏ đầu/cuối để loại nhiễu
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots, resample_to_grid

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_FILE     = "data/Data/breathing/breath_20260203_215850_realtime.csv"
DEFAULT_WINDOW   = 10.0    # giây
DEFAULT_FS       = 20.0    # Hz
DEFAULT_MIN_FS   = 15.0    # Hz — ngưỡng rớt mạng
DEFAULT_OVERLAP  = 0.5
DEFAULT_TRIM     = 15.0    # giây


# ─── Core Processing ──────────────────────────────────────────────────────────

def inspect_single_file(
    file_path: str,
    window_sec: float = DEFAULT_WINDOW,
    target_fs: float  = DEFAULT_FS,
    min_fs: float     = DEFAULT_MIN_FS,
    overlap: float    = DEFAULT_OVERLAP,
    trim_sec: float   = DEFAULT_TRIM,
) -> tuple[np.ndarray | None, list | None]:
    """
    Đọc và xử lý 1 file CSV realtime.

    Args:
        file_path: đường dẫn file CSV
        window_sec, target_fs, min_fs, overlap, trim_sec: tham số DSP

    Returns:
        (X_combined, mac_list) hoặc (None, None) nếu lỗi
        X_combined shape: (N_frames, T, 53, N_macs)
    """
    print(f"🚀 XỬ LÝ FILE: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return None, None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Lỗi đọc CSV: {e}")
        return None, None

    # Parse timestamp
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    t_max = df['timestamp'].max()
    print(f"   → Thời lượng gốc: {t_max:.2f}s")

    min_required = (trim_sec * 2) + window_sec
    if t_max < min_required:
        print(f"❌ File quá ngắn ({t_max:.1f}s). Cần tối thiểu {min_required:.1f}s.")
        return None, None

    # Trim
    df = df[(df['timestamp'] >= trim_sec) & (df['timestamp'] <= (t_max - trim_sec))]
    duration = df['timestamp'].max() - df['timestamp'].min()
    print(f"   → Sau trim {trim_sec}s: {duration:.2f}s")

    mac_list = df['mac'].unique()
    if len(mac_list) != 2:
        print(f"❌ Không đủ MIMO: chỉ có {len(mac_list)} MAC.")
        return None, None

    # Kiểm tra tần số
    for mac in mac_list:
        n_packets = len(df[df['mac'] == mac])
        actual_fs = n_packets / duration
        print(f"   → MAC {mac[-5:]}: {actual_fs:.2f} Hz")
        if actual_fs < min_fs:
            print(f"❌ Tần số quá thấp (< {min_fs}Hz), mạng bị lag!")
            return None, None

    # Lưới thời gian chung
    t_starts = [df[df['mac'] == mac]['timestamp'].min() for mac in mac_list]
    t_ends   = [df[df['mac'] == mac]['timestamp'].max() for mac in mac_list]
    t_global = np.arange(max(t_starts), min(t_ends), 1.0 / target_fs)

    mac_windows = []

    for i, mac in enumerate(mac_list):
        print(f"⚙️  Xử lý MAC {i+1}/{len(mac_list)}: {mac}")
        sub_df = df[df['mac'] == mac].sort_values('timestamp')
        t_mac  = sub_df['timestamp'].values
        sc_cols = [c for c in sub_df.columns if c.startswith('sub_')]
        csi_raw = sub_df[sc_cols].values.astype(float)   # (T, 64)

        # DSP Pipeline (import từ src.dsp — không nhân bản)
        csi = fft_shift(csi_raw)
        csi = interpolate_dc_pilots(csi)
        csi = hampel_filter(csi)
        csi = np.nan_to_num(csi, nan=0.0)
        csi = csi[:, 6:59]   # (T, 53)

        norms = np.linalg.norm(csi, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        csi = csi / norms

        csi_rs = resample_to_grid(csi, t_mac, t_global)   # (T_new, 53)

        # Sliding window
        win_samples = int(target_fs * window_sec)
        step        = int(win_samples * (1.0 - overlap))
        windows = [
            csi_rs[s : s + win_samples]
            for s in range(0, len(csi_rs) - win_samples + 1, step)
        ]
        mac_windows.append(np.array(windows))

    X_combined = np.stack(mac_windows, axis=-1)   # (N, T, 53, 2)

    print("\n" + "=" * 50)
    print(f"✅ HOÀN TẤT!")
    print(f"   Tensor shape : {X_combined.shape}")
    print(f"   N Frames     : {X_combined.shape[0]}")
    print(f"   Time steps   : {X_combined.shape[1]} ({window_sec}s @ {target_fs}Hz)")
    print(f"   Subcarriers  : {X_combined.shape[2]}")
    print(f"   MACs         : {X_combined.shape[3]}")
    return X_combined, list(mac_list)


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_random_frames(X: np.ndarray, mac_list: list, num_samples: int = 2):
    """Vẽ heatmap ngẫu nhiên để kiểm tra trực quan."""
    if X is None or len(X) == 0:
        return

    indices = random.sample(range(X.shape[0]), min(num_samples, X.shape[0]))
    print(f"\n🎨 Vẽ {len(indices)} frame ngẫu nhiên...")

    for idx in indices:
        frame = X[idx]   # (T, 53, 2)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"Frame #{idx}", fontsize=14, fontweight='bold')

        for mac_idx in range(2):
            data  = frame[:, :, mac_idx].T
            vmin  = np.nanpercentile(data, 2)
            vmax  = np.nanpercentile(data, 98)
            im    = axes[mac_idx].imshow(data, aspect='auto', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axes[mac_idx].set_title(f"MAC: {mac_list[mac_idx]}")
            axes[mac_idx].set_ylabel("Subcarrier (53)")
            axes[mac_idx].set_xlabel(f"Time steps")
            plt.colorbar(im, ax=axes[mac_idx], label="Amplitude")

        plt.tight_layout()
        plt.show()


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess WiFi CSI CSV file")
    parser.add_argument("--file",    default=DEFAULT_FILE,    help="Path to CSV file")
    parser.add_argument("--window",  default=DEFAULT_WINDOW,  type=float, help="Window length (sec)")
    parser.add_argument("--fs",      default=DEFAULT_FS,      type=float, help="Target sample rate (Hz)")
    parser.add_argument("--min-fs",  default=DEFAULT_MIN_FS,  type=float, help="Min acceptable freq (Hz)")
    parser.add_argument("--overlap", default=DEFAULT_OVERLAP, type=float, help="Window overlap ratio")
    parser.add_argument("--trim",    default=DEFAULT_TRIM,    type=float, help="Trim start/end (sec)")
    parser.add_argument("--plot",    default=3,               type=int,   help="Number of frames to plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X, macs = inspect_single_file(
        file_path  = args.file,
        window_sec = args.window,
        target_fs  = args.fs,
        min_fs     = args.min_fs,
        overlap    = args.overlap,
        trim_sec   = args.trim,
    )
    if X is not None:
        plot_random_frames(X, macs, num_samples=args.plot)
