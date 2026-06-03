"""
src/dsp/pipeline.py — Pipeline DSP đầu-cuối cho tín hiệu CSI.

Gộp toàn bộ các bước xử lý thành một hàm duy nhất:
  FFT Shift → Nội suy DC → Hampel Filter → Cắt subcarrier → Normalize → Resample

Dùng chung cho cả training (build_dataset) và inference (pi_edge_ai).
"""

import numpy as np

from config import DSP
from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots, resample_to_grid


def process_mac_stream(
    t_mac: np.ndarray,
    csi_raw: np.ndarray,
    t_global: np.ndarray,
) -> np.ndarray:
    """
    Xử lý CSI từ một MAC address theo pipeline chuẩn.

    Args:
        t_mac:   shape (T,) — timestamps gốc (giây)
        csi_raw: shape (T, 64) — amplitude thô từ 64 subcarrier
        t_global: shape (T_new,) — lưới thời gian mục tiêu

    Returns:
        shape (T_new, 53) — CSI đã xử lý, normalize, resample
    """
    # 1. FFT Shift
    csi = fft_shift(csi_raw)

    # 2. Nội suy null/pilot subcarrier
    csi = interpolate_dc_pilots(csi)

    # 3. Hampel filter (loại outlier)
    csi = hampel_filter(csi)

    # 4. Cắt lấy 53 subcarrier hữu ích
    csi = csi[:, DSP.EFFECTIVE_SC_START:DSP.EFFECTIVE_SC_END]   # (T, 53)

    # 5. Normalize L2 theo từng timestamp
    norms = np.linalg.norm(csi, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    csi = csi / norms

    # 6. Resample về lưới thời gian đều
    csi = resample_to_grid(csi, t_mac, t_global)

    return csi


def build_windows(
    csi_resampled: np.ndarray,
    window_samples: int,
    step: int,
) -> np.ndarray:
    """
    Tách CSI đã resample thành các cửa sổ chồng nhau (sliding window).

    Args:
        csi_resampled: shape (T, 53)
        window_samples: số sample trong mỗi cửa sổ
        step: bước nhảy giữa các cửa sổ

    Returns:
        shape (N_windows, window_samples, 53)
    """
    n_total = len(csi_resampled)
    windows = [
        csi_resampled[start : start + window_samples]
        for start in range(0, n_total - window_samples + 1, step)
    ]
    return np.array(windows) if windows else np.empty((0, window_samples, 53))


def build_mimo_tensor(
    mac_csi_list: list[np.ndarray],
) -> np.ndarray:
    """
    Stack MIMO: ghép CSI của nhiều MAC thành tensor đa kênh.

    Args:
        mac_csi_list: danh sách (N_macs,) shape (N_windows, T, 53)

    Returns:
        shape (N_windows, T, 53, N_macs) — MIMO tensor
    """
    min_windows = min(t.shape[0] for t in mac_csi_list)
    clipped = [t[:min_windows] for t in mac_csi_list]
    return np.stack(clipped, axis=-1)


def prepare_inference_input(
    snap_mac0: list,
    snap_mac1: list,
    t_start: float,
    t_end: float,
    target_samples: int = DSP.TARGET_SAMPLES,
    target_fs: float = DSP.TARGET_FS,
) -> np.ndarray:
    """
    Chuẩn bị đầu vào inference từ 2 snapshot buffer (dùng trên Pi).

    Args:
        snap_mac0: list of (timestamp, amp_array) cho MAC 0
        snap_mac1: list of (timestamp, amp_array) cho MAC 1
        t_start, t_end: khoảng thời gian của cửa sổ hiện tại
        target_samples: số sample mong muốn
        target_fs: tần số mục tiêu

    Returns:
        shape (1, 2, T, 53) — NCHW tensor sẵn sàng cho ONNX inference
    """
    t_global = np.arange(t_start, t_end, 1.0 / target_fs)[:target_samples]

    mac_tensors = []
    for snap in [snap_mac0, snap_mac1]:
        t_mac   = np.array([x[0] for x in snap])
        csi_raw = np.array([x[1] for x in snap])
        processed = process_mac_stream(t_mac, csi_raw, t_global)   # (T, 53)
        mac_tensors.append(processed)

    # Stack → (T, 53, 2) → transpose → (2, T, 53) → expand → (1, 2, T, 53)
    mimo = np.stack(mac_tensors, axis=-1)           # (T, 53, 2)
    nchw = np.transpose(mimo, (2, 0, 1))            # (2, T, 53)
    return np.expand_dims(nchw, axis=0).astype(np.float32)
