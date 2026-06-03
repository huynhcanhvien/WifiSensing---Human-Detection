"""
src/dsp/interpolation.py — Nội suy và resample tín hiệu CSI.

Xử lý:
- FFT shift (đưa DC về giữa)
- Nội suy tuyến tính tại các null/pilot subcarrier
- Resample về lưới thời gian đều (TARGET_FS Hz)
"""

import numpy as np
from scipy import interpolate as scipy_interp

from config import DSP


def fft_shift(csi_matrix: np.ndarray) -> np.ndarray:
    """FFT shift dọc theo trục subcarrier (axis=1)."""
    return np.fft.fftshift(csi_matrix, axes=1)


def _interp_col(xi: float, xp: np.ndarray, fp_matrix: np.ndarray) -> np.ndarray:
    """
    Nội suy tuyến tính 1 điểm xi từ mảng giá trị đã biết.
    Nhanh hơn scipy.interp1d khi gọi trong vòng lặp.

    Args:
        xi: vị trí cần nội suy
        xp: shape (M,) — các vị trí đã biết (sorted)
        fp_matrix: shape (M, T) — giá trị tại các vị trí đã biết

    Returns:
        shape (T,) — giá trị nội suy
    """
    idx = np.searchsorted(xp, xi)
    idx = np.clip(idx, 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0 = fp_matrix[idx - 1, :]
    y1 = fp_matrix[idx,     :]
    t = (xi - x0) / (x1 - x0) if x1 != x0 else 0.0
    return y0 + t * (y1 - y0)


def interpolate_dc_pilots(csi_matrix: np.ndarray) -> np.ndarray:
    """
    Nội suy tuyến tính các null subcarrier và pilot subcarrier.

    Các subcarrier null (DC, guards, pilots) không mang thông tin CSI hữu ích.
    Giá trị của chúng được nội suy từ các subcarrier hợp lệ lân cận.

    Args:
        csi_matrix: shape (T, 64)

    Returns:
        Ma trận đã nội suy, cùng shape (T, 64)
    """
    valid_idx  = np.array(DSP.VALID_SUBCARRIERS)
    output     = np.empty_like(csi_matrix, dtype=float)
    valid_vals = csi_matrix[:, valid_idx].T   # shape (len(valid_idx), T)

    for xi in range(64):
        output[:, xi] = _interp_col(xi, valid_idx, valid_vals)

    return output


def resample_to_grid(
    csi_matrix: np.ndarray,
    t_original: np.ndarray,
    t_global: np.ndarray,
) -> np.ndarray:
    """
    Resample CSI về lưới thời gian đều bằng nội suy tuyến tính (scipy).

    Args:
        csi_matrix: shape (T_orig, N_sc) — CSI thô
        t_original: shape (T_orig,) — timestamps gốc (giây)
        t_global: shape (T_new,) — lưới thời gian mới

    Returns:
        shape (T_new, N_sc) — CSI đã resample
    """
    f_resample = scipy_interp.interp1d(
        t_original, csi_matrix, axis=0,
        fill_value="extrapolate", bounds_error=False
    )
    return f_resample(t_global)
