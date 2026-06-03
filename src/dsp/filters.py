"""
src/dsp/filters.py — Bộ lọc tín hiệu CSI.

Triển khai duy nhất của Hampel filter dùng cho toàn bộ dự án.
Thay thế 3 bản cũ rải rác trong build_db.py, pi_edge_ai.py, export.py.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import DSP


def hampel_filter(input_matrix: np.ndarray) -> np.ndarray:
    """
    Áp dụng Hampel filter để loại bỏ outlier theo từng subcarrier.

    Thuật toán: với mỗi điểm, tính median và MAD trong cửa sổ lân cận.
    Điểm bị coi là outlier nếu |x - median| > sigma * k * MAD.

    Args:
        input_matrix: shape (T, N_subcarriers) — ma trận CSI theo thời gian.

    Returns:
        Ma trận đã lọc, cùng shape với input.
    """
    w   = 2 * DSP.HAMPEL_WINDOW + 1
    out = input_matrix.copy()

    for sc in range(input_matrix.shape[1]):
        series = input_matrix[:, sc]
        padded  = np.pad(series, DSP.HAMPEL_WINDOW, mode='edge')
        windows = sliding_window_view(padded, w)          # (T, w)
        medians  = np.median(windows, axis=1)
        mads     = np.median(np.abs(windows - medians[:, None]), axis=1)
        threshold = DSP.HAMPEL_SIGMA * DSP.K_HAMPEL * mads
        outliers  = np.abs(series - medians) > threshold
        out[outliers, sc] = medians[outliers]

    return out
