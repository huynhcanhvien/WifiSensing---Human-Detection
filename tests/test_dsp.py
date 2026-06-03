"""
tests/test_dsp.py — Unit tests cho src/dsp/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots, resample_to_grid


class TestHampelFilter:
    def test_output_same_shape(self):
        X = np.random.randn(100, 53)
        out = hampel_filter(X)
        assert out.shape == X.shape

    def test_clean_signal_unchanged(self):
        """Tín hiệu sạch (không outlier) phải giữ nguyên phần lớn giá trị."""
        X   = np.ones((100, 53)) * 5.0
        out = hampel_filter(X)
        np.testing.assert_allclose(out, X, atol=1e-10)

    def test_removes_obvious_outlier(self):
        """Outlier rõ ràng phải bị thay thế bằng giá trị median."""
        X = np.ones((100, 1)) * 1.0
        X[50, 0] = 1000.0   # outlier rõ ràng
        out = hampel_filter(X)
        assert abs(out[50, 0]) < 10.0, f"Outlier không được lọc: {out[50, 0]}"

    def test_dtype_preserved(self):
        X = np.random.randn(50, 10).astype(np.float64)
        out = hampel_filter(X)
        assert out.dtype == X.dtype


class TestFFTShift:
    def test_output_shape(self):
        X = np.random.randn(200, 64)
        out = fft_shift(X)
        assert out.shape == X.shape

    def test_double_shift_identity(self):
        """fftshift dùng 2 lần phải trả về mảng gốc."""
        X = np.random.randn(100, 64)
        np.testing.assert_array_equal(fft_shift(fft_shift(X)), X)


class TestInterpolateDCPilots:
    def test_output_shape(self):
        X = np.random.randn(200, 64)
        out = interpolate_dc_pilots(X)
        assert out.shape == X.shape

    def test_valid_subcarriers_unchanged(self):
        """Các subcarrier hợp lệ phải giữ nguyên giá trị."""
        from config import DSP
        X   = np.random.randn(50, 64)
        out = interpolate_dc_pilots(X)
        for sc in DSP.VALID_SUBCARRIERS:
            np.testing.assert_allclose(
                out[:, sc], X[:, sc], atol=1e-10,
                err_msg=f"Subcarrier {sc} bị thay đổi!"
            )


class TestResampleToGrid:
    def test_output_shape(self):
        t_orig = np.linspace(0, 5, 250)
        csi    = np.random.randn(250, 53)
        t_new  = np.linspace(0, 5, 100)
        out    = resample_to_grid(csi, t_orig, t_new)
        assert out.shape == (100, 53)

    def test_constant_signal(self):
        """Resample tín hiệu hằng số phải cho ra hằng số."""
        t_orig  = np.linspace(0, 5, 250)
        csi     = np.ones((250, 10)) * 3.14
        t_new   = np.linspace(0.5, 4.5, 100)
        out     = resample_to_grid(csi, t_orig, t_new)
        np.testing.assert_allclose(out, 3.14, atol=1e-6)
