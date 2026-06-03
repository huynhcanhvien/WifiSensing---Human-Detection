"""
tests/test_model.py — Unit tests cho src/models/light_resnet.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.models.light_resnet import LightResNet, ResidualBlock
from config import DSP, Hardware, DatasetConfig


class TestResidualBlock:
    def test_same_channels_same_spatial(self):
        block = ResidualBlock(16, 16, stride=1)
        x   = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_channel_increase_stride_2(self):
        block = ResidualBlock(16, 32, stride=2)
        x   = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)


class TestLightResNet:
    def setup_method(self):
        self.model = LightResNet(num_classes=DatasetConfig.NUM_CLASSES)
        self.model.eval()
        n_macs  = len(Hardware.TARGET_MACS)
        t_steps = DSP.TARGET_SAMPLES   # 250
        n_sc    = DSP.EFFECTIVE_SC_END - DSP.EFFECTIVE_SC_START  # 53
        self.dummy_input = torch.randn(4, n_macs, t_steps, n_sc)

    def test_forward_output_shape(self):
        with torch.no_grad():
            out = self.model(self.dummy_input)
        assert out.shape == (4, DatasetConfig.NUM_CLASSES)

    def test_forward_batch_size_1(self):
        with torch.no_grad():
            x   = torch.randn(1, 2, DSP.TARGET_SAMPLES, 53)
            out = self.model(x)
        assert out.shape == (1, DatasetConfig.NUM_CLASSES)

    def test_no_nan_output(self):
        with torch.no_grad():
            out = self.model(self.dummy_input)
        assert not torch.isnan(out).any(), "Model output chứa NaN!"

    def test_parameter_count(self):
        """Đảm bảo model đủ nhỏ để chạy trên Pi (< 200K params)."""
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params < 200_000, f"Model quá lớn: {n_params:,} params"

    def test_onnx_export(self, tmp_path):
        """Kiểm tra export ONNX không lỗi."""
        out_path = str(tmp_path / "test_model.onnx")
        dummy    = torch.randn(1, 2, DSP.TARGET_SAMPLES, 53)
        torch.onnx.export(
            self.model, dummy, out_path,
            opset_version=11, input_names=['input'], output_names=['output']
        )
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
