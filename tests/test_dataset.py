"""
tests/test_dataset.py — Unit tests cho src/data/dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.data.dataset import CSIDataset


class TestCSIDataset:
    def setup_method(self):
        # (N, T, SC, MACs) — đúng format đầu vào
        self.X = np.random.randn(100, 250, 53, 2).astype(np.float32)
        self.Y = np.random.randint(0, 2, size=100).astype(np.int64)
        self.ds = CSIDataset(self.X, self.Y)

    def test_length(self):
        assert len(self.ds) == 100

    def test_item_shapes(self):
        x, y = self.ds[0]
        # X phải được transpose sang (MACs, T, SC) = (2, 250, 53)
        assert x.shape == (2, 250, 53)
        assert y.shape == ()

    def test_item_dtype(self):
        x, y = self.ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_values_consistent(self):
        """Kiểm tra giá trị sau khi permute vẫn đúng."""
        x, y = self.ds[5]
        # X[5] shape (2, 250, 53) ↔ self.X[5] permuted (3,0,1) → (2, 250, 53)
        expected = torch.FloatTensor(self.X[5]).permute(2, 0, 1)
        torch.testing.assert_close(x, expected)

    def test_all_labels_present(self):
        labels = [self.ds[i][1].item() for i in range(len(self.ds))]
        assert 0 in labels and 1 in labels
