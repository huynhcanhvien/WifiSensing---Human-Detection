"""
src/models/light_resnet.py — Kiến trúc LightResNet cho phân loại CSI.

LightResNet là mô hình CNN nhẹ với Residual Blocks, được thiết kế để
chạy hiệu quả trên Raspberry Pi 4 (via ONNX) và Mac M-series.

Input: (batch, 2, 250, 53) — 2 MAC, 250 time steps, 53 subcarriers
Output: (batch, num_classes) — logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Khối Residual cơ bản với skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut nếu kích thước thay đổi
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class LightResNet(nn.Module):
    """
    LightResNet: mạng ResNet nhỏ gọn cho phân loại tín hiệu WiFi CSI.

    Architecture:
        Conv(2→16) → Layer1(16→32, stride=2) → Layer2(32→64, stride=2)
        → GlobalAvgPool → Dropout(0.5) → FC(64→num_classes)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Input channels = 2 (số MAC address)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 32, stride=2)
        self.layer2 = self._make_layer(32, 64, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(0.5)
        self.fc       = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, stride),
            ResidualBlock(out_ch, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
