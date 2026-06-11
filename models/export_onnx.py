"""
models/export_onnx.py — Chuyển đổi 4 model PyTorch (.pth) sang ONNX.

Cách dùng:
    python models/export_onnx.py --pth-dir /path/to/pth --out-dir models/onnx
    python models/export_onnx.py  # dùng đường dẫn mặc định (./onnx/)

Yêu cầu:
    pip install torch onnx onnxruntime
"""

import argparse
import os

import onnx
import torch
import torch.nn as nn

# ─── Định nghĩa kiến trúc (phải khớp với notebook) ────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class LightResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1         = nn.Conv2d(2, 16, 3, stride=1, padding=1, bias=False)
        self.bn1           = nn.BatchNorm2d(16)
        self.relu          = nn.ReLU(inplace=True)
        self.pool          = nn.MaxPool2d(2, 2)
        self.layer1        = BasicBlock(16, 16, stride=1)
        self.layer2        = BasicBlock(16, 32, stride=2)
        self.layer3        = BasicBlock(32, 64, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc            = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        return self.fc(x.view(x.size(0), -1))


class MLP(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Kiến trúc gốc được dùng khi training: pool (8,53) → flatten 848
        # Dùng dynamo_export thay vì torch.onnx.export để tránh lỗi AdaptiveAvgPool2d
        self.pool = nn.AdaptiveAvgPool2d((8, 53))
        self.net  = nn.Sequential(
            nn.Flatten(),                            # 2*8*53 = 848
            nn.Linear(848, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(self.pool(x))


class CNN2D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features   = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class CNNLSTMHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )
        self.lstm = nn.LSTM(32 * 13, 128, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        N, C, T, F = x.size()
        out = self.cnn(x)
        out = out.permute(0, 2, 1, 3).reshape(N, T, -1)
        out, _ = self.lstm(out)
        return self.head(out[:, -1])


# Registry
MODEL_REGISTRY = {
    "LightResNet":   LightResNet,
    "CNN2D":         CNN2D,
    "CNNLSTMHybrid": CNNLSTMHybrid,
    "MLP":           MLP,
}

# ─── Export ───────────────────────────────────────────────────────────────────

def _export_standard(model, dummy: torch.Tensor, onnx_path: str) -> None:
    """torch.onnx.export — hoạt động với hầu hết kiến trúc."""
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["csi_input"],
        output_names=["logits"],
        dynamic_axes={"csi_input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def _patch_adaptive_pool_for_onnx(model: nn.Module, t_in: int = 250) -> None:
    """
    Thay AdaptiveAvgPool2d(out_h, out_w) bằng AvgPool2d tương đương ONNX-compatible.
    Chỉ áp dụng khi out_h không là ước của t_in (ONNX không hỗ trợ trường hợp đó).
    Output shape giữ nguyên → checkpoint weights vẫn load được.
    Lưu ý: chỉ dùng cho benchmark latency, không dùng để test accuracy.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.AdaptiveAvgPool2d):
            out_h, out_w = module.output_size
            if t_in % out_h != 0:
                # AvgPool2d với kernel/stride cho ra đúng out_h outputs
                # floor((t_in - k) / k) + 1 = out_h  →  k = t_in // out_h
                k = t_in // out_h
                compat_pool = nn.AvgPool2d(kernel_size=(k, 1), stride=(k, 1))
                setattr(model, name, compat_pool)
        else:
            _patch_adaptive_pool_for_onnx(module, t_in)


def export_model(name: str, cls, pth_path: str, onnx_path: str) -> None:
    if not os.path.exists(pth_path):
        print(f"  [SKIP] {name}: không tìm thấy {pth_path}")
        return

    model = cls(num_classes=2)
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 2, 250, 53)
    note  = ""

    try:
        _export_standard(model, dummy, onnx_path)
    except Exception:
        # Fallback: patch AdaptiveAvgPool2d → AvgPool2d rồi export lại
        _patch_adaptive_pool_for_onnx(model, t_in=250)
        _export_standard(model, dummy, onnx_path)
        note = "  ⚠  pool patched for ONNX (latency benchmark only)"

    onnx.checker.check_model(onnx.load(onnx_path))
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  [OK]   {name:<16} → {onnx_path}  ({size_mb:.3f} MB){note}")


def main(pth_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nExport ONNX — pth: '{pth_dir}'  →  out: '{out_dir}'\n")

    for name, cls in MODEL_REGISTRY.items():
        pth_path  = os.path.join(pth_dir,  f"best_{name}.pth")
        onnx_path = os.path.join(out_dir, f"{name}.onnx")
        export_model(name, cls, pth_path, onnx_path)

    print("\nHoàn tất export.\n")


def parse_args() -> argparse.Namespace:
    here     = os.path.dirname(os.path.abspath(__file__))
    root     = os.path.dirname(here)
    ckpt_dir = os.path.join(root, "artifacts", "checkpoints")
    parser = argparse.ArgumentParser(description="Export PyTorch → ONNX")
    parser.add_argument(
        "--pth-dir", default=ckpt_dir,
        help="Thư mục chứa các file best_<Model>.pth (mặc định: artifacts/checkpoints/)",
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(here, "onnx"),
        help="Thư mục lưu file .onnx (mặc định: models/onnx/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.pth_dir, args.out_dir)
