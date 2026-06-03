"""
scripts/export_onnx.py — Export mô hình PyTorch sang định dạng ONNX.

ONNX được dùng để chạy inference trên Raspberry Pi 4 via OpenCV DNN.

Cách dùng:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model artifacts/checkpoints/best_light_resnet.pth
    python scripts/export_onnx.py --out artifacts/csi_lightresnet.onnx

Ghi chú:
    - Model export dùng cùng kiến trúc LightResNet từ src/models/
    - Opset 11 được chọn vì tương thích tốt nhất với OpenCV 4.x trên Raspberry Pi
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DatasetConfig, DSP, Hardware, Paths
from src.models.light_resnet import LightResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LightResNet to ONNX format")
    parser.add_argument("--model",  default=str(Paths.CHECKPOINTS / "best_light_resnet.pth"), help="Path to .pth weights")
    parser.add_argument("--out",    default=str(Paths.ONNX_MODEL),                            help="Output .onnx path")
    parser.add_argument("--opset",  default=11, type=int,                                     help="ONNX opset version")
    return parser.parse_args()


def export_onnx():
    args = parse_args()

    print(f"📦 Đang export ONNX...")
    print(f"   Weights: {args.model}")
    print(f"   Output:  {args.out}")

    if not os.path.exists(args.model):
        print(f"❌ Không tìm thấy model: {args.model}")
        print("   → Chạy `python scripts/train.py` trước!")
        return

    # ── Load model ──
    model = LightResNet(num_classes=DatasetConfig.NUM_CLASSES)
    model.load_state_dict(
        torch.load(args.model, map_location=torch.device('cpu'), weights_only=True)
    )
    model.eval()

    # ── Dummy input: (1, 2, 250, 53) ──
    n_macs   = len(Hardware.TARGET_MACS)
    t_steps  = DSP.TARGET_SAMPLES          # 250
    n_sc     = DSP.EFFECTIVE_SC_END - DSP.EFFECTIVE_SC_START  # 53
    dummy_input = torch.randn(1, n_macs, t_steps, n_sc)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Export ──
    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\n✅ Xuất thành công: {args.out}  ({size_kb:.1f} KB)")
    print(f"   Input shape: (batch, {n_macs}, {t_steps}, {n_sc})")
    print(f"   Opset: {args.opset}")


if __name__ == "__main__":
    export_onnx()
