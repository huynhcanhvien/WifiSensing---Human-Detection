"""
scripts/evaluate.py — Đánh giá mô hình đã huấn luyện trên tập Test.

Cách dùng:
    python scripts/evaluate.py
    python scripts/evaluate.py --model artifacts/checkpoints/best_light_resnet.pth

Output:
    - Bảng classification report (precision, recall, F1)
    - Confusion matrix lưu tại artifacts/confusion_matrix.png
"""

import argparse
import os
import sys

import torch
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DatasetConfig, Paths, TrainingConfig
from src.data.dataset import get_dataloaders
from src.models.light_resnet import LightResNet
from src.utils.engine import Engine
from src.utils.metrics import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LightResNet on WiFi CSI test set")
    parser.add_argument("--data",    default=str(Paths.DATASET_NPZ),                            help="Path to .npz dataset")
    parser.add_argument("--model",   default=str(Paths.CHECKPOINTS / "best_light_resnet.pth"),  help="Path to model weights")
    parser.add_argument("--batch",   default=TrainingConfig.BATCH_SIZE, type=int,               help="Batch size")
    parser.add_argument("--workers", default=TrainingConfig.NUM_WORKERS, type=int,              help="DataLoader workers")
    return parser.parse_args()


def evaluate():
    args   = parse_args()
    device = TrainingConfig.DEVICE

    print("\n" + "=" * 55)
    print("📊 ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 55)

    # ── Data ──
    _, _, test_loader = get_dataloaders(
        data_path=args.data,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    # ── Model ──
    model = LightResNet(num_classes=DatasetConfig.NUM_CLASSES).to(device)

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("   → Chạy `python scripts/train.py` trước!")
        return

    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"✅ Loaded weights từ: {args.model}")

    # ── Inference ──
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_targets = Engine.evaluate_epoch(
        model, test_loader, criterion, device
    )

    class_names = [DatasetConfig.CLASS_NAMES[i] for i in range(DatasetConfig.NUM_CLASSES)]

    print(f"\n📈 Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.4f}\n")
    print("📋 CLASSIFICATION REPORT:")
    print(classification_report(test_targets, test_preds, target_names=class_names, digits=4))

    # ── Confusion Matrix ──
    cm_path = str(Paths.ARTIFACTS / "confusion_matrix.png")
    os.makedirs(Paths.ARTIFACTS, exist_ok=True)
    plot_confusion_matrix(
        test_targets, test_preds,
        classes=class_names,
        save_path=cm_path,
        title="Confusion Matrix — Tập Test",
    )


if __name__ == "__main__":
    evaluate()
