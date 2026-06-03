"""
scripts/train.py — Script huấn luyện mô hình LightResNet.

Cách dùng:
    python scripts/train.py
    python scripts/train.py --epochs 100 --lr 5e-4 --batch 32

Lưu checkpoint tốt nhất tại: artifacts/checkpoints/best_light_resnet.pth
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Đảm bảo root được tìm thấy khi chạy từ mọi thư mục
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Paths, TrainingConfig, DatasetConfig
from src.data.dataset import get_dataloaders
from src.models.light_resnet import LightResNet
from src.utils.engine import Engine
from src.utils.metrics import calculate_metrics
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightResNet on WiFi CSI data")
    parser.add_argument("--data",    default=str(Paths.DATASET_NPZ),               help="Path to .npz dataset")
    parser.add_argument("--epochs",  default=TrainingConfig.EPOCHS,   type=int,    help="Number of epochs")
    parser.add_argument("--lr",      default=TrainingConfig.LEARNING_RATE, type=float, help="Learning rate")
    parser.add_argument("--batch",   default=TrainingConfig.BATCH_SIZE, type=int,   help="Batch size")
    parser.add_argument("--workers", default=TrainingConfig.NUM_WORKERS, type=int,  help="DataLoader workers")
    parser.add_argument("--seed",    default=TrainingConfig.SEED,     type=int,     help="Random seed")
    parser.add_argument("--save",    default=str(Paths.CHECKPOINTS / "best_light_resnet.pth"), help="Path to save best model")
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)

    device = TrainingConfig.DEVICE
    print("=" * 55)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN (Device: {device})")
    print("=" * 55)

    # ── Data ──
    train_loader, val_loader, _ = get_dataloaders(
        data_path=args.data,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    # ── Model ──
    model     = LightResNet(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=TrainingConfig.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ── Logging ──
    log_dir = TrainingConfig.log_dir()
    writer  = SummaryWriter(log_dir=log_dir)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = Engine.train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate (lấy cả preds để tính F1)
        val_loss, val_acc, val_preds, val_targets = Engine.evaluate_epoch(model, val_loader, criterion, device)

        _, val_prec, val_rec, val_f1 = calculate_metrics(val_targets, val_preds)

        scheduler.step(val_loss)

        # TensorBoard
        writer.add_scalar('Loss/Train',   train_loss, epoch)
        writer.add_scalar('Loss/Val',     val_loss,   epoch)
        writer.add_scalar('Acc/Train',    train_acc,  epoch)
        writer.add_scalar('Acc/Val',      val_acc,    epoch)
        writer.add_scalar('F1/Val',       val_f1,     epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save)
            print(f"   🌟 Best model saved (Val Loss: {val_loss:.4f})")

        # Periodic checkpoint
        if epoch % TrainingConfig.CHECKPOINT_INTERVAL == 0:
            ckpt_path = str(Paths.CHECKPOINTS / f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
            }, ckpt_path)

    writer.close()
    print("\n✅ HOÀN TẤT HUẤN LUYỆN!")
    print(f"   Best Val Loss : {best_val_loss:.4f}")
    print(f"   Best model    : {args.save}")
    print(f"   TensorBoard   : tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    train()
