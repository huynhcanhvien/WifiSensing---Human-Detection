"""
src/utils/logger.py — Logging utilities cho training.
"""

import logging
import os

import torch


class AverageMeter:
    """Theo dõi giá trị trung bình chạy — dùng trong training loop."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Tính accuracy từ logits và targets."""
    _, preds = torch.max(outputs, 1)
    return (preds == targets).sum().item() / targets.size(0)


class TrainingLogger:
    """Logger đơn giản ghi vào file và stdout đồng thời."""

    def __init__(self, log_dir: str, log_file: str = "training.log"):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg: str):
        self.logger.info(msg)
