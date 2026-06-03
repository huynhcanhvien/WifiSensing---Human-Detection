"""
src/utils/engine.py — Training/Validation Engine.

Engine class được sử dụng trong scripts/train.py để thực hiện
mỗi epoch training và validation, giúp train.py gọn hơn.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import AverageMeter, calculate_accuracy


class Engine:
    """Encapsulates training và evaluation logic cho một epoch."""

    @staticmethod
    def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """
        Chạy một epoch training.

        Returns:
            (avg_loss, avg_accuracy)
        """
        model.train()
        losses = AverageMeter()
        accs   = AverageMeter()

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))

        return losses.avg, accs.avg

    @staticmethod
    def evaluate_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float, list, list]:
        """
        Chạy một epoch evaluation (không tính gradient).

        Returns:
            (avg_loss, avg_accuracy, all_preds, all_targets)
        """
        model.eval()
        losses    = AverageMeter()
        accs      = AverageMeter()
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

                _, preds = torch.max(outputs, 1)
                acc      = calculate_accuracy(outputs, targets)

                losses.update(loss.item(), inputs.size(0))
                accs.update(acc, inputs.size(0))
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        return losses.avg, accs.avg, all_preds, all_targets
