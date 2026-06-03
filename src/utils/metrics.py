"""
src/utils/metrics.py — Metrics đánh giá mô hình phân loại.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def calculate_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Tính các metrics phân loại.

    Returns:
        (accuracy, precision, recall, f1) — tất cả theo macro average
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return float(acc), float(precision), float(recall), float(f1)


def plot_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    classes: list[str],
    save_path: str | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Vẽ và (tùy chọn) lưu Confusion Matrix.

    Args:
        y_true: nhãn thực tế
        y_pred: nhãn dự đoán
        classes: danh sách tên lớp
        save_path: nếu cung cấp, lưu ảnh tại đây
        title: tiêu đề biểu đồ
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
    )
    plt.title(title, fontsize=16)
    plt.ylabel('Nhãn Thực Tế (True Label)', fontsize=12)
    plt.xlabel('Mô Hình Dự Đoán (Predicted)', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"📸 Confusion matrix saved: {save_path}")
    else:
        plt.show()
    plt.close()
