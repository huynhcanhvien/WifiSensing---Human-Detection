import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Tập Test', fontsize=16)
    plt.ylabel('Nhãn Thực Tế (True Label)', fontsize=12)
    plt.xlabel('Mô Hình Dự Đoán (Predicted Label)', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📸 Đã lưu ảnh ma trận nhầm lẫn tại: {save_path}")
    else:
        plt.show()
    plt.close()