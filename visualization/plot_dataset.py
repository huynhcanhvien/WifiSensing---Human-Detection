"""
visualization/plot_dataset.py — Vẽ heatmap mẫu từ dataset .npz đã xây dựng.

Trích xuất từ hàm plot_dataset_samples() trong src/data/build_dataset.py,
được mở rộng với nhiều loại biểu đồ hơn:
  1. CSI Heatmap (từng MAC)
  2. Phân bố nhãn (class distribution)
  3. Histogram biên độ theo label

Cách dùng:
    python visualization/plot_dataset.py
    python visualization/plot_dataset.py --data data/wifi_sensing_dataset.npz --samples 5
    python visualization/plot_dataset.py --no-show
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DatasetConfig, Hardware, Paths

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ─── Plot 1: CSI Heatmap Samples ──────────────────────────────────────────────

def plot_csi_samples(
    X: np.ndarray,
    Y: np.ndarray,
    num_samples: int = 4,
    save: bool = True,
    show: bool = True,
) -> str | None:
    """
    Vẽ heatmap CSI ngẫu nhiên từ dataset, theo nhãn.

    Args:
        X: shape (N, T, 53, 2)
        Y: shape (N,)
        num_samples: số mẫu mỗi class

    Returns:
        Đường dẫn ảnh đã lưu.
    """
    class_names = DatasetConfig.CLASS_NAMES
    n_classes   = DatasetConfig.NUM_CLASSES
    n_macs      = X.shape[3]

    fig, axes = plt.subplots(
        num_samples * n_classes, n_macs,
        figsize=(6 * n_macs, 3.5 * num_samples * n_classes)
    )
    fig.suptitle("Dataset CSI Heatmap Samples", fontsize=16, fontweight='bold', y=1.002)

    row = 0
    for cls in range(n_classes):
        idx_cls = np.where(Y == cls)[0]
        chosen  = np.random.choice(idx_cls, min(num_samples, len(idx_cls)), replace=False)

        for sample_idx in chosen:
            tensor = X[sample_idx]   # (T, 53, 2)
            label  = class_names[cls]
            for mac_i in range(n_macs):
                ax  = axes[row, mac_i] if n_classes * num_samples > 1 else axes[mac_i]
                csi = tensor[:, :, mac_i]   # (T, 53)
                vmin, vmax = np.percentile(csi, 5), np.percentile(csi, 95)
                im  = ax.imshow(csi.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(
                    f"Label: {label}  |  MAC {mac_i}: {Hardware.TARGET_MACS[mac_i]}",
                    fontsize=9, fontweight='bold'
                )
                ax.set_xlabel("Time (samples)")
                ax.set_ylabel("Subcarrier")
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            row += 1

    plt.tight_layout()
    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "dataset_csi_samples.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")
    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Plot 2: Class Distribution ───────────────────────────────────────────────

def plot_class_distribution(
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    save: bool = True,
    show: bool = True,
) -> str | None:
    """Vẽ bar chart phân bố nhãn trong tập train và test."""
    class_names = [DatasetConfig.CLASS_NAMES[i] for i in range(DatasetConfig.NUM_CLASSES)]
    x     = np.arange(len(class_names))
    width = 0.35

    train_counts = [np.sum(Y_train == i) for i in range(DatasetConfig.NUM_CLASSES)]
    test_counts  = [np.sum(Y_test  == i) for i in range(DatasetConfig.NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_train = ax.bar(x - width/2, train_counts, width, label='Train', color='#4A90D9', alpha=0.88)
    bars_test  = ax.bar(x + width/2, test_counts,  width, label='Test',  color='#E87040', alpha=0.88)

    ax.bar_label(bars_train, padding=3, fontsize=11, fontweight='bold')
    ax.bar_label(bars_test,  padding=3, fontsize=11, fontweight='bold')

    ax.set_title("Class Distribution — Train vs Test", fontsize=14, fontweight='bold')
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Windows", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    total = sum(train_counts) + sum(test_counts)
    ax.set_title(
        f"Class Distribution — Train vs Test  (Total: {total:,} windows)",
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "dataset_class_distribution.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")
    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Plot 3: Amplitude Histogram ──────────────────────────────────────────────

def plot_amplitude_histogram(
    X: np.ndarray,
    Y: np.ndarray,
    save: bool = True,
    show: bool = True,
) -> str | None:
    """
    Vẽ histogram phân bố biên độ CSI theo label.
    Giúp kiểm tra xem 2 class có tách biệt về phân bố không.
    """
    class_names = DatasetConfig.CLASS_NAMES
    colors      = ['#4A90D9', '#E87040']

    fig, axes = plt.subplots(1, X.shape[3], figsize=(10 * X.shape[3], 5))
    if X.shape[3] == 1:
        axes = [axes]

    for mac_i, ax in enumerate(axes):
        for cls in range(DatasetConfig.NUM_CLASSES):
            samples = X[Y == cls, :, :, mac_i].flatten()
            ax.hist(
                samples, bins=80, alpha=0.6,
                color=colors[cls], label=class_names[cls],
                density=True, edgecolor='none'
            )
        ax.set_title(
            f"Amplitude Distribution — MAC {mac_i}: {Hardware.TARGET_MACS[mac_i]}",
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel("Normalized Amplitude", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle("Amplitude Distribution by Class", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "dataset_amplitude_histogram.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")
    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ phân tích dataset CSI")
    parser.add_argument("--data",    default=str(Paths.DATASET_NPZ),  help="Path to .npz dataset")
    parser.add_argument("--samples", default=3, type=int,              help="Số mẫu mỗi class để vẽ heatmap")
    parser.add_argument("--no-save", action="store_true",              help="Không lưu ảnh")
    parser.add_argument("--no-show", action="store_true",              help="Không hiện cửa sổ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save = not args.no_save
    show = not args.no_show

    if not os.path.exists(args.data):
        print(f"❌ Không tìm thấy dataset: {args.data}")
        print("   → Chạy `python scripts/build_dataset.py` trước!")
        sys.exit(1)

    print(f"📦 Đang load dataset từ: {args.data}")
    data    = np.load(args.data)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test  = data['X_test']
    Y_test  = data['Y_test']

    print(f"   Train: {X_train.shape[0]:,} windows  |  Test: {X_test.shape[0]:,} windows")
    print(f"   Tensor shape: {X_train.shape}\n")

    # Gộp train+test cho visualizations không cần split
    X_all = np.vstack([X_train, X_test])
    Y_all = np.concatenate([Y_train, Y_test])

    print("📊 [1/3] CSI Heatmap Samples...")
    plot_csi_samples(X_all, Y_all, num_samples=args.samples, save=save, show=show)

    print("📊 [2/3] Class Distribution...")
    plot_class_distribution(Y_train, Y_test, save=save, show=show)

    print("📊 [3/3] Amplitude Histogram...")
    plot_amplitude_histogram(X_all, Y_all, save=save, show=show)

    print(f"\n✅ Xong! Ảnh được lưu tại: visualization/output/")
