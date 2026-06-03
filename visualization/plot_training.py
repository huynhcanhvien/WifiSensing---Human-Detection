"""
visualization/plot_training.py — Vẽ kết quả training từ TensorBoard logs.

Đọc TensorBoard event files và vẽ:
  1. Loss curve (Train vs Val)
  2. Accuracy curve (Train vs Val)
  3. Learning Rate schedule
  4. Confusion Matrix (từ file nếu có)

Cách dùng:
    python visualization/plot_training.py
    python visualization/plot_training.py --logdir artifacts/runs/experiment_20260203_215850
    python visualization/plot_training.py --no-show
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Paths

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def find_latest_run(runs_dir: str) -> str | None:
    """Tìm thư mục experiment mới nhất trong runs/."""
    runs = sorted(glob.glob(os.path.join(runs_dir, "experiment_*")))
    return runs[-1] if runs else None


def load_tensorboard_scalars(logdir: str) -> dict:
    """
    Đọc scalar events từ TensorBoard log directory.
    Yêu cầu: pip install tensorboard
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("Cần cài tensorboard: pip install tensorboard")

    ea = EventAccumulator(logdir)
    ea.Reload()

    tags    = ea.Tags().get('scalars', [])
    scalars = {}
    for tag in tags:
        events      = ea.Scalars(tag)
        steps       = [e.step  for e in events]
        values      = [e.value for e in events]
        scalars[tag] = (steps, values)

    return scalars


def plot_training_curves(
    logdir: str,
    save: bool = True,
    show: bool = True,
) -> str | None:
    """
    Vẽ training curves từ TensorBoard logs.

    Returns:
        Đường dẫn ảnh đã lưu.
    """
    print(f"📖 Đang đọc TensorBoard logs từ: {logdir}")

    try:
        scalars = load_tensorboard_scalars(logdir)
    except Exception as e:
        print(f"❌ Không đọc được logs: {e}")
        return None

    if not scalars:
        print("⚠️  Không có scalar data trong log directory.")
        return None

    # Nhận diện các keys có trong log
    has_loss = 'Loss/Train' in scalars and 'Loss/Val' in scalars
    has_acc  = 'Acc/Train'  in scalars and 'Acc/Val'  in scalars
    has_f1   = 'F1/Val'     in scalars
    has_lr   = 'LR'         in scalars

    n_plots = sum([has_loss, has_acc, (has_f1 or has_lr)])
    if n_plots == 0:
        print("⚠️  Không có đủ dữ liệu để vẽ.")
        return None

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    run_name = os.path.basename(logdir)
    fig.suptitle(f"Training Results — {run_name}", fontsize=14, fontweight='bold')

    ax_idx = 0

    # ── Loss ──
    if has_loss:
        ax = axes[ax_idx]; ax_idx += 1
        steps_tr, loss_tr = scalars['Loss/Train']
        steps_vl, loss_vl = scalars['Loss/Val']
        ax.plot(steps_tr, loss_tr, color='#4A90D9', lw=2, label='Train Loss')
        ax.plot(steps_vl, loss_vl, color='#E87040', lw=2, label='Val Loss')
        best_epoch = steps_vl[np.argmin(loss_vl)]
        best_val   = min(loss_vl)
        ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.6, label=f'Best epoch {best_epoch}')
        ax.annotate(f'Best: {best_val:.4f}', xy=(best_epoch, best_val),
                    xytext=(best_epoch + 2, best_val + 0.01),
                    fontsize=9, color='gray')
        ax.set_title("Loss Curve", fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    # ── Accuracy ──
    if has_acc:
        ax = axes[ax_idx]; ax_idx += 1
        steps_tr, acc_tr = scalars['Acc/Train']
        steps_vl, acc_vl = scalars['Acc/Val']
        ax.plot(steps_tr, [v * 100 for v in acc_tr], color='#4A90D9', lw=2, label='Train Acc')
        ax.plot(steps_vl, [v * 100 for v in acc_vl], color='#E87040', lw=2, label='Val Acc')
        if has_f1:
            steps_f1, f1_vl = scalars['F1/Val']
            ax.plot(steps_f1, [v * 100 for v in f1_vl], color='#2ECC71', lw=2,
                    linestyle='--', label='Val F1')
        ax.set_ylim(0, 105)
        ax.set_title("Accuracy & F1", fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score (%)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    # ── Learning Rate ──
    if has_lr:
        ax = axes[ax_idx]; ax_idx += 1
        steps_lr, lr_vals = scalars['LR']
        ax.semilogy(steps_lr, lr_vals, color='#9B59B6', lw=2)
        ax.set_title("Learning Rate Schedule", fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate (log scale)")
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"training_{run_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")
    if show:
        plt.show()
    plt.close()
    return out_path


def plot_confusion_matrix_from_file(
    cm_path: str,
    save: bool = True,
    show: bool = True,
) -> str | None:
    """Hiển thị lại confusion matrix đã lưu (nếu có)."""
    if not os.path.exists(cm_path):
        print(f"⚠️  Không tìm thấy confusion matrix tại: {cm_path}")
        return None

    img = plt.imread(cm_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight='bold')

    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "confusion_matrix_display.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")
    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ training curves từ TensorBoard logs")
    parser.add_argument("--logdir",  default=None,         help="Path to TensorBoard run dir (mặc định: run mới nhất)")
    parser.add_argument("--no-save", action="store_true",  help="Không lưu ảnh")
    parser.add_argument("--no-show", action="store_true",  help="Không hiện cửa sổ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save = not args.no_save
    show = not args.no_show

    # Tìm logdir
    logdir = args.logdir
    if logdir is None:
        logdir = find_latest_run(str(Paths.RUNS))
        if logdir is None:
            print(f"❌ Không tìm thấy run nào trong {Paths.RUNS}")
            print("   → Chạy `python scripts/train.py` trước!")
            sys.exit(1)
        print(f"📁 Dùng run mới nhất: {logdir}")

    print("📊 [1/2] Training Curves...")
    plot_training_curves(logdir, save=save, show=show)

    print("📊 [2/2] Confusion Matrix...")
    cm_path = str(Paths.ARTIFACTS / "confusion_matrix.png")
    plot_confusion_matrix_from_file(cm_path, save=save, show=show)

    print(f"\n✅ Xong! Ảnh được lưu tại: visualization/output/")
