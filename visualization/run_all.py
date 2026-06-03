"""
visualization/run_all.py — Chạy toàn bộ visualization pipeline.

Tự động sinh tất cả ảnh vào visualization/output/:
  1. dataset_csi_samples.png       — Heatmap mẫu từ dataset
  2. dataset_class_distribution.png — Phân bố nhãn
  3. dataset_amplitude_histogram.png — Histogram biên độ
  4. training_<run>.png             — Loss/Accuracy/LR curves
  5. confusion_matrix_display.png   — Confusion matrix

Cách dùng:
    python visualization/run_all.py               # Chạy tất cả (không hiện cửa sổ)
    python visualization/run_all.py --show        # Hiện từng cửa sổ
    python visualization/run_all.py --skip-pcap   # Bỏ qua bước vẽ PCAP
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Paths

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ visualization pipeline")
    parser.add_argument("--data",       default=str(Paths.DATASET_NPZ), help="Path to .npz dataset")
    parser.add_argument("--logdir",     default=None,                    help="TensorBoard run dir")
    parser.add_argument("--samples",    default=3,    type=int,          help="Số mẫu heatmap mỗi class")
    parser.add_argument("--show",       action="store_true",             help="Hiện cửa sổ (mặc định: batch mode)")
    parser.add_argument("--skip-train", action="store_true",             help="Bỏ qua training curves")
    return parser.parse_args()


def main():
    args = parse_args()
    show = args.show
    save = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ─── 1. Dataset Visualizations ───────────────────────────────────────────
    if os.path.exists(args.data):
        import numpy as np
        from visualization.plot_dataset import (
            plot_amplitude_histogram,
            plot_class_distribution,
            plot_csi_samples,
        )

        print("\n" + "=" * 55)
        print("📦 DATASET VISUALIZATIONS")
        print("=" * 55)
        data    = np.load(args.data)
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test  = data['X_test']
        Y_test  = data['Y_test']
        X_all   = np.vstack([X_train, X_test])
        Y_all   = np.concatenate([Y_train, Y_test])

        print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  |  Shape: {X_train.shape}")

        results['csi_samples']    = plot_csi_samples(X_all, Y_all, num_samples=args.samples, save=save, show=show)
        results['class_dist']     = plot_class_distribution(Y_train, Y_test, save=save, show=show)
        results['amp_histogram']  = plot_amplitude_histogram(X_all, Y_all, save=save, show=show)
    else:
        print(f"\n⚠️  Bỏ qua dataset plots — không tìm thấy: {args.data}")
        print("   → Chạy `python scripts/build_dataset.py` để tạo dataset.")

    # ─── 2. Training Curves ───────────────────────────────────────────────────
    if not args.skip_train:
        from visualization.plot_training import find_latest_run, plot_training_curves, plot_confusion_matrix_from_file

        print("\n" + "=" * 55)
        print("📈 TRAINING VISUALIZATIONS")
        print("=" * 55)

        logdir = args.logdir or find_latest_run(str(Paths.RUNS))
        if logdir:
            results['training'] = plot_training_curves(logdir, save=save, show=show)
        else:
            print(f"⚠️  Bỏ qua training curves — không tìm thấy run trong {Paths.RUNS}")

        cm_path = str(Paths.ARTIFACTS / "confusion_matrix.png")
        results['confusion_matrix'] = plot_confusion_matrix_from_file(cm_path, save=save, show=show)

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("✅ HOÀN TẤT — Tất cả ảnh trong: visualization/output/")
    print("=" * 55)
    saved_files = [p for p in results.values() if p is not None]
    for p in saved_files:
        size_kb = os.path.getsize(p) / 1024
        print(f"   📄 {os.path.basename(p)}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
