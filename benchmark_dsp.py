"""
benchmark_dsp.py — Đo latency từng bước DSP và toàn pipeline.

Chạy trên máy local:
    python benchmark_dsp.py

Chạy trên Raspberry Pi 4:
    python benchmark_dsp.py --runs 200

Yêu cầu: numpy scipy matplotlib (đã có trong môi trường chính)
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DSP
from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots, resample_to_grid
from src.dsp.pipeline import prepare_inference_input

# ─── Tham số giả lập đầu vào ──────────────────────────────────────────────────
# Giả lập tín hiệu WiFi CSI thô: T_RAW mẫu tại ~60 Hz trước khi resample
T_RAW  = 300                    # số mẫu thô từ Pi (~60 Hz × 5 s)
N_SC   = 64                     # số subcarrier đầu vào
T_GRID = DSP.TARGET_SAMPLES     # = 250 (sau resample về 50 Hz)
N_MACS = 2


def _make_inputs(t_raw: int = T_RAW) -> dict:
    """Tạo dữ liệu giả lập có shape thực tế."""
    t_mac   = np.linspace(0, DSP.WINDOW_SEC, t_raw)
    csi_raw = np.abs(np.random.randn(t_raw, N_SC)).astype(float) * 100.0
    t_global = np.arange(0, DSP.WINDOW_SEC, 1.0 / DSP.TARGET_FS)[:T_GRID]

    # Sau fft_shift + interpolate (dùng cho bước tiếp theo)
    csi_shifted = fft_shift(csi_raw)
    csi_interp  = interpolate_dc_pilots(csi_shifted)

    # Sau hampel + slice (dùng cho L2 norm)
    csi_filtered = hampel_filter(csi_interp)
    csi_sliced   = csi_filtered[:, DSP.EFFECTIVE_SC_START:DSP.EFFECTIVE_SC_END]

    return {
        "t_mac":       t_mac,
        "csi_raw":     csi_raw,
        "t_global":    t_global,
        "csi_shifted": csi_shifted,
        "csi_interp":  csi_interp,
        "csi_filtered": csi_filtered,
        "csi_sliced":  csi_sliced,
    }


# ─── Helpers đo thời gian ─────────────────────────────────────────────────────

def _bench(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies)
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "raw":  arr,
    }


# ─── Benchmark từng bước ──────────────────────────────────────────────────────

def run_step_benchmarks(inp: dict, warmup: int, runs: int) -> dict:
    results = {}

    print("  [1/7] fft_shift          ...", end="", flush=True)
    results["1. FFT Shift"] = _bench(
        lambda: fft_shift(inp["csi_raw"]), warmup, runs)
    print(f"  {results['1. FFT Shift']['mean']:.3f} ms")

    print("  [2/7] interpolate_dc     ...", end="", flush=True)
    results["2. DC/Pilot Interp"] = _bench(
        lambda: interpolate_dc_pilots(inp["csi_shifted"]), warmup, runs)
    print(f"  {results['2. DC/Pilot Interp']['mean']:.3f} ms")

    print("  [3/7] hampel_filter      ...", end="", flush=True)
    results["3. Hampel Filter"] = _bench(
        lambda: hampel_filter(inp["csi_interp"]), warmup, runs)
    print(f"  {results['3. Hampel Filter']['mean']:.3f} ms")

    print("  [4/7] slice subcarriers  ...", end="", flush=True)
    results["4. SC Slice"] = _bench(
        lambda: inp["csi_filtered"][:, DSP.EFFECTIVE_SC_START:DSP.EFFECTIVE_SC_END],
        warmup, runs)
    print(f"  {results['4. SC Slice']['mean']:.3f} ms")

    print("  [5/7] L2 normalize       ...", end="", flush=True)
    def _l2_norm():
        norms = np.linalg.norm(inp["csi_sliced"], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return inp["csi_sliced"] / norms
    results["5. L2 Normalize"] = _bench(_l2_norm, warmup, runs)
    print(f"  {results['5. L2 Normalize']['mean']:.3f} ms")

    print("  [6/7] resample_to_grid   ...", end="", flush=True)
    results["6. Resample"] = _bench(
        lambda: resample_to_grid(inp["csi_sliced"], inp["t_mac"], inp["t_global"]),
        warmup, runs)
    print(f"  {results['6. Resample']['mean']:.3f} ms")

    print("  [7/7] full pipeline×2MAC ...", end="", flush=True)
    snap = [(t, row) for t, row in zip(inp["t_mac"], inp["csi_raw"])]
    full_key = "7. Full Pipeline (×2 MACs)"
    # Giảm runs cho bước này vì mỗi lần chạy ~20ms (hampel ×2 MAC)
    full_runs = min(runs, 100)
    results[full_key] = _bench(
        lambda: prepare_inference_input(snap, snap,
                                        inp["t_mac"][0], inp["t_mac"][-1]),
        min(warmup, 10), full_runs)
    print(f"  {results[full_key]['mean']:.3f} ms  (n={full_runs})")

    return results


# ─── In bảng kết quả ──────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    print("\n" + "=" * 75)
    print(f"  {'Bước DSP':<24} {'Mean':>7} {'Std':>6} {'Min':>6} {'P50':>6} {'P95':>6} {'P99':>6}")
    print(f"  {'':24} {'(ms)':>7} {'(ms)':>6} {'(ms)':>6} {'(ms)':>6} {'(ms)':>6} {'(ms)':>6}")
    print("-" * 75)

    total_mean = 0.0
    for name, m in results.items():
        label = name.replace("\n", " ")
        sep = "─" * 75 if "Full" in name else ""
        if sep:
            print(sep)
        print(
            f"  {label:<24} {m['mean']:>7.3f} {m['std']:>6.3f} "
            f"{m['min']:>6.3f} {m['p50']:>6.3f} {m['p95']:>6.3f} {m['p99']:>6.3f}"
        )
        if "Full" not in name:
            total_mean += m["mean"]

    print("-" * 75)
    print(f"  {'Σ bước 1–6 (1 MAC)':<24} {total_mean:>7.3f} ms  (ước tính)")
    print("=" * 75 + "\n")


# ─── Biểu đồ ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: str | None = None) -> None:
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds  = [results[n]["std"]  for n in names]

    labels    = [n.replace("\n", "\n") for n in names]
    is_full   = ["Full" in n for n in names]
    colors    = ["#e74c3c" if f else "#3498db" for f in is_full]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"DSP Pipeline Latency Benchmark — CPU\n"
        f"Input: ({T_RAW}, 64)  →  Output: ({T_GRID}, 53) per MAC",
        fontsize=12, fontweight="bold",
    )

    # ── Bar chart ──
    ax = axes[0]
    ax.set_facecolor("#f0f2f5")
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white",
                  yerr=stds, capsize=4,
                  error_kw=dict(lw=1.5, capthick=1.5, ecolor="#555"))

    for bar, mean, full in zip(bars, means, is_full):
        va = "bottom"
        offset = max(stds) * 0.4
        ax.text(bar.get_x() + bar.get_width() / 2, mean + offset,
                f"{mean:.2f}", ha="center", va=va,
                fontsize=8.5, fontweight="bold" if full else "normal",
                color="#c0392b" if full else "#2c3e50")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, ha="center")
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title("Mean ± Std theo bước", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f ms"))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#3498db", alpha=0.85, label="Bước riêng lẻ"),
            Patch(color="#e74c3c", alpha=0.85, label="Full pipeline ×2 MACs"),
        ],
        fontsize=8.5, loc="upper left",
    )

    # ── Stacked bar: cấu trúc pipeline ──
    ax2 = axes[1]
    ax2.set_facecolor("#f0f2f5")
    step_names  = [n for n in names if "Full" not in n]
    step_means  = [results[n]["mean"] for n in step_names]
    step_labels = [n.replace("\n", " ") for n in step_names]
    step_colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#f39c12"]

    left = 0.0
    y    = 0
    for label, val, color in zip(step_labels, step_means, step_colors):
        ax2.barh(y, val, left=left, color=color, alpha=0.85,
                 edgecolor="white", height=0.5)
        if val > 0.05:
            ax2.text(left + val / 2, y, f"{val:.2f}",
                     ha="center", va="center", fontsize=8, color="white",
                     fontweight="bold")
        left += val

    full_key = [n for n in names if "Full" in n][0]
    ax2.barh(y + 1, results[full_key]["mean"], color="#e74c3c",
             alpha=0.85, edgecolor="white", height=0.5)
    ax2.text(results[full_key]["mean"] / 2, y + 1,
             f"{results[full_key]['mean']:.2f} ms",
             ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Σ bước 1–6\n(1 MAC)", "Full pipeline\n(×2 MACs)"], fontsize=9)
    ax2.set_xlabel("Latency (ms)", fontsize=10)
    ax2.set_title("Cấu trúc thời gian pipeline", fontsize=10)
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f ms"))
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch as P
    ax2.legend(
        handles=[P(color=c, alpha=0.85, label=l)
                 for c, l in zip(step_colors, step_labels)],
        fontsize=7.5, loc="lower right", ncol=2,
    )

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {out_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark latency từng bước DSP")
    parser.add_argument("--runs",    default=300, type=int,
                        help="Số lần đo thực (mặc định: 300)")
    parser.add_argument("--warmup",  default=30,  type=int,
                        help="Số lần warmup (mặc định: 30)")
    parser.add_argument("--t-raw",   default=T_RAW, type=int,
                        help=f"Số mẫu CSI thô đầu vào (mặc định: {T_RAW})")
    parser.add_argument("--no-plot", action="store_true",
                        help="Chỉ in bảng, không vẽ biểu đồ")
    parser.add_argument("--save-plot", action="store_true",
                        help="Lưu biểu đồ PNG thay vì hiển thị cửa sổ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device_info = "Unknown"
    try:
        import platform
        device_info = f"{platform.processor() or platform.machine()}  |  {platform.system()} {platform.release()}"
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "Model" in line and "Raspberry" in line:
                    device_info = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    print(f"\nThiết bị  : {device_info}")
    print(f"T_raw     : {args.t_raw} mẫu  →  T_grid: {T_GRID} mẫu (sau resample)")
    print(f"Warmup    : {args.warmup} lần  |  Benchmark: {args.runs} lần\n")

    inp = _make_inputs(args.t_raw)
    results = run_step_benchmarks(inp, warmup=args.warmup, runs=args.runs)
    print_table(results)

    if not args.no_plot:
        out_img = "dsp_latency_benchmark.png" if args.save_plot else None
        plot_results(results, out_path=out_img)
