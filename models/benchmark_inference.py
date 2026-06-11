"""
models/benchmark_inference.py — Đo latency inference ONNX trên CPU.

Chạy trên máy local:
    python models/benchmark_inference.py --onnx-dir models/onnx

Chạy trên Raspberry Pi 4:
    python benchmark_inference.py --onnx-dir ./onnx --threads 4 --runs 200

Yêu cầu:
    pip install onnxruntime numpy matplotlib
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import onnxruntime as ort

# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark_model(
    onnx_path: str,
    input_shape: tuple = (1, 2, 250, 53),
    warmup: int = 50,
    runs: int = 300,
    threads: int = 0,            # 0 = dùng tất cả core có sẵn
) -> dict:
    """
    Trả về dict chứa các số liệu latency (ms) của 1 lần inference.
    Không tính thời gian DSP, chỉ đo forward pass của model.
    """
    opts = ort.SessionOptions()
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess       = ort.InferenceSession(onnx_path, sess_options=opts,
                                      providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy      = np.random.randn(*input_shape).astype(np.float32)

    # Warmup — tránh ảnh hưởng của JIT / cache miss lần đầu
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)  # → ms

    arr = np.array(latencies)
    return {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "min":   float(np.min(arr)),
        "max":   float(np.max(arr)),
        "p50":   float(np.percentile(arr, 50)),
        "p95":   float(np.percentile(arr, 95)),
        "p99":   float(np.percentile(arr, 99)),
        "raw":   arr,
    }


# ─── Report ───────────────────────────────────────────────────────────────────

def print_table(results: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  {'Model':<18} {'Mean':>8} {'Std':>7} {'Min':>7} {'P50':>7} {'P95':>7} {'P99':>7}")
    print(f"  {'':18} {'(ms)':>8} {'(ms)':>7} {'(ms)':>7} {'(ms)':>7} {'(ms)':>7} {'(ms)':>7}")
    print("-" * 78)
    for name, m in results.items():
        print(
            f"  {name:<18} {m['mean']:>8.2f} {m['std']:>7.2f} "
            f"{m['min']:>7.2f} {m['p50']:>7.2f} {m['p95']:>7.2f} {m['p99']:>7.2f}"
        )
    print("=" * 78 + "\n")


def plot_results(results: dict, out_path: str | None = None) -> None:
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds  = [results[n]["std"]  for n in names]
    p95s  = [results[n]["p95"]  for n in names]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c", "#e67e22"][:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Inference Latency Benchmark — CPU (ONNX Runtime)", fontsize=13, fontweight="bold")

    # ── Bar chart: mean ± std ──
    ax = axes[0]
    x  = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white",
                  yerr=stds, capsize=5, error_kw=dict(lw=1.5, capthick=1.5))
    ax.bar(x, p95s, color=colors, alpha=0.25, edgecolor="none")

    for bar, mean, p95 in zip(bars, means, p95s):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + max(stds) * 0.3,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title("Mean ± Std  (thanh mờ = P95)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f ms"))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Box plot: phân phối raw ──
    ax2 = axes[1]
    raw_data = [results[n]["raw"] for n in names]
    bp = ax2.boxplot(raw_data, labels=names, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     whiskerprops=dict(lw=1.2),
                     capprops=dict(lw=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax2.set_ylabel("Latency (ms)", fontsize=10)
    ax2.set_title("Phân phối latency (box plot)", fontsize=10)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f ms"))
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(onnx_dir: str, warmup: int, runs: int, threads: int,
         input_t: int, save_plot: bool) -> None:

    onnx_files = sorted(
        f for f in os.listdir(onnx_dir) if f.endswith(".onnx")
    )
    if not onnx_files:
        print(f"Không tìm thấy file .onnx trong '{onnx_dir}'")
        return

    input_shape = (1, 2, input_t, 53)
    device_info = _detect_device()

    print(f"\nThiết bị  : {device_info}")
    print(f"Input     : {input_shape}  (batch=1, 2 MACs, T={input_t}, 53 subcarriers)")
    print(f"Threads   : {'auto' if threads == 0 else threads}")
    print(f"Warmup    : {warmup} lần  |  Benchmark: {runs} lần\n")

    results = {}
    for fname in onnx_files:
        name = fname.replace(".onnx", "")
        path = os.path.join(onnx_dir, fname)
        print(f"  Đang benchmark {name}...", end="", flush=True)
        results[name] = benchmark_model(path, input_shape, warmup, runs, threads)
        print(f"  done  (mean={results[name]['mean']:.2f} ms)")

    print_table(results)

    if save_plot:
        out_img = os.path.join(onnx_dir, "latency_benchmark.png")
        plot_results(results, out_path=out_img)
    else:
        plot_results(results)


def _detect_device() -> str:
    import platform
    info = f"{platform.processor() or platform.machine()}  |  {platform.system()} {platform.release()}"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "Model" in line and "Raspberry" in line:
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass
    return info


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference latency")
    parser.add_argument(
        "--onnx-dir", default=os.path.join(here, "onnx"),
        help="Thư mục chứa các file .onnx (mặc định: models/onnx/)",
    )
    parser.add_argument(
        "--warmup", default=50, type=int,
        help="Số lần warmup trước khi đo (mặc định: 50)",
    )
    parser.add_argument(
        "--runs", default=300, type=int,
        help="Số lần đo thực (mặc định: 300)",
    )
    parser.add_argument(
        "--threads", default=0, type=int,
        help="Số CPU thread cho ORT (0 = tự động; Pi 4: dùng 4)",
    )
    parser.add_argument(
        "--input-t", default=250, type=int,
        help="Số time-step trong input CSI (mặc định: 250)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Chỉ in bảng, không vẽ biểu đồ",
    )
    parser.add_argument(
        "--save-plot", action="store_true",
        help="Lưu biểu đồ ra file PNG thay vì hiển thị cửa sổ",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.no_plot:
        main(args.onnx_dir, args.warmup, args.runs, args.threads,
             args.input_t, args.save_plot)
    else:
        onnx_files = sorted(f for f in os.listdir(args.onnx_dir) if f.endswith(".onnx"))
        results = {}
        for fname in onnx_files:
            name = fname.replace(".onnx", "")
            path = os.path.join(args.onnx_dir, fname)
            print(f"  Đang benchmark {name}...", end="", flush=True)
            results[name] = benchmark_model(path, (1, 2, args.input_t, 53),
                                            args.warmup, args.runs, args.threads)
            print(f"  done")
        print_table(results)
