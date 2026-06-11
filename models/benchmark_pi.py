"""
models/benchmark_pi.py — Đo latency inference ONNX trên Raspberry Pi 4.

Chép file này + thư mục onnx/ lên Pi rồi chạy:
    python benchmark_pi.py
    python benchmark_pi.py --onnx-dir ./onnx --threads 4 --runs 200

Yêu cầu:  pip install onnxruntime numpy
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort


def benchmark(onnx_path: str, threads: int, warmup: int, runs: int) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 2, 250, 53).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    a = np.array(times)
    return {
        "mean": float(np.mean(a)),
        "std":  float(np.std(a)),
        "min":  float(np.min(a)),
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
        "p99":  float(np.percentile(a, 99)),
        "max":  float(np.max(a)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default=os.path.join(os.path.dirname(__file__), "onnx"))
    parser.add_argument("--threads",  default=4, type=int,
                        help="Số CPU thread (Pi 4 có 4 core, mặc định: 4)")
    parser.add_argument("--warmup",   default=30,  type=int)
    parser.add_argument("--runs",     default=200, type=int)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.onnx_dir) if f.endswith(".onnx"))
    if not files:
        print(f"Không tìm thấy file .onnx trong: {args.onnx_dir}")
        return

    # Thông tin thiết bị
    try:
        with open("/proc/cpuinfo") as f:
            model_line = next((l for l in f if l.startswith("Model")), None)
        device = model_line.split(":")[-1].strip() if model_line else "Unknown"
    except FileNotFoundError:
        import platform
        device = f"{platform.processor() or platform.machine()} | {platform.system()}"

    print(f"\nThiết bị : {device}")
    print(f"Threads  : {args.threads}  |  Warmup: {args.warmup}  |  Runs: {args.runs}")
    print(f"Input    : (1, 2, 250, 53)  — batch=1, 2 MACs, 250 time-steps, 53 subcarriers\n")

    results = {}
    for fname in files:
        name = fname.replace(".onnx", "")
        path = os.path.join(args.onnx_dir, fname)
        print(f"  {name} ...", end="", flush=True)
        results[name] = benchmark(path, args.threads, args.warmup, args.runs)
        print(f" {results[name]['mean']:.1f} ms")

    # Bảng kết quả
    W = max(len(n) for n in results) + 2
    print("\n" + "=" * (W + 58))
    print(f"  {'Model':<{W}} {'Mean':>7} {'Std':>6} {'Min':>6} {'P50':>6} {'P95':>6} {'P99':>6} {'Max':>6}")
    print(f"  {'':^{W}} {'(ms)':>7} {'(ms)':>6} {'(ms)':>6} {'(ms)':>6} {'(ms)':>6} {'(ms)':>6} {'(ms)':>6}")
    print("-" * (W + 58))
    for name, m in results.items():
        print(f"  {name:<{W}} {m['mean']:>7.1f} {m['std']:>6.1f} "
              f"{m['min']:>6.1f} {m['p50']:>6.1f} {m['p95']:>6.1f} "
              f"{m['p99']:>6.1f} {m['max']:>6.1f}")
    print("=" * (W + 58) + "\n")


if __name__ == "__main__":
    main()
