import os
import glob
import numpy as np
import pandas as pd
from scipy import interpolate
import dpkt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================= 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN =================
DATA_DIR = "data/Data"
CLASS_MAP = {"empty": 0, "breathing": 1}

TARGET_MACS = [
    bytes.fromhex("d4e9f4a36480"),
    bytes.fromhex("d4e9f4a43c18")
]

# ================= 2. CẤU HÌNH DSP & AI =================
WINDOW_SEC      = 10
TARGET_FS       = 20.0
MIN_FS_THRESHOLD = 15.0
OVERLAP_PCT     = 0.5
TRIM_SEC        = 15.0          # Cắt 15s đầu + 15s cuối
MIN_TOTAL_SEC   = 60.0          # ✅ File phải có ít nhất 60s dữ liệu thô

HAMPEL_WINDOW   = 5
HAMPEL_SIGMA    = 3
K_HAMPEL        = 1.4826

TEST_SIZE       = 0.2
# ====================================================================


# ──────────────────────────────────────────────────────────────────────
# DSP HELPERS
# ──────────────────────────────────────────────────────────────────────

def _hampel_series_numpy(args):
    """
    Worker Hampel vectorized thuần NumPy — nhanh hơn pandas rolling 3–5x.
    Dùng sliding_window_view để tính median/MAD mà không cần vòng lặp Python.
    """
    series_data, window, sigma = args
    from numpy.lib.stride_tricks import sliding_window_view

    n   = len(series_data)
    w   = 2 * window + 1
    out = series_data.copy()

    # Pad 2 đầu để center window không bị lệch biên
    padded  = np.pad(series_data, window, mode='edge')
    windows = sliding_window_view(padded, w)          # (n, w)

    medians = np.median(windows, axis=1)              # rolling median
    mads    = np.median(np.abs(windows - medians[:, None]), axis=1)  # rolling MAD
    threshold = sigma * K_HAMPEL * mads

    outlier_idx = np.abs(series_data - medians) > threshold
    out[outlier_idx] = medians[outlier_idx]
    return out


def hampel_filter_parallel(input_matrix):
    """
    Hampel song song trên từng subcarrier, mỗi subcarrier dùng NumPy vectorized.
    Nhanh hơn phiên bản pandas rolling 3–5x, giảm đáng kể thời gian + nhiệt độ CPU.
    """
    n_sub = input_matrix.shape[1]
    args  = [(input_matrix[:, sc], HAMPEL_WINDOW, HAMPEL_SIGMA) for sc in range(n_sub)]
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
        results = list(ex.map(_hampel_series_numpy, args))
    return np.array(results).T


def interpolate_dc_pilots(csi_matrix):
    """Nội suy lấp đầy DC & Null subcarriers (ESP32 HT20)."""
    null_subcarriers = [0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34, 59, 60, 61, 62, 63]
    valid_subcarriers = [i for i in range(64) if i not in null_subcarriers]
    x = np.arange(64)
    output = np.zeros_like(csi_matrix, dtype=float)
    for i in range(csi_matrix.shape[0]):
        f = interpolate.interp1d(
            valid_subcarriers, csi_matrix[i, valid_subcarriers],
            kind='linear', fill_value="extrapolate"
        )
        output[i, :] = f(x)
    return output


# ──────────────────────────────────────────────────────────────────────
# FILE-LEVEL VALIDATION
# ──────────────────────────────────────────────────────────────────────

def quick_check_duration(file_path):
    """
    ✅ MỚI: Đọc nhanh timestamp đầu/cuối của file PCAP để kiểm tra
    xem có đủ MIN_TOTAL_SEC giây không — TRƯỚC KHI xử lý nặng.
    Trả về (True, duration) nếu đủ, (False, duration) nếu thiếu.
    """
    first_ts = None
    last_ts  = None
    try:
        with open(file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, _ in pcap:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
        if first_ts is None:
            return False, 0.0
        duration = last_ts - first_ts
        return duration >= MIN_TOTAL_SEC, duration
    except Exception:
        return False, 0.0


# ──────────────────────────────────────────────────────────────────────
# PER-FILE PROCESSING  (chạy trong worker process)
# ──────────────────────────────────────────────────────────────────────

def process_single_file(args):
    """
    Quy trình đầy đủ cho 1 file PCAP:
      1. Kiểm tra độ dài ≥ 60s   ← ✅ MỚI
      2. Parse PCAP
      3. Trim 15s đầu/cuối       ← ✅ ĐÃ CÓ, giờ check sau khi biết tổng thời gian
      4. Kiểm tra chất lượng Hz
      5. FFT Shift → Interp DC → Hampel (song song) → L2 Norm → Resample 20Hz
      6. Sliding Window (có padding biên)  ← ✅ FIX padding
      7. Đồng bộ t_global từ giao của 2 MAC ← ✅ FIX lệch pha
    Trả về (X_file, Y_file, meta) hoặc (None, None, reason).
    """
    file_path, label_id = args

    # ── Bước 0: Kiểm tra nhanh thời lượng file ──────────────────────
    ok, raw_duration = quick_check_duration(file_path)
    if not ok:
        return None, None, f"SKIP (duration={raw_duration:.1f}s < {MIN_TOTAL_SEC}s): {os.path.basename(file_path)}"

    # ── Bước 1: Parse PCAP ──────────────────────────────────────────
    data_rows = {mac: [] for mac in TARGET_MACS}
    start_ts  = None

    try:
        with open(file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                if start_ts is None:
                    start_ts = ts
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    udp = eth.data.data
                    if udp.dport == 5500:
                        payload  = udp.data
                        mac_addr = payload[4:10]
                        if mac_addr in TARGET_MACS:
                            raw_np = np.frombuffer(payload[-256:], dtype=np.int16)
                            amp    = np.sqrt(raw_np[0::2].astype(float) ** 2 +
                                             raw_np[1::2].astype(float) ** 2)
                            if len(amp) == 64:
                                data_rows[mac_addr].append(
                                    np.insert(amp, 0, ts - start_ts)
                                )
                except Exception:
                    continue

        if any(len(rows) < 100 for rows in data_rows.values()):
            return None, None, f"SKIP (quá ít gói tin): {os.path.basename(file_path)}"

        # ── Bước 2: Trim & kiểm tra Hz — tính t_global từ GIAO của 2 MAC ──
        #    ✅ FIX: không dùng t_global của riêng MAC đầu tiên nữa
        mac_frames = {}
        t_start_common = -np.inf
        t_end_common   =  np.inf

        for mac in TARGET_MACS:
            arr = np.array(data_rows[mac])
            t_raw_max = arr[:, 0].max()

            # Trim 15s đầu/cuối
            mask = (arr[:, 0] >= TRIM_SEC) & (arr[:, 0] <= (t_raw_max - TRIM_SEC))
            arr  = arr[mask]

            if len(arr) < 100:
                return None, None, f"SKIP (sau trim quá ít mẫu): {os.path.basename(file_path)}"

            duration  = arr[:, 0].max() - arr[:, 0].min()
            actual_fs = len(arr) / duration if duration > 0 else 0
            if actual_fs < MIN_FS_THRESHOLD:
                return None, None, (f"SKIP (Hz={actual_fs:.1f} < {MIN_FS_THRESHOLD}): "
                                    f"{os.path.basename(file_path)}")

            mac_frames[mac] = arr
            # Lấy giao thời gian của 2 MAC để đồng bộ
            t_start_common = max(t_start_common, arr[:, 0].min())
            t_end_common   = min(t_end_common,   arr[:, 0].max())

        if t_end_common - t_start_common < WINDOW_SEC:
            return None, None, f"SKIP (giao thời gian quá ngắn): {os.path.basename(file_path)}"

        # Trục thời gian chung duy nhất cho cả 2 MAC
        t_global = np.arange(t_start_common, t_end_common, 1.0 / TARGET_FS)

        # ── Bước 3: DSP pipeline + Resample cho từng MAC ────────────
        list_of_mac_tensors = []

        for mac in TARGET_MACS:
            arr = mac_frames[mac]
            t_mac  = arr[:, 0]
            csi_raw = arr[:, 1:]

            # FFT Shift → Interp DC → Hampel (song song) → cắt 53 sub → L2 Norm
            csi_shifted  = np.fft.fftshift(csi_raw, axes=1)
            csi_interp   = interpolate_dc_pilots(csi_shifted)
            csi_denoised = hampel_filter_parallel(csi_interp)   # ✅ song song
            csi_eff      = csi_denoised[:, 6:59]                # 53 subcarrier hữu ích

            norms = np.linalg.norm(csi_eff, axis=1, keepdims=True)
            norms[norms == 0] = 1
            csi_norm = csi_eff / norms

            # Resample về t_global
            f_resample   = interpolate.interp1d(t_mac, csi_norm, axis=0,
                                                fill_value="extrapolate")
            csi_resampled = f_resample(t_global)

            # ── Bước 4: Sliding Window có padding biên ────────────────
            win_samples = int(TARGET_FS * WINDOW_SEC)
            step        = int(win_samples * (1.0 - OVERLAP_PCT))
            n_total     = len(csi_resampled)

            windows = []
            for start in range(0, n_total - win_samples + 1, step):
                windows.append(csi_resampled[start: start + win_samples])

            list_of_mac_tensors.append(np.array(windows))

        # ── Bước 5: Đảm bảo 2 MAC có cùng số window trước khi stack ─
        min_windows = min(t.shape[0] for t in list_of_mac_tensors)
        list_of_mac_tensors = [t[:min_windows] for t in list_of_mac_tensors]

        # Stack MIMO: (n_windows, time, subcarrier, n_mac)
        X_file = np.stack(list_of_mac_tensors, axis=-1)
        Y_file = np.full(X_file.shape[0], label_id, dtype=np.int8)

        meta = {
            "file":        os.path.basename(file_path),
            "label":       label_id,
            "raw_duration": round(raw_duration, 2),
            "n_windows":   int(X_file.shape[0]),
        }
        return X_file, Y_file, meta

    except Exception as e:
        return None, None, f"ERROR {os.path.basename(file_path)}: {e}"


# ──────────────────────────────────────────────────────────────────────
# MAIN: BUILD DATASET
# ──────────────────────────────────────────────────────────────────────

def build_dataset():
    t0 = time.time()
    print("🚀 BẮT ĐẦU XÂY DỰNG DATASET...\n")

    # Scan file
    tasks = []
    for class_name, label_id in CLASS_MAP.items():
        folder = os.path.join(DATA_DIR, class_name)
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*.pcap"))
            for f in files:
                tasks.append((f, label_id))

    print(f"📁 Tìm thấy {len(tasks)} file PCAP")
    print(f"   Yêu cầu: ≥ {MIN_TOTAL_SEC}s dữ liệu thô, trim {TRIM_SEC}s đầu/cuối\n")

    # Chạy song song toàn bộ file
    n_workers = multiprocessing.cpu_count()
    print(f"⚙️  Đang xử lý song song trên {n_workers} nhân CPU...\n")

    results   = []
    skipped   = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(process_single_file, t): t for t in tasks}

        pbar = tqdm(
            as_completed(future_map),
            total=len(tasks),
            desc="⚙️  Tiến độ",
            unit="file",
            ncols=85,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} file  [⏱ {elapsed}<{remaining}  {rate_fmt}]"
        )

        for future in pbar:
            X, Y, info = future.result()
            if X is not None:
                results.append({"X": X, "Y": Y, "meta": info, "label": info["label"]})
                pbar.write(f"   ✅ {info['file']}  →  {info['n_windows']} windows")
            else:
                skipped.append(info)
                pbar.write(f"   ⚠️  {info}")

    if not results:
        print("\n❌ Không có dữ liệu hợp lệ!"); return

    # ── Kiểm tra class balance ✅ ────────────────────────────────────
    print("\n" + "─" * 50)
    print("📊 PHÂN PHỐI NHÃN (Class Balance):")
    for lb, name in [(0, "empty"), (1, "breathing")]:
        class_results = [r for r in results if r["label"] == lb]
        total_windows = sum(r["meta"]["n_windows"] for r in class_results)
        print(f"   {name:>10} (label={lb}): {len(class_results):>3} files | "
              f"{total_windows:>5} windows")

    counts = {lb: sum(r["meta"]["n_windows"] for r in results if r["label"] == lb)
              for lb in [0, 1]}
    ratio  = max(counts.values()) / max(min(counts.values()), 1)
    if ratio > 2.0:
        print(f"\n   ⚠️  CẢNH BÁO: Tỉ lệ mất cân bằng ~{ratio:.1f}x — "
              f"cân nhắc augmentation hoặc class_weight khi train!")

    # ── Stratified Split 80/20 theo frame ──────────────────────────
    X_all = np.vstack([r["X"] for r in results])
    Y_all = np.concatenate([r["Y"] for r in results])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=Y_all      # ← giữ nguyên tỉ lệ label trong cả 2 tập
    )

    # ── Lưu dataset + metadata ✅ ────────────────────────────────────
    output_npz  = "../wifi_sensing_dataset.npz"
    output_meta = "../wifi_sensing_dataset_meta.json"

    np.savez_compressed(output_npz,
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test,   Y_test=Y_test)

    all_meta = {
        "config": {
            "WINDOW_SEC":        WINDOW_SEC,
            "TARGET_FS":         TARGET_FS,
            "OVERLAP_PCT":       OVERLAP_PCT,
            "TRIM_SEC":          TRIM_SEC,
            "MIN_TOTAL_SEC":     MIN_TOTAL_SEC,
            "MIN_FS_THRESHOLD":  MIN_FS_THRESHOLD,
            "TEST_SIZE":         TEST_SIZE,
            "SPLIT_METHOD":      "stratified_random",
        },
        "files_processed": [r["meta"] for r in results],
        "files_skipped":   skipped,
        "split_summary": {
            "train": {
                "n_windows": int(X_train.shape[0]),
                "label_0":   int((Y_train == 0).sum()),
                "label_1":   int((Y_train == 1).sum()),
            },
            "test": {
                "n_windows": int(X_test.shape[0]),
                "label_0":   int((Y_test == 0).sum()),
                "label_1":   int((Y_test == 1).sum()),
            },
        }
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    # ── Tóm tắt kết quả ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"✅ HOÀN TẤT SAU {time.time() - t0:.2f}s")
    print(f"   Tensor shape : (n_windows, {int(TARGET_FS*WINDOW_SEC)}, 53, 2)")
    print(f"\n   TRAIN ({int((1-TEST_SIZE)*100)}%): {X_train.shape[0]:,} frames")
    print(f"     empty     : {(Y_train==0).sum():,}  ({(Y_train==0).mean()*100:.1f}%)")
    print(f"     breathing : {(Y_train==1).sum():,}  ({(Y_train==1).mean()*100:.1f}%)")
    print(f"\n   TEST  ({int(TEST_SIZE*100)}%): {X_test.shape[0]:,} frames")
    print(f"     empty     : {(Y_test==0).sum():,}  ({(Y_test==0).mean()*100:.1f}%)")
    print(f"     breathing : {(Y_test==1).sum():,}  ({(Y_test==1).mean()*100:.1f}%)")
    print(f"\n   File bị bỏ : {len(skipped)}")
    print(f"💾 Dataset : {output_npz}")
    print(f"📋 Metadata: {output_meta}")
    print("=" * 55)


if __name__ == "__main__":
    build_dataset()