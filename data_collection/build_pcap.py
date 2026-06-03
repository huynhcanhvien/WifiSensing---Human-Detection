import os
import glob
import numpy as np
from scipy import interpolate
import dpkt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import json
from tqdm import tqdm
import psutil

# ================= 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN =================
DATA_DIR = "data/Data"
CLASS_MAP = {"empty": 0, "breathing": 1}

TARGET_MACS = [
    bytes.fromhex("d4e9f4a36480"),
    bytes.fromhex("d4e9f4a43c18")
]

# ================= 2. CẤU HÌNH DSP & AI =================
WINDOW_SEC      = 5
TARGET_FS       = 50.0
MIN_FS_THRESHOLD = 40.0
OVERLAP_PCT     = 0.5
TRIM_SEC        = 10.0          # Cắt 10s đầu + 10s cuối
MIN_TOTAL_SEC   = 50.0          # ✅ File phải có ít nhất 50s dữ liệu thô

HAMPEL_WINDOW   = 5
HAMPEL_SIGMA    = 3
K_HAMPEL        = 1.4826

TEST_SIZE       = 0.2
# ====================================================================


# ──────────────────────────────────────────────────────────────────────
# DSP HELPERS
# ──────────────────────────────────────────────────────────────────────

def hampel_filter(input_matrix):
    """
    Hampel filter tiết kiệm RAM:
    - Xử lý từng subcarrier một (không tạo ma trận 3D khổng lồ)
    - Mỗi subcarrier dùng sliding_window_view 1D → chỉ tốn (n × w) RAM thay vì (n × 53 × w)
    - Không dùng pool lồng nhau, không tạo pandas object
    """
    from numpy.lib.stride_tricks import sliding_window_view

    w   = 2 * HAMPEL_WINDOW + 1
    out = input_matrix.copy()

    for sc in range(input_matrix.shape[1]):
        series = input_matrix[:, sc]
        padded  = np.pad(series, HAMPEL_WINDOW, mode='edge')
        windows = sliding_window_view(padded, w)              # (n, w)
        medians  = np.median(windows, axis=1)
        mads     = np.median(np.abs(windows - medians[:, None]), axis=1)
        threshold = HAMPEL_SIGMA * K_HAMPEL * mads
        outliers  = np.abs(series - medians) > threshold
        out[outliers, sc] = medians[outliers]

    return out


def interpolate_dc_pilots(csi_matrix):
    """
    Nội suy lấp đầy DC & Null subcarriers (ESP32 HT20).
    Vectorized theo chiều subcarrier (64 lần) thay vì theo chiều packet (2 triệu lần).
    np.interp xử lý toàn bộ n_packets cho 1 target index → nhanh và tiết kiệm RAM.
    """
    null_subcarriers  = [0, 1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34, 59, 60, 61, 62, 63]
    valid_idx = np.array([i for i in range(64) if i not in null_subcarriers])
    output    = np.empty_like(csi_matrix, dtype=float)

    # valid_vals: (n_valid, n_packets) — transpose để vectorize theo packet
    valid_vals = csi_matrix[:, valid_idx].T   # (n_valid, n_packets)

    for xi in range(64):
        # np.interp(xi, xp=valid_idx, fp=mỗi hàng của valid_vals)
        # → nội suy tại 1 điểm xi cho tất cả n_packets cùng lúc
        output[:, xi] = np.interp(xi, valid_idx, valid_vals[:, :],
                                   left=None, right=None) if False else                         _interp_col(xi, valid_idx, valid_vals)
    return output


def _interp_col(xi, xp, fp_matrix):
    """Nội suy tuyến tính tại xi cho tất cả packets — thuần NumPy, không scipy."""
    # Tìm vị trí xi nằm giữa 2 điểm nào trong xp
    idx = np.searchsorted(xp, xi)
    idx = np.clip(idx, 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0 = fp_matrix[idx - 1, :]   # (n_packets,)
    y1 = fp_matrix[idx,     :]   # (n_packets,)
    # Nội suy tuyến tính
    t = (xi - x0) / (x1 - x0) if x1 != x0 else 0.0
    return y0 + t * (y1 - y0)


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
        mac_stats  = {}          # ← lưu thống kê từng MAC cho metadata
        t_start_common = -np.inf
        t_end_common   =  np.inf

        for mac in TARGET_MACS:
            arr = np.array(data_rows[mac])
            t_raw_max = arr[:, 0].max()
            n_raw     = len(arr)
            raw_fs    = n_raw / t_raw_max if t_raw_max > 0 else 0

            # Trim 15s đầu/cuối
            mask = (arr[:, 0] >= TRIM_SEC) & (arr[:, 0] <= (t_raw_max - TRIM_SEC))
            arr  = arr[mask]

            if len(arr) < 100:
                return None, None, f"SKIP (sau trim quá ít mẫu): {os.path.basename(file_path)}"

            duration  = arr[:, 0].max() - arr[:, 0].min()
            actual_fs = len(arr) / duration if duration > 0 else 0
            gaps      = np.diff(arr[:, 0])
            if actual_fs < MIN_FS_THRESHOLD:
                return None, None, (f"SKIP (Hz={actual_fs:.1f} < {MIN_FS_THRESHOLD}): "
                                    f"{os.path.basename(file_path)}")

            mac_frames[mac] = arr
            mac_stats[mac]  = {
                "mac":             mac.hex(':').upper(),
                "n_packets_raw":   int(n_raw),
                "n_packets_trim":  int(len(arr)),
                "raw_fs_hz":       round(raw_fs, 2),
                "trim_fs_hz":      round(actual_fs, 2),
                "duration_trim_s": round(duration, 2),
                "gap_mean_ms":     round(float(np.mean(gaps)) * 1000, 2),
                "gap_max_ms":      round(float(np.max(gaps))  * 1000, 2),
                "gap_std_ms":      round(float(np.std(gaps))  * 1000, 2),
            }

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

            # FFT Shift → Interp DC → Hampel → cắt 53 sub → L2 Norm
            # del sau mỗi bước để giải phóng RAM ngay lập tức
            csi_shifted  = np.fft.fftshift(csi_raw, axes=1);  del csi_raw
            csi_interp   = interpolate_dc_pilots(csi_shifted); del csi_shifted
            csi_denoised = hampel_filter(csi_interp);          del csi_interp
            csi_eff      = csi_denoised[:, 6:59];              del csi_denoised

            norms = np.linalg.norm(csi_eff, axis=1, keepdims=True)
            norms[norms == 0] = 1
            csi_norm = csi_eff / norms;  del csi_eff

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

        overlap_duration = round(t_end_common - t_start_common, 2)
        meta = {
            "file":             os.path.basename(file_path),
            "label":            label_id,
            "raw_duration_s":   round(raw_duration, 2),
            "overlap_duration_s": overlap_duration,
            "n_windows":        int(X_file.shape[0]),
            "tensor_shape":     list(X_file.shape),
            "macs": [mac_stats[mac] for mac in TARGET_MACS],
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
    # Giới hạn worker thông minh theo RAM thực tế (mỗi file lớn ~2–3 GB RAM)
    available_gb = psutil.virtual_memory().available / 1e9
    # Mỗi file lớn cần ~3–4GB RAM (raw parse + DSP intermediate)
    # Cap tối đa 2 worker để tránh OOM dù máy có nhiều nhân
    ram_based    = max(1, int(available_gb // 4.0))
    n_workers    = min(2, multiprocessing.cpu_count(), ram_based)
    print(f"⚙️  RAM khả dụng: {available_gb:.1f} GB  →  dùng {n_workers} worker(s) (tối đa 2 để tránh OOM)\n")

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

    # ── Chia Train/Test theo thứ tự thời gian trong từng file ────────
    # 80 windows đầu → Train, 20 windows cuối → Test
    # Đảm bảo tính liên tục chuỗi thời gian, cân đối theo từng file
    train_X, train_Y = [], []
    test_X,  test_Y  = [], []
    file_split_meta  = []

    for r in results:
        X, Y = r["X"], r["Y"]
        n    = len(Y)
        n_train = max(1, int(n * (1 - TEST_SIZE)))  # 80% đầu
        n_test  = n - n_train                        # 20% cuối

        train_X.append(X[:n_train])
        train_Y.append(Y[:n_train])
        test_X.append(X[n_train:])
        test_Y.append(Y[n_train:])

        file_split_meta.append({
            **r["meta"],
            "n_train": n_train,
            "n_test":  n_test,
        })

    X_train = np.vstack(train_X);      Y_train = np.concatenate(train_Y)
    X_test  = np.vstack(test_X);       Y_test  = np.concatenate(test_Y)

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
            "SPLIT_METHOD":      "temporal_per_file",
        },
        "files_processed": file_split_meta,
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