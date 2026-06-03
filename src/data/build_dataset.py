"""
src/data/build_dataset.py — Xây dựng dataset .npz từ SQLite database.

Pipeline:
  1. Đọc dữ liệu labeled từ bảng csi_raw trong wifi_sensing.db
  2. Sessionize: phát hiện các khối liên tục cùng nhãn
  3. Với mỗi session: FFT Shift → Nội suy DC → Hampel → Normalize → Resample → Sliding Window
  4. Train/Test split theo session (không shuffle để tránh leakage)
  5. Lưu ra .npz + metadata JSON

Refactored từ build_db.py — DSP logic được import từ src.dsp.
"""

import json
import os
import sqlite3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DSP, DatasetConfig, Hardware, Paths
from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots, resample_to_grid


# ─── Session Processing ───────────────────────────────────────────────────────

def process_session(
    session_df: pd.DataFrame,
    session_id: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict | str]:
    """
    Xử lý một session liên tục thành tensor CSI windows.

    Args:
        session_df: DataFrame của session (đã có cột ts_float, mac, amp, label)
        session_id: ID của session

    Returns:
        (X_session, Y_session, meta) nếu thành công, hoặc (None, None, msg) nếu bị bỏ qua.
        X_session shape: (N_windows, T, 53, N_macs)
    """
    label_str = session_df['label'].iloc[0]
    if label_str not in DatasetConfig.CLASS_MAP:
        return None, None, f"SKIP (Unknown label '{label_str}'): Session {session_id}"

    label_id = DatasetConfig.CLASS_MAP[label_str]

    # ── Parse amplitudes ──
    data_rows = {mac: [] for mac in Hardware.TARGET_MACS}
    start_ts  = session_df['ts_float'].min()

    for _, row in session_df.iterrows():
        mac = row['mac']
        if mac not in Hardware.TARGET_MACS:
            continue
        try:
            amp = np.array(json.loads(row['amp']), dtype=float)
        except (json.JSONDecodeError, TypeError):
            continue
        if len(amp) == 64:
            data_rows[mac].append(np.insert(amp, 0, row['ts_float'] - start_ts))

    if any(len(rows) < 100 for rows in data_rows.values()):
        return None, None, f"SKIP (Not enough packets): Session {session_id}"

    # ── Per-MAC processing ──
    mac_frames = {}
    mac_stats  = {}
    t_start_common = -np.inf
    t_end_common   =  np.inf

    for mac in Hardware.TARGET_MACS:
        arr = np.array(data_rows[mac])
        t_raw_max = arr[:, 0].max()
        n_raw     = len(arr)

        # Trim đầu/cuối
        mask = (arr[:, 0] >= DSP.TRIM_SEC) & (arr[:, 0] <= (t_raw_max - DSP.TRIM_SEC))
        arr  = arr[mask]

        if len(arr) < 100:
            return None, None, f"SKIP (Too short after trim): Session {session_id}"

        duration  = arr[:, 0].max() - arr[:, 0].min()
        actual_fs = len(arr) / duration if duration > 0 else 0

        if actual_fs < DSP.MIN_FS_THRESHOLD:
            return None, None, f"SKIP (Low freq {actual_fs:.1f}Hz): Session {session_id}"

        mac_frames[mac] = arr
        mac_stats[mac]  = {
            "mac": mac,
            "n_packets_raw":  int(n_raw),
            "n_packets_trim": int(len(arr)),
            "trim_fs_hz":     round(actual_fs, 2),
            "duration_trim_s": round(duration, 2),
        }
        t_start_common = max(t_start_common, arr[:, 0].min())
        t_end_common   = min(t_end_common,   arr[:, 0].max())

    if t_end_common - t_start_common < DSP.WINDOW_SEC:
        return None, None, f"SKIP (Common duration < {DSP.WINDOW_SEC}s): Session {session_id}"

    # ── Resample + DSP pipeline ──
    t_global     = np.arange(t_start_common, t_end_common, 1.0 / DSP.TARGET_FS)
    win_samples  = int(DSP.TARGET_FS * DSP.WINDOW_SEC)
    step         = int(win_samples * (1.0 - DSP.OVERLAP_PCT))

    mac_window_list = []

    for mac in Hardware.TARGET_MACS:
        arr     = mac_frames[mac]
        t_mac   = arr[:, 0]
        csi_raw = arr[:, 1:]   # (T, 64)

        csi = fft_shift(csi_raw)
        csi = interpolate_dc_pilots(csi)
        csi = hampel_filter(csi)
        csi = csi[:, DSP.EFFECTIVE_SC_START:DSP.EFFECTIVE_SC_END]   # (T, 53)

        norms = np.linalg.norm(csi, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        csi = csi / norms

        csi_rs = resample_to_grid(csi, t_mac, t_global)   # (T_new, 53)

        n_total = len(csi_rs)
        windows = [
            csi_rs[s : s + win_samples]
            for s in range(0, n_total - win_samples + 1, step)
        ]
        mac_window_list.append(np.array(windows))   # (N_win, T, 53)

    min_wins = min(t.shape[0] for t in mac_window_list)
    mac_window_list = [t[:min_wins] for t in mac_window_list]

    # Stack MIMO: (N_win, T, 53, N_macs)
    X_session = np.stack(mac_window_list, axis=-1)
    Y_session = np.full(X_session.shape[0], label_id, dtype=np.int8)

    meta = {
        "session_id":         session_id,
        "label_str":          label_str,
        "label_id":           label_id,
        "overlap_duration_s": round(t_end_common - t_start_common, 2),
        "n_windows":          int(X_session.shape[0]),
        "macs":               [mac_stats[m] for m in Hardware.TARGET_MACS],
    }
    return X_session, Y_session, meta


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_dataset_samples(X: np.ndarray, Y: np.ndarray, num_samples: int = 3, save_path: str = "dataset_samples.png"):
    """Vẽ heatmap mẫu từ dataset để kiểm tra trực quan."""
    print(f"\n[INFO] Generating {num_samples} sample plots...")
    indices = np.random.choice(X.shape[0], min(num_samples, X.shape[0]), replace=False)
    class_names_inv = DatasetConfig.CLASS_NAMES

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, idx in enumerate(indices):
        tensor     = X[idx]   # (T, 53, 2)
        label_name = class_names_inv.get(int(Y[idx]), "Unknown")
        for j, mac in enumerate(Hardware.TARGET_MACS):
            ax  = axes[i, j]
            cax = ax.imshow(tensor[:, :, j].T, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"MAC {j} [{mac}] — {label_name}")
            ax.set_xlabel("Time (samples)")
            ax.set_ylabel("Subcarrier index")
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[SUCCESS] Saved to '{save_path}'")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_db_dataset(
    db_path: str | None = None,
    output_npz: str | None = None,
    output_meta: str | None = None,
):
    """
    Entry-point: đọc DB → xử lý sessions → lưu .npz + meta JSON.

    Args:
        db_path: đường dẫn DB (default: config.Paths.DB_PATH)
        output_npz: đường dẫn output .npz (default: data/wifi_sensing_dataset.npz)
        output_meta: đường dẫn output .json (default: data/wifi_sensing_dataset_meta.json)
    """
    db_path     = db_path     or str(Paths.DB_PATH)
    output_npz  = output_npz  or str(Paths.DATASET_NPZ)
    output_meta = output_meta or str(Paths.DATA_DIR / "wifi_sensing_dataset_meta.json")

    t0 = time.time()
    print("🚀 STARTING DATABASE DATASET BUILD...\n")

    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return

    # ── Load ──
    conn  = sqlite3.connect(db_path)
    query = "SELECT timestamp, mac, amp, label FROM csi_raw WHERE label IS NOT NULL ORDER BY id ASC"
    df    = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ No labeled data found in the database!")
        return

    # ── Sessionize ──
    df['ts_float']      = pd.to_datetime(df['timestamp']).astype(int) / 10**9
    df['label_changed'] = df['label'] != df['label'].shift(1)
    df['time_gap']      = df['ts_float'].diff() > 2.0
    df['session_id']    = (df['label_changed'] | df['time_gap']).cumsum()

    sessions = df.groupby('session_id')
    print(f"📁 Found {len(sessions)} labeled recording sessions.\n")

    results, skipped = [], []

    for session_id, session_df in sessions:
        duration = session_df['ts_float'].max() - session_df['ts_float'].min()
        if duration < DSP.MIN_TOTAL_SEC:
            skipped.append(f"SKIP (dur={duration:.1f}s < {DSP.MIN_TOTAL_SEC}s): Session {session_id}")
            continue

        X, Y, meta = process_session(session_df, session_id)
        if X is not None:
            results.append({"X": X, "Y": Y, "meta": meta, "label": meta["label_id"]})
            print(f"   ✅ Session {session_id:02d} ({meta['label_str']}) → {meta['n_windows']} windows")
        else:
            skipped.append(meta)
            print(f"   ⚠️  {meta}")

    if not results:
        print("\n❌ No valid data extracted after DSP filtering!")
        return

    # ── Class balance ──
    print("\n" + "─" * 50)
    print("📊 CLASS BALANCE:")
    for name, lb in DatasetConfig.CLASS_MAP.items():
        class_res   = [r for r in results if r["label"] == lb]
        total_wins  = sum(r["meta"]["n_windows"] for r in class_res)
        print(f"   {name:>10} (label={lb}): {len(class_res):>3} sessions | {total_wins:>5} windows")

    # ── Train/Test split ──
    train_X, train_Y, test_X, test_Y = [], [], [], []
    split_meta = []

    for r in results:
        X, Y  = r["X"], r["Y"]
        n      = len(Y)
        n_train = max(1, int(n * (1 - DatasetConfig.TEST_SIZE)))
        n_test  = n - n_train

        train_X.append(X[:n_train])
        train_Y.append(Y[:n_train])
        test_X.append(X[n_train:])
        test_Y.append(Y[n_train:])
        split_meta.append({**r["meta"], "n_train": n_train, "n_test": n_test})

    X_train = np.vstack(train_X)
    Y_train = np.concatenate(train_Y)
    X_test  = np.vstack(test_X)
    Y_test  = np.concatenate(test_Y)

    # ── Save ──
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

    all_meta = {
        "config": {
            "WINDOW_SEC":  DSP.WINDOW_SEC,
            "TARGET_FS":   DSP.TARGET_FS,
            "OVERLAP_PCT": DSP.OVERLAP_PCT,
            "TRIM_SEC":    DSP.TRIM_SEC,
        },
        "sessions_processed": split_meta,
        "split_summary": {
            "train": {"n_windows": int(X_train.shape[0])},
            "test":  {"n_windows": int(X_test.shape[0])},
        },
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print(f"✅ FINISHED in {time.time() - t0:.2f}s")
    print(f"   Tensor shape : (N, {int(DSP.TARGET_FS * DSP.WINDOW_SEC)}, 53, {len(Hardware.TARGET_MACS)})")
    print(f"   TRAIN        : {X_train.shape[0]:,} frames")
    print(f"   TEST         : {X_test.shape[0]:,} frames")
    print(f"💾 Saved: {output_npz}")
    print("=" * 55)

    plot_dataset_samples(X_train, Y_train, num_samples=3, save_path="dataset_samples.png")
