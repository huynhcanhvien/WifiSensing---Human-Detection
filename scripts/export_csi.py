"""
scripts/export_csi.py — Trích xuất CSI raw từ wifi_sensing.db ra CSV/NPY.

Refactored từ export.py — DSP logic được import từ src.dsp.

Cách dùng:
    python scripts/export_csi.py                         # Xuất tất cả MACs
    python scripts/export_csi.py --mac d4e9f4a36480      # Chỉ 1 MAC
    python scripts/export_csi.py --session 0             # Chỉ session đầu
    python scripts/export_csi.py --list                  # Liệt kê sessions
    python scripts/export_csi.py --fs 50 --trim 2.0      # Cấu hình tùy chỉnh

Output (mặc định tại exports/):
    session00_<mac>_csi.npy      # (T, 64) float32
    session00_<mac>_csi.csv      # dạng CSV
    session00_<mac>_time.npy     # (T,) timestamps
    session00_<mac>_meta.json    # metadata session
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExportConfig, Hardware, Paths
from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots, resample_to_grid


# ─── DB Helpers ───────────────────────────────────────────────────────────────

def load_from_db(db_path: str, mac: str | None = None) -> pd.DataFrame:
    conn  = sqlite3.connect(db_path)
    query = "SELECT timestamp, mac, amp FROM csi_raw"
    params = []
    if mac:
        query += " WHERE mac = ?"
        params.append(mac)
    query += " ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    print(f"  Đọc {len(df):,} rows từ DB (mac={mac or 'all'})")
    return df


def parse_timestamps(df: pd.DataFrame) -> np.ndarray:
    try:
        ts = pd.to_datetime(df["timestamp"])
        t0 = ts.iloc[0]
        return (ts - t0).dt.total_seconds().values
    except Exception:
        print("  [WARN] Không parse được timestamp, dùng index fallback")
        return np.arange(len(df)) / ExportConfig.TARGET_FS


def parse_amp_column(df: pd.DataFrame) -> np.ndarray:
    amps = []
    for row in df["amp"]:
        try:
            arr = json.loads(row)
            if len(arr) >= 64:
                amps.append(arr[:64])
            else:
                amps.append(arr + [0.0] * (64 - len(arr)))
        except Exception:
            amps.append([0.0] * 64)
    return np.array(amps, dtype=np.float32)


def detect_sessions(time_arr: np.ndarray, gap_sec: float = 5.0) -> list[tuple[int, int]]:
    if len(time_arr) == 0:
        return []
    breaks = np.where(np.diff(time_arr) > gap_sec)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends   = np.concatenate([breaks, [len(time_arr)]])
    return list(zip(starts.tolist(), ends.tolist()))


# ─── Export Per-MAC ───────────────────────────────────────────────────────────

def export_mac(
    mac: str,
    cfg: dict,
    session_id: int | None = None,
):
    print(f"\n{'=' * 55}\n  MAC: {mac}\n{'=' * 55}")
    df = load_from_db(cfg["db_path"], mac=mac)
    if df.empty:
        print("  [SKIP] Không có dữ liệu.")
        return

    time_raw = parse_timestamps(df)
    csi_raw  = parse_amp_column(df)
    sessions = detect_sessions(time_raw)
    print(f"  Phát hiện {len(sessions)} session(s)")

    for s_idx, (s_start, s_end) in enumerate(sessions):
        if session_id is not None and s_idx != session_id:
            continue

        t_seg    = time_raw[s_start:s_end] - time_raw[s_start]
        csi_seg  = csi_raw[s_start:s_end]
        n_frames = s_end - s_start
        duration = t_seg[-1] if len(t_seg) > 1 else 0
        actual_fs = n_frames / duration if duration > 0 else 0

        print(f"\n  [Session {s_idx}] frames={n_frames:,}  duration={duration:.1f}s  ~{actual_fs:.1f}Hz")

        if n_frames < 10:
            print("  [SKIP] Quá ít frame.")
            continue

        # DSP Pipeline
        print("  → FFT shift...")
        csi = fft_shift(csi_seg)

        print("  → Nội suy DC pilots...")
        csi = interpolate_dc_pilots(csi)

        print("  → Hampel filter...")
        csi = hampel_filter(csi)

        # Resample
        trim_start = cfg["trim_start"]
        window_sec = cfg["window_sec"]
        t_fs       = cfg["target_fs"]

        t_min = t_seg.min() + trim_start
        t_max = t_seg.max() if window_sec is None else t_min + window_sec
        t_max = min(t_max, t_seg.max())
        t_grid = np.arange(t_min, t_max, 1.0 / t_fs)

        if len(t_grid) == 0:
            print(f"  [ERR] Không có dữ liệu sau khi trim {trim_start}s")
            continue

        mask     = (t_seg >= t_min - 1.0) & (t_seg <= t_max + 1.0)
        csi_seg2 = csi[mask]
        t_seg2   = t_seg[mask]

        if len(t_seg2) < 2:
            print("  [ERR] Không đủ frames để resample")
            continue

        print(f"  → Resample về {t_fs} Hz...")
        csi_rs = resample_to_grid(csi_seg2, t_seg2, t_grid).astype(np.float32)
        t_out  = t_grid - t_grid[0]

        print(f"  ✓ Output: {csi_rs.shape}  ({t_out[-1]:.1f}s @ {t_fs}Hz)")

        output_dir = cfg["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        tag = f"session{s_idx:02d}_{mac}"

        np.save(os.path.join(output_dir, f"{tag}_csi.npy"),  csi_rs)
        np.save(os.path.join(output_dir, f"{tag}_time.npy"), t_out)

        df_out = pd.DataFrame(csi_rs, columns=[f"sc{i}" for i in range(64)])
        df_out.insert(0, "time_sec", t_out)
        df_out.to_csv(os.path.join(output_dir, f"{tag}_csi.csv"), index=False, float_format="%.4f")

        meta = {
            "mac":                  mac,
            "session_idx":          s_idx,
            "n_frames_raw":         int(n_frames),
            "n_frames_resampled":   int(len(t_out)),
            "duration_sec":         float(round(t_out[-1], 3)),
            "target_fs_hz":         t_fs,
            "trim_start_sec":       trim_start,
            "window_sec":           window_sec,
            "exported_at":          datetime.now().isoformat(),
        }
        with open(os.path.join(output_dir, f"{tag}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Đã lưu: {tag}_csi.npy / .csv / _time.npy / _meta.json")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trích xuất CSI từ wifi_sensing.db")
    parser.add_argument("--db",      default=str(Paths.DB_PATH),                        help="Path to .db file")
    parser.add_argument("--out",     default=str(ExportConfig.OUTPUT_DIR),              help="Output directory")
    parser.add_argument("--mac",     default=None,                                       help="Chỉ xuất 1 MAC")
    parser.add_argument("--session", default=None,  type=int,                           help="Chỉ xuất session index N")
    parser.add_argument("--fs",      default=ExportConfig.TARGET_FS, type=float,        help="Target sample rate (Hz)")
    parser.add_argument("--trim",    default=0.0,   type=float,                         help="Bỏ N giây đầu")
    parser.add_argument("--window",  default=None,  type=float,                         help="Chỉ lấy N giây (None=tất cả)")
    parser.add_argument("--list",    action="store_true",                                help="Liệt kê sessions, không xuất")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = {
        "db_path":    args.db,
        "output_dir": args.out,
        "target_fs":  args.fs,
        "trim_start": args.trim,
        "window_sec": args.window,
    }

    if not os.path.exists(cfg["db_path"]):
        print(f"[ERR] Không tìm thấy: {cfg['db_path']}")
        sys.exit(1)

    conn  = sqlite3.connect(cfg["db_path"])
    macs  = pd.read_sql_query("SELECT mac, COUNT(*) as n FROM csi_raw GROUP BY mac", conn)
    conn.close()

    if args.list:
        for mac in ([args.mac] if args.mac else macs["mac"].tolist()):
            df  = load_from_db(cfg["db_path"], mac=mac)
            t   = parse_timestamps(df)
            ses = detect_sessions(t)
            print(f"\n  MAC {mac} — {len(ses)} session(s):")
            for i, (s, e) in enumerate(ses):
                dur = t[e-1] - t[s] if e - s > 1 else 0
                print(f"    [{i}] frames={e-s:,}  duration={dur:.1f}s")
    else:
        macs_to_export = [args.mac] if args.mac else Hardware.TARGET_MACS
        for mac in macs_to_export:
            export_mac(mac, cfg, session_id=args.session)
        print(f"\n✅ Xong! Files: {os.path.abspath(cfg['output_dir'])}/")
