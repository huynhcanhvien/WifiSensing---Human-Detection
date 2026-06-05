"""
visualization/plot_comparison.py — Layout 3×2 so sánh CSI heatmap + waveform: Breathing vs Empty.

Layout:
    Hàng 0 (Breathing):  [MAC 0 heatmap]      |  [MAC 1 heatmap]
    Hàng 1 (Empty):      [MAC 0 heatmap]      |  [MAC 1 heatmap]
    Hàng 2 (Waveform):   [Breathing waveform] |  [Empty waveform]

Pipeline xử lý: fft_shift → interpolate_dc_pilots → hampel_filter → L2 normalize → slice 53 SC.
Tự động chọn file dài nhất; hiển thị cửa sổ 60 giây (tuỳ chỉnh được).

Cách dùng:
    python visualization/plot_comparison.py                         # tự tìm file dài nhất
    python visualization/plot_comparison.py --window 60 --start 5
    python visualization/plot_comparison.py \\
        --breathing data/Data/breathing/breath_xxx_realtime.csv \\
        --empty     data/Data/empty/empty_xxx_realtime.csv
    python visualization/plot_comparison.py --no-show
"""

import argparse
import glob
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DSP, Hardware
from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_ROOT  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Data")

N_SC = DSP.EFFECTIVE_SC_END - DSP.EFFECTIVE_SC_START  # 53 subcarrier hữu ích


# ─── DSP Pipeline ─────────────────────────────────────────────────────────────

def _apply_dsp(csi_raw: np.ndarray) -> np.ndarray:
    """Áp dụng pipeline DSP chuẩn và trả về (T, 53) L2-normalized."""
    csi = fft_shift(csi_raw)
    csi = interpolate_dc_pilots(csi)
    csi = hampel_filter(csi)
    csi = csi[:, DSP.EFFECTIVE_SC_START:DSP.EFFECTIVE_SC_END]
    norms = np.linalg.norm(csi, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return csi / norms


# ─── Data Loading ─────────────────────────────────────────────────────────────

def _normalize_mac(mac: str) -> str:
    return mac.lower().replace(':', '').replace('-', '')


def load_csv(file_path: str) -> dict[str, dict]:
    """Đọc CSV realtime, áp dụng DSP, trả về {mac: {'time', 'csi'(T,53)}}."""
    df = pd.read_csv(file_path)

    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    result = {}
    for mac in df['mac'].unique():
        sub     = df[df['mac'] == mac].sort_values('timestamp')
        t       = sub['timestamp'].values
        sc_cols = [c for c in sub.columns if c.startswith('sub_')]
        csi_raw = sub[sc_cols].values.astype(float)
        result[_normalize_mac(str(mac))] = {'time': t, 'csi': _apply_dsp(csi_raw)}

    return result


def load_pcap(file_path: str) -> dict[str, dict]:
    """Đọc PCAP, áp dụng DSP, trả về {mac_hex: {'time', 'csi'(T,53)}}."""
    try:
        import dpkt
    except ImportError:
        raise ImportError("Cần cài dpkt: pip install dpkt")

    data_rows: dict[bytes, list] = {mac: [] for mac in Hardware.TARGET_MACS_BYTES}
    start_ts = None

    with open(file_path, 'rb') as f:
        for ts, buf in dpkt.pcap.Reader(f):
            if start_ts is None:
                start_ts = ts
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                udp = eth.data.data
                if udp.dport == 5500:
                    payload  = udp.data
                    mac_addr = payload[4:10]
                    if mac_addr in Hardware.TARGET_MACS_BYTES:
                        raw = np.frombuffer(payload[-256:], dtype=np.int16)
                        amp = np.sqrt(raw[0::2].astype(float)**2 + raw[1::2].astype(float)**2)
                        data_rows[mac_addr].append(np.insert(amp, 0, ts - start_ts))
            except Exception:
                continue

    result = {}
    for mac_bytes, rows in data_rows.items():
        if len(rows) < 10:
            continue
        arr = np.array(rows)
        result[mac_bytes.hex()] = {'time': arr[:, 0], 'csi': _apply_dsp(arr[:, 1:])}

    return result


def load_file(file_path: str) -> dict[str, dict]:
    """Auto-detect định dạng theo extension (.csv / .pcap)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return load_csv(file_path)
    if ext == '.pcap':
        return load_pcap(file_path)
    raise ValueError(f"Định dạng không hỗ trợ: {ext!r}. Dùng .csv hoặc .pcap")


def _find_longest(label_dir: str, exts: tuple[str, ...] = ('.csv', '.pcap')) -> str | None:
    """Tìm file dữ liệu dài nhất trong thư mục (ưu tiên CSV realtime, sắp xếp theo size)."""
    candidates = []
    for ext in exts:
        candidates += glob.glob(os.path.join(label_dir, f"*{ext}"))
    csv_files   = [f for f in candidates if f.endswith('_realtime.csv')]
    other_files = [f for f in candidates if f not in csv_files]
    for group in [csv_files, other_files]:
        if group:
            return max(group, key=os.path.getsize)
    return None


# ─── MAC Selection ────────────────────────────────────────────────────────────

def _select_two_macs(data: dict[str, dict]) -> list[tuple[str, dict]]:
    """
    Chọn 2 MAC theo thứ tự ưu tiên từ Hardware.TARGET_MACS.
    Nếu không khớp MAC cấu hình, lấy 2 MAC đầu tiên trong dict.
    """
    preferred = [_normalize_mac(m) for m in Hardware.TARGET_MACS]
    ordered: list[tuple[str, dict]] = []

    for target in preferred:
        match = next((k for k in data if _normalize_mac(k) == target), None)
        if match:
            ordered.append((match, data[match]))

    if not ordered:
        ordered = list(data.items())

    if len(ordered) == 0:
        return []

    # Đảm bảo luôn trả về đúng 2 phần tử (lặp lại nếu chỉ có 1 MAC)
    while len(ordered) < 2:
        ordered.append(ordered[0])

    return ordered[:2]


# ─── Time Window ──────────────────────────────────────────────────────────────

def _slice_window(
    data: dict[str, dict],
    start_sec: float,
    window_sec: float,
) -> dict[str, dict]:
    """
    Cắt tất cả MAC trong data xuống cửa sổ [start_sec, start_sec + window_sec].
    Thời gian được reset về 0 từ điểm bắt đầu cửa sổ.
    """
    end_sec = start_sec + window_sec
    result  = {}
    for mac, d in data.items():
        t    = d['time']
        mask = (t >= start_sec) & (t <= end_sec)
        if not mask.any():
            result[mac] = d
            continue
        t_win = t[mask] - t[mask][0]
        result[mac] = {'time': t_win, 'csi': d['csi'][mask]}
    return result


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling average căn giữa, không zero-pad ở biên (dùng pandas min_periods)."""
    if window < 2 or len(x) < window:
        return x
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _draw_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    csi: np.ndarray,
    t: np.ndarray,
    title: str,
    cmap: str,
) -> None:
    """Vẽ heatmap CSI 2D (subcarrier × time)."""
    vmin = np.nanpercentile(csi, 5)
    vmax = np.nanpercentile(csi, 95)
    im = ax.imshow(
        csi.T,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        extent=[float(t[0]), float(t[-1]), 0, N_SC],
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )
    ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
    ax.set_xlabel("Thời gian (s)", fontsize=8)
    ax.set_ylabel("Subcarrier", fontsize=8)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Biên độ L2", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    n_samples = csi.shape[0]
    duration  = float(t[-1]) - float(t[0])
    ax.text(
        0.01, 0.97,
        f"{n_samples} mẫu  |  {duration:.1f}s",
        transform=ax.transAxes,
        fontsize=6, va='top', ha='left', color='white',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.35),
    )


def _draw_waveform(
    ax: plt.Axes,
    mac_list: list[tuple[str, dict]],
    state_label: str,
    colors: list[str],
    smooth_win: int = 30,
) -> None:
    """
    Vẽ waveform trung bình biên độ các subcarrier theo thời gian.

    mean_amp[t] = mean(csi[t, :])  → tín hiệu 1-D phản ánh sự thay đổi
    tổng thể của kênh WiFi: dao động rõ khi có hơi thở, phẳng khi trống.
    """
    ax.set_facecolor('#fafafa')

    for i, (mac_str, d) in enumerate(mac_list):
        t        = d['time']
        mean_amp = np.nanmean(d['csi'], axis=1)

        # Raw signal (mờ)
        ax.plot(t, mean_amp, color=colors[i], alpha=0.18, lw=0.7)

        # Smoothed signal (nổi bật)
        smoothed = _smooth(mean_amp, smooth_win)
        short_mac = Hardware.TARGET_MACS[i] if i < len(Hardware.TARGET_MACS) else mac_str
        ax.plot(t, smoothed, color=colors[i], alpha=0.92, lw=1.6,
                label=f"MAC {i} ({short_mac[-6:]})")

    ax.set_title(f"Waveform Biên Độ Trung Bình — {state_label}", fontsize=9, fontweight='bold', pad=5)
    ax.set_xlabel("Thời gian (s)", fontsize=8)
    ax.set_ylabel("Mean Amplitude\n(53 subcarriers)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.spines[['top', 'right']].set_visible(False)


# ─── Main Plot ────────────────────────────────────────────────────────────────

def plot_comparison(
    breathing_path: str,
    empty_path: str,
    window_sec: float = 60.0,
    start_sec: float  = 3.0,
    save: bool        = True,
    show: bool        = True,
) -> str | None:
    """
    Vẽ layout 3 hàng × 2 cột:
        Hàng 0 — Breathing heatmap  MAC 0  |  MAC 1
        Hàng 1 — Empty heatmap      MAC 0  |  MAC 1
        Hàng 2 — Waveform Breathing        |  Waveform Empty

    Args:
        breathing_path: Đường dẫn file CSV/PCAP trạng thái có người thở.
        empty_path:     Đường dẫn file CSV/PCAP trạng thái phòng trống.
        window_sec:     Độ dài cửa sổ thời gian cần hiển thị (giây).
        start_sec:      Điểm bắt đầu cửa sổ (giây) — bỏ qua nhiễu đầu file.

    Returns:
        Đường dẫn ảnh đã lưu (nếu save=True), None nếu không lưu.
    """
    print(f"📂 Breathing : {os.path.basename(breathing_path)}")
    print(f"📂 Empty     : {os.path.basename(empty_path)}")
    print(f"⏱️  Cửa sổ   : {start_sec:.0f}s → {start_sec + window_sec:.0f}s  ({window_sec:.0f}s)")
    print("⚙️  Đang áp dụng DSP pipeline...")

    breath_raw = _slice_window(load_file(breathing_path), start_sec, window_sec)
    empty_raw  = _slice_window(load_file(empty_path),     start_sec, window_sec)
    breath_macs = _select_two_macs(breath_raw)
    empty_macs  = _select_two_macs(empty_raw)

    if not breath_macs or not empty_macs:
        print("❌ Không tìm thấy dữ liệu MAC hợp lệ trong một hoặc cả hai file.")
        return None

    print(f"   Breathing MACs: {[m for m, _ in breath_macs]}")
    print(f"   Empty     MACs: {[m for m, _ in empty_macs]}")

    # ── Figure + GridSpec ──
    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[2.6, 2.6, 1.8],
        hspace=0.58, wspace=0.38,
        top=0.85, bottom=0.07, left=0.08, right=0.97,
    )

    fig.suptitle(
        f"So Sánh CSI Heatmap — Breathing vs Empty  "
        f"[{start_sec:.0f}s – {start_sec + window_sec:.0f}s]\n"
        "Pipeline: FFT Shift → DC/Pilot Interpolation → Hampel Filter → L2 Normalize",
        fontsize=12, fontweight='bold', y=0.97, linespacing=1.6,
    )

    # Nhãn hàng bên trái
    fig.text(0.013, 0.76, "BREATHING", ha='center', va='center',
             fontsize=10, fontweight='bold', rotation=90, color='#a0005e')
    fig.text(0.013, 0.50, "EMPTY",     ha='center', va='center',
             fontsize=10, fontweight='bold', rotation=90, color='#005ea0')
    fig.text(0.013, 0.18, "WAVEFORM",  ha='center', va='center',
             fontsize=10, fontweight='bold', rotation=90, color='#2d6a2d')

    # ── Hàng 0 & 1: Heatmaps ──
    heatmap_rows = [
        (0, breath_macs, "plasma",  "Breathing"),
        (1, empty_macs,  "viridis", "Empty"),
    ]
    for row_idx, mac_list, cmap, state_label in heatmap_rows:
        for col_idx in range(2):
            mac_str, d = mac_list[col_idx]
            short_mac  = (
                Hardware.TARGET_MACS[col_idx]
                if col_idx < len(Hardware.TARGET_MACS)
                else mac_str
            )
            ax = fig.add_subplot(gs[row_idx, col_idx])
            _draw_heatmap(
                ax=ax, fig=fig,
                csi=d['csi'], t=d['time'],
                title=f"[{state_label}]  MAC {col_idx} — {short_mac}",
                cmap=cmap,
            )

    # ── Hàng 2: Waveforms ──
    # Breathing — màu đỏ/cam (gợi plasma)
    ax_wave_breath = fig.add_subplot(gs[2, 0])
    _draw_waveform(ax_wave_breath, breath_macs, "Breathing",
                   colors=['#c0392b', '#e67e22'])

    # Empty — màu xanh lam/teal (gợi viridis)
    ax_wave_empty = fig.add_subplot(gs[2, 1])
    _draw_waveform(ax_wave_empty, empty_macs, "Empty",
                   colors=['#2471a3', '#1abc9c'])

    # Căn chỉnh trục Y giữa 2 waveform để so sánh công bằng
    all_vals = []
    for mac_list in [breath_macs, empty_macs]:
        for _, d in mac_list:
            all_vals.append(np.nanmean(d['csi'], axis=1))
    global_min = min(v.min() for v in all_vals)
    global_max = max(v.max() for v in all_vals)
    margin = (global_max - global_min) * 0.08
    for ax_w in [ax_wave_breath, ax_wave_empty]:
        ax_w.set_ylim(global_min - margin, global_max + margin)

    # ── Save ──
    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "csi_comparison_2x2.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")

    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vẽ layout 3×2 CSI heatmap + waveform so sánh Breathing vs Empty"
    )
    parser.add_argument(
        "--breathing", default=None,
        help="File CSV/PCAP trạng thái breathing (mặc định: tự tìm file dài nhất)",
    )
    parser.add_argument(
        "--empty", default=None,
        help="File CSV/PCAP trạng thái empty (mặc định: tự tìm file dài nhất)",
    )
    parser.add_argument(
        "--window", default=60.0, type=float, metavar="SEC",
        help="Độ dài cửa sổ thời gian hiển thị (giây, mặc định: 60)",
    )
    parser.add_argument(
        "--start", default=3.0, type=float, metavar="SEC",
        help="Điểm bắt đầu cửa sổ (giây, mặc định: 3 — bỏ qua nhiễu đầu file)",
    )
    parser.add_argument("--no-save", action="store_true", help="Không lưu ảnh")
    parser.add_argument("--no-show", action="store_true", help="Không hiện cửa sổ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    breath_path = args.breathing
    empty_path  = args.empty

    if breath_path is None:
        breath_path = _find_longest(os.path.join(DATA_ROOT, "breathing"))
        if breath_path is None:
            print(f"❌ Không tìm thấy file trong {DATA_ROOT}/breathing/")
            sys.exit(1)
        print(f"🔍 Auto-detect breathing : {os.path.basename(breath_path)}")

    if empty_path is None:
        empty_path = _find_longest(os.path.join(DATA_ROOT, "empty"))
        if empty_path is None:
            print(f"❌ Không tìm thấy file trong {DATA_ROOT}/empty/")
            sys.exit(1)
        print(f"🔍 Auto-detect empty     : {os.path.basename(empty_path)}")

    plot_comparison(
        breathing_path = breath_path,
        empty_path     = empty_path,
        window_sec     = args.window,
        start_sec      = args.start,
        save           = not args.no_save,
        show           = not args.no_show,
    )
