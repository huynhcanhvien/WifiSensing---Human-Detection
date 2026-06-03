"""
visualization/plot_csv.py — Vẽ dashboard CSI từ file CSV realtime.

Tái tạo từ data_collection/plot_csv.py, được refactor để:
- Dùng DSP từ src.dsp
- Tự động lưu ảnh vào visualization/output/
- Dashboard gồm 3 cột: Packet Interval / CSI Heatmap / Amplitude Shape
- Slider tương tác để cuộn theo thời gian

Cách dùng:
    python visualization/plot_csv.py --file data/Data/breathing/breath_xxx_realtime.csv
    python visualization/plot_csv.py --file data/Data/breathing/breath_xxx_realtime.csv --no-show
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_csv(file_path: str) -> dict | None:
    """
    Đọc file CSV realtime, áp dụng DSP pipeline, trả về dict theo MAC.

    Returns:
        {mac: {'time': np.ndarray, 'csi': np.ndarray(T,64), 'interval': np.ndarray}}
    """
    print(f"🚀 Đang đọc: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None

    # Parse timestamp
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    mac_list = df['mac'].unique()
    result   = {}

    for mac in mac_list:
        print(f"⚙️  Xử lý MAC: {mac}")
        sub   = df[df['mac'] == mac].sort_values('timestamp')
        t     = sub['timestamp'].values
        sc_cols = [c for c in sub.columns if c.startswith('sub_')]
        csi_raw = sub[sc_cols].values.astype(float)   # (T, 64)

        # DSP pipeline
        csi = fft_shift(csi_raw)
        csi = interpolate_dc_pilots(csi)
        csi = hampel_filter(csi)

        # L2 normalize
        norms = np.linalg.norm(csi, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        csi = csi / norms

        # Đặt NaN vào vùng null subcarrier (để không vẽ)
        from config import DSP
        null_mask = [i for i in range(64) if i not in DSP.VALID_SUBCARRIERS]
        csi[:, null_mask] = np.nan

        interval = np.diff(t, prepend=t[0]) * 1000   # ms

        result[mac] = {'time': t, 'csi': csi, 'interval': interval}

    return result


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_csv_dashboard(
    file_path: str,
    window_duration: float = 10.0,
    save: bool             = True,
    show: bool             = True,
) -> str | None:
    """
    Vẽ dashboard 3 cột (Interval / Heatmap / Shape) với slider tương tác.

    Returns:
        Đường dẫn ảnh đã lưu (snapshot tại t=0) nếu save=True.
    """
    data = load_csv(file_path)
    if not data:
        return None

    basename  = os.path.splitext(os.path.basename(file_path))[0]
    n_devices = len(data)

    fig, axes = plt.subplots(n_devices, 3, figsize=(20, 5 * n_devices + 1))
    plt.subplots_adjust(bottom=0.12, hspace=0.45, wspace=0.28)
    fig.suptitle(f"CSI Dashboard — {basename}", fontsize=15, fontweight='bold', y=0.99)

    if n_devices == 1:
        axes = np.expand_dims(axes, 0)

    plots   = {}
    max_time = 0

    for i, (mac, d) in enumerate(data.items()):
        t        = d['time']
        csi      = d['csi']
        interval = d['interval']
        max_time = max(max_time, t[-1])

        avg_freq = 1000 / np.nanmean(interval[interval > 0]) if len(interval) > 0 else 0

        # ── Col 0: Packet Interval ──
        ax0 = axes[i, 0]
        ax0.plot(t, interval, color='#2196F3', lw=0.8, alpha=0.7)
        ax0.set_title(f"MAC: {mac}\nPacket Interval  (avg {avg_freq:.1f} Hz)", fontweight='bold', fontsize=9)
        ax0.set_ylabel("Interval (ms)")
        ax0.set_ylim(0, 60)
        ax0.grid(True, linestyle=':', alpha=0.4)

        # ── Col 1: CSI Heatmap ──
        ax1   = axes[i, 1]
        vmin, vmax = np.nanpercentile(csi, 5), np.nanpercentile(csi, 95)
        im = ax1.imshow(
            csi.T, aspect='auto', cmap='viridis', origin='lower',
            extent=[t[0], t[-1], 0, 64], vmin=vmin, vmax=vmax
        )
        ax1.set_title("CSI Heatmap (L2 Normalized)", fontweight='bold', fontsize=9)
        ax1.set_ylabel("Subcarrier Index")
        fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.04)

        # ── Col 2: Amplitude Shape ──
        ax2       = axes[i, 2]
        avg_shape = np.nanmean(csi, axis=0)
        std_shape = np.nanstd(csi, axis=0)
        x_axis    = np.arange(64)

        # Vẽ toàn bộ csi (sample mỏng)
        step = max(1, len(csi) // 300)
        ax2.plot(x_axis, csi[::step].T, color='royalblue', alpha=0.015, lw=0.5)
        line_shape, = ax2.plot(x_axis, avg_shape, color='navy', lw=2, label='Mean')
        ax2.fill_between(x_axis, avg_shape - std_shape, avg_shape + std_shape,
                         alpha=0.2, color='royalblue', label='±1 STD')
        ax2.set_title("Amplitude Shape", fontweight='bold', fontsize=9)
        ax2.set_xlabel("Subcarrier Index")
        ax2.set_xlim(0, 63)
        y_max = np.nanmax(avg_shape) * 1.6 if not np.isnan(avg_shape).all() else 1
        ax2.set_ylim(0, y_max)
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.legend(fontsize=8)

        plots[mac] = {
            'data': d,
            'ax0': ax0, 'ax1': ax1, 'ax2': ax2,
            'line_shape': line_shape,
        }

    # ── Save snapshot (trước khi thêm slider) ──
    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"csv_{basename}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")

    # ── Slider tương tác ──
    ax_slider = plt.axes([0.15, 0.03, 0.7, 0.025], facecolor='#e8e8e8')
    slider    = Slider(ax_slider, 'Time (s)', 0, max(0.0, max_time - window_duration), valinit=0)

    def update(val):
        t_start = slider.val
        t_end   = t_start + window_duration
        for mac, p in plots.items():
            t   = p['data']['time']
            csi = p['data']['csi']
            p['ax0'].set_xlim(t_start, t_end)
            p['ax1'].set_xlim(t_start, t_end)
            # Update shape theo cửa sổ hiện tại
            idx_s = np.searchsorted(t, t_start)
            idx_e = np.searchsorted(t, t_end)
            if idx_e > idx_s:
                win_csi = csi[idx_s:idx_e]
                p['line_shape'].set_ydata(np.nanmean(win_csi, axis=0))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

    print("📊 Dashboard sẵn sàng!")
    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ dashboard CSI từ file CSV realtime")
    parser.add_argument("--file",    required=True,            help="Đường dẫn file .csv")
    parser.add_argument("--window",  default=10.0, type=float, help="Khoảng thời gian hiển thị (giây)")
    parser.add_argument("--no-save", action="store_true",      help="Không lưu ảnh")
    parser.add_argument("--no-show", action="store_true",      help="Không hiện cửa sổ (batch mode)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_csv_dashboard(
        file_path       = args.file,
        window_duration = args.window,
        save            = not args.no_save,
        show            = not args.no_show,
    )
