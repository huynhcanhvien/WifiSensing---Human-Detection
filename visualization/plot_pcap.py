"""
visualization/plot_pcap.py — Vẽ heatmap CSI từ file PCAP.

Tái tạo từ data_collection/plot.py, được refactor để:
- Dùng DSP từ src.dsp (không nhân bản)
- Tự động lưu ảnh vào visualization/output/
- Hỗ trợ argparse

Cách dùng:
    python visualization/plot_pcap.py --file data/Data/breathing/breath_xxx.pcap
    python visualization/plot_pcap.py --file data/Data/empty/breath_xxx.pcap --no-show
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DSP, Hardware
from src.dsp.filters import hampel_filter
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_pcap(file_path: str, window_sec: float, trim_sec: float) -> dict:
    """Đọc file PCAP và trả về CSI raw cho từng MAC."""
    try:
        import dpkt
    except ImportError:
        raise ImportError("Cần cài dpkt: pip install dpkt")

    max_time = trim_sec + window_sec + 5.0
    data_rows = {mac: [] for mac in Hardware.TARGET_MACS_BYTES}
    start_ts  = None

    print(f"🔍 Đang đọc: {os.path.basename(file_path)}")
    with open(file_path, 'rb') as f:
        for ts, buf in dpkt.pcap.Reader(f):
            if start_ts is None:
                start_ts = ts
            if ts - start_ts > max_time:
                break
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

    results = {}
    for mac, rows in data_rows.items():
        if len(rows) < 10:
            continue
        arr = np.array(rows)
        csi = fft_shift(arr[:, 1:])
        csi = interpolate_dc_pilots(csi)
        csi = hampel_filter(csi)
        results[mac.hex()] = {'time': arr[:, 0], 'csi': csi}

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_pcap(
    file_path: str,
    window_sec: float = 10.0,
    trim_sec: float   = 15.0,
    save: bool        = True,
    show: bool        = True,
) -> str | None:
    """
    Vẽ heatmap 2D + 3D surface cho CSI từ file PCAP.

    Returns:
        Đường dẫn file ảnh đã lưu (nếu save=True), None nếu không save.
    """
    raw_data = parse_pcap(file_path, window_sec, trim_sec)
    if not raw_data:
        print("❌ Không có dữ liệu hợp lệ trong file.")
        return None

    basename = os.path.splitext(os.path.basename(file_path))[0]
    mac_keys = list(raw_data.keys())
    n_macs   = len(mac_keys)

    fig = plt.figure(figsize=(18, 6 * n_macs + 4))
    fig.suptitle(
        f"CSI Analysis — {basename}",
        fontsize=16, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(n_macs + 1, 2, figure=fig, hspace=0.45, wspace=0.3)

    processed = []
    for i, mac in enumerate(mac_keys):
        d   = raw_data[mac]
        t   = d['time']
        csi = d['csi'][:, DSP.EFFECTIVE_SC_START:DSP.EFFECTIVE_SC_END]  # (T, 53)

        # Normalize L2
        norms = np.linalg.norm(csi, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        csi_norm = csi / norms
        processed.append((t, csi_norm))

        # Heatmap
        ax_heat = fig.add_subplot(gs[i, 0])
        vmin, vmax = np.nanpercentile(csi_norm, 5), np.nanpercentile(csi_norm, 95)
        im = ax_heat.imshow(
            csi_norm.T, aspect='auto', cmap='viridis', origin='lower',
            extent=[t[0], t[-1], 0, 53], vmin=vmin, vmax=vmax
        )
        ax_heat.set_title(f"MAC {i}: {mac}", fontweight='bold')
        ax_heat.set_xlabel("Time (s)")
        ax_heat.set_ylabel("Subcarrier (53 effective)")
        fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.04, label="Norm. Amplitude")

        # Amplitude mean shape
        ax_shape = fig.add_subplot(gs[i, 1])
        avg_shape = np.nanmean(csi_norm, axis=0)
        std_shape = np.nanstd(csi_norm, axis=0)
        x_axis    = np.arange(53)
        ax_shape.fill_between(x_axis, avg_shape - std_shape, avg_shape + std_shape, alpha=0.25, color='royalblue', label='±1 STD')
        ax_shape.plot(x_axis, avg_shape, color='royalblue', lw=2, label='Mean')
        ax_shape.set_title(f"Amplitude Shape — MAC {i}: {mac}", fontweight='bold')
        ax_shape.set_xlabel("Subcarrier Index")
        ax_shape.set_ylabel("Norm. Amplitude")
        ax_shape.legend(fontsize=9)
        ax_shape.grid(True, linestyle=':', alpha=0.5)

    # 3D Surface cho MAC đầu tiên
    if processed:
        t0, csi0 = processed[0]
        ax3d = fig.add_subplot(gs[n_macs, :], projection='3d')
        n_t_down = min(200, csi0.shape[0])  # downsample cho 3D
        step_t   = max(1, csi0.shape[0] // n_t_down)
        csi_down = csi0[::step_t]
        t_down   = t0[::step_t]
        X, Y = np.meshgrid(np.arange(53), t_down)
        z_lim = np.nanpercentile(csi_down, 99)
        Z     = np.clip(csi_down, 0, z_lim)
        ax3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.88)
        ax3d.set_title(f"3D Surface — MAC 0: {mac_keys[0]}", fontweight='bold')
        ax3d.set_xlabel("Subcarrier")
        ax3d.set_ylabel("Time (s)")
        ax3d.set_zlabel("Amplitude")

    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"pcap_{basename}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")

    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Vẽ heatmap CSI từ file PCAP")
    parser.add_argument("--file",    required=True,             help="Đường dẫn file .pcap")
    parser.add_argument("--window",  default=10.0, type=float,  help="Độ dài cửa sổ xem (giây)")
    parser.add_argument("--trim",    default=15.0, type=float,  help="Cắt bỏ đầu/cuối (giây)")
    parser.add_argument("--no-save", action="store_true",       help="Không lưu ảnh")
    parser.add_argument("--no-show", action="store_true",       help="Không hiện cửa sổ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_pcap(
        file_path  = args.file,
        window_sec = args.window,
        trim_sec   = args.trim,
        save       = not args.no_save,
        show       = not args.no_show,
    )
