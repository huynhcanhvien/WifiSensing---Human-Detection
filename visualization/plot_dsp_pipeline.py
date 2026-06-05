"""
visualization/plot_dsp_pipeline.py — Minh họa từng bước DSP Pipeline trên dữ liệu thực.

Lấy 1 đoạn CSI (64 subcarrier) từ file breathing, áp dụng tuần tự từng bước,
vẽ theo swim lane: mỗi hàng = 1 bước xử lý, 2 cột:
  · Trái  — Profile biên độ theo subcarrier tại 1 thời điểm đại diện
  · Phải  — Heatmap CSI (thời gian × subcarrier) toàn bộ đoạn

Bước:  Raw → FFT Shift → Nội suy null/pilot → Hampel Filter → Cắt [6:59] → L2 Norm

Cách dùng:
    python visualization/plot_dsp_pipeline.py
    python visualization/plot_dsp_pipeline.py --file data/Data/breathing/breath_xxx.csv
    python visualization/plot_dsp_pipeline.py --frames 200 --no-show
"""

import argparse
import glob
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DSP, Hardware
from src.dsp.interpolation import fft_shift, interpolate_dc_pilots

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_ROOT  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Data"
)

# ─── Swim-lane styling ────────────────────────────────────────────────────────

LANES = [
    dict(label="① Raw",                 bg='#F8FAFC', border='#94A3B8', acc='#475569', cmap='Greys_r'),
    dict(label="② FFT Shift",           bg='#EFF6FF', border='#3B82F6', acc='#1D4ED8', cmap='Blues'),
    dict(label="③ Cắt Null SC",         bg='#FFF1F2', border='#F43F5E', acc='#BE123C', cmap='RdPu'),
    dict(label="④ Nội Suy DC",          bg='#F0FDF4', border='#22C55E', acc='#15803D', cmap='Greens'),
    dict(label="⑤ Hampel Filter",       bg='#FFF7ED', border='#FB923C', acc='#C2410C', cmap='Oranges'),
    dict(label="⑥ Chuẩn Hóa L2",       bg='#F0FDFA', border='#2DD4BF', acc='#0F766E', cmap='YlGnBu'),
]

# Vị trí null SC gốc và sau FFT-shift (để vẽ vùng tô bóng)
NULL_ORIG    = set(DSP.NULL_SUBCARRIERS)
NULL_SHIFTED = {(i + 32) % 64 for i in NULL_ORIG}


# ─── Hampel với theo dõi outlier ─────────────────────────────────────────────

def _hampel_tracked(mat: np.ndarray):
    """Trả về (filtered, outlier_mask) — outlier_mask[t, sc] = True nếu bị thay."""
    w    = 2 * DSP.HAMPEL_WINDOW + 1
    out  = mat.copy()
    mask = np.zeros_like(mat, dtype=bool)
    for sc in range(mat.shape[1]):
        series  = mat[:, sc]
        padded  = np.pad(series, DSP.HAMPEL_WINDOW, mode='edge')
        windows = sliding_window_view(padded, w)
        medians = np.median(windows, axis=1)
        mads    = np.median(np.abs(windows - medians[:, None]), axis=1)
        thr     = DSP.HAMPEL_SIGMA * DSP.K_HAMPEL * mads
        bad     = np.abs(series - medians) > thr
        out[bad, sc]  = medians[bad]
        mask[bad, sc] = True
    return out, mask


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_sample(file_path: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Đọc n_frames từ CSV breathing, chọn MAC có nhiều gói nhất.
    Returns: (t, csi_raw)  shapes (T,)  và  (T, 64)
    """
    df = pd.read_csv(file_path)
    mac = df['mac'].value_counts().index[0]
    sub = df[df['mac'] == mac].sort_values('timestamp').copy()

    if sub['timestamp'].dtype == object:
        sub['timestamp'] = pd.to_datetime(sub['timestamp'])
        sub['timestamp'] = (
            sub['timestamp'] - sub['timestamp'].min()
        ).dt.total_seconds()

    # Bắt đầu từ 20% để tránh nhiễu khởi động
    start = max(0, int(len(sub) * 0.20))
    end   = start + n_frames
    sel   = sub.iloc[start:end]

    sc_cols = [c for c in sub.columns if c.startswith('sub_')]
    t   = sel['timestamp'].values - sel['timestamp'].values[0]
    csi = sel[sc_cols].values.astype(float)
    return t, csi


# ─── Build all pipeline stages ───────────────────────────────────────────────

def build_stages(csi_raw: np.ndarray):
    """Trả về list [(key, data, extra)] theo thứ tự pipeline."""
    stages = []

    # 1 Raw
    c1 = csi_raw.copy()
    stages.append(('raw', c1, {}))

    # 2 FFT Shift — DC về giữa phổ (idx 32)
    c2 = fft_shift(c1)
    stages.append(('fft', c2, {}))

    # 3 Cắt bỏ null SC — set các vị trí null sau shift → NaN để hiện rõ vùng bị loại
    c3_vis = c2.copy().astype(float)
    for pos in NULL_SHIFTED:
        c3_vis[:, pos] = np.nan
    stages.append(('remove_null', c3_vis, {}))

    # 4 Nội suy DC — fill null positions bằng nội suy tuyến tính từ valid SC lân cận
    c4 = interpolate_dc_pilots(c2)
    stages.append(('interp', c4, {}))

    # 5 Hampel filter — loại outlier từng SC
    c5, outlier_mask = _hampel_tracked(c4)
    stages.append(('hampel', c5, {'mask': outlier_mask}))

    # 6 L2 normalize — cắt [6:59] lấy 53 SC hữu ích rồi normalize
    c6_slice = c5[:, DSP.EFFECTIVE_SC_START:DSP.EFFECTIVE_SC_END]
    norms = np.linalg.norm(c6_slice, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    c6 = c6_slice / norms
    stages.append(('l2', c6, {}))

    return stages


# ─── Per-step plot helpers ────────────────────────────────────────────────────

def _shade_nulls(ax, null_set: set, n_sc: int, alpha: float = 0.18, color='red'):
    """Tô vùng bóng đỏ tại các vị trí null subcarrier."""
    positions = sorted(i for i in null_set if i < n_sc)
    if not positions:
        return
    groups, group = [], [positions[0]]
    for p in positions[1:]:
        if p == group[-1] + 1:
            group.append(p)
        else:
            groups.append(group)
            group = [p]
    groups.append(group)
    for g in groups:
        ax.axvspan(g[0] - 0.5, g[-1] + 0.5, alpha=alpha, color=color, zorder=0)


def _heatmap(ax, fig, csi: np.ndarray, t: np.ndarray, cmap: str):
    vmin = np.nanpercentile(csi, 2)
    vmax = np.nanpercentile(csi, 98)
    im = ax.imshow(
        csi.T, aspect='auto', origin='lower', cmap=cmap,
        extent=[float(t[0]), float(t[-1]), 0, csi.shape[1]],
        vmin=vmin, vmax=vmax, interpolation='nearest',
    )
    ax.set_xlabel("Thời gian (s)", fontsize=7)
    ax.set_ylabel("Subcarrier", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=5.5)


def _profile(ax, csi: np.ndarray, t_idx: int, acc: str,
             null_set: set | None = None,
             outlier_mask: np.ndarray | None = None):
    n_sc = csi.shape[1]
    x    = np.arange(n_sc)
    y    = csi[t_idx]

    ax.plot(x, y, color=acc, lw=1.4, zorder=3)

    if null_set:
        _shade_nulls(ax, null_set, n_sc)

    if outlier_mask is not None:
        bad_sc = np.where(outlier_mask[t_idx])[0]
        if len(bad_sc):
            ax.scatter(bad_sc, csi[t_idx, bad_sc],
                       color='#EF4444', s=18, zorder=5, label=f'Outlier ({len(bad_sc)})')
            ax.legend(fontsize=6, loc='upper right', framealpha=0.75,
                      borderpad=0.3, labelspacing=0.2,
                      handlelength=1.0, handletextpad=0.3)

    ax.set_xlabel("Subcarrier index", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle=':', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)


# ─── Main plot ────────────────────────────────────────────────────────────────

def plot_dsp_pipeline(
    file_path: str,
    n_frames: int = 160,
    save: bool = True,
    show: bool = True,
) -> str | None:
    """
    Vẽ swim-lane DSP pipeline.

    Args:
        file_path: CSV breathing file.
        n_frames:  Số frame để hiển thị heatmap.
        save/show: Lưu / hiện cửa sổ.
    """
    print(f"📂 File  : {os.path.basename(file_path)}")
    print(f"🎞️  Frames: {n_frames}")

    t, csi_raw = load_sample(file_path, n_frames)
    stages  = build_stages(csi_raw)
    n_steps = len(stages)

    # Chọn frame có biên độ tổng cao nhất trên vùng SC hợp lệ (đỉnh breathing)
    valid_sc = [i for i in range(csi_raw.shape[1]) if i not in NULL_ORIG]
    energy   = csi_raw[:, valid_sc].mean(axis=1)
    t_idx    = int(np.argmax(energy))

    print(f"   Raw shape: {csi_raw.shape}  |  t_idx={t_idx}  (t={t[t_idx]:.2f}s)")

    # ── Figure & GridSpec ──
    fig = plt.figure(figsize=(15, 3.5 * n_steps))

    # Outer GridSpec: n_steps hàng — left넓혀서 swim lane label 들어갈 여백 확보
    outer = gridspec.GridSpec(
        n_steps, 1, figure=fig,
        hspace=0.18,
        top=0.95, bottom=0.04, left=0.20, right=0.98,
    )

    fig.suptitle(
        f"DSP Pipeline — CSI Signal Processing  |  "
        f"{os.path.basename(file_path)}  |  {n_frames} frames",
        fontsize=10, fontweight='bold', y=0.98,
    )

    _ann_kw = dict(
        xycoords='axes fraction',
        fontsize=6.5,
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.72, edgecolor='none'),
    )

    for row, ((key, csi, extra), lane) in enumerate(zip(stages, LANES)):
        # ── Inner GridSpec cho 2 subplot ──
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row],
            wspace=0.42, width_ratios=[1, 1],
        )

        # Nền toàn hàng (swim lane background)
        bg_ax = fig.add_subplot(outer[row])
        bg_ax.set_facecolor(lane['bg'])
        bg_ax.set_xticks([])
        bg_ax.set_yticks([])
        for spine in bg_ax.spines.values():
            spine.set_edgecolor(lane['border'])
            spine.set_linewidth(1.6)

        # Nhãn swim lane — xoay 90°, đặt trong lề trái, dùng toạ độ axes
        bg_ax.text(
            -0.115, 0.5, lane['label'],
            transform=bg_ax.transAxes,
            ha='center', va='center',
            fontsize=7.5, fontweight='bold',
            rotation=90, color=lane['acc'],
            multialignment='center',
        )

        ax_prof = fig.add_subplot(inner[0, 0])
        ax_heat = fig.add_subplot(inner[0, 1])
        ax_prof.set_facecolor(lane['bg'])
        ax_heat.set_facecolor(lane['bg'])

        omask  = extra.get('mask')

        # Null set phù hợp với bước
        # Null-shading: Raw dùng vị trí gốc, FFT+remove_null dùng vị trí sau shift
        if key == 'raw':
            null_set = NULL_ORIG
        elif key == 'fft':
            null_set = NULL_SHIFTED
        else:
            null_set = None    # step 3+ không cần shade thêm (NaN đã hiện gap)

        # ── Profile plot ──
        _profile(
            ax=ax_prof, csi=csi, t_idx=t_idx, acc=lane['acc'],
            null_set=null_set,
            outlier_mask=omask,
        )

        # Annotation per-step
        if key == 'fft':
            ax_prof.axvline(32, color='#F59E0B', lw=1.2, ls='--', alpha=0.75)
            ax_prof.annotate("DC → 32", xy=(0.52, 0.96), **_ann_kw, color='#D97706')
        elif key == 'remove_null':
            ax_prof.annotate("Null SC = NaN\n(gaps)", xy=(0.5, 0.96), **_ann_kw,
                             color='#BE123C')
        elif key == 'l2':
            ax_prof.set_xlabel(
                f"Subcarrier index (SC gốc {DSP.EFFECTIVE_SC_START}–{DSP.EFFECTIVE_SC_END-1})",
                fontsize=6.5,
            )
            ax_prof.set_ylabel("Amplitude (L2)", fontsize=7)

        # ── Heatmap ──
        _heatmap(ax=ax_heat, fig=fig, csi=csi, t=t, cmap=lane['cmap'])

    # ── Save ──
    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "dsp_pipeline_swimlane.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {out_path}")

    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vẽ swim-lane DSP pipeline trên dữ liệu CSI thực"
    )
    parser.add_argument(
        "--file", default=None,
        help="File CSV breathing (mặc định: tự tìm file dài nhất)",
    )
    parser.add_argument(
        "--frames", default=160, type=int, metavar="N",
        help="Số frame sử dụng (mặc định: 160)",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    file_path = args.file
    if file_path is None:
        candidates = glob.glob(
            os.path.join(DATA_ROOT, "breathing", "*_realtime.csv")
        )
        if not candidates:
            print(f"❌ Không tìm thấy file trong {DATA_ROOT}/breathing/")
            sys.exit(1)
        file_path = max(candidates, key=os.path.getsize)
        print(f"🔍 Auto-detect: {os.path.basename(file_path)}")

    plot_dsp_pipeline(
        file_path = file_path,
        n_frames  = args.frames,
        save      = not args.no_save,
        show      = not args.no_show,
    )
