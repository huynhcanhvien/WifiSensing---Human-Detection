"""
visualization/plot_model_tradeoff.py — Bubble Chart phân tích sự đánh đổi
Accuracy vs Latency vs Model Size của các kiến trúc AI cho WiFi Sensing.

Trục X  : Latency trên Raspberry Pi 4 (ms)
Trục Y  : Độ chính xác phân loại (%)
Bong bóng: Kích thước = Model Size (MB); màu sắc theo nhóm kiến trúc.

Cách dùng:
    python visualization/plot_model_tradeoff.py
    python visualization/plot_model_tradeoff.py --no-show
    python visualization/plot_model_tradeoff.py --no-save
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ─── Dữ liệu mô hình ──────────────────────────────────────────────────────────
# Mỗi mô hình: (tên, accuracy%, latency_ms, model_size_MB, nhóm)
# Latency đo trên Raspberry Pi 4 với batch_size=1, tính trung bình 100 lần.
MODELS = [
    # ── Proposed ──
    ("LightResnet\n(Proposed)",  96.3,  212,   0.82,  "Proposed"),

    # ── Deep Learning ──
    ("2D-CNN",                   91.8,  165,   2.10,  "Deep Learning"),
    ("CNN-LSTM",                 90.5,  340,   3.65,  "Deep Learning"),
    ("MLP",                      82.5,   45,   0.95,  "Deep Learning"),

    # ── Traditional ML ──
    ("SVM",                      78.3,  185,  11.50,  "Traditional ML"),
    ("Decision Tree",            71.2,    8,   0.28,  "Traditional ML"),
]

# Màu sắc theo nhóm kiến trúc
GROUP_COLORS = {
    "Proposed":      "#e74c3c",  # đỏ nổi bật
    "Deep Learning": "#3498db",  # xanh dương
    "Traditional ML": "#f39c12", # vàng cam
}

# Ngưỡng "vùng triển khai thực tế" trên Raspberry Pi 4
LATENCY_THRESHOLD_MS   = 500   # dưới 500ms → phản hồi real-time chấp nhận được
ACCURACY_THRESHOLD_PCT = 88.0  # trên 88% → đủ tin cậy cho ứng dụng thực tế


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bubble_scale(sizes_mb: np.ndarray, min_px: float = 120, max_px: float = 2800) -> np.ndarray:
    """Chuyển model size (MB) sang diện tích bong bóng (points²) theo log scale."""
    log_s = np.log1p(sizes_mb)
    norm  = (log_s - log_s.min()) / (log_s.max() - log_s.min() + 1e-9)
    return min_px + norm * (max_px - min_px)


def _annotation_offset(name: str) -> tuple[float, float]:
    """Điều chỉnh vị trí nhãn để tránh chồng chéo."""
    offsets = {
        "LightResnet\n(Proposed)": ( 38,  14),
        "2D-CNN":                  (-70, -22),
        "CNN-LSTM":                ( 18,  12),
        "MLP":                     ( 18, -20),
        "SVM":                     ( 18,  12),
        "Decision Tree":           ( 18, -20),
    }
    return offsets.get(name, (18, 8))


# ─── Main Plot ────────────────────────────────────────────────────────────────

def plot_model_tradeoff(save: bool = True, show: bool = True) -> str | None:
    """
    Vẽ Bubble Chart đánh đổi Accuracy–Latency–ModelSize.
    LightResnet (Proposed) nằm trong góc tối ưu: accuracy cao, latency thấp.

    Returns:
        Đường dẫn ảnh đã lưu, hoặc None.
    """
    names, accs, lats, sizes, groups = zip(*MODELS)
    accs   = np.array(accs,  dtype=float)
    lats   = np.array(lats,  dtype=float)
    sizes  = np.array(sizes, dtype=float)
    bubble = _bubble_scale(sizes)
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f0f2f5")

    # ── Vùng tối ưu (deployment-ready zone) ──
    ax.axvspan(0, LATENCY_THRESHOLD_MS,
               ymin=0, ymax=1,
               color="#2ecc71", alpha=0.07, zorder=0)
    ax.axhspan(ACCURACY_THRESHOLD_PCT, 100,
               color="#3498db", alpha=0.06, zorder=0)

    # Giao vùng tối ưu — highlight nhẹ hơn
    ax.fill_betweenx(
        [ACCURACY_THRESHOLD_PCT, 100],
        0, LATENCY_THRESHOLD_MS,
        color="#27ae60", alpha=0.10, zorder=0,
    )
    ax.text(
        LATENCY_THRESHOLD_MS * 0.50, ACCURACY_THRESHOLD_PCT + 0.35,
        "Deployment-ready zone",
        fontsize=8.5, color="#1a7a3a", fontstyle="italic", alpha=0.75,
        ha="center", va="bottom",
    )

    # Đường ngưỡng
    ax.axvline(LATENCY_THRESHOLD_MS, color="#27ae60",
               lw=1.2, ls="--", alpha=0.55)
    ax.axhline(ACCURACY_THRESHOLD_PCT, color="#2980b9",
               lw=1.2, ls="--", alpha=0.55)

    # ── Vẽ bong bóng ──
    sc = ax.scatter(
        lats, accs,
        s=bubble,
        c=colors,
        alpha=0.80,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    # ── Nhãn mô hình ──
    for name, acc, lat, size_mb in zip(names, accs, lats, sizes):
        dx, dy = _annotation_offset(name)
        is_proposed = "Proposed" in name

        fontsize  = 9.5 if is_proposed else 7.8
        fontweight = "bold" if is_proposed else "normal"
        color      = "#c0392b" if is_proposed else "#2c3e50"

        ax.annotate(
            name,
            xy=(lat, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color="#aab0b8",
                lw=0.8,
                connectionstyle="arc3,rad=0.1",
            ),
            zorder=5,
        )

        # Thêm nhãn kích thước nhỏ bên dưới mỗi bong bóng
        ax.annotate(
            f"{size_mb:.1f} MB",
            xy=(lat, acc),
            xytext=(0, -bubble[list(names).index(name)] ** 0.5 * 0.55 - 8),
            textcoords="offset points",
            fontsize=6.0,
            color="#7f8c8d",
            ha="center",
            va="top",
            zorder=5,
        )

    # ── Đánh dấu đặc biệt LightResnet ──
    light_idx = list(names).index("LightResnet\n(Proposed)")
    ax.scatter(
        lats[light_idx], accs[light_idx],
        s=bubble[light_idx] + 300,
        facecolors="none",
        edgecolors="#e74c3c",
        linewidths=2.5,
        zorder=4,
    )
    ax.scatter(
        lats[light_idx], accs[light_idx],
        s=20,
        c="white",
        zorder=6,
    )

    # ── Star marker cho điểm tối ưu ──
    ax.plot(
        lats[light_idx], accs[light_idx],
        marker="*", color="#e74c3c",
        markersize=14, zorder=7,
        markeredgecolor="white", markeredgewidth=0.8,
    )

    # ── Callout box cho LightResnet ──
    callout_text = (
        f"LightResnet (Proposed)\n"
        f"Accuracy : {accs[light_idx]:.1f}%\n"
        f"Latency  : {lats[light_idx]:.0f} ms\n"
        f"Size     : {sizes[light_idx]:.2f} MB"
    )
    ax.annotate(
        callout_text,
        xy=(lats[light_idx], accs[light_idx]),
        xytext=(lats[light_idx] + 160, accs[light_idx] - 2.8),
        fontsize=8.5,
        color="#7f0000",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#fff5f5",
            edgecolor="#e74c3c",
            linewidth=1.5,
            alpha=0.92,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color="#e74c3c",
            lw=1.5,
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=8,
    )

    # ── Legend nhóm kiến trúc ──
    legend_patches = [
        mpatches.Patch(color=c, label=g, alpha=0.85)
        for g, c in GROUP_COLORS.items()
    ]
    legend_arch = ax.legend(
        handles=legend_patches,
        title="Kiến trúc",
        title_fontsize=9,
        fontsize=8.5,
        loc="lower right",
        framealpha=0.88,
        edgecolor="#ccc",
        frameon=True,
    )
    ax.add_artist(legend_arch)

    # ── Legend kích thước bong bóng (scale reference) ──
    ref_sizes_mb = [1, 10, 100]
    ref_bubbles  = _bubble_scale(np.array(ref_sizes_mb, dtype=float))
    handles_size = [
        plt.scatter([], [], s=s, color="#aab0b8", alpha=0.75, edgecolors="white", lw=1)
        for s in ref_bubbles
    ]
    ax.legend(
        handles_size,
        [f"{s} MB" for s in ref_sizes_mb],
        title="Model Size",
        title_fontsize=8.5,
        fontsize=8,
        loc="lower left",
        framealpha=0.88,
        edgecolor="#ccc",
        scatterpoints=1,
        handletextpad=1.2,
    )

    # ── Trục và tiêu đề ──
    ax.set_xlabel("Latency trên Raspberry Pi 4  (ms) ↓ Thấp hơn = Tốt hơn",
                  fontsize=11, labelpad=8)
    ax.set_ylabel("Độ chính xác (Accuracy %)  ↑ Cao hơn = Tốt hơn",
                  fontsize=11, labelpad=8)
    ax.set_title(
        "Đánh đổi Accuracy – Latency – Model Size\n"
        "Các mô hình AI cho bài toán WiFi Sensing trên thiết bị Edge (Raspberry Pi 4)",
        fontsize=13, fontweight="bold", pad=14,
    )

    # Giới hạn trục — để vùng bong bóng không bị cắt
    ax.set_xlim(-30, max(lats) * 1.35)
    ax.set_ylim(accs.min() - 4.0, 100.5)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.45, color="#bdc3c7")
    ax.spines[["top", "right"]].set_visible(False)

    # Caption dưới biểu đồ
    fig.text(
        0.5, 0.005,
        "Kích thước bong bóng ∝ log(Model Size MB)  |  "
        "Latency: trung bình 100 lần suy luận, batch=1, FP32  |  "
        "Vùng xanh = đáp ứng yêu cầu triển khai real-time (< 500ms, > 90%)",
        ha="center", fontsize=7.5, color="#7f8c8d",
    )

    plt.tight_layout(rect=[0, 0.025, 1, 1])

    out_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "model_tradeoff_bubble.png")
        plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close()
    return out_path


# ─── Entry-point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bubble Chart đánh đổi Accuracy–Latency–ModelSize"
    )
    parser.add_argument("--no-save", action="store_true", help="Không lưu ảnh PNG")
    parser.add_argument("--no-show", action="store_true", help="Không hiện cửa sổ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_model_tradeoff(save=not args.no_save, show=not args.no_show)
