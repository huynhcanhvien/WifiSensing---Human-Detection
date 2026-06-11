"""
visualization/plot_pi_tradeoff.py — Bubble Chart học thuật
Accuracy vs Latency (Pi 4), bubble size = số tham số model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Dữ liệu ───────────────────────────────────────────────────────────────────
# (tên, accuracy %, latency ms Pi4, params)
MODELS = [
    ("LightResNet",    100.00, 203,  77_378),
    ("CNN2D",           99.76, 223,  93_666),
    ("CNNLSTMHybrid",   99.91, 243, 281_586),
    ("MLP",             96.61, 184, 600_706),
    ("XGBoost*",        91.30,  78,  45_000),
    ("SVM*",            88.50, 115, 975_790),
]

names  = [m[0] for m in MODELS]
acc    = np.array([m[1] for m in MODELS])
lat    = np.array([m[2] for m in MODELS])
params = np.array([m[3] for m in MODELS])

# Bubble size tỉ lệ log(params) → area
size = (np.log10(params) - np.log10(params.min())) / \
       (np.log10(params.max()) - np.log10(params.min()))
area = 80 + size * 900   # điểm²

# Màu: proposed=đỏ, DL=xanh, ML=cam
color = ["#d62728", "#1f77b4", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]

# ── Vẽ ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(lat, acc, s=area, c=color, alpha=0.75, edgecolors="white", linewidths=1.2, zorder=3)

# Nhãn tên model — đặt phía dưới bubble
# (dx, dy) tuỳ chỉnh cho các model bị chồng nhau
label_offset = {
    "LightResNet":   (-18, -18),
    "CNN2D":         (  0, -18),
    "CNNLSTMHybrid": ( 18, -28),
    "MLP":           (  0, -38),
    "XGBoost*":      (  0, -16),
    "SVM*":          (  0, -34),
}
for name, a, l, s in zip(names, acc, lat, area):
    dx, dy = label_offset.get(name, (0, -18))
    ax.annotate(name, xy=(l, a), xytext=(dx, dy),
                textcoords="offset points", fontsize=9,
                ha="center", va="top",
                fontweight="bold" if name == "LightResNet" else "normal")

# ── Trục ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Inference Latency on Raspberry Pi 4 (ms)", fontsize=11)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("Accuracy vs. Latency Trade-off\n"
             "(bubble size ∝ number of parameters)", fontsize=12)

ax.set_xlim(0, 310)
ax.set_ylim(85, 102)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)


plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
out = os.path.join(OUTPUT_DIR, "pi4_tradeoff_bubble.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
