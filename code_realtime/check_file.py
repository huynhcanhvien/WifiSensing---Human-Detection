import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dpkt
from scipy import interpolate

# ══════════════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════════════
FILE_PATH = "data/Data/breathing/breath_20260315_084446.pcap"

TARGET_MACS = [
    bytes.fromhex("d4e9f4a36480"),
    bytes.fromhex("d4e9f4a43c18"),
]

# Ngưỡng khoảng trống bị coi là "mất gói" (giây)
GAP_THRESHOLD_SEC = 1.0

# Chia file thành N đoạn để vẽ Hz theo thời gian
N_SEGMENTS = 60
# ══════════════════════════════════════════════════════


def parse_pcap(file_path):
    """Đọc toàn bộ file, trả về timestamps cho từng MAC."""
    print(f"📂 Đang đọc: {os.path.basename(file_path)}")
    print(f"   Kích thước: {os.path.getsize(file_path) / 1e9:.2f} GB\n")

    data = {mac: {"ts": [], "csi": []} for mac in TARGET_MACS}
    start_ts = None
    total_packets = 0

    with open(file_path, "rb") as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            if start_ts is None:
                start_ts = ts
            rel_ts = ts - start_ts
            total_packets += 1

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
                            data[mac_addr]["ts"].append(rel_ts)
                            data[mac_addr]["csi"].append(amp)
            except Exception:
                continue

    print(f"   Tổng gói tin (tất cả loại): {total_packets:,}")
    return data, start_ts


def analyze_mac(mac, ts_list, label):
    """Phân tích chi tiết 1 MAC."""
    ts  = np.array(ts_list)
    n   = len(ts)

    if n == 0:
        print(f"   ❌ {label}: Không có gói tin nào!")
        return None

    duration = ts[-1] - ts[0]
    avg_hz   = n / duration if duration > 0 else 0
    gaps     = np.diff(ts)

    big_gaps = gaps[gaps > GAP_THRESHOLD_SEC]
    gap_idx  = np.where(gaps > GAP_THRESHOLD_SEC)[0]

    print(f"\n{'─'*55}")
    print(f"  MAC: {mac.hex(':').upper()}  ({label})")
    print(f"{'─'*55}")
    print(f"  Tổng gói tin     : {n:,}")
    print(f"  Thời lượng       : {duration:.1f}s  ({duration/3600:.2f} giờ)")
    print(f"  Tần suất TB      : {avg_hz:.2f} Hz")
    print(f"  Khoảng cách TB   : {np.mean(gaps)*1000:.1f} ms")
    print(f"  Khoảng cách MAX  : {np.max(gaps):.2f}s")
    print(f"\n  Khoảng trống > {GAP_THRESHOLD_SEC}s : {len(big_gaps)} lần")

    if len(big_gaps) > 0:
        print(f"  Chi tiết khoảng trống lớn nhất (top 10):")
        top_idx = np.argsort(big_gaps)[::-1][:10]
        for i in top_idx:
            t_start = ts[gap_idx[i]]
            t_end   = ts[gap_idx[i] + 1]
            print(f"    [{t_start/60:6.1f} phút → {t_end/60:6.1f} phút]  "
                  f"mất {big_gaps[i]:.1f}s")

    # Tính Hz theo từng đoạn thời gian
    seg_edges = np.linspace(ts[0], ts[-1], N_SEGMENTS + 1)
    seg_hz    = []
    seg_count = []
    for i in range(N_SEGMENTS):
        mask  = (ts >= seg_edges[i]) & (ts < seg_edges[i + 1])
        count = mask.sum()
        dur   = seg_edges[i + 1] - seg_edges[i]
        seg_hz.append(count / dur if dur > 0 else 0)
        seg_count.append(count)

    return {
        "ts":       ts,
        "n":        n,
        "duration": duration,
        "avg_hz":   avg_hz,
        "gaps":     gaps,
        "big_gaps": big_gaps,
        "gap_ts":   ts[gap_idx] if len(gap_idx) > 0 else np.array([]),
        "seg_edges": seg_edges,
        "seg_hz":   np.array(seg_hz),
        "seg_count": np.array(seg_count),
    }


def analyze_sync(stats_dict):
    """Kiểm tra độ lệch pha giữa 2 MAC."""
    print(f"\n{'─'*55}")
    print("  ĐỒNG BỘ 2 MAC")
    print(f"{'─'*55}")

    items = list(stats_dict.items())
    if len(items) < 2:
        print("  Chỉ có 1 MAC — không thể so sánh.")
        return

    (mac1, s1), (mac2, s2) = items[0], items[1]

    t_start_diff = abs(s1["ts"][0]  - s2["ts"][0])
    t_end_diff   = abs(s1["ts"][-1] - s2["ts"][-1])
    overlap_start = max(s1["ts"][0],  s2["ts"][0])
    overlap_end   = min(s1["ts"][-1], s2["ts"][-1])
    overlap_dur   = max(0, overlap_end - overlap_start)

    print(f"  Lệch điểm bắt đầu : {t_start_diff*1000:.1f} ms")
    print(f"  Lệch điểm kết thúc: {t_end_diff:.2f}s")
    print(f"  Thời gian GIAO nhau: {overlap_dur:.1f}s  ({overlap_dur/3600:.2f} giờ)")
    print(f"  Số gói tin MAC1   : {s1['n']:,}")
    print(f"  Số gói tin MAC2   : {s2['n']:,}")
    print(f"  Chênh lệch gói   : {abs(s1['n'] - s2['n']):,}  "
          f"({abs(s1['n']-s2['n'])/max(s1['n'],s2['n'])*100:.1f}%)")

    if overlap_dur < 30:
        print("\n  🔴 CẢNH BÁO: 2 MAC gần như không overlap — đây là nguyên nhân chính!")
    elif abs(s1['n'] - s2['n']) / max(s1['n'], s2['n']) > 0.3:
        print("\n  🟡 CẢNH BÁO: Số gói tin 2 MAC lệch nhau >30%")
    else:
        print("\n  ✅ 2 MAC đồng bộ tốt")


def plot_diagnostics(stats_dict, file_path):
    """Vẽ dashboard chẩn đoán toàn diện."""
    mac_labels = [f"MAC {i+1}: {m.hex(':').upper()}" for i, m in enumerate(stats_dict.keys())]
    stats_list = list(stats_dict.values())
    n_mac      = len(stats_list)

    fig = plt.figure(figsize=(20, 5 * (n_mac + 1)))
    fig.suptitle(
        f"CHẨN ĐOÁN FILE: {os.path.basename(file_path)}\n"
        f"({os.path.getsize(file_path)/1e9:.2f} GB  |  "
        f"{stats_list[0]['duration']/3600:.2f} giờ)",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(n_mac + 1, 3, figure=fig, hspace=0.45, wspace=0.3)

    colors = ["#1f77b4", "#ff7f0e"]

    for i, (mac, s) in enumerate(stats_dict.items()):
        lbl   = mac_labels[i]
        color = colors[i % 2]
        t_min = s["ts"] / 60  # đổi sang phút

        # ── Cột 1: Hz theo thời gian ──────────────────────────────
        ax1 = fig.add_subplot(gs[i, 0])
        seg_mid = (s["seg_edges"][:-1] + s["seg_edges"][1:]) / 2 / 60
        ax1.bar(seg_mid, s["seg_hz"],
                width=(s["seg_edges"][1] - s["seg_edges"][0]) / 60 * 0.9,
                color=color, alpha=0.75)
        ax1.axhline(s["avg_hz"], color="red", lw=1.5, linestyle="--",
                    label=f"TB: {s['avg_hz']:.1f} Hz")
        ax1.axhline(15, color="orange", lw=1, linestyle=":",
                    label="Ngưỡng tối thiểu (15 Hz)")
        ax1.set_title(f"{lbl}\nTần suất theo thời gian", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Thời gian (phút)")
        ax1.set_ylabel("Hz")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # ── Cột 2: Khoảng cách giữa các gói tin ──────────────────
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(t_min[1:], s["gaps"] * 1000, color=color, lw=0.4, alpha=0.6)
        ax2.axhline(GAP_THRESHOLD_SEC * 1000, color="red", lw=1.5,
                    linestyle="--", label=f"Ngưỡng gap ({GAP_THRESHOLD_SEC}s)")
        ax2.set_title(f"{lbl}\nKhoảng cách giữa gói tin", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Thời gian (phút)")
        ax2.set_ylabel("ms")
        ax2.set_ylim(0, min(GAP_THRESHOLD_SEC * 3000, s["gaps"].max() * 1100))
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # ── Cột 3: Số gói tin tích lũy ───────────────────────────
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(t_min, np.arange(len(t_min)), color=color, lw=1.5)
        # Vẽ đường lý tưởng (Hz đều)
        ideal = np.linspace(0, len(t_min) - 1, len(t_min))
        ax3.plot(t_min, ideal, color="gray", lw=1, linestyle="--",
                 alpha=0.6, label="Lý tưởng (đều)")
        # Đánh dấu khoảng trống lớn
        if len(s["gap_ts"]) > 0:
            for gt in s["gap_ts"]:
                ax3.axvline(gt / 60, color="red", lw=0.5, alpha=0.4)
        ax3.set_title(f"{lbl}\nGói tin tích lũy (đường đỏ = gap > {GAP_THRESHOLD_SEC}s)",
                      fontsize=9, fontweight="bold")
        ax3.set_xlabel("Thời gian (phút)")
        ax3.set_ylabel("Số gói tích lũy")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

    # ── Hàng cuối: So sánh Hz của 2 MAC cùng trục ────────────────
    if n_mac >= 2:
        ax_cmp = fig.add_subplot(gs[n_mac, :])
        for i, (mac, s) in enumerate(stats_dict.items()):
            seg_mid = (s["seg_edges"][:-1] + s["seg_edges"][1:]) / 2 / 60
            ax_cmp.plot(seg_mid, s["seg_hz"],
                        color=colors[i % 2], lw=1.5, alpha=0.8,
                        label=mac_labels[i])
        ax_cmp.axhline(15, color="orange", lw=1.5, linestyle=":",
                       label="Ngưỡng tối thiểu (15 Hz)")
        ax_cmp.set_title("SO SÁNH TẦN SUẤT 2 MAC THEO THỜI GIAN",
                         fontsize=10, fontweight="bold")
        ax_cmp.set_xlabel("Thời gian (phút)")
        ax_cmp.set_ylabel("Hz")
        ax_cmp.legend()
        ax_cmp.grid(True, alpha=0.3)

    out_path = os.path.splitext(file_path)[0] + "_diagnosis.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n📊 Biểu đồ đã lưu: {out_path}")
    plt.show()


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH

    if not os.path.exists(target):
        print(f"❌ Không tìm thấy file: {target}")
        return

    # 1. Parse
    raw_data, _ = parse_pcap(target)

    # 2. Phân tích từng MAC
    stats_dict = {}
    for mac in TARGET_MACS:
        ts_list = raw_data[mac]["ts"]
        label   = f"MAC {TARGET_MACS.index(mac)+1}"
        if len(ts_list) > 0:
            s = analyze_mac(mac, ts_list, label)
            if s:
                stats_dict[mac] = s
        else:
            print(f"\n  ❌ {label} ({mac.hex(':')}): Không có dữ liệu trong file này!")

    if not stats_dict:
        print("\n❌ Không tìm thấy dữ liệu cho bất kỳ MAC nào!")
        return

    # 3. Kiểm tra đồng bộ 2 MAC
    analyze_sync(stats_dict)

    # 4. Kết luận nguyên nhân ít windows
    print(f"\n{'═'*55}")
    print("  KẾT LUẬN")
    print(f"{'═'*55}")
    for mac, s in stats_dict.items():
        bad_segs = (s["seg_hz"] < 15).sum()
        pct_bad  = bad_segs / N_SEGMENTS * 100
        print(f"  {mac.hex(':').upper()}: "
              f"{bad_segs}/{N_SEGMENTS} đoạn dưới ngưỡng 15Hz ({pct_bad:.0f}%)")

    total_gap_time = sum(s["big_gaps"].sum() for s in stats_dict.values()) / len(stats_dict)
    print(f"\n  Tổng thời gian mất tín hiệu TB: {total_gap_time:.1f}s "
          f"({total_gap_time/list(stats_dict.values())[0]['duration']*100:.1f}% file)")

    # 5. Vẽ biểu đồ
    print("\n🖼  Đang vẽ biểu đồ chẩn đoán...")
    plot_diagnostics(stats_dict, target)


if __name__ == "__main__":
    main()