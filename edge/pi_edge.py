"""
edge/pi_standalone.py — Chạy trên Raspberry Pi 4, hoàn toàn độc lập.

Chỉ cần copy 2 file sang Pi:
    scp edge/pi_standalone.py artifacts/checkpoints/csi_lightresnet.onnx pi@192.168.10.2:~/

Yêu cầu (cài 1 lần trên Pi):
    sudo apt install -y python3-opencv
    pip3 install "numpy<2" scipy

Chạy:
    python3 pi_standalone.py
"""

import queue
import socket
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate as scipy_interp

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH — chỉnh ở đây nếu cần
# ══════════════════════════════════════════════════════════════════════════════

# Đường dẫn model ONNX (cùng thư mục với script này)
import os
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "csi_lightresnet.onnx")

# MAC address 2 thiết bị WiFi (chữ thường, không dấu phẩy)
TARGET_MACS = ["d4e9f4a36480", "d4e9f4a43c18"]
TARGET_MACS_BYTES = [bytes.fromhex(m) for m in TARGET_MACS]

# Mạng
GATEWAY_IP          = "192.168.10.1"   # IP máy Mac nhận kết quả
GATEWAY_PORT        = 9999
UDP_PORT            = 5500
INFERENCE_INTERVAL  = 2.5              # giây giữa 2 lần inference
STARTUP_DELAY       = 10.0             # giây chờ buffer đủ dữ liệu

# DSP
WINDOW_SEC      = 5.0
TARGET_FS       = 50.0
TARGET_SAMPLES  = int(WINDOW_SEC * TARGET_FS)   # = 250
BUFFER_KEEP_SEC = 7.0

HAMPEL_WINDOW = 5
HAMPEL_SIGMA  = 3.0
K_HAMPEL      = 1.4826

NULL_SC          = {0,1,2,3,4,5,29,30,31,32,33,34,59,60,61,62,63}
VALID_SC         = [i for i in range(64) if i not in NULL_SC]
EFFECTIVE_SC_START = 6
EFFECTIVE_SC_END   = 59   # → 53 subcarrier hữu ích

CLASS_NAMES = {0: "Empty", 1: "Breathing"}

# ══════════════════════════════════════════════════════════════════════════════
# DSP PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def fft_shift(csi: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(csi, axes=1)


def interpolate_dc_pilots(csi: np.ndarray) -> np.ndarray:
    valid_idx  = np.array(VALID_SC)
    output     = np.empty_like(csi, dtype=float)
    valid_vals = csi[:, valid_idx].T   # (M, T)
    for xi in range(64):
        idx = np.searchsorted(valid_idx, xi)
        idx = np.clip(idx, 1, len(valid_idx) - 1)
        x0, x1 = valid_idx[idx - 1], valid_idx[idx]
        y0, y1  = valid_vals[idx - 1], valid_vals[idx]
        t = (xi - x0) / (x1 - x0) if x1 != x0 else 0.0
        output[:, xi] = y0 + t * (y1 - y0)
    return output


def hampel_filter(csi: np.ndarray) -> np.ndarray:
    w   = 2 * HAMPEL_WINDOW + 1
    out = csi.copy()
    for sc in range(csi.shape[1]):
        series   = csi[:, sc]
        padded   = np.pad(series, HAMPEL_WINDOW, mode="edge")
        windows  = sliding_window_view(padded, w)
        medians  = np.median(windows, axis=1)
        mads     = np.median(np.abs(windows - medians[:, None]), axis=1)
        outliers = np.abs(series - medians) > HAMPEL_SIGMA * K_HAMPEL * mads
        out[outliers, sc] = medians[outliers]
    return out


def resample_to_grid(csi: np.ndarray, t_orig: np.ndarray,
                     t_grid: np.ndarray) -> np.ndarray:
    f = scipy_interp.interp1d(t_orig, csi, axis=0,
                               fill_value="extrapolate", bounds_error=False)
    return f(t_grid)


def prepare_input(snap0: list, snap1: list,
                  t_start: float, t_end: float) -> np.ndarray:
    """DSP pipeline → tensor (1, 2, T, 53) sẵn sàng cho ONNX."""
    t_grid = np.arange(t_start, t_end, 1.0 / TARGET_FS)[:TARGET_SAMPLES]
    tensors = []
    for snap in [snap0, snap1]:
        t_mac   = np.array([x[0] for x in snap])
        csi_raw = np.array([x[1] for x in snap])
        csi = fft_shift(csi_raw)
        csi = interpolate_dc_pilots(csi)
        csi = hampel_filter(csi)
        csi = csi[:, EFFECTIVE_SC_START:EFFECTIVE_SC_END]   # (T, 53)
        norms = np.linalg.norm(csi, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        csi = csi / norms
        csi = resample_to_grid(csi, t_mac, t_grid)          # (250, 53)
        tensors.append(csi)
    mimo = np.stack(tensors, axis=-1)           # (250, 53, 2)
    nchw = np.transpose(mimo, (2, 0, 1))        # (2, 250, 53)
    return np.expand_dims(nchw, 0).astype(np.float32)   # (1, 2, 250, 53)


# ══════════════════════════════════════════════════════════════════════════════
# THREADS
# ══════════════════════════════════════════════════════════════════════════════

tcp_queue  = queue.Queue(maxsize=500)
ai_buffers = {mac: deque() for mac in TARGET_MACS_BYTES}
buf_lock   = threading.Lock()


def tcp_sender():
    import json

    def connect():
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((GATEWAY_IP, GATEWAY_PORT))
                print("[INFO] TCP connected to gateway.")
                return s
            except Exception:
                print("[WARN] Gateway not reachable. Retry in 2s...")
                time.sleep(2)

    client = connect()
    while True:
        payload = tcp_queue.get()
        try:
            client.sendall((json.dumps(payload) + "\n").encode())
        except Exception as e:
            print(f"[WARN] TCP dropped, reconnecting... ({e})")
            client = connect()
        tcp_queue.task_done()


def ai_worker():
    print(f"[INFO] AI sleeping {STARTUP_DELAY}s for buffer warmup...")
    time.sleep(STARTUP_DELAY)

    net = cv2.dnn.readNetFromONNX(ONNX_PATH)
    print(f"[INFO] AI Engine ready — {ONNX_PATH}")

    while True:
        time.sleep(INFERENCE_INTERVAL)

        now     = time.time()
        t_end   = now
        t_start = now - WINDOW_SEC

        with buf_lock:
            raw0 = list(ai_buffers[TARGET_MACS_BYTES[0]])
            raw1 = list(ai_buffers[TARGET_MACS_BYTES[1]])

        snap0 = [(t, a) for t, a in raw0 if t_start <= t <= t_end]
        snap1 = [(t, a) for t, a in raw1 if t_start <= t <= t_end]

        if len(snap0) < 100 or len(snap1) < 100:
            print(f"[WARN] Not enough data: MAC0={len(snap0)}, MAC1={len(snap1)}")
            continue

        t0 = datetime.now()
        X  = prepare_input(snap0, snap1, t_start, t_end)
        net.setInput(X)
        logits    = net.forward()[0]
        exp_l     = np.exp(logits - np.max(logits))
        probs     = exp_l / exp_l.sum()
        class_id  = int(np.argmax(probs))
        latency   = round((datetime.now() - t0).total_seconds() * 1000, 2)

        label = CLASS_NAMES[class_id]
        tcp_queue.put({
            "type":       "ai_result",
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "action":     label,
            "confidence": round(float(probs[class_id] * 100), 2),
            "latency_ms": latency,
        })
        print(f"[AI] {label}  {probs[class_id]*100:.1f}%  ({latency}ms)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — UDP Listener
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    threading.Thread(target=tcp_sender, daemon=True).start()
    threading.Thread(target=ai_worker,  daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    print(f"[INFO] Listening UDP :{UDP_PORT}")

    raw_batch = []
    try:
        while True:
            data, _ = sock.recvfrom(4096)
            mac = data[4:10]
            if mac not in TARGET_MACS_BYTES:
                continue

            ts  = time.time()
            raw = np.frombuffer(data[-256:], dtype=np.int16)
            amp = np.sqrt(raw[0::2].astype(float)**2 + raw[1::2].astype(float)**2)
            if len(amp) != 64:
                continue

            with buf_lock:
                ai_buffers[mac].append((ts, amp))
                while (ai_buffers[mac] and
                       ts - ai_buffers[mac][0][0] > BUFFER_KEEP_SEC):
                    ai_buffers[mac].popleft()

            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            raw_batch.append({
                "timestamp": dt,
                "mac":       mac.hex(),
                "amp":       amp.tolist(),
            })
            if len(raw_batch) >= 50:
                tcp_queue.put({"type": "raw_data", "data": raw_batch.copy()})
                raw_batch.clear()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down.")
        sock.close()
