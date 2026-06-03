"""
edge/pi_edge_ai.py — Chạy trên Raspberry Pi 4.

Thu thập UDP packets từ WiFi CSI tool → DSP pipeline → ONNX inference → gửi về Mac Gateway qua TCP.

Cách dùng:
    python edge/pi_edge_ai.py

Yêu cầu:
    - opencv-python (cho cv2.dnn.readNetFromONNX)
    - scipy, numpy
    - File ONNX: artifacts/csi_lightresnet.onnx (copy từ Mac sang Pi)

Sơ đồ:
    [WiFi CSI Tool] --UDP:5500--> [pi_edge_ai.py] --TCP:9999--> [mac_gateway.py]
"""

import queue
import socket
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from config import DSP, DatasetConfig, Hardware, NetworkConfig
from src.dsp.pipeline import prepare_inference_input


# ─── Thread-safe resources ────────────────────────────────────────────────────

tcp_queue   = queue.Queue(maxsize=500)
# deque: O(1) popleft
ai_buffers  = {mac: deque() for mac in Hardware.TARGET_MACS_BYTES}
buffer_lock = threading.Lock()


# ─── TCP Sender ───────────────────────────────────────────────────────────────

def tcp_sender_worker():
    """Worker thread: dequeue payload → gửi đến Mac Gateway qua TCP."""
    import json

    def make_conn() -> socket.socket:
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((NetworkConfig.MAC_GATEWAY_IP, NetworkConfig.GATEWAY_PORT))
                print("[INFO] TCP connected to gateway.")
                return s
            except Exception:
                print("[WARN] Gateway not reachable. Retry in 2s...")
                time.sleep(2)

    client = make_conn()
    while True:
        payload = tcp_queue.get()
        try:
            client.sendall((json.dumps(payload) + "\n").encode('utf-8'))
        except Exception as e:
            print(f"[WARN] TCP dropped, reconnecting... ({e})")
            client = make_conn()
        tcp_queue.task_done()


# ─── AI Inference ─────────────────────────────────────────────────────────────

def ai_inference_worker():
    """Worker thread: định kỳ lấy snapshot buffer → DSP → ONNX → gửi kết quả."""
    from config import Paths

    print(f"[INFO] AI sleeping {NetworkConfig.STARTUP_DELAY}s for buffer warmup...")
    time.sleep(NetworkConfig.STARTUP_DELAY)

    onnx_path = str(Paths.ONNX_MODEL)
    net = cv2.dnn.readNetFromONNX(onnx_path)
    print(f"[INFO] AI Engine ready — model: {onnx_path}")

    while True:
        time.sleep(NetworkConfig.INFERENCE_INTERVAL)

        now     = time.time()
        t_end   = now
        t_start = now - DSP.WINDOW_SEC

        with buffer_lock:
            raw0 = list(ai_buffers[Hardware.TARGET_MACS_BYTES[0]])
            raw1 = list(ai_buffers[Hardware.TARGET_MACS_BYTES[1]])

        snap0 = [(t, a) for t, a in raw0 if t_start <= t <= t_end]
        snap1 = [(t, a) for t, a in raw1 if t_start <= t <= t_end]

        if len(snap0) < 100 or len(snap1) < 100:
            print(f"[WARN] Not enough data: MAC0={len(snap0)}, MAC1={len(snap1)}")
            continue

        t_infer_start = datetime.now()

        # DSP pipeline → ONNX input tensor
        X = prepare_inference_input(snap0, snap1, t_start, t_end)

        net.setInput(X)
        logits     = net.forward()[0]
        exp_l      = np.exp(logits - np.max(logits))
        probs      = exp_l / exp_l.sum()
        class_id   = int(np.argmax(probs))
        latency_ms = round((datetime.now() - t_infer_start).total_seconds() * 1000, 2)

        label = DatasetConfig.CLASS_NAMES[class_id]
        tcp_queue.put({
            "type":       "ai_result",
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "action":     label,
            "confidence": round(float(probs[class_id] * 100), 2),
            "latency_ms": latency_ms,
        })
        print(f"[AI] {label}  {probs[class_id]*100:.1f}%  ({latency_ms}ms)")


# ─── Main: UDP Listener ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    threading.Thread(target=tcp_sender_worker,   daemon=True).start()
    threading.Thread(target=ai_inference_worker, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', NetworkConfig.PI_UDP_PORT))
    print(f"[INFO] Listening UDP :{NetworkConfig.PI_UDP_PORT}")

    raw_batch = []

    try:
        while True:
            data, _ = sock.recvfrom(4096)
            mac_addr = data[4:10]
            if mac_addr not in Hardware.TARGET_MACS_BYTES:
                continue

            ts  = time.time()
            raw = np.frombuffer(data[-256:], dtype=np.int16)
            amp = np.sqrt(raw[0::2].astype(float)**2 + raw[1::2].astype(float)**2)
            if len(amp) != 64:
                continue

            with buffer_lock:
                ai_buffers[mac_addr].append((ts, amp))
                # Xóa dữ liệu cũ: O(1) popleft
                while ai_buffers[mac_addr] and \
                      (ts - ai_buffers[mac_addr][0][0]) > DSP.BUFFER_KEEP_SEC:
                    ai_buffers[mac_addr].popleft()

            dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            raw_batch.append({
                "timestamp": dt_str,
                "mac":       mac_addr.hex(),
                "amp":       amp.tolist(),
            })

            # Gửi theo batch 50 frames
            if len(raw_batch) >= 50:
                tcp_queue.put({"type": "raw_data", "data": raw_batch.copy()})
                raw_batch.clear()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down.")
        sock.close()
