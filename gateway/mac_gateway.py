"""
gateway/mac_gateway.py — Chạy trên Mac (PC trung tâm).

Nhận dữ liệu từ Raspberry Pi qua TCP → lưu vào SQLite → broadcast qua WebSocket → Dashboard.

Cách dùng:
    python gateway/mac_gateway.py

Sơ đồ:
    [Pi] --TCP:9999--> [mac_gateway.py] --WebSocket:8765--> [dashboard.html]
                              ↓
                      [wifi_sensing.db]

Labeling qua Dashboard:
    - Mở dashboard.html trong trình duyệt
    - Nhấn "Start Label" → gán nhãn cho dữ liệu đang thu thập
    - Nhấn "Stop Label" → dừng gán nhãn
"""

import asyncio
import json
import os
import socket
import sqlite3
import threading
from datetime import datetime

import websockets

from config import NetworkConfig, Paths


# ─── Global state ─────────────────────────────────────────────────────────────

CURRENT_LABEL: str | None = None   # Nhãn đang được gán (None = không gán)
ws_clients: set = set()
ws_loop: asyncio.AbstractEventLoop | None = None


# ─── WebSocket Bridge ─────────────────────────────────────────────────────────

async def ws_handler(websocket):
    """Xử lý mỗi WebSocket client (Dashboard)."""
    global CURRENT_LABEL
    ws_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("cmd") == "start_label":
                    CURRENT_LABEL = data.get("label", "")
                    print(f"\n[LABELING] 🔴 STARTED recording label: '{CURRENT_LABEL}'")
                elif data.get("cmd") == "stop_label":
                    CURRENT_LABEL = None
                    print("\n[LABELING] ⏹️ STOPPED. Label cleared.")
            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)


async def ws_main():
    async with websockets.serve(ws_handler, "0.0.0.0", NetworkConfig.WS_PORT):
        print(f"[INFO] WebSocket server on :{NetworkConfig.WS_PORT}")
        await asyncio.Future()   # run forever


def start_ws_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(ws_main())


def broadcast(payload_dict: dict):
    """Thread-safe broadcast đến tất cả WebSocket clients."""
    global ws_clients, ws_loop
    if not ws_loop:
        return
    payload_dict["current_label"] = CURRENT_LABEL
    msg = json.dumps(payload_dict)
    for ws in ws_clients.copy():
        try:
            asyncio.run_coroutine_threadsafe(ws.send(msg), ws_loop)
        except Exception:
            ws_clients.discard(ws)


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db(db_name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_name, check_same_thread=False)
    c    = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            predicted_action TEXT,
            confidence_score REAL,
            latency_ms       REAL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS csi_raw (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mac       TEXT,
            amp       TEXT,
            label     TEXT
        )
    ''')

    # Thêm cột label nếu database cũ chưa có
    try:
        c.execute("ALTER TABLE csi_raw ADD COLUMN label TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    return conn


# ─── Payload Handler ──────────────────────────────────────────────────────────

def handle_payload(payload: dict, cursor: sqlite3.Cursor):
    """Xử lý payload từ Pi: lưu DB và broadcast."""
    global CURRENT_LABEL
    ptype = payload.get("type")

    if ptype == "ai_result":
        ts         = payload.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        action     = payload.get("action", "Unknown")
        confidence = payload.get("confidence", 0.0)
        latency    = payload.get("latency_ms", 0.0)

        cursor.execute(
            "INSERT INTO inference_logs (timestamp, predicted_action, confidence_score, latency_ms) VALUES (?, ?, ?, ?)",
            (ts, action, confidence, latency),
        )

        bar     = "█" * int(confidence / 5) + "░" * (20 - int(confidence / 5))
        lbl_tag = f"[LBL: {CURRENT_LABEL}]" if CURRENT_LABEL else ""
        print(f"[{ts}]  {action:<10}  |{bar}|  {confidence:>5.1f}%  {latency:>6.1f}ms  {lbl_tag}")
        broadcast(payload)

    elif ptype == "raw_data":
        frames = payload.get("data", [])
        rows   = [
            (f["timestamp"], f["mac"], json.dumps(f["amp"]), CURRENT_LABEL)
            for f in frames
        ]
        cursor.executemany(
            "INSERT INTO csi_raw (timestamp, mac, amp, label) VALUES (?, ?, ?, ?)",
            rows,
        )
        broadcast(payload)

    else:
        print(f"  [WARN] Unknown payload type: {ptype}")


# ─── TCP Server ───────────────────────────────────────────────────────────────

def recv_line(sock: socket.socket) -> str | None:
    """Đọc một dòng JSON kết thúc bằng '\\n' từ socket."""
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return None
        buf += chunk
        if b"\n" in buf:
            line, _ = buf.split(b"\n", 1)
            return line.decode("utf-8")


def run_server():
    db_name  = str(Paths.DB_PATH)
    conn_db  = init_db(db_name)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((NetworkConfig.GATEWAY_HOST, NetworkConfig.GATEWAY_PORT))

    threading.Thread(target=start_ws_server, daemon=True).start()
    server.listen(1)

    print(f"[INFO] Gateway listening on :{NetworkConfig.GATEWAY_PORT}")
    print(f"[INFO] Database: {os.path.abspath(db_name)}\n")

    while True:
        print("[INFO] Waiting for Pi connection...")
        try:
            client, addr = server.accept()
            print(f"[OK]   Pi connected from {addr}\n")
            cursor = conn_db.cursor()

            while True:
                line = recv_line(client)
                if line is None:
                    print("[WARN] Pi disconnected. Waiting for reconnect...\n")
                    break
                try:
                    payload = json.loads(line)
                    handle_payload(payload, cursor)
                    conn_db.commit()
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Bad JSON: {e}")

        except KeyboardInterrupt:
            print("\n[INFO] Shutting down.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            try:
                client.close()
            except Exception:
                pass

    server.close()
    conn_db.close()
    print("[INFO] Closed.")


if __name__ == "__main__":
    run_server()
