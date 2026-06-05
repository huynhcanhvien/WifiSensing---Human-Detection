"""
gateway/api_server.py — REST API server chạy trên Mac.

Cung cấp dữ liệu từ SQLite cho React frontend (History + Statistics).

Cách dùng:
    python gateway/api_server.py

Endpoints:
    GET /api/inference          — lịch sử AI results (phân trang, lọc)
    GET /api/inference/stats    — thống kê tổng hợp
    GET /api/inference/export   — tải CSV
    GET /api/csi                — raw CSI frames (phân trang, lọc theo label)
    GET /api/labels             — danh sách labels có trong csi_raw
    GET /api/health             — kiểm tra trạng thái
"""

import csv
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sqlite3

from config import Paths

app = FastAPI(title="WiFi Sensing API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(Paths.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ─── Inference Logs ───────────────────────────────────────────────────────────

@app.get("/api/inference")
def get_inference(
    page:   int            = Query(1, ge=1),
    limit:  int            = Query(50, ge=1, le=200),
    action: Optional[str]  = Query(None),
    from_:  Optional[str]  = Query(None, alias="from"),
    to:     Optional[str]  = Query(None),
):
    conn   = get_db()
    cursor = conn.cursor()

    conditions, params = [], []
    if action:
        conditions.append("predicted_action = ?")
        params.append(action)
    if from_:
        conditions.append("timestamp >= ?")
        params.append(from_)
    if to:
        conditions.append("timestamp <= ?")
        params.append(to)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    total = cursor.execute(
        f"SELECT COUNT(*) FROM inference_logs {where}", params
    ).fetchone()[0]

    offset = (page - 1) * limit
    rows   = cursor.execute(
        f"SELECT id, timestamp, predicted_action, confidence_score, latency_ms "
        f"FROM inference_logs {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()

    conn.close()
    return {
        "data":       [dict(r) for r in rows],
        "total":      total,
        "page":       page,
        "limit":      limit,
        "totalPages": (total + limit - 1) // limit,
    }


@app.get("/api/inference/stats")
def get_inference_stats():
    conn   = get_db()
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
    if total == 0:
        conn.close()
        return {"total": 0, "byAction": [], "avgConfidence": 0, "avgLatency": 0,
                "confidenceTimeline": [], "latencyHistogram": []}

    by_action = cursor.execute("""
        SELECT predicted_action,
               COUNT(*) as count,
               ROUND(AVG(confidence_score), 2) as avg_confidence,
               ROUND(AVG(latency_ms), 2) as avg_latency
        FROM inference_logs
        GROUP BY predicted_action
        ORDER BY count DESC
    """).fetchall()

    avg_conf = cursor.execute(
        "SELECT ROUND(AVG(confidence_score), 2) FROM inference_logs"
    ).fetchone()[0]

    avg_lat = cursor.execute(
        "SELECT ROUND(AVG(latency_ms), 2) FROM inference_logs"
    ).fetchone()[0]

    # Timeline: confidence trung bình theo từng phút (200 điểm gần nhất)
    timeline = cursor.execute("""
        SELECT strftime('%Y-%m-%d %H:%M', timestamp) as minute,
               ROUND(AVG(confidence_score), 2) as avg_conf,
               COUNT(*) as count
        FROM inference_logs
        GROUP BY minute
        ORDER BY minute DESC
        LIMIT 60
    """).fetchall()

    # Latency histogram
    lat_hist = cursor.execute("""
        SELECT
            CASE
                WHEN latency_ms < 100  THEN '<100ms'
                WHEN latency_ms < 200  THEN '100-200ms'
                WHEN latency_ms < 300  THEN '200-300ms'
                WHEN latency_ms < 500  THEN '300-500ms'
                ELSE '>500ms'
            END as bucket,
            COUNT(*) as count
        FROM inference_logs
        GROUP BY bucket
        ORDER BY MIN(latency_ms)
    """).fetchall()

    conn.close()
    return {
        "total":               total,
        "byAction":            [dict(r) for r in by_action],
        "avgConfidence":       avg_conf,
        "avgLatency":          avg_lat,
        "confidenceTimeline":  [dict(r) for r in reversed(timeline)],
        "latencyHistogram":    [dict(r) for r in lat_hist],
    }


@app.get("/api/inference/export")
def export_inference_csv(
    action: Optional[str] = Query(None),
    from_:  Optional[str] = Query(None, alias="from"),
    to:     Optional[str] = Query(None),
):
    conn   = get_db()
    cursor = conn.cursor()

    conditions, params = [], []
    if action:
        conditions.append("predicted_action = ?")
        params.append(action)
    if from_:
        conditions.append("timestamp >= ?")
        params.append(from_)
    if to:
        conditions.append("timestamp <= ?")
        params.append(to)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows  = cursor.execute(
        f"SELECT id, timestamp, predicted_action, confidence_score, latency_ms "
        f"FROM inference_logs {where} ORDER BY timestamp DESC",
        params,
    ).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "predicted_action", "confidence_score", "latency_ms"])
    for row in rows:
        writer.writerow(list(row))

    output.seek(0)
    filename = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─── CSI Raw ──────────────────────────────────────────────────────────────────

@app.get("/api/labels")
def get_labels():
    conn  = get_db()
    rows  = conn.execute(
        "SELECT DISTINCT label FROM csi_raw WHERE label IS NOT NULL ORDER BY label"
    ).fetchall()
    conn.close()
    return {"labels": [r[0] for r in rows]}


@app.get("/api/csi")
def get_csi(
    page:  int           = Query(1, ge=1),
    limit: int           = Query(100, ge=1, le=500),
    label: Optional[str] = Query(None),
    mac:   Optional[str] = Query(None),
):
    conn   = get_db()
    cursor = conn.cursor()

    conditions, params = [], []
    if label:
        conditions.append("label = ?")
        params.append(label)
    if mac:
        conditions.append("mac = ?")
        params.append(mac)

    where  = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    total  = cursor.execute(
        f"SELECT COUNT(*) FROM csi_raw {where}", params
    ).fetchone()[0]

    offset = (page - 1) * limit
    rows   = cursor.execute(
        f"SELECT id, timestamp, mac, label FROM csi_raw {where} "
        f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()

    conn.close()
    return {
        "data":       [dict(r) for r in rows],
        "total":      total,
        "page":       page,
        "limit":      limit,
        "totalPages": (total + limit - 1) // limit,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("[INFO] API server on http://localhost:8000")
    print("[INFO] Docs:       http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
