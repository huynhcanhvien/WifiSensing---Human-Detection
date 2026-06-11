"""
config.py — Cấu hình tập trung cho toàn bộ dự án PBL5.

Tất cả hằng số dùng chung giữa các module (DSP, training, edge, gateway)
đều được định nghĩa tại đây. Không khai báo lại ở file khác.
"""

import os
from datetime import datetime
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

class Paths:
    ROOT        = ROOT_DIR
    DATA_DIR    = ROOT_DIR / "data"
    DATASET_NPZ = ROOT_DIR / "data" / "wifi_sensing_dataset.npz"
    DB_PATH     = ROOT_DIR / "wifi_sensing.db"
    ARTIFACTS   = ROOT_DIR / "artifacts"
    CHECKPOINTS = ROOT_DIR / "artifacts" / "checkpoints"
    RUNS        = ROOT_DIR / "artifacts" / "runs"
    ONNX_MODEL  = ROOT_DIR / "artifacts" / "csi_lightresnet.onnx"

# ─── Hardware / MAC Addresses ─────────────────────────────────────────────────
class Hardware:
    TARGET_MACS = [
        "d4e9f4a36480",
        "d4e9f4a43c18",
    ]
    # Bytes form dùng cho pi_edge_ai.py (so sánh với UDP packet bytes)
    TARGET_MACS_BYTES = [bytes.fromhex(m) for m in TARGET_MACS]

# ─── DSP (Signal Processing) ──────────────────────────────────────────────────
class DSP:
    WINDOW_SEC       = 5.0          # Độ dài cửa sổ (giây)
    TARGET_FS        = 50.0         # Tần số resample (Hz)
    TARGET_SAMPLES   = int(WINDOW_SEC * TARGET_FS)   # = 250
    MIN_FS_THRESHOLD = 40.0         # Tần số tối thiểu để không bỏ session
    OVERLAP_PCT      = 0.5          # Tỷ lệ overlap giữa các cửa sổ
    TRIM_SEC         = 2.0          # Cắt bỏ đầu/cuối (giây) để loại nhiễu chuột
    MIN_TOTAL_SEC    = 10.0         # Session phải dài tối thiểu (giây)
    BUFFER_KEEP_SEC  = 7.0          # Giữ buffer bao lâu trên Pi

    # Hampel filter
    HAMPEL_WINDOW = 5
    HAMPEL_SIGMA  = 3.0
    K_HAMPEL      = 1.4826          # = 1 / Phi^{-1}(3/4) ≈ 1/qnorm(3/4)

    # Subcarrier mapping (802.11n / HT20, 64 subcarriers)
    NULL_SUBCARRIERS  = [0,1,2,3,4,5,29,30,31,32,33,34,59,60,61,62,63]
    VALID_SUBCARRIERS = [i for i in range(64) if i not in [0,1,2,3,4,5,29,30,31,32,33,34,59,60,61,62,63]]
    EFFECTIVE_SC_START = 6   # sau khi FFTshift + nội suy, lấy [6:59]
    EFFECTIVE_SC_END   = 59  # → 53 subcarrier hữu ích

# ─── Dataset ──────────────────────────────────────────────────────────────────
class DatasetConfig:
    CLASS_MAP   = {"Empty": 0, "Breathing": 1}
    CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
    NUM_CLASSES = 2
    TEST_SIZE   = 0.2
    VAL_SIZE    = 0.1   # % tập train dùng làm validation

# ─── Training ─────────────────────────────────────────────────────────────────
class TrainingConfig:
    BATCH_SIZE          = 64
    NUM_WORKERS         = 2
    EPOCHS              = 50
    LEARNING_RATE       = 1e-3
    WEIGHT_DECAY        = 1e-4
    CHECKPOINT_INTERVAL = 10
    SEED                = 42

    # TensorBoard log dir — tạo theo timestamp mỗi lần train
    @staticmethod
    def log_dir():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Paths.RUNS / f"experiment_{ts}")

    # Tự động nhận diện device — lazy import để không bắt buộc torch trên Pi
    try:
        import torch as _torch
        if _torch.backends.mps.is_available():
            DEVICE = _torch.device("mps")
        elif _torch.cuda.is_available():
            DEVICE = _torch.device("cuda")
        else:
            DEVICE = _torch.device("cpu")
    except ImportError:
        DEVICE = "cpu"

# ─── Network (Gateway & Edge) ─────────────────────────────────────────────────
class NetworkConfig:
    GATEWAY_HOST    = "0.0.0.0"
    GATEWAY_PORT    = 9999
    WS_PORT         = 8765
    PI_UDP_PORT     = 5500
    MAC_GATEWAY_IP  = "192.168.10.1"   # IP Mac trên Pi
    INFERENCE_INTERVAL = 2.5            # Giây giữa mỗi lần inference
    STARTUP_DELAY   = 10.0              # Thời gian chờ trước khi AI bắt đầu

# ─── Export ───────────────────────────────────────────────────────────────────
class ExportConfig:
    TARGET_FS  = 20.0   # Hz — resample khi export CSV (khác training)
    OUTPUT_DIR = ROOT_DIR / "exports"

# ─── Tạo thư mục quan trọng nếu chưa có ──────────────────────────────────────
for _dir in [Paths.CHECKPOINTS, Paths.RUNS, ExportConfig.OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)
