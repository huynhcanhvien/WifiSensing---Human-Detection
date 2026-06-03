# PBL5 — WiFi Sensing for Breathing Detection

Hệ thống phát hiện nhịp thở thời gian thực sử dụng **Channel State Information (CSI)** từ tín hiệu WiFi, kết hợp Raspberry Pi 4 (edge AI) và Mac (gateway + dashboard).

## Kiến Trúc Hệ Thống

```
[WiFi CSI Tool]
       ↓ UDP :5500
[Raspberry Pi 4]  ←── edge/pi_edge_ai.py ───→ ONNX Inference
       ↓ TCP :9999
[Mac Gateway]     ←── gateway/mac_gateway.py → SQLite DB
       ↓ WebSocket :8765
[Dashboard]       ←── gateway/dashboard.html
```

## Cấu Trúc Thư Mục

```
PBL5/
├── config.py               # Cấu hình tập trung (DSP, Training, Network, ...)
├── requirements.txt        # Toàn bộ dependencies
│
├── src/                    # Source code chính
│   ├── dsp/                # Xử lý tín hiệu (DSP)
│   │   ├── filters.py      # Hampel filter
│   │   ├── interpolation.py # FFT Shift, nội suy DC, resample
│   │   └── pipeline.py     # Pipeline DSP đầu-cuối
│   ├── data/               # Dataset loading
│   │   ├── dataset.py      # CSIDataset + get_dataloaders
│   │   └── build_dataset.py # Xây dựng .npz từ DB
│   ├── models/
│   │   └── light_resnet.py # Kiến trúc LightResNet
│   └── utils/
│       ├── engine.py       # Training/Eval engine
│       ├── metrics.py      # Metrics + confusion matrix
│       ├── logger.py       # AverageMeter, Logger
│       └── seed.py         # Reproducibility
│
├── scripts/                # Entry-point scripts
│   ├── train.py            # Huấn luyện mô hình
│   ├── evaluate.py         # Đánh giá mô hình
│   ├── build_dataset.py    # Xây dựng dataset từ DB
│   ├── export_onnx.py      # Export sang ONNX
│   └── export_csi.py       # Export CSI raw ra CSV/NPY
│
├── edge/                   # Code chạy trên Raspberry Pi
│   └── pi_edge_ai.py
│
├── gateway/                # Code chạy trên Mac
│   ├── mac_gateway.py
│   └── dashboard.html
│
├── visualization/          # Tất cả visualization scripts
│   ├── plot_pcap.py        # Heatmap + 3D surface từ file PCAP
│   ├── plot_csv.py         # Dashboard từ file CSV (có slider)
│   ├── plot_dataset.py     # Phân tích dataset .npz
│   ├── plot_training.py    # Training curves từ TensorBoard
│   ├── run_all.py          # Chạy toàn bộ pipeline một lệnh
│   └── output/             # ← Tất cả ảnh xuất ra đây
│
├── data_collection/        # Scripts thu thập và debug PCAP/CSV
│   ├── preprocess.py       # Tiền xử lý file CSV
│   ├── build_pcap.py
│   └── ...
│
├── data/                   # Dữ liệu thô (không commit lên git)
│   ├── Data/breathing/     # PCAP + CSV cho hoạt động thở
│   ├── Data/empty/         # PCAP + CSV cho phòng trống
│   └── wifi_sensing_dataset.npz  # Dataset đã xử lý
│
├── artifacts/              # Model files
│   ├── checkpoints/        # PyTorch checkpoints (.pth)
│   ├── runs/               # TensorBoard logs
│   └── csi_lightresnet.onnx
│
└── tests/                  # Unit tests
    ├── test_dsp.py
    ├── test_model.py
    └── test_dataset.py
```

## Cài Đặt

```bash
# Clone và cài dependencies
pip install -r requirements.txt
```

## Workflow

### 1. Thu Thập Dữ Liệu

```bash
# Trên Raspberry Pi: gửi CSI packets
bash data/receiver.sh

# Trên Mac: khởi động gateway để nhận và lưu vào DB
python gateway/mac_gateway.py

# Mở dashboard để gán nhãn
open gateway/dashboard.html
```

### 2. Xây Dựng Dataset

```bash
# Từ SQLite DB → wifi_sensing_dataset.npz
python scripts/build_dataset.py
# hoặc chỉ định path:
python scripts/build_dataset.py --db wifi_sensing.db --out data/wifi_sensing_dataset.npz
```

### 3. Huấn Luyện

```bash
python scripts/train.py
# Tùy chỉnh:
python scripts/train.py --epochs 100 --lr 5e-4 --batch 32
```

Theo dõi bằng TensorBoard:
```bash
tensorboard --logdir artifacts/runs
```

### 4. Đánh Giá

```bash
python scripts/evaluate.py
```

### 5. Export ONNX (để chạy trên Pi)

```bash
python scripts/export_onnx.py
# → artifacts/csi_lightresnet.onnx
```

Sao chép file ONNX sang Raspberry Pi và chạy:
```bash
python edge/pi_edge_ai.py
```

### 6. Export CSI Raw (để phân tích)

```bash
python scripts/export_csi.py --list          # Liệt kê sessions
python scripts/export_csi.py --session 0     # Xuất session 0
```

## Visualization

Tất cả ảnh được lưu tự động vào `visualization/output/`.

```bash
# Chạy toàn bộ pipeline một lệnh (batch mode, không hiện cửa sổ)
python visualization/run_all.py

# Hoặc từng script riêng:

# Vẽ từ file PCAP (heatmap 2D + 3D surface)
python visualization/plot_pcap.py --file data/Data/breathing/breath_xxx.pcap

# Vẽ dashboard từ file CSV (có slider tương tác)
python visualization/plot_csv.py --file data/Data/breathing/breath_xxx_realtime.csv

# Phân tích dataset (heatmap samples + class distribution + amplitude histogram)
python visualization/plot_dataset.py

# Training curves + Confusion Matrix
python visualization/plot_training.py
```

| Script | Output | Mô tả |
|--------|--------|-------|
| `plot_pcap.py` | `pcap_<name>.png` | Heatmap CSI + 3D surface từ PCAP |
| `plot_csv.py` | `csv_<name>.png` | Dashboard 3 cột với slider thời gian |
| `plot_dataset.py` | `dataset_*.png` (3 file) | Heatmap samples, class distribution, amplitude histogram |
| `plot_training.py` | `training_<run>.png` | Loss/Accuracy/LR curves từ TensorBoard |
| `run_all.py` | tất cả ảnh trên | Master script chạy mọi thứ |

## Tests

```bash
# Chạy toàn bộ tests
python -m pytest tests/ -v

# Chạy test cụ thể
python -m pytest tests/test_dsp.py -v
```

## Cấu Hình

Tất cả tham số được quản lý tập trung tại [`config.py`](config.py):

| Class | Mục đích |
|-------|---------|
| `Paths` | Đường dẫn file/thư mục |
| `Hardware` | MAC addresses |
| `DSP` | Tham số xử lý tín hiệu |
| `DatasetConfig` | Nhãn, số lớp, tỷ lệ split |
| `TrainingConfig` | Batch size, epochs, learning rate |
| `NetworkConfig` | Port TCP/WebSocket/UDP |
| `ExportConfig` | Cấu hình export CSV |

## Nhãn Dataset

| Label | Mô tả |
|-------|-------|
| `0` — Empty | Phòng trống, không có người |
| `1` — Breathing | Có người đang thở bình thường |

## Tác Giả

Huỳnh Cảnh Viên · Phạm Nhật Khoa · Phạm Quang Thắng
