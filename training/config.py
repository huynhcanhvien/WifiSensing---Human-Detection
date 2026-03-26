import os
import torch
from datetime import datetime

class Config:
    # ── PATHS (Đường dẫn Local) ──
    # Trỏ ra ngoài thư mục src, vào thư mục data
    DATA_PATH = 'data/wifi_sensing_dataset.npz'
    
    # Thư mục lưu log của TensorBoard và checkpoint mô hình
    LOG_DIR = "training/model/runs/experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = 'training/model/checkpoints'
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'best_light_resnet.pth')
    
    CHECKPOINT_INTERVAL = 10 
    
    # ── DATASET ──
    BATCH_SIZE = 64
    # Mac M1 chạy đa nhân rất tốt, có thể set num_workers = 2 hoặc 4
    NUM_WORKERS = 2     
    
    # ── MODEL & TRAINING ──
    NUM_CLASSES = 2
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4 
    
    # ── SYSTEM ──
    SEED = 42
    # Tự động nhận diện GPU (MPS cho Mac M1/M2/M3, CUDA cho Nvidia, hoặc CPU)
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

# Tự động tạo thư mục nếu chưa có
os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '../models/runs'), exist_ok=True)