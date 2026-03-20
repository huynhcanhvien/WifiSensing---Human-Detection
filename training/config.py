import os
import torch

# ================= ĐƯỜNG DẪN (PATHS) =================
# Tự động lấy đường dẫn của thư mục 'training' hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Trỏ ra ngoài thư mục PBL5 để lấy dataset
DATA_PATH = os.path.join(BASE_DIR, "..", "wifi_sensing_dataset.npz")

# Thư mục lưu log và trọng số mô hình
LOG_DIR = os.path.join(BASE_DIR, "log")
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard_runs")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_resnet_mimo.pth")

# ================= HYPERPARAMETERS =================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
NUM_CLASSES = 2

# Tự động nhận diện phần cứng (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")