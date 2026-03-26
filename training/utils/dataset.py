import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CSI_Dataset(Dataset):
    def __init__(self, X, Y):
        # Đầu vào X đang có shape: (N, 150, 53, 2)
        # PyTorch Conv2d yêu cầu shape: (N, Channels, H, W)
        # Nên ta chuyển (N, Time, Subcarrier, MACs) -> (N, MACs, Time, Subcarrier)
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        self.Y = torch.LongTensor(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataloaders(cfg):
    print(f"⏳ Đang load dữ liệu từ file .npz...")
    try:
        data = np.load(cfg.DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại: {cfg.DATA_PATH}. Vui lòng kiểm tra lại!")

    X_train_full = data['X_train']
    Y_train_full = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Cắt 10% tập Train làm tập Validation
    val_size = int(len(X_train_full) * 0.1)
    X_train = X_train_full[:-val_size]
    Y_train = Y_train_full[:-val_size]
    X_val   = X_train_full[-val_size:]
    Y_val   = Y_train_full[-val_size:]

    train_dataset = CSI_Dataset(X_train, Y_train)
    val_dataset   = CSI_Dataset(X_val, Y_val)
    test_dataset  = CSI_Dataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    print(f"✅ Đã chia Dataloaders: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")
    return train_loader, val_loader, test_loader