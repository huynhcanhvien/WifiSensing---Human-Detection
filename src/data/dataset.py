"""
src/data/dataset.py — CSI Dataset và DataLoader cho PyTorch.

CSI_Dataset nhận dữ liệu dạng numpy và thực hiện transpose
để phù hợp với Conv2d (NCHW format).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TrainingConfig


class CSIDataset(Dataset):
    """
    Dataset cho dữ liệu CSI WiFi Sensing.

    Đầu vào X có shape: (N, Time, Subcarrier, MACs)
    Được chuyển về:     (N, MACs, Time, Subcarrier)  ← NCHW cho Conv2d

    Args:
        X: numpy array shape (N, T, 53, 2)
        Y: numpy array shape (N,) — nhãn integer
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # (N, T, SC, MACs) → (N, MACs, T, SC)
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        self.Y = torch.LongTensor(Y)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def get_dataloaders(
    data_path: str,
    batch_size: int = TrainingConfig.BATCH_SIZE,
    num_workers: int = TrainingConfig.NUM_WORKERS,
    val_size: float = TrainingConfig.VAL_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load dataset từ file .npz và trả về 3 DataLoader (train/val/test).

    Validation set được tách từ 10% cuối tập train (không random shuffle
    để tránh data leakage từ sliding window).

    Args:
        data_path: đường dẫn đến file .npz
        batch_size: kích thước batch
        num_workers: số worker cho DataLoader
        val_size: tỷ lệ tập validation từ train set

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print(f"⏳ Đang load dữ liệu từ: {data_path}")
    try:
        data = np.load(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ Không tìm thấy file dataset tại: {data_path}\n"
            "   → Chạy `python scripts/build_dataset.py` để tạo dataset."
        )

    X_train_full = data['X_train']
    Y_train_full = data['Y_train']
    X_test       = data['X_test']
    Y_test       = data['Y_test']

    # Tách validation từ cuối tập train
    n_val    = max(1, int(len(X_train_full) * val_size))
    X_train  = X_train_full[:-n_val]
    Y_train  = Y_train_full[:-n_val]
    X_val    = X_train_full[-n_val:]
    Y_val    = Y_train_full[-n_val:]

    train_ds = CSIDataset(X_train, Y_train)
    val_ds   = CSIDataset(X_val,   Y_val)
    test_ds  = CSIDataset(X_test,  Y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(
        f"✅ Dataset sẵn sàng — "
        f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader
