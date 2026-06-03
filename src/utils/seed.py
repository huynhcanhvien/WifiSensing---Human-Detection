"""
src/utils/seed.py — Reproducibility helpers.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho tất cả random number generators để đảm bảo reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Tắt non-deterministic operations trên CUDA nếu muốn hoàn toàn deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
