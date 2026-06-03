"""
scripts/build_dataset.py — Entry-point để xây dựng dataset .npz từ SQLite DB.

Cách dùng:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --db wifi_sensing.db --out data/wifi_sensing_dataset.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Paths
from src.data.build_dataset import build_db_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WiFi Sensing dataset from SQLite DB")
    parser.add_argument("--db",  default=str(Paths.DB_PATH),     help="Đường dẫn file .db")
    parser.add_argument("--out", default=str(Paths.DATASET_NPZ), help="Đường dẫn output .npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_db_dataset(db_path=args.db, output_npz=args.out)
