import os
import numpy as np
import matplotlib.pyplot as plt

# Import cấu hình từ file config.py cùng thư mục
from config import Config

def inspect_dataset():
    cfg = Config()
    print("="*50)
    print("🔍 KIỂM TRA VÀ ĐÁNH GIÁ DỮ LIỆU (.npz)")
    print("="*50)

    # 1. Load data
    print(f"📂 Đang đọc file: {cfg.DATA_PATH}")
    if not os.path.exists(cfg.DATA_PATH):
        print(f"❌ Lỗi: Không tìm thấy file tại {cfg.DATA_PATH}")
        print("Hãy chắc chắn bạn đã copy file wifi_sensing_dataset.npz vào thư mục data/")
        return

    data = np.load(cfg.DATA_PATH)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test  = data['X_test']
    Y_test  = data['Y_test']

    # 2. In kích thước (Shapes)
    print("\n📏 1. KÍCH THƯỚC DỮ LIỆU (SHAPES):")
    print(f"   X_train: {X_train.shape} | Y_train: {Y_train.shape}")
    print(f"   X_test : {X_test.shape}  | Y_test : {Y_test.shape}")
    
    # Kiểm tra xem có đúng định dạng (N, 150, 53, 2) cho 3s @ 50Hz không
    expected_time_steps = int(3.0 * 50.0) # 150
    if X_train.shape[1] == expected_time_steps:
        print(f"   ✅ Trục thời gian chuẩn xác: {expected_time_steps} samples (3 giây ở 50Hz).")
    else:
        print(f"   ⚠️ Cảnh báo: Trục thời gian là {X_train.shape[1]}, khác với dự kiến {expected_time_steps}.")

    # 3. Kiểm tra dữ liệu hỏng (NaN / Inf)
    print("\n⚠️ 2. KIỂM TRA DỮ LIỆU LỖI (NaN/Inf):")
    has_nan = np.isnan(X_train).any() or np.isnan(X_test).any()
    has_inf = np.isinf(X_train).any() or np.isinf(X_test).any()
    print(f"   Chứa giá trị NaN (Not a Number): {'Có ❌' if has_nan else 'Không ✅'}")
    print(f"   Chứa giá trị Inf (Vô cực):       {'Có ❌' if has_inf else 'Không ✅'}")

    # 4. Phân phối nhãn
    print("\n📊 3. PHÂN PHỐI NHÃN (CLASS BALANCE):")
    train_labels, train_counts = np.unique(Y_train, return_counts=True)
    test_labels, test_counts = np.unique(Y_test, return_counts=True)

    print("   Tập Train:")
    for label, count in zip(train_labels, train_counts):
        name = "Empty" if label == 0 else "Presence"
        print(f"      - {name:<10} (Label {label}): {count:,} mẫu ({(count/len(Y_train))*100:.1f}%)")

    print("   Tập Test:")
    for label, count in zip(test_labels, test_counts):
        name = "Empty" if label == 0 else "Presence"
        print(f"      - {name:<10} (Label {label}): {count:,} mẫu ({(count/len(Y_test))*100:.1f}%)")

    # 5. Trực quan hóa 1 mẫu ngẫu nhiên
    print("\n🖼️ 4. TRỰC QUAN HÓA (HEATMAP):")
    presence_idx = np.where(Y_train == 1)[0]
    
    if len(presence_idx) > 0:
        idx = np.random.choice(presence_idx)
        sample = X_train[idx] # Shape: (150, 53, 2)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Mẫu Presence ngẫu nhiên (Index: {idx} trong tập Train)", fontsize=14)
        
        # Trực quan hóa MAC 1
        im1 = axes[0].imshow(sample[:, :, 0].T, aspect='auto', cmap='jet', origin='lower')
        axes[0].set_title("MAC 1 Heatmap (D4:E9:F4:A3:64:80)")
        axes[0].set_xlabel("Time Steps (150 samples = 3s)")
        axes[0].set_ylabel("Subcarriers (53)")
        fig.colorbar(im1, ax=axes[0])
        
        # Trực quan hóa MAC 2
        im2 = axes[1].imshow(sample[:, :, 1].T, aspect='auto', cmap='jet', origin='lower')
        axes[1].set_title("MAC 2 Heatmap (D4:E9:F4:A4:3C:18)")
        axes[1].set_xlabel("Time Steps (150 samples = 3s)")
        axes[1].set_ylabel("Subcarriers (53)")
        fig.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        print("   Đang hiển thị biểu đồ Heatmap... Đóng cửa sổ biểu đồ để kết thúc script.")
        plt.show()

    print("="*50 + "\n")

if __name__ == "__main__":
    inspect_dataset()