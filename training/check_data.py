import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Đường dẫn tới file data đã lưu
DATA_PATH = "data/wifi_sensing_dataset.npz"

def plot_random_samples():
    print("Đang load dữ liệu...")
    data = np.load(DATA_PATH)
    X_train = data['X_train']
    Y_train = data['Y_train']
    
    # Tìm index của nhãn 0 (empty) và nhãn 1 (breathing)
    empty_idx = np.where(Y_train == 0)[0]
    breath_idx = np.where(Y_train == 1)[0]
    
    if len(empty_idx) == 0 or len(breath_idx) == 0:
        print("Thiếu dữ liệu của một trong hai nhãn để plot!")
        return

    # Chọn ngẫu nhiên 1 mẫu empty và 1 mẫu breathing
    idx_e = random.choice(empty_idx)
    idx_b = random.choice(breath_idx)
    
    samples = [
        (idx_e, "EMPTY (Label 0)", X_train[idx_e]),
        (idx_b, "BREATHING (Label 1)", X_train[idx_b])
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle("Kiểm tra chất lượng CSI Data: Empty vs Breathing", fontsize=16)

    for col, (idx, title, tensor) in enumerate(samples):
        # tensor shape: (200, 53, 2) -> (time, subcarriers, macs)
        mac1_data = tensor[:, :, 0]
        mac2_data = tensor[:, :, 1]
        
        # Hàng 1: Heatmap MAC 1 (Transposed để trục X là thời gian)
        im1 = axes[0, col].imshow(mac1_data.T, aspect='auto', cmap='jet', origin='lower')
        axes[0, col].set_title(f"{title} - MAC 1 Heatmap")
        axes[0, col].set_ylabel("Subcarrier Index")
        fig.colorbar(im1, ax=axes[0, col])

        # Hàng 2: Heatmap MAC 2
        im2 = axes[1, col].imshow(mac2_data.T, aspect='auto', cmap='jet', origin='lower')
        axes[1, col].set_title(f"{title} - MAC 2 Heatmap")
        axes[1, col].set_ylabel("Subcarrier Index")
        fig.colorbar(im2, ax=axes[1, col])

        # Hàng 3: Biểu đồ dạng sóng (Lấy trung bình các subcarriers)
        mean_mac1 = np.mean(mac1_data, axis=1)
        mean_mac2 = np.mean(mac2_data, axis=1)
        time_axis = np.linspace(0, 10, 200) # 200 samples = 10s (do 20Hz)
        
        axes[2, col].plot(time_axis, mean_mac1, label="MAC 1", alpha=0.8)
        axes[2, col].plot(time_axis, mean_mac2, label="MAC 2", alpha=0.8)
        axes[2, col].set_title(f"{title} - Waveform (Mean over Subcarriers)")
        axes[2, col].set_xlabel("Time (seconds)")
        axes[2, col].set_ylabel("Normalized Amplitude")
        axes[2, col].legend()
        axes[2, col].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_random_samples()