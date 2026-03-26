# PBL5 - WiFi Sensing for Breathing Detection

## Mô tả dự án
Dự án PBL5 là một hệ thống phát hiện hơi thở dựa trên tín hiệu WiFi (WiFi Sensing) sử dụng Channel State Information (CSI). Dự án bao gồm việc thu thập dữ liệu từ PCAP files, tiền xử lý dữ liệu realtime, và huấn luyện mô hình Machine Learning (PyTorch) để phân loại các hoạt động như thở, không thở, và giữ hơi.

## Cài đặt
1. Đảm bảo bạn có Python 3.8+ cài đặt.
2. Cài đặt các thư viện cần thiết:
   ```
   cd training
   pip install -r requirements.txt
   ```
3. Dự án hỗ trợ GPU MPS (Mac M1/M2/M3), CUDA, hoặc CPU.

## Cấu trúc dự án

### 1. Folder `code_realtime/`
Chứa các script xử lý dữ liệu từ PCAP và CSV files.

- `build_pcap.py`: Xây dựng dataset, script xây dựng và xử lý file PCAP. Sử dụng để chuyển đổi dữ liệu thô thành định dạng có thể sử dụng training.
- `check_file.py`: Kiểm tra tính toàn vẹn của file dữ liệu. Chạy trước khi xử lý để đảm bảo file không bị lỗi.
- `check_header.py`: Kiểm tra header của file PCAP. Sử dụng để xác minh cấu trúc packet.
- `check_mac.py`: Kiểm tra địa chỉ MAC trong dữ liệu. Hữu ích để lọc dữ liệu từ thiết bị cụ thể.
- `conver_pcap.py`: Chuyển đổi file PCAP sang định dạng khác (có thể là CSV). Sử dụng khi cần export dữ liệu.
- `debug_pcap.py`: Script debug cho file PCAP. Sử dụng để phân tích và sửa lỗi trong dữ liệu.
- `plot_csv.py`: Vẽ biểu đồ từ file CSV. 
- `plot.py`: Vẽ biểu đồ cho file pcap.
- `preprocess.py`: Tiền xử lý dữ liệu realtime. Bao gồm Hampel filter, interpolation. Chạy script này để làm sạch dữ liệu trước khi feed vào model.
- Cách sử dụng: Chạy từng script riêng lẻ, ví dụ `python preprocess.py` để tiền xử lý file CSV cụ thể (cấu hình trong script).

### 2. Folder `data/`
Chứa dữ liệu thô và đã xử lý.

- `receiver.sh`: Script shell để thu thập dữ liệu từ thiết bị. Chạy `./receiver.sh` để bắt đầu thu thập packet WiFi.
- `wifi_sensing_dataset_meta.json`: Metadata của dataset. Chứa thông tin cấu hình như WINDOW_SEC, TARGET_FS, và chi tiết từng file đã xử lý (số windows, shape tensor, MAC addresses).
- `wifi_sensing_dataset.npz`: Dataset đã xử lý dưới dạng NumPy compressed. Sử dụng trong training.
- Subfolder `Data/` chứa các file raw thu thập:
  - `breathing/`: File PCAP và CSV cho hoạt động thở. Ví dụ: `breath_20260203_215850.pcap` và `breath_20260203_215850_realtime.csv`.
  - `empty/`: File cho trạng thái không có hoạt động (empty room).
  - `empty_tmp/`: File tạm thời cho empty, có thể dùng để test.
- Cách sử dụng: Sử dụng `receiver.sh` kết hợp với `sender.sh` trên raspberry pi 4 để thu thập dữ liệu mới. Đọc metadata từ JSON để hiểu cấu trúc dataset.

### 3. Folder `models/`
Chứa checkpoints và logs của mô hình.

- `checkpoints/`: Lưu trữ các checkpoint mô hình đã huấn luyện.
- `runs/`: Logs từ TensorBoard.
- Cách sử dụng: Sau khi training, kiểm tra checkpoints để load model. Sử dụng TensorBoard để visualize training: `tensorboard --logdir models/runs`.

### 4. Folder `paper/`
Chứa tài liệu nghiên cứu hoặc báo cáo liên quan đến dự án. (Hiện tại trống, có thể thêm PDF hoặc docs.)

### 5. Folder `training/`
Chứa code huấn luyện và đánh giá mô hình.

- `check_data.py`: Kiểm tra dữ liệu trước khi training. Chạy `python check_data.py` để validate dataset.
- `config.py`: File cấu hình. Định nghĩa paths, batch size, epochs, device (MPS/CUDA/CPU). Sửa đổi các tham số ở đây trước khi training.
- `evaluate.py`: Đánh giá mô hình trên test set. Chạy `python evaluate.py` sau khi training để xem metrics và confusion matrix.
- `requirements.txt`: Danh sách thư viện cần thiết (torch, numpy, v.v.).
- `train.py`: Script chính để huấn luyện mô hình. Chạy `python train.py` để bắt đầu training với LightResNet.
- `__pycache__/`: Thư mục cache Python (tự động tạo).
- `log/`: Logs từ training.
- `model/`:
  - `model.py`: Định nghĩa mô hình LightResNet.
  - `checkpoints/`: Checkpoint từ training.
  - `runs/`: TensorBoard logs.
- `utils/`:
  - `dataset.py`: Xử lý dataset, tạo DataLoaders.
  - `engine.py`: Engine training/validation.
  - `logger.py`: Logging utilities.
  - `utils.py`: Các hàm tiện ích, như calculate_metrics, plot_confusion_matrix.
- Cách sử dụng:
  1. Cấu hình trong `config.py`.
  2. Chạy `python train.py` để huấn luyện.
  3. Chạy `python evaluate.py` để đánh giá.
  4. Sử dụng TensorBoard: `tensorboard --logdir training/model/runs`.

## Hướng dẫn sử dụng tổng thể
1. **Thu thập dữ liệu**: Chạy `data/receiver.sh` để thu thập PCAP.
2. **Xử lý realtime**: Sử dụng scripts trong `code_realtime/` để preprocess và visualize.
3. **Chuẩn bị dataset**: Đảm bảo `data/wifi_sensing_dataset.npz` có sẵn.
4. **Huấn luyện**: Chạy `training/train.py`.
5. **Đánh giá**: Chạy `training/evaluate.py`.

## Lưu ý
- Đảm bảo paths trong config đúng.
- Dataset labels: 0 (empty), 1 (breathing).
## Tác giả
Huỳnh Cảnh Viên - Phạm Nhật Khoa - Phạm Quang Thắng
