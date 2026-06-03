import sys
import os
from scapy.all import rdpcap, UDP
from collections import Counter

# ================= CẤU HÌNH =================
# Thay đường dẫn file pcap cần kiểm tra vào đây
FILE_PATH = "data/Data/breathing/breath_20260307_072231.pcap" 
# ============================================

def get_mac_from_payload(payload_bytes):
    """Trích xuất MAC từ payload (theo chuẩn firmware ESP32 CSI)"""
    try:
        # MAC nằm ở byte thứ 4 đến 10 trong payload
        mac_bytes = payload_bytes[4:10]
        mac_str = ':'.join('%02x' % b for b in mac_bytes)
        return mac_str
    except:
        return None

def analyze_pcap(file_path):
    print(f"📂 Đang đọc file: {file_path}...")
    
    if not os.path.exists(file_path):
        print("❌ Lỗi: Không tìm thấy file!")
        return

    try:
        packets = rdpcap(file_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file pcap: {e}")
        return

    total_packets = len(packets)
    csi_packets = 0
    
    # Bộ đếm cho từng MAC
    mac_counts = Counter()

    print(f"🔍 Đang phân tích {total_packets} gói tin...")

    for pkt in packets:
        # Chỉ kiểm tra gói tin UDP Port 5500 (Gói tin CSI)
        if UDP in pkt and pkt[UDP].dport == 5500:
            csi_packets += 1
            payload = bytes(pkt[UDP].payload)
            
            # Trích xuất MAC từ nội dung gói tin
            # (Chính xác hơn là xem header Wifi vì header có thể bị giả lập)
            src_mac = get_mac_from_payload(payload)
            
            if src_mac:
                mac_counts[src_mac] += 1

    print("\n" + "="*40)
    print("   KẾT QUẢ PHÂN TÍCH THIẾT BỊ")
    print("="*40)
    print(f"∑ Tổng số gói tin bắt được: {total_packets}")
    print(f"∑ Số gói tin CSI (Port 5500): {csi_packets}")
    print("-" * 40)
    print(f"{'MAC ADDRESS':<20} | {'SỐ GÓI':<10} | {'TỶ LỆ':<10}")
    print("-" * 40)

    for mac, count in mac_counts.most_common():
        percent = (count / csi_packets) * 100 if csi_packets > 0 else 0
        print(f"{mac:<20} | {count:<10} | {percent:.1f}%")
    
    print("="*40)

    # Đánh giá nhanh
    if len(mac_counts) == 2:
        print("✅ TỐT: Phát hiện đúng 2 thiết bị (Khớp với giả thuyết 2 ESP32).")
    elif len(mac_counts) > 2:
        print("⚠️ CẢNH BÁO: Có nhiều hơn 2 thiết bị đang phát vào port 5500!")
    elif len(mac_counts) == 1:
        print("ℹ️ CHÚ Ý: Chỉ thu được dữ liệu từ 1 thiết bị.")

if __name__ == "__main__":
    # Hỗ trợ chạy từ dòng lệnh: python check_mac.py ten_file.pcap
    target_file = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH
    analyze_pcap(target_file)