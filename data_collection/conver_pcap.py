import pandas as pd
import numpy as np
from scapy.all import rdpcap, UDP
import sys
import os
from datetime import datetime, timedelta

# ================= CẤU HÌNH =================
INPUT_PCAP = "data/Data/empty/empty_20260315_022732.pcap" # File pcap gốc

# Chọn định dạng thời gian bạn muốn:
# 'datetime' : Ngày tháng cụ thể (VD: 2026-01-29 23:06:27)
# 'unix'     : Số giây gốc (VD: 1769789187.123)
# 'relative' : Giây tính từ 0 (VD: 0.0, 0.1, 0.2...) -> Cách cũ
TIME_FORMAT = 'datetime' 

# Lọc MAC (Chỉ lấy 2 con TX của bạn)
TARGET_MACS = [
    "d4:e9:f4:a3:64:80", 
    "d4:e9:f4:a4:3c:18"
]
# ============================================

def get_mac(payload):
    try:
        return ':'.join('%02x' % b for b in payload[4:10])
    except: return None

def process_pcap(file_path):
    print(f"📂 Đang đọc: {file_path}...")
    try:
        packets = rdpcap(file_path)
    except:
        print("❌ Lỗi: Không tìm thấy file!")
        return

    data_rows = []
    print(f"⚙️ Đang xử lý {len(packets)} gói tin...")
    
    for pkt in packets:
        if UDP in pkt and pkt[UDP].dport == 5500:
            payload = bytes(pkt[UDP].payload)
            src_mac = get_mac(payload)
            
            if src_mac in TARGET_MACS and len(payload) >= 274:
                # Giải mã CSI
                raw_np = np.frombuffer(payload[-256:], dtype=np.int16)
                amp = np.sqrt(raw_np[0::2].astype(float)**2 + raw_np[1::2].astype(float)**2)
                
                if len(amp) == 64:
                    row = {
                        'timestamp': float(pkt.time),
                        'mac': src_mac,
                        'rssi': pkt.dBm_AntSignal if hasattr(pkt, 'dBm_AntSignal') else 0
                    }
                    for i in range(64): row[f'sub_{i}'] = amp[i]
                    data_rows.append(row)

    if not data_rows:
        print("⚠️ Không có dữ liệu hợp lệ.")
        return

    df = pd.DataFrame(data_rows)
    
    # --- XỬ LÝ THỜI GIAN THEO YÊU CẦU CỦA BẠN ---
    if TIME_FORMAT == 'datetime':
        # Chuyển đổi sang datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Chuyển sang múi giờ Việt Nam (UTC+7)
        # Cách thủ công an toàn nhất: Cộng thêm 7 giờ
        df['timestamp'] = df['timestamp'] + timedelta(hours=7)
        
        print("🕒 Đã chuyển đổi sang ngày giờ thực tế (Múi giờ VN).")
        
    elif TIME_FORMAT == 'relative':
        df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]
        print("⏱ Đã chuyển đổi sang thời gian tương đối (bắt đầu từ 0).")
        
    else:
        print("💾 Giữ nguyên Timestamp gốc (Unix Epoch).")

    # --- LƯU FILE ---
    base_name = os.path.splitext(file_path)[0]
    csv_name = base_name + "_realtime.csv"
    
    # Định dạng lại chuỗi thời gian khi lưu để không bị lỗi mili-giây
    if TIME_FORMAT == 'datetime':
        df.to_csv(csv_name, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
    else:
        df.to_csv(csv_name, index=False)
        
    print(f"✅ Xong! File mới: {csv_name}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else INPUT_PCAP
    process_pcap(target)