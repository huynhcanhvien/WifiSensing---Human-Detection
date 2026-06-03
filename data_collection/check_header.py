from scapy.all import *
import binascii
import sys

# Thay đường dẫn file của bạn vào đây
FILE_PATH = "dataset_breath/BREATHING/BREATHING_035.pcap"

def debug_packet_header(pcap_file):
    print(f"🔍 Đang soi file: {pcap_file}")
    
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    for pkt in packets:
        if UDP in pkt and pkt[UDP].dport == 5500:
            payload = bytes(pkt[UDP].payload)
            
            # In ra độ dài
            print(f"📦 Độ dài Payload: {len(payload)} bytes")
            
            # In ra 40 byte đầu tiên dưới dạng HEX để tìm MAC
            raw_hex = binascii.hexlify(payload[:40]).decode('utf-8')
            
            # Tách thành từng cặp cho dễ nhìn
            formatted_hex = ' '.join(raw_hex[i:i+2] for i in range(0, len(raw_hex), 2))
            
            print("\n👇 HEADER (40 bytes đầu tiên):")
            print(formatted_hex)
            print("-" * 50)
            print("Hãy copy dòng số HEX ở trên và gửi cho tôi!")
            return # Chỉ cần soi 1 gói là đủ

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_packet_header(sys.argv[1])
    else:
        debug_packet_header(FILE_PATH)