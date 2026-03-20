import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config
from utils.logger import Logger
from utils.dataset import get_dataloaders
from model.resnet import LightResNet
from utils.engine import Engine

def main():
    logger = Logger(config.LOG_DIR, "train_resnet.log")
    logger.info(f"🚀 Khởi động Training Pipeline trên: {config.DEVICE.type.upper()}")
    
    if not os.path.exists(config.DATA_PATH):
        logger.info(f"❌ Không tìm thấy file {config.DATA_PATH}. Hãy chạy build_dataset.py trước!")
        return

    # Khởi tạo TensorBoard Writer
    writer = SummaryWriter(log_dir=config.TENSORBOARD_DIR)
    logger.info(f"📈 Đã khởi tạo TensorBoard tại: {config.TENSORBOARD_DIR}")

    # 1. Nạp Dữ liệu
    logger.info("📦 Đang tải dữ liệu...")
    train_loader, test_loader, train_size, test_size = get_dataloaders(config.DATA_PATH, config.BATCH_SIZE)
    logger.info(f"📊 Dataset: {train_size} Train samples | {test_size} Test samples")
    
    # 2. Khởi tạo Model
    model = LightResNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Đã load LightResNet ({total_params:,} parameters)")
    
    # 3. Mất mát (Loss) và Tối ưu (Optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 4. Vòng lặp Học (Training Loop)
    best_acc = 0.0
    engine = Engine()
    
    logger.info("🔥 Bắt đầu quá trình học...")
    start_time = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = engine.train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        test_loss, test_acc = engine.evaluate_epoch(model, test_loader, criterion, config.DEVICE)
        
        scheduler.step(test_loss)
        
        # Ghi biểu đồ vào TensorBoard
        writer.add_scalars('Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        logger.info(f"Epoch [{epoch:02d}/{config.EPOCHS}] | "
                    f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
                    f"Test Loss: {test_loss:.4f} - Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logger.info(f"   🌟 Đã lưu Model mới đạt đỉnh Accuracy: {best_acc*100:.2f}%")
            
    total_time = (time.time() - start_time) / 60
    logger.info(f"✅ Huấn luyện hoàn tất sau {total_time:.2f} phút!")
    logger.info(f"🏆 Best Test Accuracy đạt được: {best_acc*100:.2f}%")
    
    writer.close() # Đóng TensorBoard khi hoàn tất

if __name__ == "__main__":
    main()