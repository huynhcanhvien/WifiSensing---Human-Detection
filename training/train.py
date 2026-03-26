import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np

# Nhập các module cục bộ
from config import Config
from utils.dataset import get_dataloaders
from model import LightResNet
from utils import calculate_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    cfg = Config()
    set_seed(cfg.SEED)
    
    print("="*50)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN (Thiết bị: {cfg.DEVICE})")
    print("="*50)

    train_loader, val_loader, _ = get_dataloaders(cfg)

    model = LightResNet(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    best_val_loss = float('inf')

    for epoch in range(1, cfg.EPOCHS + 1):
        # ── TRAIN PHASE ──
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} [Train]", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss = train_loss / len(train_loader.dataset)
        train_acc, train_prec, train_rec, train_f1 = calculate_metrics(train_targets, train_preds)

        # ── VAL PHASE ──
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calculate_metrics(val_targets, val_preds)
        
        scheduler.step(val_loss)

        # ── LOGGING TENSORBOARD ──
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # ── SAVE BEST MODEL ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.SAVE_MODEL_PATH)
            print(f"   🌟 Đã lưu Best Model tại Val Loss: {val_loss:.4f}")

        # ── SAVE CHECKPOINT ĐỊNH KỲ ──
        if epoch % cfg.CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(cfg.SAVE_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, ckpt_path)

    writer.close()
    print("✅ HOÀN TẤT HUẤN LUYỆN!")

if __name__ == "__main__":
    train()