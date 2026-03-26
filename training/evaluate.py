import os
import torch
from sklearn.metrics import classification_report

from config import Config
from dataset import get_dataloaders
from model import LightResNet
from utils import plot_confusion_matrix

def evaluate():
    cfg = Config()
    
    print("\n" + "="*50)
    print("📊 ĐÁNH GIÁ CHI TIẾT TRÊN TẬP TEST")
    print("="*50)

    _, _, test_loader = get_dataloaders(cfg)

    model = LightResNet(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    
    if not os.path.exists(cfg.SAVE_MODEL_PATH):
        print(f"❌ Không tìm thấy model tại {cfg.SAVE_MODEL_PATH}. Vui lòng chạy train.py trước!")
        return
        
    model.load_state_dict(torch.load(cfg.SAVE_MODEL_PATH, map_location=cfg.DEVICE))
    model.eval()

    test_preds, test_targets = [], []
    print("⏳ Đang chạy suy luận (inference) trên tập Test...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(cfg.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())

    class_names = ["Empty (0)", "Breathing/Presence (1)"]
    print("\n📋 BẢNG BÁO CÁO PHÂN LOẠI:")
    print(classification_report(test_targets, test_preds, target_names=class_names, digits=4))

    # Vẽ và lưu Confusion Matrix
    cm_path = os.path.join(os.path.dirname(__file__), '../models/confusion_matrix.png')
    plot_confusion_matrix(test_targets, test_preds, classes=class_names, save_path=cm_path)

if __name__ == "__main__":
    evaluate()