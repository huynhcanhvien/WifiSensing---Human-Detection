import os
import logging
import torch

class Logger:
    def __init__(self, log_dir, log_file="training.log"):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )
        self.logger = logging.getLogger()
        
    def info(self, msg):
        self.logger.info(msg)

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == targets).item()
    return correct / targets.size(0)