import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CSIDataset(Dataset):
    def __init__(self, X, Y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.X = X_tensor.permute(0, 3, 1, 2) # Xoay trục MAC lên làm Channel
        self.Y = torch.tensor(Y, dtype=torch.long)
        
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataloaders(data_path, batch_size):
    data = np.load(data_path)
    
    train_dataset = CSIDataset(data['X_train'], data['Y_train'])
    test_dataset = CSIDataset(data['X_test'], data['Y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(train_dataset), len(test_dataset)