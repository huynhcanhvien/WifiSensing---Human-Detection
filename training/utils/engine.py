import torch
from utils.logger import AverageMeter, calculate_accuracy

class Engine:
    @staticmethod
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train() 
        losses = AverageMeter()
        accs = AverageMeter()
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()      
            outputs = model(inputs)    
            loss = criterion(outputs, targets) 
            
            loss.backward()            
            optimizer.step()           
            
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))
            
        return losses.avg, accs.avg

    @staticmethod
    def evaluate_epoch(model, dataloader, criterion, device):
        model.eval() 
        losses = AverageMeter()
        accs = AverageMeter()
        
        with torch.no_grad(): 
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                acc = calculate_accuracy(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                accs.update(acc, inputs.size(0))
                
        return losses.avg, accs.avg