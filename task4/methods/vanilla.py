import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train_vanilla(model, train_loader, val_loader, device, save_path, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = -1.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> Saving best model (Val Acc: {best_val_acc:.2f}%)")
            torch.save(model.state_dict(), save_path)
