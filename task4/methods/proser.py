import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

def proser_classifier_placeholder_loss(logits, targets, num_known=10, num_dummy=5):
    """
    L_CP: Exclude the correct class, and encourage the remaining probability 
    mass to be uniformly distributed among the dummy classifiers.
    """
    N = logits.size(0)
    
    # Mask out the correct class
    logits_masked = logits.clone()
    logits_masked[torch.arange(N), targets] = -1e9
    
    # Compute log probabilities over the remaining classes
    log_probs = F.log_softmax(logits_masked, dim=1)
    
    # We want to maximize the probability of the dummy classes
    # Which means minimizing the negative log probability
    dummy_log_probs = log_probs[:, num_known:num_known+num_dummy]
    
    # Mean over the K dummy classes, and mean over the batch
    loss_cp = -dummy_log_probs.mean()
    return loss_cp

def proser_data_placeholder_loss(logits, num_known=10, num_dummy=5):
    """
    L_DP: For proxy unknowns (mixup features), encourage them to be predicted 
    as one of the dummy classes (uniform over dummy classes).
    """
    log_probs = F.log_softmax(logits, dim=1)
    dummy_log_probs = log_probs[:, num_known:num_known+num_dummy]
    loss_dp = -dummy_log_probs.mean()
    return loss_dp

def train_proser(model, train_loader, val_loader, device, save_path, epochs=50):
    # Standard classification loss for known classes
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer settings for PROSER fine-tuning
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    num_known = 10
    num_dummy = 5
    beta = 1.0 # weight for L_CP
    gamma = 0.1 # weight for L_DP
    
    best_val_acc = -1.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [PROSER]"):
            inputs, targets = inputs.to(device), targets.to(device)
            bsz = inputs.size(0)
            
            # Split mini-batch into two equal parts
            half = bsz // 2
            
            # Part 1: Classifier placeholder training
            inputs_cp = inputs[:half]
            targets_cp = targets[:half]
            
            optimizer.zero_grad()
            
            # Forward pass without mixup
            logits_cp, _ = model(inputs_cp, mixup=False)
            
            # Standard CE on known classes (ignore dummy classes for CE)
            loss_ce = criterion(logits_cp[:, :num_known], targets_cp)
            
            # Classifier placeholder loss
            loss_cp_val = proser_classifier_placeholder_loss(logits_cp, targets_cp, num_known, num_dummy)
            
            # Part 2: Data placeholder training (Manifold Mixup)
            inputs_dp = inputs[half:]
            targets_dp = targets[half:]
            
            if inputs_dp.size(0) > 1:
                # Need to mix examples from different classes
                # Simplest way: shift the batch by 1 and check if targets differ.
                # Since we want them to differ, we can just randomly permute until they differ,
                # or just use shifted and accept that most will differ (in CIFAR-10, 90% differ).
                indices = torch.randperm(inputs_dp.size(0)).to(device)
                
                # To be strict about different classes:
                for i in range(inputs_dp.size(0)):
                    while targets_dp[i] == targets_dp[indices[i]]:
                        indices[i] = torch.randint(0, inputs_dp.size(0), (1,)).item()
                        
                mixup_lambda = np.random.beta(2.0, 2.0)
                
                logits_dp, _ = model(inputs_dp, mixup=True, mixup_lambda=mixup_lambda, indices=indices)
                
                loss_dp_val = proser_data_placeholder_loss(logits_dp, num_known, num_dummy)
            else:
                loss_dp_val = 0.0
                
            loss = loss_ce + beta * loss_cp_val + gamma * loss_dp_val
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits_cp[:, :num_known].max(1)
            total += targets_cp.size(0)
            correct += predicted.eq(targets_cp).sum().item()
            
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
                logits, _ = model(inputs, mixup=False)
                loss = criterion(logits[:, :num_known], targets)
                val_loss += loss.item()
                _, predicted = logits[:, :num_known].max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> Saving best PROSER model (Val Acc: {best_val_acc:.2f}%)")
            torch.save(model.state_dict(), save_path)
