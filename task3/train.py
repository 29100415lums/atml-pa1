import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import torchvision.transforms as T
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK3_ROOT = CURRENT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(TASK3_ROOT, '..'))
SHARED_ROOT = os.path.join(PROJECT_ROOT, 'shared')

sys.path.insert(0, TASK3_ROOT)
sys.path.insert(0, SHARED_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from shared.pacs import PACSDataset, InfiniteDataLoader, PACS_DOMAINS
from shared.pacs_protocol import generate_pacs_splits

from models.backbone import PACSBackbone
from models.classifier_head import PACSClassifierHead
from methods.dan_dg import mmd_loss
from methods.sam import SAM
from evaluation.domain_metrics import compute_macro_f1, evaluate_model_on_loader

SOURCE_DOMAINS = ['art_painting', 'cartoon', 'photo']
TARGET_DOMAIN = 'sketch'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms():
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def build_source_loaders(splits, train_tf, val_tf, source_batch=8):
    train_loaders, val_loaders = {}, {}
    for domain in SOURCE_DOMAINS:
        domain_splits = splits['source_splits'][domain]
        train_ds = PACSDataset(
            list(zip(domain_splits['train']['paths'],
                     domain_splits['train']['labels'],
                     [PACS_DOMAINS.index(domain)] * len(domain_splits['train']['paths']))),
            transform=train_tf
        )
        val_ds = PACSDataset(
            list(zip(domain_splits['val']['paths'],
                     domain_splits['val']['labels'],
                     [PACS_DOMAINS.index(domain)] * len(domain_splits['val']['paths']))),
            transform=val_tf
        )
        train_loaders[domain] = InfiniteDataLoader(train_ds, batch_size=source_batch, shuffle=True, num_workers=2)
        val_loaders[domain] = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    return train_loaders, val_loaders

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PACSBackbone()
        self.head = PACSClassifierHead()
    def forward(self, x):
        f = self.backbone(x)
        logits = self.head(f)
        return logits, f
    def freeze_bn_running_stats(self):
        self.backbone.freeze_bn_running_stats()

def train(config, args):
    set_seed(config.get('seed', 6304))
    
    splits = generate_pacs_splits(args.pacs_root, seed=config.get('seed', 6304))
    train_tf, val_tf = get_transforms()
    train_loaders, val_loaders = build_source_loaders(splits, train_tf, val_tf, source_batch=config.get('source_batch_size', 8))
    
    model = FullModel().to(DEVICE)
    method = config['method']
    
    ckpt_dir = os.path.join(TASK3_ROOT, 'results', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{method}.pt")
    
    if method == 'erm':
        print("[ERM] Baseline reuse. Skipping training.")
        return
        
    lr = 1e-4
    wd = 1e-4
    max_epochs = config.get('max_epochs', 30)
    patience = config.get('patience', 5)
    
    if method == 'sam':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=config.get('rho', 0.05), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
    criterion = nn.CrossEntropyLoss()
    
    best_mean_f1 = -1.0
    patience_counter = 0
    best_state = None
    steps_per_epoch = 50
    
    history = {'train_loss': [], 'val_f1': [], 'cls_loss': [], 'mmd_loss': []}
    
    print(f"--- Training {method.upper()} ---")
    
    for epoch in range(max_epochs):
        model.train()
        model.freeze_bn_running_stats()
        
        epoch_cls, epoch_mmd, epoch_total = 0.0, 0.0, 0.0
        
        for step in range(steps_per_epoch):
            # Fetch 8 examples per source
            src_imgs, src_labels, src_feats_list = [], [], []
            for domain in SOURCE_DOMAINS:
                imgs, labels, _ = next(train_loaders[domain])
                src_imgs.append(imgs.to(DEVICE))
                src_labels.append(labels.to(DEVICE))
            
            # Combine all for classification loss
            all_imgs = torch.cat(src_imgs, dim=0)
            all_labels = torch.cat(src_labels, dim=0)
            
            if method == 'sam':
                # First pass
                logits, _ = model(all_imgs)
                loss = criterion(logits, all_labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                model.freeze_bn_running_stats()
                
                # Second pass
                logits, _ = model(all_imgs)
                loss_second = criterion(logits, all_labels)
                loss_second.backward()
                optimizer.second_step(zero_grad=True)
                model.freeze_bn_running_stats()
                
                epoch_cls += loss.item()
                epoch_total += loss.item()
                
            elif method == 'dan_dg':
                optimizer.zero_grad()
                logits_list, feats_list = [], []
                for imgs in src_imgs:
                    l, f = model(imgs)
                    logits_list.append(l)
                    feats_list.append(f)
                
                all_logits = torch.cat(logits_list, dim=0)
                cls_loss = criterion(all_logits, all_labels)
                
                # Pairwise MMD
                mmd = 0.0
                pairs = [(0,1), (0,2), (1,2)]
                for i, j in pairs:
                    mmd += mmd_loss(feats_list[i], feats_list[j])
                mmd = mmd / 3.0
                
                lambda_dg = config.get('lambda_dg', 1.0)
                total_loss = cls_loss + lambda_dg * mmd
                
                total_loss.backward()
                optimizer.step()
                model.freeze_bn_running_stats()
                
                epoch_cls += cls_loss.item()
                epoch_mmd += mmd.item()
                epoch_total += total_loss.item()
        
        # Evaluate on source domains
        mean_f1 = _evaluate_source_domains(model, val_loaders)
        
        avg_total = epoch_total / steps_per_epoch
        avg_cls = epoch_cls / steps_per_epoch
        avg_mmd = epoch_mmd / steps_per_epoch
        
        history['train_loss'].append(avg_total)
        history['cls_loss'].append(avg_cls)
        history['mmd_loss'].append(avg_mmd)
        history['val_f1'].append(mean_f1)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Total Loss: {avg_total:.4f} | Mean Source F1: {mean_f1:.4f}")
        
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break
                
    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"Saved best model to {save_path} (F1={best_mean_f1:.4f})")
        
    hist_path = os.path.join(TASK3_ROOT, 'results', f"{method}_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f)
    print(f"Saved training history to {hist_path}")

def _evaluate_source_domains(model, val_loaders):
    model.eval()
    domain_f1s = []
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            all_preds, all_labels = [], []
            for imgs, labels, _ in loader:
                imgs = imgs.to(DEVICE)
                logits, _ = model(imgs)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
            f1 = compute_macro_f1(all_labels, all_preds)
            domain_f1s.append(f1)
    model.train()
    model.freeze_bn_running_stats()
    return sum(domain_f1s) / len(domain_f1s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3 DG Training')
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config file')
    parser.add_argument('--pacs_root', type=str, default='./shared/data/PACS', help='Path to PACS data directory.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train(config, args)
