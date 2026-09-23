import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def evaluate_model_on_loader(model, loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    macro_f1 = compute_macro_f1(all_labels, all_preds)
    
    # Optional: per-class accuracy
    per_class_acc = {}
    for c in range(7):
        idx = np.array(all_labels) == c
        if idx.sum() > 0:
            per_class_acc[c] = (np.array(all_preds)[idx] == c).mean()
        else:
            per_class_acc[c] = 0.0
            
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }
