"""
task2/evaluation/metrics.py
Accuracy, Macro-F1, and per-class accuracy computation.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']


def compute_accuracy(y_true, y_pred):
    """Top-1 accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true, y_pred):
    """Macro-averaged F1 score across all 7 classes."""
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


def compute_per_class_accuracy(y_true, y_pred, num_classes=7):
    """Per-class accuracy as a dict {class_name: accuracy}."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    per_class = {}
    for c in range(num_classes):
        mask = (y_true == c)
        if mask.sum() == 0:
            per_class[PACS_CLASSES[c]] = 0.0
        else:
            per_class[PACS_CLASSES[c]] = float((y_pred[mask] == c).mean())
    return per_class


def compute_confusion_matrix(y_true, y_pred, num_classes=7):
    """Normalized confusion matrix (rows = true, cols = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    return cm_norm


def evaluate_model_on_loader(model, loader, device):
    """
    Evaluates a model on a DataLoader.

    Returns:
        dict with 'accuracy', 'macro_f1', 'per_class_accuracy', 'predictions', 'labels'.
    """
    import torch
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return {
        'accuracy': compute_accuracy(all_labels, all_preds),
        'macro_f1': compute_macro_f1(all_labels, all_preds),
        'per_class_accuracy': compute_per_class_accuracy(all_labels, all_preds),
        'predictions': all_preds,
        'labels': all_labels
    }
