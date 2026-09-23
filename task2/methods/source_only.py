"""
task2/methods/source_only.py
Method 1: Source-Only ERM Baseline.
Trains ResNet-18 on domain-balanced source batches (8 Photo + 8 Art + 8 Cartoon = 24).
No adaptation. Establishes the raw domain gap baseline.
Checkpoint is saved and reused unchanged as the Task 3 ERM baseline.
"""
import os
import sys
import torch
import torch.nn as nn

TASK2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK2_ROOT)
sys.path.insert(0, os.path.join(TASK2_ROOT, '..'))

from models.backbone import PACSBackbone


def train_source_only(train_loaders, val_loaders, device, max_epochs=30, patience=5,
                      lr=1e-4, weight_decay=1e-4, save_path=None):
    """
    Source-Only ERM training loop.

    Args:
        train_loaders: dict mapping domain name -> InfiniteDataLoader for training.
        val_loaders: dict mapping domain name -> DataLoader for validation.
        device: torch device.
        max_epochs: maximum training epochs.
        patience: early stopping patience on mean source val macro-F1.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        save_path: path to save best model checkpoint.

    Returns:
        (model, history) where history is a dict of loss/metric curves.
    """
    model = PACSBackbone(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'train_cls_loss': [], 'val_macro_f1': {d: [] for d in val_loaders}, 'mean_val_f1': []}

    best_mean_f1 = -1.0
    patience_counter = 0
    best_state = None
    source_domains = list(train_loaders.keys())

    for epoch in range(max_epochs):
        model.train()
        model.freeze_bn_running_stats()  # Critical: freeze BN running stats

        epoch_loss = 0.0
        # Each step: draw one batch from each source domain
        steps_per_epoch = 50  # Fixed steps per epoch
        for step in range(steps_per_epoch):
            all_imgs, all_labels = [], []
            for domain in source_domains:
                imgs, labels, _ = next(train_loaders[domain])
                all_imgs.append(imgs)
                all_labels.append(labels)

            imgs = torch.cat(all_imgs, dim=0).to(device)
            labels = torch.cat(all_labels, dim=0).to(device)

            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Re-enforce BN policy after optimizer step
            model.freeze_bn_running_stats()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        history['train_cls_loss'].append(avg_loss)

        # Validation: compute macro-F1 per source domain
        mean_f1 = _evaluate_all_domains(model, val_loaders, history, device)
        history['mean_val_f1'].append(mean_f1)

        print(f"  [Source-Only] Epoch {epoch+1}/{max_epochs} | Loss: {avg_loss:.4f} | Mean Source F1: {mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Source-Only] Early stopping at epoch {epoch+1} (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)
            print(f"  [Source-Only] Best model saved to {save_path} (F1={best_mean_f1:.4f})")

    return model, history


def _evaluate_all_domains(model, val_loaders, history, device):
    """Compute per-domain macro-F1 and return mean across domains."""
    from evaluation.metrics import compute_macro_f1
    model.eval()
    domain_f1s = []
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            all_preds, all_labels = [], []
            for imgs, labels, _ in loader:
                imgs = imgs.to(device)
                logits, _ = model(imgs)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())
            f1 = compute_macro_f1(all_labels, all_preds)
            history['val_macro_f1'][domain].append(f1)
            domain_f1s.append(f1)
    model.train()
    model.freeze_bn_running_stats()
    return sum(domain_f1s) / len(domain_f1s)
