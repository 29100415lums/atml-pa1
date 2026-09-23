"""
task2/methods/dann.py
Method 3: DANN - Domain-Adversarial Neural Networks.
Binary domain discriminator attached to 512-d features via GRL.
Alpha schedule: alpha(p) = 2/(1+exp(-10*p)) - 1, where p = epoch/max_epochs.
"""
import os
import sys
import torch
import torch.nn as nn

TASK2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK2_ROOT)
sys.path.insert(0, os.path.join(TASK2_ROOT, '..'))

from models.backbone import PACSBackbone
from models.domain_discriminator import DomainDiscriminator, grl_schedule


def train_dann(train_loaders, target_loader, val_loaders, device,
               max_epochs=30, patience=5, lr=1e-4, weight_decay=1e-4,
               save_path=None):
    """
    DANN training loop.

    Args:
        train_loaders: dict domain -> InfiniteDataLoader (source domains).
        target_loader: InfiniteDataLoader for Sketch (target, labels ignored).
        val_loaders: dict domain -> DataLoader for source validation.
        device: torch device.
        max_epochs: max training epochs.
        patience: early stopping patience.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        save_path: path to save best checkpoint.

    Returns:
        (model, domain_disc, history)
    """
    model = PACSBackbone(num_classes=7).to(device)

    # Initialize from source-only checkpoint if available (stabilizes cls from epoch 1)
    src_ckpt = save_path.replace('dann.pt', 'source_only.pt') if save_path else None
    if src_ckpt and os.path.exists(src_ckpt):
        model.load_state_dict(torch.load(src_ckpt, map_location=device))
        print("  [DANN] Initialized from source_only checkpoint.")

    disc = DomainDiscriminator(in_dim=512).to(device)

    # Separate optimizers: backbone trains slowly, discriminator trains faster
    backbone_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=lr * 10, weight_decay=weight_decay)
    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()

    history = {
        'train_cls_loss': [], 'train_dom_loss': [], 'train_total_loss': [],
        'val_macro_f1': {d: [] for d in val_loaders}, 'mean_val_f1': []
    }

    best_mean_f1 = -1.0
    patience_counter = 0
    best_model_state = None
    best_disc_state = None
    source_domains = list(train_loaders.keys())
    steps_per_epoch = 50

    for epoch in range(max_epochs):
        model.train()
        model.freeze_bn_running_stats()
        disc.train()

        # Set GRL alpha based on schedule
        alpha = grl_schedule(epoch, max_epochs)
        disc.set_alpha(alpha)

        epoch_cls, epoch_dom, epoch_total = 0.0, 0.0, 0.0

        for step in range(steps_per_epoch):
            src_imgs, src_labels = [], []
            for domain in source_domains:
                imgs, labels, _ = next(train_loaders[domain])
                src_imgs.append(imgs)
                src_labels.append(labels)

            tgt_imgs, _, _ = next(target_loader)

            src_imgs = torch.cat(src_imgs, dim=0).to(device)
            src_labels = torch.cat(src_labels, dim=0).to(device)
            tgt_imgs = tgt_imgs.to(device)

            n_src = src_imgs.shape[0]
            n_tgt = tgt_imgs.shape[0]

            # Domain labels: source=0, target=1
            src_dom_labels = torch.zeros(n_src, dtype=torch.long, device=device)
            tgt_dom_labels = torch.ones(n_tgt, dtype=torch.long, device=device)

            backbone_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            # Source forward
            src_logits, src_feats = model(src_imgs)
            cls_loss = cls_criterion(src_logits, src_labels)

            # Target forward (features only)
            _, tgt_feats = model(tgt_imgs)

            # Discriminator forward (through GRL)
            src_dom_logits = disc(src_feats)
            tgt_dom_logits = disc(tgt_feats)
            dom_loss = (dom_criterion(src_dom_logits, src_dom_labels) +
                        dom_criterion(tgt_dom_logits, tgt_dom_labels)) / 2.0

            total_loss = cls_loss + 0.1 * dom_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            backbone_optimizer.step()
            disc_optimizer.step()
            model.freeze_bn_running_stats()

            epoch_cls += cls_loss.item()
            epoch_dom += dom_loss.item()
            epoch_total += total_loss.item()

        history['train_cls_loss'].append(epoch_cls / steps_per_epoch)
        history['train_dom_loss'].append(epoch_dom / steps_per_epoch)
        history['train_total_loss'].append(epoch_total / steps_per_epoch)

        mean_f1 = _evaluate_all_domains(model, val_loaders, history, device)
        history['mean_val_f1'].append(mean_f1)

        print(f"  [DANN] Epoch {epoch+1}/{max_epochs} | Cls: {epoch_cls/steps_per_epoch:.4f} | "
              f"Dom: {epoch_dom/steps_per_epoch:.4f} | Alpha: {alpha:.3f} | Mean F1: {mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_disc_state = {k: v.cpu().clone() for k, v in disc.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [DANN] Early stopping at epoch {epoch+1}.")
                break

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        disc.load_state_dict({k: v.to(device) for k, v in best_disc_state.items()})
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'model': best_model_state, 'disc': best_disc_state}, save_path)
            print(f"  [DANN] Best model saved to {save_path} (F1={best_mean_f1:.4f})")

    return model, disc, history


def _evaluate_all_domains(model, val_loaders, history, device):
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
