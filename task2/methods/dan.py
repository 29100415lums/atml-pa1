"""
task2/methods/dan.py
Method 2: DAN - Deep Adaptation Networks (MMD Alignment).
Loss: L_cls(Xs, Ys) + lambda_MMD * MMD^2(F(Xs), F(Xt))
MMD uses sum of 3 Gaussian RBF kernels with bandwidths {0.5, 1.0, 2.0} * median pairwise squared distance.
"""
import os
import sys
import torch
import torch.nn as nn

TASK2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK2_ROOT)
sys.path.insert(0, os.path.join(TASK2_ROOT, '..'))

from models.backbone import PACSBackbone


def mmd_loss(source_feats, target_feats, sigmas=(0.5, 1.0, 2.0)):
    """
    Computes the Maximum Mean Discrepancy (MMD^2) between source and target feature distributions
    using a mixture of Gaussian RBF kernels.

    Bandwidth is set to sigma_i * median_pairwise_sq_dist for each sigma_i.
    Combined kernel: K(x,y) = sum_i exp(-||x-y||^2 / (2 * sigma_i * median_dist))

    Args:
        source_feats: (N, D) source feature tensor.
        target_feats: (M, D) target feature tensor.
        sigmas: tuple of kernel bandwidth scaling factors.

    Returns:
        Scalar MMD^2 loss.
    """
    combined = torch.cat([source_feats, target_feats], dim=0)
    # Compute median pairwise squared distance for bandwidth selection
    dists = torch.cdist(combined, combined, p=2).pow(2)
    mask = dists > 0
    median_dist = dists[mask].median().item() if mask.any() else 1.0
    if median_dist < 1e-8:
        median_dist = 1.0

    def rbf_kernel_matrix(X, Y):
        dxy = torch.cdist(X, Y, p=2).pow(2)
        K = sum(torch.exp(-dxy / (2.0 * s * median_dist)) for s in sigmas)
        return K

    n = source_feats.shape[0]
    m = target_feats.shape[0]

    K_ss = rbf_kernel_matrix(source_feats, source_feats)
    K_tt = rbf_kernel_matrix(target_feats, target_feats)
    K_st = rbf_kernel_matrix(source_feats, target_feats)

    mmd = (K_ss.sum() / (n * n) + K_tt.sum() / (m * m)
           - 2.0 * K_st.sum() / (n * m))
    return mmd


def train_dan(train_loaders, target_loader, val_loaders, device,
              lambda_mmd=1.0, max_epochs=30, patience=5,
              lr=1e-4, weight_decay=1e-4, save_path=None):
    """
    DAN training loop.

    Args:
        train_loaders: dict domain -> InfiniteDataLoader (source domains).
        target_loader: InfiniteDataLoader for Sketch (target, labels ignored).
        val_loaders: dict domain -> DataLoader for source validation.
        device: torch device.
        lambda_mmd: MMD loss weight (default 1.0 per spec).
        max_epochs: max training epochs.
        patience: early stopping patience.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        save_path: path to save best checkpoint.

    Returns:
        (model, history)
    """
    model = PACSBackbone(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_cls_loss': [], 'train_mmd_loss': [], 'train_total_loss': [],
        'val_macro_f1': {d: [] for d in val_loaders}, 'mean_val_f1': []
    }

    best_mean_f1 = -1.0
    patience_counter = 0
    best_state = None
    source_domains = list(train_loaders.keys())
    steps_per_epoch = 50

    for epoch in range(max_epochs):
        model.train()
        model.freeze_bn_running_stats()

        epoch_cls, epoch_mmd, epoch_total = 0.0, 0.0, 0.0

        for step in range(steps_per_epoch):
            # Collect 24 source + 24 target
            src_imgs, src_labels = [], []
            for domain in source_domains:
                imgs, labels, _ = next(train_loaders[domain])
                src_imgs.append(imgs)
                src_labels.append(labels)

            tgt_imgs, _, _ = next(target_loader)

            src_imgs = torch.cat(src_imgs, dim=0).to(device)
            src_labels = torch.cat(src_labels, dim=0).to(device)
            tgt_imgs = tgt_imgs.to(device)

            optimizer.zero_grad()
            # Forward source
            src_logits, src_feats = model(src_imgs)
            cls_loss = criterion(src_logits, src_labels)

            # Forward target (features only, no labels used)
            _, tgt_feats = model(tgt_imgs)

            # MMD alignment on 512-d penultimate features
            mmd = mmd_loss(src_feats, tgt_feats)
            total_loss = cls_loss + lambda_mmd * mmd

            total_loss.backward()
            optimizer.step()
            model.freeze_bn_running_stats()

            epoch_cls += cls_loss.item()
            epoch_mmd += mmd.item()
            epoch_total += total_loss.item()

        history['train_cls_loss'].append(epoch_cls / steps_per_epoch)
        history['train_mmd_loss'].append(epoch_mmd / steps_per_epoch)
        history['train_total_loss'].append(epoch_total / steps_per_epoch)

        mean_f1 = _evaluate_all_domains(model, val_loaders, history, device)
        history['mean_val_f1'].append(mean_f1)

        print(f"  [DAN] Epoch {epoch+1}/{max_epochs} | Cls: {epoch_cls/steps_per_epoch:.4f} | "
              f"MMD: {epoch_mmd/steps_per_epoch:.4f} | Mean F1: {mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [DAN] Early stopping at epoch {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)
            print(f"  [DAN] Best model saved to {save_path} (F1={best_mean_f1:.4f})")

    return model, history


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
