"""
task2/evaluation/domain_separability.py
Computes Domain Separability Score:
  - Freeze backbone, extract balanced source-val and target features.
  - Split 70/30 using seed 6304.
  - Train balanced Logistic Regression (C=1.0) to classify Source vs Target.
  - Test accuracy is the Domain Separability Score.
  - 50% = perfect alignment (chance), 100% = completely separable.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 6304


def extract_features(model, loader, device, max_samples=None):
    """Extracts 512-d penultimate features from a DataLoader."""
    model.eval()
    all_feats = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            _, feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            count += feats.shape[0]
            if max_samples and count >= max_samples:
                break
    return np.concatenate(all_feats, axis=0)[:max_samples]


def compute_domain_separability(model, source_val_loaders, target_loader, device, seed=SEED):
    """
    Computes the Domain Separability Score between source validation and target features.

    Protocol:
      1. Extract features from all source validation domains combined.
      2. Extract features from target domain.
      3. Balance: take min(N_src, N_tgt) samples from each.
      4. 70/30 stratified train/test split (seed 6304).
      5. Train Logistic Regression (C=1.0, balanced class weights).
      6. Return test accuracy as the Domain Separability Score.

    Args:
        model: PACSBackbone model (features extracted from penultimate 512-d layer).
        source_val_loaders: dict domain_name -> DataLoader for source validation.
        target_loader: DataLoader for target domain.
        device: torch device.
        seed: random seed for reproducibility.

    Returns:
        float: Domain Separability Score in [0, 1]. 0.5 = ideal (undetectable).
    """
    # Extract source validation features
    src_feats_list = []
    for loader in source_val_loaders.values():
        f = extract_features(model, loader, device)
        src_feats_list.append(f)
    src_feats = np.concatenate(src_feats_list, axis=0)  # (N_src, 512)

    # Extract target features
    tgt_feats = extract_features(model, target_loader, device)  # (N_tgt, 512)

    # Balance
    n = min(len(src_feats), len(tgt_feats))
    rng = np.random.RandomState(seed)
    src_idx = rng.choice(len(src_feats), n, replace=False)
    tgt_idx = rng.choice(len(tgt_feats), n, replace=False)

    X = np.concatenate([src_feats[src_idx], tgt_feats[tgt_idx]], axis=0)
    y = np.array([0] * n + [1] * n)  # 0 = source, 1 = target

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 70/30 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )

    # Train logistic regression
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    return float(score)
