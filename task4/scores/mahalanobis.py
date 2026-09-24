import torch
from tqdm import tqdm

def fit_mahalanobis_estimators(model, unaug_train_loader, device, num_classes=10):
    """
    Estimates class means (mu_c) and one shared diagonal covariance (Sigma) 
    from unaugmented CIFAR-10 training features.
    """
    model.eval()
    features_list = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in unaug_train_loader:
            inputs = inputs.to(device)
            _, features = model(inputs)
            features_list.append(features.cpu())
            targets_list.append(targets.cpu())
            
    all_features = torch.cat(features_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    
    # Calculate class means
    mu_c = []
    for c in range(num_classes):
        class_features = all_features[all_targets == c]
        if len(class_features) > 0:
            mu_c.append(class_features.mean(dim=0))
        else:
            mu_c.append(torch.zeros(all_features.size(1)))
    mu_c = torch.stack(mu_c)
    
    # Calculate shared diagonal covariance
    centered_features = all_features.clone()
    for c in range(num_classes):
        mask = (all_targets == c)
        if mask.any():
            centered_features[mask] -= mu_c[c]
            
    # Variance per dimension
    variance = centered_features.var(dim=0, unbiased=True)
    # Add 10^-6 to every diagonal entry
    variance = variance + 1e-6
    
    return mu_c.to(device), variance.to(device)

def get_mahalanobis_score(features, mu_c, variance):
    """
    Mahalanobis Score
    u_Mah(x) = min_c (f(x) - mu_c)^T \Sigma^-1 (f(x) - mu_c)
    variance is the diagonal of \Sigma.
    """
    num_classes = mu_c.size(0)
    scores = []
    
    # variance shape: (D,) -> we can divide by it directly (equivalent to \Sigma^-1 for diagonal)
    for c in range(num_classes):
        diff = features - mu_c[c] # (N, D)
        dist = (diff ** 2) / variance # (N, D)
        dist = dist.sum(dim=1) # (N,)
        scores.append(dist)
        
    scores = torch.stack(scores, dim=1) # (N, C)
    min_dist, _ = scores.min(dim=1)
    
    return min_dist
