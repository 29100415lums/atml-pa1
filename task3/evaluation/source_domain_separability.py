import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from shared.pacs import PACS_DOMAINS

def compute_source_domain_separability(model, val_loaders, device='cuda' if torch.cuda.is_available() else 'cpu', seed=6304):
    """
    Computes source-domain separability using a logistic regression classifier on frozen features.
    """
    model.eval()
    
    all_features = []
    all_domains = []
    
    with torch.no_grad():
        for domain, loader in val_loaders.items():
            domain_idx = PACS_DOMAINS.index(domain)
            domain_features = []
            
            for imgs, _, _ in loader:
                imgs = imgs.to(device)
                _, feats = model(imgs)
                domain_features.append(feats.cpu())
                
            domain_features = torch.cat(domain_features, dim=0).numpy()
            
            all_features.append(domain_features)
            all_domains.extend([domain_idx] * len(domain_features))
            
    # Combine features and labels
    # To collect *balanced* features, we take the minimum number of examples across source domains
    min_count = min(len(f) for f in all_features)
    
    balanced_features = []
    balanced_domains = []
    
    for i, feats in enumerate(all_features):
        # We sample deterministically using the seed
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(feats))[:min_count]
        balanced_features.append(feats[indices])
        domain_idx = PACS_DOMAINS.index(list(val_loaders.keys())[i])
        balanced_domains.extend([domain_idx] * min_count)
        
    X = np.concatenate(balanced_features, axis=0)
    y = np.array(balanced_domains)
    
    # 70/30 split using seed 6304
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    
    # Multinomial logistic regression C=1
    clf = LogisticRegression(C=1.0, multi_class='multinomial', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    return score
