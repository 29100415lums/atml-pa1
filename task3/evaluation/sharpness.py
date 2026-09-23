import torch
import torch.nn as nn
import numpy as np

def compute_sharpness_proxy(model, val_loaders, device='cuda' if torch.cuda.is_available() else 'cpu', seed=6304):
    """
    Measures local sharpness by computing the increase in cross-entropy loss
    after a single gradient ascent step of size rho=0.05.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Select fixed validation batch: 32 examples from each source using seed
    # Since loaders are deterministic (shuffle=False), we can just take the first 32 examples from each
    
    batch_imgs = []
    batch_labels = []
    
    for domain, loader in val_loaders.items():
        domain_imgs = []
        domain_labels = []
        for imgs, labels, _ in loader:
            domain_imgs.append(imgs)
            domain_labels.append(labels)
        
        domain_imgs = torch.cat(domain_imgs, dim=0)
        domain_labels = torch.cat(domain_labels, dim=0)
        
        # We need exactly 32
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(domain_imgs))[:32]
        
        batch_imgs.append(domain_imgs[indices])
        batch_labels.append(domain_labels[indices])
        
    X = torch.cat(batch_imgs, dim=0).to(device)
    y = torch.cat(batch_labels, dim=0).to(device)
    
    # Save original parameters
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Requires grad for parameters to compute ascent direction
    for p in model.parameters():
        p.requires_grad = True
        
    model.zero_grad()
    
    # Base loss
    logits, _ = model(X)
    loss_val_theta = criterion(logits, y)
    loss_val_theta.backward()
    
    # Compute normalized gradient-ascent perturbation
    rho = 0.05
    grad_norm = torch.norm(
        torch.stack([p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]),
        p=2
    )
    
    # Apply perturbation: theta + eps
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                eps = rho * p.grad / (grad_norm + 1e-12)
                p.add_(eps)
                
    # Perturbed loss
    model.zero_grad()
    with torch.no_grad():
        logits_perturbed, _ = model(X)
        loss_val_theta_eps = criterion(logits_perturbed, y)
        
    delta_sharpness = (loss_val_theta_eps - loss_val_theta).item()
    
    # Restore model state
    model.load_state_dict(orig_state)
    
    return delta_sharpness
