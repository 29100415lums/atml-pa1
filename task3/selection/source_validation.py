import torch
from evaluation.domain_metrics import compute_macro_f1

def evaluate_source_validation(model, val_loaders, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluates the model on all source validation sets.
    Returns the mean macro-F1 across source domains.
    """
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
            domain_f1s.append(f1)
            
    # Do not call model.train() here to keep it pure evaluation
    return sum(domain_f1s) / len(domain_f1s)
