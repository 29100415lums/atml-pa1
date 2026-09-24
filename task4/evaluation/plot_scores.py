import os
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, TASK4_ROOT)

from models.resnet_cifar import get_cifar_resnet18, ResNet18Penultimate
from data.cifar10 import get_cifar10_loaders
from data.cifar100_unknowns import get_cifar100_unknown_loaders
from scores.msp import get_msp_score
from scores.mls import get_mls_score
from scores.mahalanobis import fit_mahalanobis_estimators, get_mahalanobis_score
from sklearn.metrics import roc_curve

def extract_features_and_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_features = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits, features = model(inputs)
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
    return torch.cat(all_logits), torch.cat(all_features)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, val_loader, test_loader, unaug_train_loader = get_cifar10_loaders()
    near_loader, far_loader = get_cifar100_unknown_loaders()
    
    vanilla_ckpt = os.path.join(TASK4_ROOT, 'checkpoints', 'vanilla.pt')
    if not os.path.exists(vanilla_ckpt):
        print("Vanilla checkpoint not found. Cannot plot.")
        return
        
    base_model = get_cifar_resnet18(num_classes=10)
    model = ResNet18Penultimate(base_model).to(device)
    model.load_state_dict(torch.load(vanilla_ckpt, map_location=device))
    
    print("Extracting features...")
    mu_c, variance = fit_mahalanobis_estimators(model, unaug_train_loader, device)
    mu_c, variance = mu_c.cpu(), variance.cpu()
    
    test_logits, test_feats = extract_features_and_logits(model, test_loader, device)
    near_logits, near_feats = extract_features_and_logits(model, near_loader, device)
    far_logits, far_feats = extract_features_and_logits(model, far_loader, device)
    
    unknown_logits = torch.cat([near_logits, far_logits])
    unknown_feats = torch.cat([near_feats, far_feats])
    
    # Calculate scores
    scores_dict = {
        'MSP': {
            'known': get_msp_score(test_logits).numpy(),
            'unknown': get_msp_score(unknown_logits).numpy()
        },
        'MLS': {
            'known': get_mls_score(test_logits).numpy(),
            'unknown': get_mls_score(unknown_logits).numpy()
        },
        'Mahalanobis': {
            'known': get_mahalanobis_score(test_feats, mu_c, variance).numpy(),
            'unknown': get_mahalanobis_score(unknown_feats, mu_c, variance).numpy()
        }
    }
    
    # Plot ROC curves
    plt.figure(figsize=(15, 5))
    
    for i, (name, s_dict) in enumerate(scores_dict.items()):
        plt.subplot(1, 3, i+1)
        
        y_true = np.concatenate([np.zeros(len(s_dict['known'])), np.ones(len(s_dict['unknown']))])
        y_scores = np.concatenate([s_dict['known'], s_dict['unknown']])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        plt.plot(fpr, tpr, label=f'{name} ROC', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title(f'{name} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(alpha=0.3)
        
    plt.tight_layout()
    plot_path = os.path.join(TASK4_ROOT, 'results', 'roc_curves.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"ROC plot saved to {plot_path}")

if __name__ == '__main__':
    main()
