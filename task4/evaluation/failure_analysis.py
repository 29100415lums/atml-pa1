import os
import torch
import numpy as np
import sys
import torchvision.transforms as T

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, TASK4_ROOT)

from models.resnet_cifar import get_cifar_resnet18, ResNet18Penultimate
from data.cifar10 import get_cifar10_loaders
from data.cifar100_unknowns import get_cifar100_unknown_loaders
from scores.mls import get_mls_score
from evaluation.metrics import get_threshold_at_fpr
from torchvision.datasets import CIFAR10, CIFAR100

CIFAR10.url = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/cifar-10-python.tar.gz"
CIFAR100.url = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar-100-python.tar.gz"

def get_classes_maps():
    cifar10 = CIFAR10(root='./shared/data', download=True)
    cifar100 = CIFAR100(root='./shared/data', download=True)
    return cifar10.classes, cifar100.classes

def get_failures(scores, logits, targets, threshold, cifar10_classes, cifar100_classes, num_failures=3):
    # Failure: accepted unknown (score < threshold)
    accepted_mask = scores <= threshold
    
    failures = []
    
    indices = torch.where(accepted_mask)[0].tolist()
    
    for idx in indices[:num_failures]:
        pred_class_idx = logits[idx].argmax().item()
        pred_class = cifar10_classes[pred_class_idx]
        true_unknown_class = cifar100_classes[targets[idx].item()]
        score = scores[idx].item()
        
        failures.append({
            'true_unknown': true_unknown_class,
            'predicted_known': pred_class,
            'score': score,
            'threshold': threshold
        })
    return failures

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    _, val_loader, _, _ = get_cifar10_loaders()
    near_loader, far_loader = get_cifar100_unknown_loaders()
    
    c10_classes, c100_classes = get_classes_maps()
    
    ckpt_dir = os.path.join(TASK4_ROOT, 'checkpoints')
    vanilla_ckpt = os.path.join(ckpt_dir, "vanilla.pt")
    
    if not os.path.exists(vanilla_ckpt):
        print(f"Vanilla checkpoint not found at {vanilla_ckpt}")
        return
        
    base_model = get_cifar_resnet18(num_classes=10)
    model = ResNet18Penultimate(base_model).to(device)
    model.load_state_dict(torch.load(vanilla_ckpt, map_location=device))
    model.eval()
    
    print("Extracting Val features for threshold...")
    val_logits = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            logits, _ = model(inputs.to(device))
            val_logits.append(logits.cpu())
    val_logits = torch.cat(val_logits)
    val_scores = get_mls_score(val_logits)
    tau = get_threshold_at_fpr(val_scores.numpy(), fpr=0.05)
    
    def extract_all(loader):
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in loader:
                logits, _ = model(inputs.to(device))
                all_logits.append(logits.cpu())
                all_targets.append(targets)
        return torch.cat(all_logits), torch.cat(all_targets)
        
    near_logits, near_targets = extract_all(near_loader)
    far_logits, far_targets = extract_all(far_loader)
    
    near_scores = get_mls_score(near_logits)
    far_scores = get_mls_score(far_logits)
    
    near_failures = get_failures(near_scores, near_logits, near_targets, tau, c10_classes, c100_classes)
    far_failures = get_failures(far_scores, far_logits, far_targets, tau, c10_classes, c100_classes)
    
    print(f"\n--- MLS Threshold: {tau:.4f} ---")
    print("\n[ Near Unknown Failures (Accepted) ]")
    for f in near_failures:
        print(f"Unknown: {f['true_unknown']:15} | Pred: {f['predicted_known']:10} | Score: {f['score']:.4f}")
        
    print("\n[ Far Unknown Failures (Accepted) ]")
    for f in far_failures:
        print(f"Unknown: {f['true_unknown']:15} | Pred: {f['predicted_known']:10} | Score: {f['score']:.4f}")

if __name__ == '__main__':
    main()
