import os
import torch
import numpy as np
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_ROOT = CURRENT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(TASK4_ROOT, '..'))
sys.path.insert(0, TASK4_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from models.resnet_cifar import get_cifar_resnet18, ResNet18Penultimate, ResNet18PROSER
from data.cifar10 import get_cifar10_loaders
from data.cifar100_unknowns import get_cifar100_unknown_loaders
from scores.msp import get_msp_score
from scores.mls import get_mls_score
from scores.energy import get_energy_score
from scores.mahalanobis import fit_mahalanobis_estimators, get_mahalanobis_score
from evaluation.metrics import compute_auroc, get_threshold_at_fpr, compute_rejection_rates

def extract_features_and_logits(model, loader, device, is_proser=False, num_known=10):
    model.eval()
    all_logits = []
    all_features = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if is_proser:
                logits, features = model(inputs, mixup=False)
            else:
                logits, features = model(inputs)
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_targets.append(targets.cpu())
            
    return torch.cat(all_logits), torch.cat(all_features), torch.cat(all_targets)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading data...")
    train_loader, val_loader, test_loader, unaug_train_loader = get_cifar10_loaders()
    near_loader, far_loader = get_cifar100_unknown_loaders()
    
    ckpt_dir = os.path.join(TASK4_ROOT, 'checkpoints')
    results = {}
    
    # Evaluate Vanilla
    vanilla_ckpt = os.path.join(ckpt_dir, "vanilla.pt")
    if os.path.exists(vanilla_ckpt):
        print("\nEvaluating Vanilla...")
        base_model = get_cifar_resnet18(num_classes=10)
        vanilla_model = ResNet18Penultimate(base_model).to(device)
        vanilla_model.load_state_dict(torch.load(vanilla_ckpt, map_location=device))
        
        # Fit Mahalanobis
        mu_c, variance = fit_mahalanobis_estimators(vanilla_model, unaug_train_loader, device)
        mu_c, variance = mu_c.cpu(), variance.cpu()
        
        # Extract for Val (calibration)
        val_logits, val_feats, _ = extract_features_and_logits(vanilla_model, val_loader, device)
        
        # Extract for Test (known)
        test_logits, test_feats, test_targets = extract_features_and_logits(vanilla_model, test_loader, device)
        
        # Extract for Near/Far
        near_logits, near_feats, _ = extract_features_and_logits(vanilla_model, near_loader, device)
        far_logits, far_feats, _ = extract_features_and_logits(vanilla_model, far_loader, device)
        
        # Calculate CSA
        preds = test_logits.argmax(dim=1)
        csa = (preds == test_targets).float().mean().item()
        print(f"Vanilla CSA: {csa*100:.2f}%")
        
        results['vanilla'] = {'csa': csa}
        
        def eval_score(score_fn, name):
            if name == 'Mahalanobis':
                val_scores = score_fn(val_feats, mu_c, variance).numpy()
                test_scores = score_fn(test_feats, mu_c, variance).numpy()
                near_scores = score_fn(near_feats, mu_c, variance).numpy()
                far_scores = score_fn(far_feats, mu_c, variance).numpy()
            else:
                val_scores = score_fn(val_logits).numpy()
                test_scores = score_fn(test_logits).numpy()
                near_scores = score_fn(near_logits).numpy()
                far_scores = score_fn(far_logits).numpy()
                
            all_unknown = np.concatenate([near_scores, far_scores])
            
            auroc_near = compute_auroc(test_scores, near_scores)
            auroc_far = compute_auroc(test_scores, far_scores)
            auroc_all = compute_auroc(test_scores, all_unknown)
            
            tau = get_threshold_at_fpr(val_scores, fpr=0.05)
            
            accept_test = 1.0 - compute_rejection_rates(test_scores, tau)
            reject_near = compute_rejection_rates(near_scores, tau)
            reject_far = compute_rejection_rates(far_scores, tau)
            
            results['vanilla'][name] = {
                'auroc_near': auroc_near,
                'auroc_far': auroc_far,
                'auroc_all': auroc_all,
                'accept_test': accept_test,
                'reject_near': reject_near,
                'reject_far': reject_far
            }
            
            print(f"  {name}:")
            print(f"    AUROC (Near/Far/All): {auroc_near:.4f} / {auroc_far:.4f} / {auroc_all:.4f}")
            print(f"    Rejection @ FPR95 (Test_Accept/Near_Reject/Far_Reject): {accept_test:.4f} / {reject_near:.4f} / {reject_far:.4f}")
            
        eval_score(get_msp_score, "MSP")
        eval_score(get_mls_score, "MLS")
        eval_score(get_energy_score, "Energy")
        eval_score(get_mahalanobis_score, "Mahalanobis")
        
    # Evaluate GCSC
    gcsc_ckpt = os.path.join(ckpt_dir, "gcsc.pt")
    if os.path.exists(gcsc_ckpt):
        print("\nEvaluating GCSC...")
        base_model = get_cifar_resnet18(num_classes=10)
        gcsc_model = ResNet18Penultimate(base_model).to(device)
        gcsc_model.load_state_dict(torch.load(gcsc_ckpt, map_location=device))
        
        val_logits, _, _ = extract_features_and_logits(gcsc_model, val_loader, device)
        test_logits, _, test_targets = extract_features_and_logits(gcsc_model, test_loader, device)
        near_logits, _, _ = extract_features_and_logits(gcsc_model, near_loader, device)
        far_logits, _, _ = extract_features_and_logits(gcsc_model, far_loader, device)
        
        preds = test_logits.argmax(dim=1)
        csa = (preds == test_targets).float().mean().item()
        print(f"GCSC CSA: {csa*100:.2f}%")
        
        val_scores = get_mls_score(val_logits).numpy()
        test_scores = get_mls_score(test_logits).numpy()
        near_scores = get_mls_score(near_logits).numpy()
        far_scores = get_mls_score(far_logits).numpy()
        
        tau = get_threshold_at_fpr(val_scores, fpr=0.05)
        
        results['gcsc'] = {
            'csa': csa,
            'MLS': {
                'auroc_near': compute_auroc(test_scores, near_scores),
                'auroc_far': compute_auroc(test_scores, far_scores),
                'accept_test': 1.0 - compute_rejection_rates(test_scores, tau),
                'reject_near': compute_rejection_rates(near_scores, tau),
                'reject_far': compute_rejection_rates(far_scores, tau)
            }
        }
        
        print("  MLS:")
        print(f"    AUROC (Near/Far): {results['gcsc']['MLS']['auroc_near']:.4f} / {results['gcsc']['MLS']['auroc_far']:.4f}")
        print(f"    Rejection @ FPR95 (Test/Near/Far): {results['gcsc']['MLS']['accept_test']:.4f} / {results['gcsc']['MLS']['reject_near']:.4f} / {results['gcsc']['MLS']['reject_far']:.4f}")
        
    # Evaluate PROSER
    proser_ckpt = os.path.join(ckpt_dir, "proser.pt")
    if os.path.exists(proser_ckpt):
        print("\nEvaluating PROSER...")
        from scores.proser_score import get_proser_score
        
        base_model = get_cifar_resnet18(num_classes=10)
        proser_model = ResNet18PROSER(base_model).to(device)
        proser_model.load_state_dict(torch.load(proser_ckpt, map_location=device))
        
        val_logits, _, _ = extract_features_and_logits(proser_model, val_loader, device, is_proser=True)
        test_logits, _, test_targets = extract_features_and_logits(proser_model, test_loader, device, is_proser=True)
        near_logits, _, _ = extract_features_and_logits(proser_model, near_loader, device, is_proser=True)
        far_logits, _, _ = extract_features_and_logits(proser_model, far_loader, device, is_proser=True)
        
        # CSA on known 10 classes
        preds = test_logits[:, :10].argmax(dim=1)
        csa = (preds == test_targets).float().mean().item()
        print(f"PROSER CSA: {csa*100:.2f}%")
        
        results['proser'] = {'csa': csa}
        
        def eval_proser(score_fn, name):
            if name == "MLS (Known)":
                val_scores = score_fn(val_logits[:, :10]).numpy()
                test_scores = score_fn(test_logits[:, :10]).numpy()
                near_scores = score_fn(near_logits[:, :10]).numpy()
                far_scores = score_fn(far_logits[:, :10]).numpy()
            else:
                val_scores = score_fn(val_logits).numpy()
                test_scores = score_fn(test_logits).numpy()
                near_scores = score_fn(near_logits).numpy()
                far_scores = score_fn(far_logits).numpy()
                
            tau = get_threshold_at_fpr(val_scores, fpr=0.05)
            
            auroc_near = compute_auroc(test_scores, near_scores)
            auroc_far = compute_auroc(test_scores, far_scores)
            accept_test = 1.0 - compute_rejection_rates(test_scores, tau)
            reject_near = compute_rejection_rates(near_scores, tau)
            reject_far = compute_rejection_rates(far_scores, tau)
            
            results['proser'][name] = {
                'auroc_near': auroc_near,
                'auroc_far': auroc_far,
                'accept_test': accept_test,
                'reject_near': reject_near,
                'reject_far': reject_far
            }
            
            print(f"  {name}:")
            print(f"    AUROC (Near/Far): {auroc_near:.4f} / {auroc_far:.4f}")
            print(f"    Rejection @ FPR95 (Test/Near/Far): {accept_test:.4f} / {reject_near:.4f} / {reject_far:.4f}")
            
        eval_proser(get_mls_score, "MLS (Known)")
        eval_proser(get_proser_score, "PROSER Detection Score")
        
    # Save to JSON
    results_dir = os.path.join(TASK4_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'task4_final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults successfully saved to task4/results/task4_final_results.json")

if __name__ == '__main__':
    main()
