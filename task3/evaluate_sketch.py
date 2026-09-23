import os
import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as T
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK3_ROOT = CURRENT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(TASK3_ROOT, '..'))
SHARED_ROOT = os.path.join(PROJECT_ROOT, 'shared')

sys.path.insert(0, TASK3_ROOT)
sys.path.insert(0, SHARED_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from shared.pacs import PACSDataset, PACS_DOMAINS
from shared.pacs_protocol import load_pacs_splits
from models.backbone import PACSBackbone
from models.classifier_head import PACSClassifierHead
from train import FullModel
from evaluation.domain_metrics import evaluate_model_on_loader
from evaluation.source_domain_separability import compute_source_domain_separability
from evaluation.sharpness import compute_sharpness_proxy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main(args):
    print("=" * 60)
    print("Task 3 Final Evaluation (Sketch Target)")
    print("=" * 60)

    val_tf = get_eval_transform()
    splits = load_pacs_splits()
    
    # Load Source Val Loaders
    val_loaders = {}
    for domain in ['art_painting', 'cartoon', 'photo']:
        domain_splits = splits['source_splits'][domain]
        val_ds = PACSDataset(
            list(zip(domain_splits['val']['paths'],
                     domain_splits['val']['labels'],
                     [PACS_DOMAINS.index(domain)] * len(domain_splits['val']['paths']))),
            transform=val_tf
        )
        val_loaders[domain] = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
        
    # Load Target (Sketch) Loader
    target_info = splits['target_all']
    target_ds = PACSDataset(
        list(zip(target_info['paths'], target_info['labels'],
                 [PACS_DOMAINS.index('sketch')] * len(target_info['paths']))),
        transform=val_tf
    )
    target_loader = DataLoader(target_ds, batch_size=64, shuffle=False, num_workers=2)
    
    methods = ['erm', 'dan_dg', 'sam']
    results = {}
    
    ckpt_dir = os.path.join(TASK3_ROOT, 'results', 'checkpoints')
    task2_erm_ckpt = os.path.join(PROJECT_ROOT, 'task2', 'results', 'checkpoints', 'source_only.pt')
    
    for method in methods:
        print(f"\nEvaluating {method.upper()}...")
        results[method] = {}
        
        model = FullModel().to(DEVICE)
        
        if method == 'erm':
            if os.path.exists(task2_erm_ckpt):
                print(f"  -> Loading Task 2 ERM baseline: {task2_erm_ckpt}")
                # Task 2 backbone encompasses the whole model
                model_state = torch.load(task2_erm_ckpt, map_location=DEVICE)
                
                # Remap Task 2 state dict keys to Task 3 FullModel keys
                # Task 2: feature_extractor.*, classifier.*
                # Task 3: backbone.feature_extractor.*, head.classifier.*
                new_state = {}
                for k, v in model_state.items():
                    if k.startswith('feature_extractor'):
                        new_state['backbone.' + k] = v
                    elif k.startswith('classifier'):
                        new_state['head.' + k] = v
                model.load_state_dict(new_state)
            else:
                print(f"  -> [WARNING] Task 2 ERM not found at {task2_erm_ckpt}")
                continue
        else:
            ckpt_path = os.path.join(ckpt_dir, f"{method}.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            else:
                print(f"  -> [WARNING] Checkpoint not found at {ckpt_path}")
                continue
                
        # 1. Source domains
        src_accs = []
        src_f1s = []
        for domain, loader in val_loaders.items():
            res = evaluate_model_on_loader(model, loader, DEVICE)
            results[method][domain] = {'accuracy': res['accuracy'], 'macro_f1': res['macro_f1']}
            src_accs.append(res['accuracy'])
            src_f1s.append(res['macro_f1'])
            
        results[method]['mean_source'] = {'accuracy': float(np.mean(src_accs)), 'macro_f1': float(np.mean(src_f1s))}
        results[method]['worst_source'] = {'accuracy': float(np.min(src_accs)), 'macro_f1': float(np.min(src_f1s))}
        
        # 2. Sketch domain
        tgt_res = evaluate_model_on_loader(model, target_loader, DEVICE)
        results[method]['sketch'] = {
            'accuracy': tgt_res['accuracy'], 
            'macro_f1': tgt_res['macro_f1'],
            'per_class_accuracy': tgt_res['per_class_accuracy']
        }
        
        # 3. Diagnostics
        sep = compute_source_domain_separability(model, val_loaders, DEVICE)
        sharp = compute_sharpness_proxy(model, val_loaders, DEVICE)
        
        results[method]['separability'] = float(sep)
        results[method]['sharpness'] = float(sharp)
        
        print(f"  Mean Source Acc: {np.mean(src_accs)*100:.2f}% | Worst: {np.min(src_accs)*100:.2f}%")
        print(f"  Sketch Acc:      {tgt_res['accuracy']*100:.2f}%")
        print(f"  Separability:    {sep*100:.2f}%")
        print(f"  Sharpness Proxy: {sharp:.4f}")

    # Compute delta relative to ERM
    if 'erm' in results and 'sketch' in results['erm']:
        erm_acc = results['erm']['sketch']['accuracy']
        for method in methods:
            if method in results and 'sketch' in results[method]:
                delta = results[method]['sketch']['accuracy'] - erm_acc
                results[method]['sketch']['delta_acc_vs_erm'] = float(delta)

    # Save results
    os.makedirs(os.path.join(TASK3_ROOT, 'results'), exist_ok=True)
    out_path = os.path.join(TASK3_ROOT, 'results', 'task3_final_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
