import os
import sys
import yaml
import json
import torch
import argparse
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK3_ROOT = CURRENT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(TASK3_ROOT, '..'))

sys.path.insert(0, TASK3_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from train import train
from evaluate_sketch import get_eval_transform
from evaluation.domain_metrics import evaluate_model_on_loader
from evaluation.sharpness import compute_sharpness_proxy
from shared.pacs import PACSDataset, PACS_DOMAINS
from shared.pacs_protocol import load_pacs_splits
from train import FullModel
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print("=" * 60)
    print("Task 3: SAM Controlled Design Study (Varying Rho)")
    print("=" * 60)
    
    rhos = [0.01, 0.05, 0.1]
    results = {}
    
    # Setup data loaders for evaluation
    val_tf = get_eval_transform()
    splits = load_pacs_splits()
    
    # Load Source Val Loaders for sharpness
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
    
    config_path = os.path.join(TASK3_ROOT, 'configs', 'sam.yaml')
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    # Parse dummy args for train
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacs_root', type=str, default=os.path.join(PROJECT_ROOT, 'shared', 'data', 'PACS'))
    args, _ = parser.parse_known_args()
        
    for rho in rhos:
        print(f"\n--- Running SAM with rho = {rho} ---")
        
        # 1. Modify config and train
        config = base_config.copy()
        config['rho'] = rho
        # Temporarily change method name so checkpoint saves as sam_0.01.pt etc
        config['method'] = f"sam_{rho}" 
        
        train(config, args)
        
        # 2. Evaluate
        ckpt_path = os.path.join(TASK3_ROOT, 'results', 'checkpoints', f"sam_{rho}.pt")
        model = FullModel().to(DEVICE)
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        else:
            print(f"[ERROR] Checkpoint not found at {ckpt_path}. Skipping.")
            continue
            
        tgt_res = evaluate_model_on_loader(model, target_loader, DEVICE)
        sharp = compute_sharpness_proxy(model, val_loaders, DEVICE)
        
        results[f"rho_{rho}"] = {
            'sketch_accuracy': tgt_res['accuracy'],
            'sharpness': float(sharp)
        }
        
        print(f"  Sketch Acc:      {tgt_res['accuracy']*100:.2f}%")
        print(f"  Sharpness Proxy: {sharp:.4f}")
        
    # Save JSON
    out_dir = os.path.join(TASK3_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'sam_controlled_study.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved controlled study results to {out_json}")
    
    # Plot trade-off
    try:
        x = rhos
        acc = [results[f"rho_{r}"]['sketch_accuracy'] for r in rhos]
        sharp = [results[f"rho_{r}"]['sharpness'] for r in rhos]
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('SAM Radius (\u03C1)', fontsize=12)
        ax1.set_ylabel('Target (Sketch) Accuracy', color=color, fontsize=12)
        ax1.plot(x, acc, marker='o', color=color, linewidth=2, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Sharpness Proxy (Loss Increase)', color=color, fontsize=12)
        ax2.plot(x, sharp, marker='s', color=color, linestyle='--', linewidth=2, label='Sharpness')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('SAM Controlled Study: Radius vs Accuracy vs Sharpness')
        fig.tight_layout()
        
        plot_path = os.path.join(out_dir, 'plots')
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(os.path.join(plot_path, 'sam_controlled_study.png'), dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.join(plot_path, 'sam_controlled_study.png')}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == '__main__':
    main()
