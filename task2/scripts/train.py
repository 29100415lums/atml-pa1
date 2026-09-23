"""
task2/scripts/train.py
Master orchestration script for Task 2 — Unsupervised Domain Adaptation on PACS.
Runs all 4 methods (Source-Only, DAN, DANN, CDAN), controlled study (DAN lambda_MMD),
computes all required evaluation metrics, and generates all report figures.

Usage (from project root /content/drive/MyDrive/ATML/PA1):
    python task2/scripts/train.py --pacs_root shared/data/PACS

Outputs:
    task2/results/checkpoints/  — saved model weights
    task2/results/plots/        — all publication-quality figures
    task2/results/task2_final_results.json — complete numerical summary
"""
import os
import sys
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T

# -- Path Setup --
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK2_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(TASK2_ROOT, '..'))
SHARED_ROOT = os.path.join(PROJECT_ROOT, 'shared')

os.chdir(PROJECT_ROOT)
sys.path.insert(0, TASK2_ROOT)
sys.path.insert(0, SHARED_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# -- Imports --
from pacs import PACSDataset, InfiniteDataLoader, PACS_DOMAINS, PACS_CLASSES
from pacs_protocol import generate_pacs_splits

from models.backbone import PACSBackbone
from methods.source_only import train_source_only
from methods.dan import train_dan
from methods.dann import train_dann
from methods.cdan import train_cdan

from evaluation.metrics import evaluate_model_on_loader, compute_per_class_accuracy
from evaluation.domain_separability import compute_domain_separability
from evaluation.class_analysis import (
    compute_class_delta, plot_class_transfer_analysis, plot_confusion_matrix
)
from evaluation.plot_utils import (
    plot_training_curves, plot_results_comparison,
    plot_domain_separability, plot_controlled_study
)

import argparse

SEED = 6304
SOURCE_DOMAINS = ['art_painting', 'cartoon', 'photo']
TARGET_DOMAIN = 'sketch'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


def build_loaders(splits, train_tf, val_tf, source_batch=8, target_batch=24):
    """
    Builds InfiniteDataLoaders for training (one per source domain + one for target)
    and standard DataLoaders for source validation and target evaluation.
    """
    train_loaders, val_loaders = {}, {}

    for domain in SOURCE_DOMAINS:
        domain_splits = splits['source_splits'][domain]
        train_ds = PACSDataset(
            list(zip(domain_splits['train']['paths'],
                     domain_splits['train']['labels'],
                     [PACS_DOMAINS.index(domain)] * len(domain_splits['train']['paths']))),
            transform=train_tf
        )
        val_ds = PACSDataset(
            list(zip(domain_splits['val']['paths'],
                     domain_splits['val']['labels'],
                     [PACS_DOMAINS.index(domain)] * len(domain_splits['val']['paths']))),
            transform=val_tf
        )
        train_loaders[domain] = InfiniteDataLoader(train_ds, batch_size=source_batch,
                                                   shuffle=True, num_workers=2)
        val_loaders[domain] = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    # Target domain: all Sketch images (labels stored but never used for training/selection)
    target_info = splits['target_all']
    target_ds_train = PACSDataset(
        list(zip(target_info['paths'], target_info['labels'],
                 [PACS_DOMAINS.index(TARGET_DOMAIN)] * len(target_info['paths']))),
        transform=train_tf
    )
    target_ds_eval = PACSDataset(
        list(zip(target_info['paths'], target_info['labels'],
                 [PACS_DOMAINS.index(TARGET_DOMAIN)] * len(target_info['paths']))),
        transform=val_tf
    )
    target_train_loader = InfiniteDataLoader(target_ds_train, batch_size=target_batch,
                                             shuffle=True, num_workers=2)
    target_eval_loader = DataLoader(target_ds_eval, batch_size=64, shuffle=False, num_workers=2)

    return train_loaders, val_loaders, target_train_loader, target_eval_loader


def main(args):
    set_seed(SEED)
    print(f"=" * 60)
    print(f"Task 2: Unsupervised Domain Adaptation (PACS → Sketch)")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)

    # Directories
    ckpt_dir = './task2/results/checkpoints'
    plots_dir = './task2/results/plots'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Generate or load splits
    splits = generate_pacs_splits(args.pacs_root, seed=SEED)
    train_tf, val_tf = get_transforms()
    train_loaders, val_loaders, target_train_loader, target_eval_loader = build_loaders(
        splits, train_tf, val_tf
    )

    final_results = {}
    histories = {}
    all_models = {}

    # -------------------------------------------------------
    # METHOD 1: Source-Only ERM
    # -------------------------------------------------------
    print("\n[1/4] Source-Only ERM ...")
    erm_ckpt = os.path.join(ckpt_dir, 'source_only.pt')
    if os.path.exists(erm_ckpt):
        print(f"  -> Loading existing checkpoint from {erm_ckpt}")
        model = PACSBackbone().to(DEVICE)
        model.load_state_dict(torch.load(erm_ckpt, map_location=DEVICE))
        hist = {}
    else:
        model, hist = train_source_only(
            train_loaders, val_loaders, DEVICE,
            max_epochs=30, patience=5, save_path=erm_ckpt
        )
    histories['source_only'] = hist
    all_models['source_only'] = model

    # -------------------------------------------------------
    # METHOD 2: DAN
    # -------------------------------------------------------
    print("\n[2/4] DAN (MMD Alignment, lambda=1.0) ...")
    dan_ckpt = os.path.join(ckpt_dir, 'dan.pt')
    if os.path.exists(dan_ckpt):
        print(f"  -> Loading existing checkpoint from {dan_ckpt}")
        model_dan = PACSBackbone().to(DEVICE)
        model_dan.load_state_dict(torch.load(dan_ckpt, map_location=DEVICE))
        hist_dan = {}
    else:
        model_dan, hist_dan = train_dan(
            train_loaders, target_train_loader, val_loaders, DEVICE,
            lambda_mmd=1.0, max_epochs=30, patience=5, save_path=dan_ckpt
        )
    histories['dan'] = hist_dan
    all_models['dan'] = model_dan

    # -------------------------------------------------------
    # METHOD 3: DANN
    # -------------------------------------------------------
    print("\n[3/4] DANN (Adversarial GRL) ...")
    dann_ckpt = os.path.join(ckpt_dir, 'dann.pt')
    if os.path.exists(dann_ckpt):
        print(f"  -> Loading existing checkpoint from {dann_ckpt}")
        model_dann = PACSBackbone().to(DEVICE)
        ckpt_data = torch.load(dann_ckpt, map_location=DEVICE)
        model_dann.load_state_dict(ckpt_data['model'] if 'model' in ckpt_data else ckpt_data)
        hist_dann = {}
        disc_dann = None
    else:
        model_dann, disc_dann, hist_dann = train_dann(
            train_loaders, target_train_loader, val_loaders, DEVICE,
            max_epochs=30, patience=5, save_path=dann_ckpt
        )
    histories['dann'] = hist_dann
    all_models['dann'] = model_dann

    # -------------------------------------------------------
    # METHOD 4: CDAN
    # -------------------------------------------------------
    print("\n[4/4] CDAN (Class-Conditional Adversarial) ...")
    cdan_ckpt = os.path.join(ckpt_dir, 'cdan.pt')
    if os.path.exists(cdan_ckpt):
        print(f"  -> Loading existing checkpoint from {cdan_ckpt}")
        model_cdan = PACSBackbone().to(DEVICE)
        ckpt_data = torch.load(cdan_ckpt, map_location=DEVICE)
        model_cdan.load_state_dict(ckpt_data['model'] if 'model' in ckpt_data else ckpt_data)
        hist_cdan = {}
    else:
        model_cdan, disc_cdan, hist_cdan = train_cdan(
            train_loaders, target_train_loader, val_loaders, DEVICE,
            max_epochs=30, patience=5, save_path=cdan_ckpt
        )
    histories['cdan'] = hist_cdan
    all_models['cdan'] = model_cdan

    # -------------------------------------------------------
    # EVALUATION: All domains + Target (using labels only here)
    # -------------------------------------------------------
    print("\n[Eval] Evaluating all methods on all domains ...")
    separability_scores = {}

    for method_name, model in all_models.items():
        final_results[method_name] = {}
        model.eval()

        # Source validation domains
        for domain, loader in val_loaders.items():
            res = evaluate_model_on_loader(model, loader, DEVICE)
            final_results[method_name][domain] = {
                'accuracy': res['accuracy'],
                'macro_f1': res['macro_f1'],
                'per_class_accuracy': res['per_class_accuracy']
            }

        # Mean source
        mean_src_acc = np.mean([final_results[method_name][d]['accuracy'] for d in SOURCE_DOMAINS])
        mean_src_f1 = np.mean([final_results[method_name][d]['macro_f1'] for d in SOURCE_DOMAINS])
        final_results[method_name]['mean_source'] = {'accuracy': float(mean_src_acc), 'macro_f1': float(mean_src_f1)}
        final_results[method_name]['mean_source_f1'] = float(mean_src_f1)

        # Target evaluation (Sketch — labels allowed ONLY here at final eval)
        tgt_res = evaluate_model_on_loader(model, target_eval_loader, DEVICE)
        final_results[method_name]['sketch'] = {
            'accuracy': tgt_res['accuracy'],
            'macro_f1': tgt_res['macro_f1'],
            'per_class_accuracy': tgt_res['per_class_accuracy'],
            'predictions': tgt_res['predictions'],
            'labels': tgt_res['labels']
        }

        # Domain Separability Score
        sep = compute_domain_separability(model, val_loaders, target_eval_loader, DEVICE)
        separability_scores[method_name] = sep
        final_results[method_name]['domain_separability'] = sep

        print(f"  [{method_name.upper()}] "
              f"Mean Src Acc: {mean_src_acc*100:.2f}% | "
              f"Target Acc: {tgt_res['accuracy']*100:.2f}% | "
              f"Separability: {sep*100:.2f}%")

    # Compute Delta Acc vs Source-Only
    src_only_tgt_acc = final_results['source_only']['sketch']['accuracy']
    for method_name in ['dan', 'dann', 'cdan']:
        delta = final_results[method_name]['sketch']['accuracy'] - src_only_tgt_acc
        final_results[method_name]['sketch']['delta_acc'] = float(delta)

    # -------------------------------------------------------
    # CONTROLLED STUDY: DAN lambda_MMD in {0.1, 1.0, 10.0}
    # -------------------------------------------------------
    print("\n[Controlled Study] DAN lambda_MMD ∈ {0.1, 1.0, 10.0} ...")
    lambda_values = [0.1, 1.0, 10.0]
    study_results = {}

    for lam in lambda_values:
        study_ckpt = os.path.join(ckpt_dir, f'dan_lambda{lam}.pt')
        if lam == 1.0 and os.path.exists(dan_ckpt):
            # Reuse already-trained DAN with lambda=1.0
            study_model = all_models['dan']
            study_sep = separability_scores['dan']
            study_tgt_res = final_results['dan']['sketch']
            study_mean_f1 = final_results['dan']['mean_source_f1']
        else:
            set_seed(SEED)
            if os.path.exists(study_ckpt):
                study_model = PACSBackbone().to(DEVICE)
                study_model.load_state_dict(torch.load(study_ckpt, map_location=DEVICE))
            else:
                study_model, _ = train_dan(
                    train_loaders, target_train_loader, val_loaders, DEVICE,
                    lambda_mmd=lam, max_epochs=30, patience=5, save_path=study_ckpt
                )
            study_tgt_res = evaluate_model_on_loader(study_model, target_eval_loader, DEVICE)
            study_sep = compute_domain_separability(study_model, val_loaders, target_eval_loader, DEVICE)
            mean_src_accs = []
            for domain, loader in val_loaders.items():
                res = evaluate_model_on_loader(study_model, loader, DEVICE)
                mean_src_accs.append(res['macro_f1'])
            study_mean_f1 = float(np.mean(mean_src_accs))

        study_results[lam] = {
            'sketch': {'accuracy': study_tgt_res['accuracy'], 'macro_f1': study_tgt_res['macro_f1']},
            'domain_separability': study_sep,
            'mean_source_f1': study_mean_f1
        }
        print(f"  [DAN lambda={lam}] Target Acc: {study_tgt_res['accuracy']*100:.2f}% | "
              f"Separability: {study_sep*100:.2f}%")

    final_results['controlled_study_dan_lambda'] = {str(k): v for k, v in study_results.items()}

    # -------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------
    print("\n[Plots] Generating all required evidence plots ...")

    # Training curves
    if any(histories[m] for m in histories):
        plot_training_curves(histories, plots_dir)

    # Results comparison
    plot_results_comparison(final_results,
                            save_path=os.path.join(plots_dir, 'results_comparison.png'))

    # Domain separability
    plot_domain_separability(separability_scores,
                             save_path=os.path.join(plots_dir, 'domain_separability.png'))

    # Controlled study
    plot_controlled_study(study_results, 'λ_MMD', lambda_values,
                          save_path=os.path.join(plots_dir, 'controlled_study_dan_lambda.png'))

    # Per-class transfer analysis
    src_only_per_class = final_results['source_only']['sketch']['per_class_accuracy']
    all_deltas = []
    adaptation_methods = []
    for method in ['dan', 'dann', 'cdan']:
        delta = compute_class_delta(src_only_per_class, final_results[method]['sketch']['per_class_accuracy'])
        all_deltas.append(delta)
        adaptation_methods.append(METHOD_LABELS.get(method, method) if 'METHOD_LABELS' in dir() else method)

    from evaluation.plot_utils import METHOD_LABELS
    plot_class_transfer_analysis(
        all_deltas,
        [METHOD_LABELS.get(m, m) for m in ['dan', 'dann', 'cdan']],
        save_path=os.path.join(plots_dir, 'class_transfer_analysis.png')
    )

    # Confusion matrices for each method on target
    for method in ['source_only', 'dan', 'dann', 'cdan']:
        tgt_data = final_results[method]['sketch']
        plot_confusion_matrix(
            tgt_data['labels'], tgt_data['predictions'],
            method_name=METHOD_LABELS.get(method, method),
            save_path=os.path.join(plots_dir, f'confusion_{method}.png')
        )

    # -------------------------------------------------------
    # Save full numerical results
    # -------------------------------------------------------
    results_path = './task2/results/task2_final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Task 2 Complete!")
    print(f"Results: {results_path}")
    print(f"Plots:   {plots_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 2: UDA on PACS')
    parser.add_argument('--pacs_root', type=str, default='./shared/data/PACS',
                        help='Path to PACS data directory.')
    args = parser.parse_args()
    main(args)
