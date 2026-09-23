"""
shared/pacs_protocol.py
Generates stratified 80/20 train/validation splits for each source domain (Photo, Art Painting, Cartoon)
using seed 6304, and saves them to shared/splits/pacs_sketch_seed6304.json.
This file is shared across Task 2 (UDA) and Task 3.
"""
import os
import sys
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict

SHARED_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SHARED_ROOT)

from pacs import scan_pacs_root, PACS_DOMAINS, PACS_CLASSES

SEED = 6304
SOURCE_DOMAINS = ['art_painting', 'cartoon', 'photo']
TARGET_DOMAIN = 'sketch'
SPLITS_FILE = os.path.join(SHARED_ROOT, 'splits', 'pacs_sketch_seed6304.json')


def generate_pacs_splits(pacs_root, seed=SEED, force=False):
    """
    Generates and saves stratified 80/20 splits for all source domains.
    Loads from disk if already computed (unless force=True).

    Args:
        pacs_root: path to PACS data directory containing domain subdirectories.
        seed: random seed (must match assignment specification = 6304).
        force: overwrite existing split file.

    Returns:
        dict with keys 'source_splits', 'target_all', 'classes', 'domains'.
    """
    os.makedirs(os.path.dirname(SPLITS_FILE), exist_ok=True)

    if os.path.exists(SPLITS_FILE) and not force:
        with open(SPLITS_FILE, 'r') as f:
            splits = json.load(f)
        print(f"[Protocol] Loaded existing PACS splits from {SPLITS_FILE}")
        return splits

    print(f"[Protocol] Scanning PACS at {pacs_root} ...")
    all_samples = scan_pacs_root(pacs_root)
    if not all_samples:
        raise FileNotFoundError(
            f"No samples found in {pacs_root}. "
            "Ensure the PACS dataset is placed at shared/data/PACS/ with subfolders for each domain."
        )

    # Separate by domain
    domain_samples = defaultdict(list)
    for i, (path, class_idx, domain_idx) in enumerate(all_samples):
        domain_name = PACS_DOMAINS[domain_idx]
        domain_samples[domain_name].append({'global_idx': i, 'path': path, 'class_idx': class_idx})

    splits = {
        'seed': seed,
        'classes': PACS_CLASSES,
        'domains': PACS_DOMAINS,
        'source_domains': SOURCE_DOMAINS,
        'target_domain': TARGET_DOMAIN,
        'source_splits': {},
        'target_all': []
    }

    # Stratified 80/20 split for each source domain
    for domain in SOURCE_DOMAINS:
        domain_data = domain_samples[domain]
        paths = [s['path'] for s in domain_data]
        labels = [s['class_idx'] for s in domain_data]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_rel_idx, val_rel_idx = next(sss.split(np.zeros(len(labels)), labels))

        splits['source_splits'][domain] = {
            'train': {
                'paths': [paths[i] for i in train_rel_idx],
                'labels': [labels[i] for i in train_rel_idx]
            },
            'val': {
                'paths': [paths[i] for i in val_rel_idx],
                'labels': [labels[i] for i in val_rel_idx]
            }
        }
        print(f"  [{domain}] Train: {len(train_rel_idx)}, Val: {len(val_rel_idx)}")

    # Target domain: all Sketch images (no labels used during training)
    target_data = domain_samples[TARGET_DOMAIN]
    splits['target_all'] = {
        'paths': [s['path'] for s in target_data],
        'labels': [s['class_idx'] for s in target_data]  # stored but never used during adaptation
    }
    print(f"  [sketch] Total target images: {len(target_data)}")

    with open(SPLITS_FILE, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"[Protocol] Splits saved to {SPLITS_FILE}")

    return splits


def load_pacs_splits():
    """Loads pre-computed splits. Raises FileNotFoundError if not generated yet."""
    if not os.path.exists(SPLITS_FILE):
        raise FileNotFoundError(
            f"Split file not found at {SPLITS_FILE}. "
            "Run generate_pacs_splits() first by executing shared/pacs_protocol.py directly."
        )
    with open(SPLITS_FILE, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacs_root', type=str, default='./shared/data/PACS',
                        help='Path to PACS data directory.')
    parser.add_argument('--force', action='store_true', help='Regenerate splits even if they exist.')
    args = parser.parse_args()
    generate_pacs_splits(args.pacs_root, seed=SEED, force=args.force)
