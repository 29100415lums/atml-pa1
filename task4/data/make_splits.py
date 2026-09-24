import os
import json
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split

CIFAR10.url = "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-10-python.tar.gz"
CIFAR100.url = "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-100-python.tar.gz"

def create_cifar_splits(data_root='./shared/data', save_path='./task4/data/splits.json', seed=6304):
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load CIFAR-10 Training Set
    cifar10_train = CIFAR10(root=data_root, train=True, download=True)
    
    # Stratified 90/10 split
    indices = np.arange(len(cifar10_train))
    targets = cifar10_train.targets
    
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.1, 
        random_state=seed, 
        stratify=targets
    )
    
    # Save indices to JSON
    splits = {
        'cifar10_train_idx': train_idx.tolist(),
        'cifar10_val_idx': val_idx.tolist(),
        'seed': seed
    }
    
    with open(save_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Splits saved to {save_path}")
    print(f"CIFAR-10 Train: {len(train_idx)} samples")
    print(f"CIFAR-10 Val: {len(val_idx)} samples")

if __name__ == "__main__":
    create_cifar_splits()
