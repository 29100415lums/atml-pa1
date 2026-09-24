import os
import json
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

def get_cifar10_loaders(data_root='./shared/data', splits_path='./task4/data/splits.json', batch_size=128, use_randaug=False):
    # Base transforms
    if use_randaug:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    val_test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Unaugmented features for Mahalanobis
    unaug_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train_ds = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    val_ds = CIFAR10(root=data_root, train=True, download=True, transform=val_test_transform)
    unaug_train_ds = CIFAR10(root=data_root, train=True, download=True, transform=unaug_transform)
    
    test_ds = CIFAR10(root=data_root, train=False, download=True, transform=val_test_transform)

    # Load splits
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits not found at {splits_path}. Run make_splits.py first.")
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    train_idx = splits['cifar10_train_idx']
    val_idx = splits['cifar10_val_idx']

    train_subset = Subset(full_train_ds, train_idx)
    val_subset = Subset(val_ds, val_idx)
    unaug_train_subset = Subset(unaug_train_ds, train_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    unaug_train_loader = DataLoader(unaug_train_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, unaug_train_loader
