import os
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset

CIFAR100.url = "https://ossci-datasets.s3.amazonaws.com/cifar/cifar-100-python.tar.gz"

NEAR_UNKNOWN_CLASSES = [
    'bus', 'pickup_truck', 'motorcycle', 'tractor', 
    'wolf', 'fox', 'leopard', 'camel'
]
FAR_UNKNOWN_CLASSES = [
    'bottle', 'bowl', 'chair', 'clock', 
    'keyboard', 'mushroom', 'sunflower', 'wardrobe'
]

def get_cifar100_unknown_loaders(data_root='./shared/data', batch_size=128):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # We only use CIFAR-100 test set
    cifar100_test = CIFAR100(root=data_root, train=False, download=True, transform=transform)
    
    # Get mapping from class name to index
    class_to_idx = cifar100_test.class_to_idx
    
    near_indices = []
    far_indices = []
    
    for i, (_, target) in enumerate(cifar100_test):
        class_name = cifar100_test.classes[target]
        if class_name in NEAR_UNKNOWN_CLASSES:
            near_indices.append(i)
        elif class_name in FAR_UNKNOWN_CLASSES:
            far_indices.append(i)
            
    near_subset = Subset(cifar100_test, near_indices)
    far_subset = Subset(cifar100_test, far_indices)
    
    near_loader = DataLoader(near_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    far_loader = DataLoader(far_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return near_loader, far_loader
