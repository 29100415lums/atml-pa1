import argparse
import os
import torch
import yaml
import sys

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK4_ROOT = CURRENT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(TASK4_ROOT, '..'))
sys.path.insert(0, TASK4_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from models.resnet_cifar import get_cifar_resnet18, ResNet18Penultimate, ResNet18PROSER
from data.cifar10 import get_cifar10_loaders
from methods.vanilla import train_vanilla
from methods.proser import train_proser

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config.get('seed', 6304))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    method = config.get('method', 'vanilla')
    use_randaug = config.get('use_randaug', False)
    
    train_loader, val_loader, test_loader, _ = get_cifar10_loaders(
        data_root=config.get('data_root', './shared/data'),
        splits_path=config.get('splits_path', './task4/data/splits.json'),
        batch_size=config.get('batch_size', 128),
        use_randaug=use_randaug
    )
    
    base_model = get_cifar_resnet18(num_classes=10)
    
    ckpt_dir = os.path.join(TASK4_ROOT, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{method}.pt")
    
    if method in ['vanilla', 'gcsc']:
        model = ResNet18Penultimate(base_model).to(device)
        train_vanilla(model, train_loader, val_loader, device, save_path, epochs=config.get('epochs', 100))
    elif method == 'proser':
        # PROSER initializes from Vanilla
        vanilla_ckpt = os.path.join(ckpt_dir, "vanilla.pt")
        if not os.path.exists(vanilla_ckpt):
            raise FileNotFoundError(f"PROSER needs {vanilla_ckpt} to initialize.")
            
        model = ResNet18PROSER(base_model).to(device)
        # Load vanilla weights into PROSER model. We need to be careful with mapping.
        # But wait, ResNet18Penultimate state_dict has 'features' and 'fc'. 
        # ResNet18PROSER has 'pre_mixup', 'post_mixup', 'fc'. 
        # Actually it's easier to load vanilla.pt into ResNet18Penultimate first, 
        # then copy the base_model back to ResNet18PROSER.
        vanilla_wrapper = ResNet18Penultimate(base_model)
        vanilla_wrapper.load_state_dict(torch.load(vanilla_ckpt, map_location='cpu'))
        
        # Now base_model is updated with vanilla weights. 
        # We can just initialize PROSER with this base_model.
        model = ResNet18PROSER(base_model).to(device)
        
        train_proser(model, train_loader, val_loader, device, save_path, epochs=config.get('epochs', 50))
    else:
        raise ValueError(f"Unknown method {method}")

if __name__ == '__main__':
    main()
