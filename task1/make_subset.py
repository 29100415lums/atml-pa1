"""
make_subset.py
"""
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 6304

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_stl10_splits(data_dir="./task1/data", seed=SEED):
    set_seed(seed)
    splits_file = "./task1/results/dataset_splits.json"
    if os.path.exists(splits_file):
        with open(splits_file, "r") as f_in:
            split_info = json.load(f_in)
        print(f"[Dataset] Using existing STL-10 splits from {splits_file}. (No download needed)")
        return split_info

    os.makedirs(data_dir, exist_ok=True)
    need_download = not os.path.exists(os.path.join(data_dir, "stl10_binary"))
    raw_train = torchvision.datasets.STL10(root=data_dir, split='train', download=need_download)
    raw_test = torchvision.datasets.STL10(root=data_dir, split='test', download=need_download)

    classes = raw_train.classes
    targets_train = np.array(raw_train.labels)
    targets_test = np.array(raw_test.labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets_train)), targets_train))

    num_classes = len(classes)
    samples_per_class = 500 // num_classes

    test_subset_indices = []
    for c in range(num_classes):
        class_indices = np.where(targets_test == c)[0]
        np.random.shuffle(class_indices)
        test_subset_indices.extend(class_indices[:samples_per_class].tolist())

    test_subset_indices = sorted(test_subset_indices)

    split_info = {
        "dataset": "STL-10",
        "classes": classes,
        "seed": seed,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_subset_indices": test_subset_indices,
        "num_classes": num_classes
    }

    os.makedirs("./task1/results", exist_ok=True)
    with open("./task1/results/dataset_splits.json", "w") as f_out:
        json.dump(split_info, f_out, indent=2)

    print(f"[Dataset] STL-10 splits ready. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_subset_indices)}")
    return split_info

class BaseDatasetWrapper(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img, target = item[0], item[1]
        if self.transform is not None:
            img = self.transform(img)
        return img, target, idx
