"""
shared/pacs.py
PACS Dataset loader - works with the standard folder structure:
  shared/data/PACS/{domain}/{class}/image.jpg
"""
import os
from PIL import Image
from torch.utils.data import Dataset

PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
CLASS_TO_IDX = {c: i for i, c in enumerate(PACS_CLASSES)}
DOMAIN_TO_IDX = {d: i for i, d in enumerate(PACS_DOMAINS)}


def scan_pacs_root(root):
    """
    Scans the PACS root directory and returns a list of (filepath, class_idx, domain_idx).
    Handles both flat (root/domain/class/img.jpg) and nested structures.
    """
    samples = []
    for domain in PACS_DOMAINS:
        domain_dir = os.path.join(root, domain)
        if not os.path.isdir(domain_dir):
            # Try capitalized variants
            for candidate in os.listdir(root):
                if candidate.lower().replace(' ', '_') == domain:
                    domain_dir = os.path.join(root, candidate)
                    break
        if not os.path.isdir(domain_dir):
            print(f"[WARNING] Domain folder not found: {domain_dir}")
            continue
        domain_idx = DOMAIN_TO_IDX[domain]
        for cls_name in os.listdir(domain_dir):
            cls_dir = os.path.join(domain_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            cls_name_lower = cls_name.lower()
            if cls_name_lower not in CLASS_TO_IDX:
                continue
            class_idx = CLASS_TO_IDX[cls_name_lower]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    samples.append((os.path.join(cls_dir, fname), class_idx, domain_idx))
    return samples


class PACSDataset(Dataset):
    """
    PACS dataset loader.
    Returns (image_tensor, class_label, domain_label) tuples.
    """
    def __init__(self, samples, transform=None):
        """
        Args:
            samples: list of (filepath, class_idx, domain_idx) tuples.
            transform: torchvision transforms to apply.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx, domain_idx = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, class_idx, domain_idx


class InfiniteDataLoader:
    """
    Wraps a DataLoader so that it cycles infinitely.
    Used to synchronize source domain batches with target batches.
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=2, drop_last=True):
        from torch.utils.data import DataLoader
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers,
                                 drop_last=drop_last, pin_memory=True)
        self._iter = iter(self.loader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)

    def __iter__(self):
        return self
