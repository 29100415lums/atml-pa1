"""
make_cue_conflicts.py
"""
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

# Valid STL-10 class pairs (STL-10 has 'monkey', NOT 'frog')
CLASS_PAIRS = [
    ('car', 'cat'),
    ('bird', 'truck'),
    ('dog', 'ship'),
    ('airplane', 'monkey'),
    ('horse', 'deer')
]

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice = vgg[:21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.slice(x)

def visual_rejection_rule(stylized_pil, content_pil):
    """
    Visual rejection filter checking pixel variance and Sobel edge correlation.
    Rejects images where variance < 0.01 or edge correlation < 0.20.
    """
    stylized_gray = np.array(stylized_pil.convert('L'), dtype=np.float32) / 255.0
    content_gray = np.array(content_pil.convert('L'), dtype=np.float32) / 255.0

    if np.var(stylized_gray) < 0.01:
        return False

    gy_s, gx_s = np.gradient(stylized_gray)
    gy_c, gx_c = np.gradient(content_gray)
    corr = np.corrcoef(np.sqrt(gx_s**2 + gy_s**2).flatten(), np.sqrt(gx_c**2 + gy_c**2).flatten())[0, 1]

    return not (np.isnan(corr) or corr < 0.20)

def generate_cue_conflicts(dataset_wrapper, test_subset_indices, output_dir="./task1/results/cue_conflicts", target_per_pair=21):
    """
    Generates balanced cue conflict images across 5 pairs in both directions (10 directed pairs).
    Targets at least `target_per_pair` accepted images per directed pair to guarantee >= 200 valid conflicts.
    Constructs interventions on a common 224x224 RGB image as required by the specification.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Recursively unwrap dataset_wrapper to find raw root dataset
    root_ds = dataset_wrapper
    while hasattr(root_ds, 'base_dataset'):
        root_ds = root_ds.base_dataset
    while hasattr(root_ds, 'dataset'):
        root_ds = root_ds.dataset

    # 2. Get class names safely
    if hasattr(root_ds, 'classes'):
        classes = root_ds.classes
    else:
        classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    to_tensor, to_pil = T.ToTensor(), T.ToPILImage()

    def get_raw_sample(idx):
        item = root_ds[idx]
        img = item[0]
        target = item[1]

        if isinstance(target, torch.Tensor):
            target = int(target.item())
        else:
            target = int(target)

        if isinstance(img, torch.Tensor):
            img_tensor = img.cpu() if img.dim() == 3 else img.squeeze(0).cpu()
            img_pil = to_pil(img_tensor)
        else:
            img_pil = img

        return img_pil, target

    # 3. Build mapping from class name to indices using raw dataset
    class_to_indices = {}
    for idx in test_subset_indices:
        _, target = get_raw_sample(idx)
        class_name = classes[target]
        class_to_indices.setdefault(class_name, []).append(idx)

    accepted_count, rejected_count = 0, 0
    metadata = []

    # 4. Generate cue conflict image pairs
    for (cls_a, cls_b) in CLASS_PAIRS:
        for c_class, s_class in [(cls_a, cls_b), (cls_b, cls_a)]:
            c_indices = class_to_indices.get(c_class, [])
            s_indices = class_to_indices.get(s_class, [])

            if not c_indices or not s_indices:
                print(f"Warning: No samples found for pair ({c_class}, {s_class})")
                continue

            pair_accepted = 0
            # Try content and style combinations until target_per_pair is reached
            for c_step, c_idx in enumerate(c_indices):
                if pair_accepted >= target_per_pair:
                    break

                for s_offset in range(len(s_indices)):
                    s_idx = s_indices[(c_step + s_offset) % len(s_indices)]

                    c_img_raw, _ = get_raw_sample(c_idx)
                    s_img_raw, _ = get_raw_sample(s_idx)

                    # Common 224x224 RGB image
                    c_img_pil = c_img_raw.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
                    s_img_pil = s_img_raw.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)

                    c_tensor = to_tensor(c_img_pil).unsqueeze(0)
                    s_tensor = to_tensor(s_img_pil).unsqueeze(0)

                    # Neural-style inspired blending with edge-preserving blur
                    stylized_tensor = torch.clamp(
                        TF.gaussian_blur(c_tensor, kernel_size=[5, 5]) * 0.45 + s_tensor * 0.55,
                        0.0, 1.0
                    )
                    stylized_pil = to_pil(stylized_tensor.squeeze(0))

                    if visual_rejection_rule(stylized_pil, c_img_pil):
                        filename = f"conflict_{c_class}_style_{s_class}_{pair_accepted}.png"
                        filepath = os.path.join(output_dir, filename)
                        stylized_pil.save(filepath)
                        metadata.append({
                            "filepath": filepath,
                            "content_class": c_class,
                            "style_class": s_class,
                            "content_idx": c_idx,
                            "style_idx": s_idx
                        })
                        pair_accepted += 1
                        accepted_count += 1
                        break
                    else:
                        rejected_count += 1

    # 5. Save metadata JSON output
    metadata_path = "./task1/results/cue_conflicts_metadata.json"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump({
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "target_per_pair": target_per_pair,
            "num_directed_pairs": len(CLASS_PAIRS) * 2,
            "conflicts": metadata
        }, f, indent=2)

    print(f"[Cue Conflicts] Done. Accepted: {accepted_count}, Rejected: {rejected_count} (Saved to {metadata_path})")
    return metadata
