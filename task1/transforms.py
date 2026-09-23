"""
transforms.py
"""
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

class ImageNormalizer:
    def __init__(self, mean, std):
        self.norm = T.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        return self.norm(tensor)

def ensure_rgb_224(pil_img):
    """Ensures input PIL image is RGB and resized to standard 224x224."""
    if isinstance(pil_img, Image.Image):
        img = pil_img.convert('RGB')
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.Resampling.BILINEAR)
        return img
    return pil_img

def apply_grayscale(pil_img):
    """Converts image to luminance grayscale and maps back to 3-channel RGB."""
    img = ensure_rgb_224(pil_img)
    return img.convert('L').convert('RGB')

def apply_hue_rotation(pil_img, hue_factor=0.5):
    """Shifts the hue channel by hue_factor (0.5 = 180 degrees in HSV space)."""
    img = ensure_rgb_224(pil_img)
    return TF.adjust_hue(img, hue_factor)

def apply_translation(pil_img, dx, dy):
    """
    Translates image by (dx, dy) pixels using reflection padding followed by shifted crop.
    dx > 0 shifts right, dx < 0 shifts left.
    dy > 0 shifts down, dy < 0 shifts up.
    """
    img = ensure_rgb_224(pil_img)
    w, h = img.size
    pad_x, pad_y = abs(dx), abs(dy)
    if pad_x == 0 and pad_y == 0:
        return img
    padded = TF.pad(img, padding=(pad_x, pad_y), padding_mode='reflect')
    return TF.crop(padded, top=pad_y - dy, left=pad_x - dx, height=h, width=w)

def get_patch_shuffled_image(pil_img, grid_size=4, seed=6304):
    """
    Divides image into grid_size x grid_size patches and applies a non-identity permutation.
    Preserves local features while destroying global geometry.
    """
    img = ensure_rgb_224(pil_img)
    w, h = img.size
    patch_w, patch_h = w // grid_size, h // grid_size

    patches = [
        img.crop((j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h))
        for i in range(grid_size) for j in range(grid_size)
    ]

    num_patches = grid_size * grid_size
    rng = np.random.RandomState(seed)

    perm = list(range(num_patches))
    while True:
        rng.shuffle(perm)
        if any(perm[i] != i for i in range(num_patches)):
            break

    shuffled_img = Image.new('RGB', (w, h))
    for idx, orig_idx in enumerate(perm):
        i, j = idx // grid_size, idx % grid_size
        shuffled_img.paste(patches[orig_idx], (j * patch_w, i * patch_h))

    return shuffled_img
