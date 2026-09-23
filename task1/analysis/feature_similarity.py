"""
analysis/feature_similarity.py
"""
import torch
import torch.nn.functional as F

def compute_cosine_stability(clean_features, transformed_features):
    clean_norm = F.normalize(torch.tensor(clean_features, dtype=torch.float32), p=2, dim=-1)
    trans_norm = F.normalize(torch.tensor(transformed_features, dtype=torch.float32), p=2, dim=-1)
    cosine_sims = torch.sum(clean_norm * trans_norm, dim=-1)

    return {
        "cosine_stability_mean": float(torch.mean(cosine_sims).item()),
        "cosine_stability_std": float(torch.std(cosine_sims).item())
    }
