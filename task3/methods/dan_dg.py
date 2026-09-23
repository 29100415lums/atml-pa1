import os
import sys
import torch
import torch.nn as nn

def mmd_loss(source_feats_1, source_feats_2, sigmas=(0.5, 1.0, 2.0)):
    combined = torch.cat([source_feats_1, source_feats_2], dim=0)
    dists = torch.cdist(combined, combined, p=2).pow(2)
    mask = dists > 0
    median_dist = dists[mask].median().item() if mask.any() else 1.0
    if median_dist < 1e-8:
        median_dist = 1.0

    def rbf_kernel_matrix(X, Y):
        dxy = torch.cdist(X, Y, p=2).pow(2)
        K = sum(torch.exp(-dxy / (2.0 * s * median_dist)) for s in sigmas)
        return K

    n = source_feats_1.shape[0]
    m = source_feats_2.shape[0]

    K_ss = rbf_kernel_matrix(source_feats_1, source_feats_1)
    K_tt = rbf_kernel_matrix(source_feats_2, source_feats_2)
    K_st = rbf_kernel_matrix(source_feats_1, source_feats_2)

    mmd = (K_ss.sum() / (n * n) + K_tt.sum() / (m * m) - 2.0 * K_st.sum() / (n * m))
    return mmd
