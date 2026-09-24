import torch

def get_mls_score(logits):
    """
    MLS (Maximum Logit Score)
    u_MLS(x) = -max_k z_k(x)
    """
    max_logits, _ = logits.max(dim=1)
    return -max_logits
