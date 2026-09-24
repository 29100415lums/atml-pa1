import torch

def get_energy_score(logits):
    """
    Energy Score
    u_Energy(x) = -log(sum(exp(z_k(x))))
    """
    return -torch.logsumexp(logits, dim=1)
