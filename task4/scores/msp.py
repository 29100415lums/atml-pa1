import torch
import torch.nn.functional as F

def get_msp_score(logits):
    """
    MSP (Maximum Softmax Probability)
    u_MSP(x) = 1 - max_k p_k(x)
    """
    probs = F.softmax(logits, dim=1)
    max_probs, _ = probs.max(dim=1)
    return 1.0 - max_probs
