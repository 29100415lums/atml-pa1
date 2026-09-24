import torch

def get_proser_score(logits, num_known=10):
    """
    PROSER placeholder-based detection score.
    Higher means more likely to be unknown.
    A common formulation is the maximum response among the dummy classes
    minus the maximum response among the known classes, or simply the maximum 
    probability of a dummy class.
    
    Here we use the difference between the max dummy logit and max known logit.
    u_PROSER(x) = max(z_dummy) - max(z_known)
    """
    known_logits = logits[:, :num_known]
    dummy_logits = logits[:, num_known:]
    
    max_known, _ = known_logits.max(dim=1)
    max_dummy, _ = dummy_logits.max(dim=1)
    
    return max_dummy - max_known
