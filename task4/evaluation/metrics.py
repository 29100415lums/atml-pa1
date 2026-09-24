import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc(known_scores, unknown_scores):
    """
    Computes the Area Under the Receiver Operating Characteristic curve (AUROC).
    known_scores: novelty scores for known examples (should be lower)
    unknown_scores: novelty scores for unknown examples (should be higher)
    """
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y_true, y_scores)

def get_threshold_at_fpr(known_scores, fpr=0.05):
    """
    Finds the threshold tau such that FPR (False Positive Rate) is at the desired level.
    By default, fpr=0.05 corresponds to the 95th percentile of unknownness on known validation data.
    Reject if score > tau (positive class = unknown).
    FPR = fraction of knowns incorrectly rejected.
    """
    # 95th percentile means 95% of known data is below tau.
    tau = np.percentile(known_scores, (1 - fpr) * 100)
    return tau

def compute_rejection_rates(scores, tau):
    """
    Computes the fraction of examples rejected based on threshold tau.
    Reject if score > tau.
    """
    rejected = np.sum(scores > tau)
    return rejected / len(scores)
