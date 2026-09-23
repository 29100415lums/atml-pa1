"""
analysis/evaluate_bias.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(y_true, logits_or_probs):
    # Ensure y_true is a numpy array
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Process logits / probabilities
    if isinstance(logits_or_probs, torch.Tensor):
        probs = F.softmax(logits_or_probs, dim=-1).detach().cpu().numpy()
    else:
        probs = np.array(logits_or_probs)

    preds = np.argmax(probs, axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average='macro')),
        "mean_max_confidence": float(np.mean(np.max(probs, axis=1))),
        "predictions": preds.tolist()
    }

def compute_prediction_consistency(clean_preds, transformed_preds):
    # Ensure inputs are converted properly if they are tensors
    if isinstance(clean_preds, torch.Tensor):
        clean_preds = clean_preds.detach().cpu().numpy()
    if isinstance(transformed_preds, torch.Tensor):
        transformed_preds = transformed_preds.detach().cpu().numpy()

    clean_arr = np.array(clean_preds)
    trans_arr = np.array(transformed_preds)

    return float(np.mean(clean_arr == trans_arr))

def evaluate_cue_conflicts(model_predict_fn, conflict_metadata, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_shape, n_texture, n_other = 0, 0, 0

    for item in conflict_metadata:
        shape_label = class_to_idx[item["content_class"]]
        texture_label = class_to_idx[item["style_class"]]
        pred_class, _ = model_predict_fn(item["filepath"])

        # Ensure pred_class is a standard Python int
        if isinstance(pred_class, torch.Tensor):
            pred_class = int(pred_class.detach().cpu().item())
        elif isinstance(pred_class, str) and pred_class in class_to_idx:
            pred_class = class_to_idx[pred_class]
        else:
            pred_class = int(pred_class)

        if pred_class == shape_label:
            n_shape += 1
        elif pred_class == texture_label:
            n_texture += 1
        else:
            n_other += 1

    n_total = len(conflict_metadata)
    denom = n_shape + n_texture
    return {
        "N_shape": n_shape, "N_texture": n_texture, "N_other": n_other, "N_total": n_total,
        "shape_bias_percent": float((n_shape / denom * 100.0) if denom > 0 else 0.0),
        "coverage_percent": float(((n_shape + n_texture) / n_total * 100.0) if n_total > 0 else 0.0)
    }
