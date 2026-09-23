"""
task2/evaluation/class_analysis.py
Per-class transfer analysis and confusion matrix plots.
Identifies classes with greatest positive and negative transfer.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']


def compute_class_delta(source_only_per_class, method_per_class):
    """
    Computes per-class accuracy change: delta[cls] = method_acc[cls] - source_only_acc[cls].
    Positive = improvement, Negative = degradation.
    """
    delta = {}
    for cls in PACS_CLASSES:
        delta[cls] = method_per_class.get(cls, 0.0) - source_only_per_class.get(cls, 0.0)
    return delta


def plot_class_transfer_analysis(all_deltas, methods, save_path, class_names=PACS_CLASSES):
    """
    Bar chart comparing per-class accuracy change (Delta Acc vs Source-Only) across methods.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = np.arange(len(class_names))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)
    for i, (method, delta) in enumerate(zip(methods, all_deltas)):
        vals = [delta.get(c, 0.0) * 100 for c in class_names]
        bars = ax.bar(x + i * width - width, vals, width, label=method,
                      color=colors[i % len(colors)], edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("Per-Class Accuracy Change Relative to Source-Only Baseline\n(Target: Sketch Domain)",
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel("ΔAccuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in class_names], fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved class transfer analysis to {save_path}")


def plot_confusion_matrix(y_true, y_pred, method_name, save_path, class_names=PACS_CLASSES):
    """Plots a normalized confusion matrix for a given method."""
    from evaluation.metrics import compute_confusion_matrix
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = compute_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[c.capitalize() for c in class_names],
                yticklabels=[c.capitalize() for c in class_names],
                ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_title(f"Normalized Confusion Matrix — {method_name.upper()} (Target: Sketch)",
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved confusion matrix for {method_name} to {save_path}")
