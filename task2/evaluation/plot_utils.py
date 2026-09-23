"""
task2/evaluation/plot_utils.py
Publication-quality plot generators for all required evidence in Task 2.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
METHODS = ['source_only', 'dan', 'dann', 'cdan']
METHOD_LABELS = {'source_only': 'Source-Only', 'dan': 'DAN', 'dann': 'DANN', 'cdan': 'CDAN'}
COLORS = {'source_only': '#7f7f7f', 'dan': '#1f77b4', 'dann': '#ff7f0e', 'cdan': '#2ca02c'}


def plot_training_curves(histories, save_dir):
    """
    Plots training loss curves (cls loss + domain/alignment loss) for all methods.
    """
    os.makedirs(save_dir, exist_ok=True)

    for method, hist in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

        # Classification loss
        ax1 = axes[0]
        if 'train_cls_loss' in hist:
            ax1.plot(hist['train_cls_loss'], label='Cls Loss', color='#1f77b4', linewidth=2)
        ax1.set_title(f"{METHOD_LABELS.get(method, method)} — Classification Loss", fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        # Domain/Alignment loss
        ax2 = axes[1]
        if 'train_mmd_loss' in hist:
            ax2.plot(hist['train_mmd_loss'], label='MMD Loss', color='#ff7f0e', linewidth=2)
        elif 'train_dom_loss' in hist:
            ax2.plot(hist['train_dom_loss'], label='Domain Loss', color='#ff7f0e', linewidth=2)
        else:
            ax2.text(0.5, 0.5, 'N/A (Source-Only)', ha='center', va='center', transform=ax2.transAxes, fontsize=12)

        ax2.plot(hist.get('mean_val_f1', []), label='Mean Val F1', color='#2ca02c',
                 linewidth=2, linestyle='--')
        ax2.set_title(f"{METHOD_LABELS.get(method, method)} — Domain/Alignment Loss & Val F1", fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Value")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{method}_training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Saved {method} training curves to {save_path}")


def plot_results_comparison(final_results, save_path):
    """
    Grouped bar chart comparing accuracy across methods on each source domain and target.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    domain_labels = ['Art Painting', 'Cartoon', 'Photo', 'Target (Sketch)']
    methods_present = [m for m in METHODS if m in final_results]

    x = np.arange(len(domains))
    width = 0.20
    n = len(methods_present)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for ax_idx, metric in enumerate(['accuracy', 'macro_f1']):
        ax = axes[ax_idx]
        for i, method in enumerate(methods_present):
            vals = []
            for dom in domains:
                res = final_results.get(method, {}).get(dom, {})
                vals.append(res.get(metric, 0.0) * 100)
            bars = ax.bar(x + offsets[i], vals, width, label=METHOD_LABELS.get(method, method),
                          color=COLORS.get(method, 'gray'), edgecolor='black', linewidth=0.5, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_title(f"{'Accuracy' if metric == 'accuracy' else 'Macro-F1'} by Domain",
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(f"{'Accuracy (%)' if metric == 'accuracy' else 'Macro-F1 (%)'}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(domain_labels, fontsize=10, rotation=10)
        ax.set_ylim(0, 115)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved results comparison chart to {save_path}")


def plot_domain_separability(separability_scores, save_path):
    """
    Bar chart comparing Domain Separability Scores across methods.
    Annotates 50% chance level.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    methods_present = [m for m in METHODS if m in separability_scores]
    scores = [separability_scores[m] * 100 for m in methods_present]
    labels = [METHOD_LABELS.get(m, m) for m in methods_present]
    colors = [COLORS.get(m, 'gray') for m in methods_present]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(labels, scores, color=colors, edgecolor='black', linewidth=0.6, alpha=0.88)
    ax.axhline(50, color='red', linewidth=1.5, linestyle='--', label='Chance Level (50%)')

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{s:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title("Domain Separability Score (Source vs. Target)\n100% = Completely Separable | 50% = Ideal Alignment",
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel("Logistic Regression Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved domain separability plot to {save_path}")


def plot_controlled_study(study_results, param_name, param_values, save_path):
    """
    Plots controlled design study results (e.g., lambda_MMD) vs accuracy, F1, and separability.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = [str(v) for v in param_values]

    target_accs = [study_results[v].get('sketch', {}).get('accuracy', 0) * 100 for v in param_values]
    mean_src_f1s = [study_results[v].get('mean_source_f1', 0) * 100 for v in param_values]
    sep_scores = [study_results[v].get('domain_separability', 0) * 100 for v in param_values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for ax, vals, title, ylabel, color in zip(
        axes,
        [target_accs, mean_src_f1s, sep_scores],
        ['Target Accuracy vs Alignment Strength',
         'Mean Source F1 vs Alignment Strength',
         'Domain Separability vs Alignment Strength'],
        ['Target Acc (%)', 'Mean Source F1 (%)', 'Separability (%)'],
        ['#1f77b4', '#ff7f0e', '#2ca02c']
    ):
        ax.plot(x, vals, marker='o', linewidth=2.2, markersize=8, color=color)
        for xi, v in zip(x, vals):
            ax.annotate(f'{v:.1f}', xy=(xi, v), xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(f"Controlled Design Study: {param_name}", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved controlled study plot to {save_path}")
