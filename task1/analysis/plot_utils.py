"""
analysis/plot_utils.py
Dedicated publication-quality plotting utilities matching all required evidence in the PA1 specification.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

def plot_translation_curves(translation_data, save_path="./task1/results/plots/translation_curves.png"):
    """
    Plots Top-1 Accuracy and Prediction Consistency against displacement delta in [0, 8, 16, 32].
    Matches requirement on PDF Page 3 & 4.
    """
    set_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    deltas = [0, 8, 16, 32]
    model_styles = {
        'resnet50': {'label': 'ResNet-50', 'color': '#1f77b4', 'marker': 'o'},
        'vit_b_16': {'label': 'ViT-B/16', 'color': '#ff7f0e', 'marker': 's'},
        'clip_vit_b32': {'label': 'CLIP ViT-B/32', 'color': '#2ca02c', 'marker': '^'}
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel 1: Top-1 Accuracy vs Displacement
    ax1 = axes[0]
    for model_name, style in model_styles.items():
        if model_name in translation_data:
            accs = [translation_data[model_name][str(d)]['accuracy'] * 100 for d in deltas]
            ax1.plot(deltas, accs, label=style['label'], color=style['color'],
                     marker=style['marker'], linewidth=2.2, markersize=7)

    ax1.set_title("Top-1 Accuracy vs. Spatial Displacement", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Displacement $\\delta$ (pixels)", fontsize=11)
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax1.set_xticks(deltas)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, fontsize=10)

    # Panel 2: Prediction Consistency vs Displacement
    ax2 = axes[1]
    for model_name, style in model_styles.items():
        if model_name in translation_data:
            cons = [translation_data[model_name][str(d)]['consistency'] * 100 for d in deltas]
            ax2.plot(deltas, cons, label=style['label'], color=style['color'],
                     marker=style['marker'], linewidth=2.2, markersize=7)

    ax2.set_title("Prediction Consistency vs. Spatial Displacement", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Displacement $\\delta$ (pixels)", fontsize=11)
    ax2.set_ylabel("Consistency relative to $\\delta=0$ (%)", fontsize=11)
    ax2.set_xticks(deltas)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved translation sensitivity curves to {save_path}")

def plot_interventions_benchmark(results, save_path="./task1/results/plots/interventions_benchmark.png"):
    """
    Bar chart comparing Clean, Grayscale, Hue Rotation, and Patch Shuffle performance across backbones.
    Matches PDF Page 4 Required Evidence #1.
    """
    set_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    conditions = ['Clean', 'Grayscale', 'Hue Rotation', 'Patch Shuffle']
    models = ['resnet50', 'vit_b_16', 'clip_vit_b32']
    model_labels = ['ResNet-50', 'ViT-B/16', 'CLIP ViT-B/32']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    x = np.arange(len(conditions))
    width = 0.25

    # Panel 1: Accuracy comparison
    ax1 = axes[0]
    for idx, (m, label, color) in enumerate(zip(models, model_labels, colors)):
        accs = [
            results.get('clean_baseline', {}).get(m, {}).get('accuracy', 0) * 100,
            results.get('color_bias', {}).get('grayscale', {}).get(m, {}).get('accuracy', 0) * 100,
            results.get('color_bias', {}).get('hue_rotation', {}).get(m, {}).get('accuracy', 0) * 100,
            results.get('patch_shuffle', {}).get(m, {}).get('accuracy', 0) * 100
        ]
        ax1.bar(x + idx * width - width, accs, width, label=label, color=color, alpha=0.88, edgecolor='black', linewidth=0.6)

    ax1.set_title("Top-1 Accuracy Across Controlled Interventions", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(frameon=True, fontsize=10)

    # Panel 2: Prediction Consistency
    ax2 = axes[1]
    for idx, (m, label, color) in enumerate(zip(models, model_labels, colors)):
        cons = [
            100.0,
            results.get('color_bias', {}).get('grayscale', {}).get(m, {}).get('consistency', 0) * 100,
            results.get('color_bias', {}).get('hue_rotation', {}).get(m, {}).get('consistency', 0) * 100,
            results.get('patch_shuffle', {}).get(m, {}).get('consistency', 0) * 100
        ]
        ax2.bar(x + idx * width - width, cons, width, label=label, color=color, alpha=0.88, edgecolor='black', linewidth=0.6)

    ax2.set_title("Prediction Consistency Relative to Clean Inputs", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Prediction Consistency (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved interventions benchmark plot to {save_path}")

def plot_shape_bias_analysis(cue_conflict_results, save_path="./task1/results/plots/shape_bias_analysis.png"):
    """
    Plots Shape Bias (%), Coverage (%), and decision count breakdown (Shape, Texture, Other).
    Matches PDF Page 4 Required Evidence #2.
    """
    set_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = ['resnet50', 'vit_b_16', 'clip_vit_b32']
    model_labels = ['ResNet-50', 'ViT-B/16', 'CLIP ViT-B/32']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel 1: Shape Bias % and Coverage %
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.35

    shape_biases = [cue_conflict_results.get(m, {}).get('shape_bias_percent', 0) for m in models]
    coverages = [cue_conflict_results.get(m, {}).get('coverage_percent', 0) for m in models]

    b1 = ax1.bar(x - width/2, shape_biases, width, label='Shape Bias (%)', color='#3498db', edgecolor='black', linewidth=0.6)
    b2 = ax1.bar(x + width/2, coverages, width, label='Coverage (%)', color='#9b59b6', edgecolor='black', linewidth=0.6)

    for bar in b1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    for bar in b2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)

    ax1.set_title("Shape Bias (%) & Coverage (%)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax1.legend(frameon=True, fontsize=10)

    # Panel 2: Absolute Decision Breakdown
    ax2 = axes[1]
    n_shape = np.array([cue_conflict_results.get(m, {}).get('N_shape', 0) for m in models])
    n_texture = np.array([cue_conflict_results.get(m, {}).get('N_texture', 0) for m in models])
    n_other = np.array([cue_conflict_results.get(m, {}).get('N_other', 0) for m in models])

    p1 = ax2.bar(model_labels, n_shape, label='Shape Decisions', color='#2ecc71', edgecolor='black', linewidth=0.6)
    p2 = ax2.bar(model_labels, n_texture, bottom=n_shape, label='Texture Decisions', color='#e67e22', edgecolor='black', linewidth=0.6)
    p3 = ax2.bar(model_labels, n_other, bottom=n_shape + n_texture, label='Other Decisions', color='#95a5a6', edgecolor='black', linewidth=0.6)

    ax2.set_title("Decision Breakdown on Cue-Conflict Images", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Number of Images", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved shape bias analysis plot to {save_path}")

def plot_cosine_stability_summary(stability_results, save_path="./task1/results/plots/representation_cosine_stability.png"):
    """
    Bar plot comparing Cosine Stability I_T across all interventions.
    Matches PDF Page 4 Required Evidence #4.
    """
    set_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    interventions = ['grayscale', 'hue_rotation', 'translation', 'patch_shuffle', 'cue_conflict']
    intervention_labels = ['Grayscale', 'Hue Rotation', 'Translation (32px)', 'Patch Shuffle', 'Cue Conflict']
    models = ['resnet50', 'vit_b_16', 'clip_vit_b32']
    model_labels = ['ResNet-50', 'ViT-B/16', 'CLIP ViT-B/32']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    x = np.arange(len(interventions))
    width = 0.25

    for idx, (m, label, color) in enumerate(zip(models, model_labels, colors)):
        means = []
        stds = []
        for itv in interventions:
            stat = stability_results.get(itv, {}).get(m, {})
            means.append(stat.get('cosine_stability_mean', 0.0))
            stds.append(stat.get('cosine_stability_std', 0.0))
        ax.bar(x + idx * width - width, means, width, yerr=stds, capsize=4,
               label=label, color=color, alpha=0.88, edgecolor='black', linewidth=0.6)

    ax.set_title("Representation Stability: Mean Cosine Similarity ($I_T$) Between Clean & Perturbed Features",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Cosine Similarity ($I_T$)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(intervention_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved cosine stability summary plot to {save_path}")

def plot_cue_conflict_examples(conflict_metadata, model_preds, classes, save_path="./task1/results/plots/cue_conflict_examples.png", num_examples=6):
    """
    Renders sample cue-conflict images with their ground-truth content/shape, style/texture,
    and model predictions. Matches PDF Page 4 Required Evidence #5.
    """
    set_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not conflict_metadata:
        return

    # Select representative samples
    step = max(1, len(conflict_metadata) // num_examples)
    samples = conflict_metadata[::step][:num_examples]

    cols = min(3, len(samples))
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.5 * rows), dpi=250)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (item, ax) in enumerate(zip(samples, axes)):
        if os.path.exists(item['filepath']):
            img = Image.open(item['filepath']).convert('RGB')
            ax.imshow(img)
        ax.axis('off')

        c_cls = item['content_class']
        s_cls = item['style_class']

        res_p = classes[model_preds['resnet50'][idx]] if idx < len(model_preds.get('resnet50', [])) else 'N/A'
        vit_p = classes[model_preds['vit_b_16'][idx]] if idx < len(model_preds.get('vit_b_16', [])) else 'N/A'
        clip_p = classes[model_preds['clip_vit_b32'][idx]] if idx < len(model_preds.get('clip_vit_b32', [])) else 'N/A'

        title_text = (f"Content (Shape): {c_cls.upper()}\n"
                      f"Style (Texture): {s_cls.upper()}\n"
                      f"ResNet: {res_p} | ViT: {vit_p} | CLIP: {clip_p}")
        ax.set_title(title_text, fontsize=9, pad=6)

    for j in range(len(samples), len(axes)):
        axes[j].axis('off')

    plt.suptitle("Representative Cue-Conflict Examples & Model Predictions", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved cue conflict examples grid to {save_path}")
