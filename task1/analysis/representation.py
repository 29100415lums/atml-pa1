"""
analysis/representation.py
"""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.manifold import TSNE

SEED = 6304

def visualize_combined_representation(clean_feats, transformed_feats, labels, class_names, backbone_name, intervention_name, save_dir="./task1/results/plots"):
    """
    Fits a single 2D t-SNE projection to the combined clean and transformed features.
    Renders publication-ready plots with complete legends for classes and condition markers.
    """
    os.makedirs(save_dir, exist_ok=True)
    N_clean = len(clean_feats)
    N_trans = len(transformed_feats)
    combined = np.concatenate([clean_feats, transformed_feats], axis=0)

    # Fit t-SNE on combined space
    embeds_2d = TSNE(
        n_components=2,
        random_state=SEED,
        perplexity=min(30, max(5, (N_clean + N_trans) // 10)),
        learning_rate='auto',
        init='pca'
    ).fit_transform(combined)

    clean_2d = embeds_2d[:N_clean]
    trans_2d = embeds_2d[N_clean:]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cmap = plt.colormaps.get_cmap("tab10")

    # Plot per-class scatter
    for c_idx in range(len(class_names)):
        mask_clean = (labels[:N_clean] == c_idx)
        if np.any(mask_clean):
            ax.scatter(
                clean_2d[mask_clean, 0], clean_2d[mask_clean, 1],
                color=cmap(c_idx), marker='o', s=35, alpha=0.75,
                edgecolors='black', linewidths=0.6
            )

        mask_trans = (labels[:N_trans] == c_idx) if len(labels) >= N_trans else (labels == c_idx)
        if np.any(mask_trans):
            ax.scatter(
                trans_2d[mask_trans, 0], trans_2d[mask_trans, 1],
                color=cmap(c_idx), marker='^', s=45, alpha=0.75,
                edgecolors='red', linewidths=0.8
            )

    # Class legend elements
    class_handles = [
        Line2D([0], [0], marker='s', color='w', label=cls_name,
               markerfacecolor=cmap(i), markersize=8)
        for i, cls_name in enumerate(class_names)
    ]

    # Condition marker legend elements
    condition_handles = [
        Line2D([0], [0], marker='o', color='w', label='Clean Reference',
               markerfacecolor='gray', markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='^', color='w', label=f'Perturbed ({intervention_name})',
               markerfacecolor='gray', markeredgecolor='red', markersize=9)
    ]

    leg_classes = ax.legend(handles=class_handles, title="Classes", loc='upper left',
                            bbox_to_anchor=(1.02, 1), frameon=True, fontsize=9, title_fontsize=10)
    ax.add_artist(leg_classes)
    ax.legend(handles=condition_handles, title="Condition", loc='upper left',
              bbox_to_anchor=(1.02, 0.45), frameon=True, fontsize=9, title_fontsize=10)

    clean_name = backbone_name.replace('_', '-').upper()
    ax.set_title(f"t-SNE Feature Geometry: {clean_name} under {intervention_name.replace('_', ' ').title()}", fontsize=13, pad=12, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_filename = os.path.join(save_dir, f"{backbone_name}_{intervention_name}_tsne.png")
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Saved t-SNE visualization to {out_filename}")
