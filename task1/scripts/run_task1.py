"""
scripts/run_task1.py
Complete PA1 Task 1 Pipeline:
1. Clean Baselines & Classifier Head Probing (ResNet-50, ViT-B/16, CLIP ViT-B/32, CLIP Zero-Shot)
2. Color Bias (Grayscale, Hue Rotation)
3. Shape vs. Texture Bias (Cue Conflicts >= 200 valid images)
4. Translation Sensitivity (Displacements 0, 8, 16, 32 across 4 cardinal directions)
5. Patch Shuffling (4x4 spatial scrambling)
6. Representation Stability (Cosine similarity across all interventions & combined t-SNE)
7. Full Evidence Plotting (Translation curves, benchmark bar charts, shape bias, cue-conflict examples)
"""
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Ensure working directory is project root and task1 is on Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(TASK1_ROOT, ".."))

os.chdir(PROJECT_ROOT)
print(f"Working directory automatically set to: {PROJECT_ROOT}")

if TASK1_ROOT not in sys.path:
    sys.path.insert(0, TASK1_ROOT)

from make_subset import prepare_stl10_splits, BaseDatasetWrapper, SEED, set_seed
from models.backbones import FrozenBackbone, train_classifier_head, zero_shot_clip_predict
try:
    from models.backbones import ClassifierHead
except ImportError:
    ClassifierHead = lambda in_dim, out_dim: nn.Linear(in_dim, out_dim)

from transforms import (
    ImageNormalizer, ensure_rgb_224, apply_grayscale,
    apply_hue_rotation, apply_translation, get_patch_shuffled_image
)
from make_cue_conflicts import generate_cue_conflicts
from analysis.evaluate_bias import compute_metrics, compute_prediction_consistency, evaluate_cue_conflicts
from analysis.feature_similarity import compute_cosine_stability
from analysis.representation import visualize_combined_representation
from analysis.plot_utils import (
    plot_translation_curves, plot_interventions_benchmark,
    plot_shape_bias_analysis, plot_cosine_stability_summary,
    plot_cue_conflict_examples
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_features_and_logits(backbone, head, pil_images, norm, to_tensor, device=DEVICE, batch_size=64):
    """Batched forward pass extracting features and logits without GPU OOM."""
    feats_list, logits_list = [], []
    for i in range(0, len(pil_images), batch_size):
        batch = [norm(to_tensor(img)) for img in pil_images[i:i + batch_size]]
        batch_tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            f = backbone(batch_tensor)
            l = head(f) if head is not None else f
        feats_list.append(f.cpu())
        logits_list.append(l.cpu())
    return torch.cat(feats_list, dim=0), torch.cat(logits_list, dim=0)

def main():
    set_seed(SEED)
    print(f"============================================================")
    print(f"Executing Full PA1 Task 1 Pipeline on device: {DEVICE}")
    print(f"============================================================")

    # Directories
    ckpt_dir = "./task1/results/checkpoints"
    plots_dir = "./task1/results/plots"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Dataset splits
    split_info = prepare_stl10_splits(data_dir="./task1/data", seed=SEED)
    classes = split_info["classes"]

    raw_train = torchvision.datasets.STL10(root="./task1/data", split='train', download=False)
    raw_test = torchvision.datasets.STL10(root="./task1/data", split='test', download=False)

    train_subset = Subset(raw_train, split_info["train_indices"])
    val_subset = Subset(raw_train, split_info["val_indices"])
    test_eval_subset = Subset(raw_test, split_info["test_subset_indices"])

    test_wrapper = BaseDatasetWrapper(test_eval_subset)
    test_labels = np.array([test_wrapper[i][1] for i in range(len(test_wrapper))])

    # Convert all 500 test images to common 224x224 RGB images as required
    clean_test_pils = [ensure_rgb_224(test_wrapper[i][0]) for i in range(len(test_wrapper))]

    # Load frozen backbones
    backbones = {
        'resnet50': FrozenBackbone('resnet50', device=DEVICE).to(DEVICE),
        'vit_b_16': FrozenBackbone('vit_b_16', device=DEVICE).to(DEVICE),
        'clip_vit_b32': FrozenBackbone('clip_vit_b32', device=DEVICE).to(DEVICE)
    }

    heads = {}
    to_tensor = T.ToTensor()

    # -------------------------------------------------------------
    # STEP 1: Train or Load Classifier Heads & Clean Baseline
    # -------------------------------------------------------------
    print("\n[Step 1/6] Training or Loading Classifier Heads & Clean Baselines...")
    for model_name, backbone in backbones.items():
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_head.pt")

        if os.path.exists(ckpt_path):
            print(f"  -> Found saved checkpoint for {model_name}. Loading weights...")
            head = ClassifierHead(backbone.embed_dim, len(classes)).to(DEVICE)
            head.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            head.eval()
            heads[model_name] = head
        else:
            print(f"  -> No checkpoint found for {model_name}. Training classifier head...")
            norm = ImageNormalizer(backbone.mean, backbone.std)
            transform_fn = lambda img: norm(to_tensor(ensure_rgb_224(img)))

            train_ds = BaseDatasetWrapper(train_subset, transform=transform_fn)
            val_ds = BaseDatasetWrapper(val_subset, transform=transform_fn)

            head, best_val_acc = train_classifier_head(
                backbone, DataLoader(train_ds, batch_size=64, shuffle=True),
                DataLoader(val_ds, batch_size=64, shuffle=False),
                num_classes=len(classes), epochs=50, lr=1e-3, weight_decay=1e-4, patience=5, device=DEVICE
            )
            torch.save(head.state_dict(), ckpt_path)
            heads[model_name] = head
            print(f"  -> {model_name} Val Acc: {best_val_acc:.4f} (Saved to {ckpt_path})")

    results = {
        "clean_baseline": {},
        "color_bias": {},
        "cue_conflict": {},
        "translation": {},
        "patch_shuffle": {},
        "representation_stability": {}
    }
    clean_preds_dict, clean_feats_dict = {}, {}

    for model_name in backbones:
        norm = ImageNormalizer(backbones[model_name].mean, backbones[model_name].std)
        feats, logits = extract_features_and_logits(backbones[model_name], heads[model_name], clean_test_pils, norm, to_tensor)
        m = compute_metrics(test_labels, logits)
        results["clean_baseline"][model_name] = m
        clean_preds_dict[model_name] = m["predictions"]
        clean_feats_dict[model_name] = feats.numpy()
        print(f"  -> {model_name} Clean Top-1 Acc: {m['accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f} | Conf: {m['mean_max_confidence']:.4f}")

    # CLIP Zero-Shot Baseline
    norm_clip = ImageNormalizer(backbones['clip_vit_b32'].mean, backbones['clip_vit_b32'].std)
    clip_tensors = torch.stack([norm_clip(to_tensor(img)) for img in clean_test_pils])
    _, clip_probs = zero_shot_clip_predict(backbones['clip_vit_b32'], clip_tensors, classes, device=DEVICE)
    m_zs = compute_metrics(test_labels, clip_probs)
    results["clean_baseline"]["clip_zero_shot"] = m_zs
    print(f"  -> CLIP Zero-Shot Top-1 Acc: {m_zs['accuracy']:.4f} | Macro-F1: {m_zs['macro_f1']:.4f} | Conf: {m_zs['mean_max_confidence']:.4f}")

    # -------------------------------------------------------------
    # STEP 2: Color Bias Interventions (Grayscale & Hue Rotation)
    # -------------------------------------------------------------
    print("\n[Step 2/6] Evaluating Color Bias Interventions...")
    gray_feats_dict = {}
    hue_feats_dict = {}

    for color_name, color_fn in [("grayscale", apply_grayscale), ("hue_rotation", apply_hue_rotation)]:
        print(f"  -> Processing {color_name}...")
        tf_pils = [color_fn(img) for img in clean_test_pils]

        for model_name in backbones:
            norm = ImageNormalizer(backbones[model_name].mean, backbones[model_name].std)
            feats, logits = extract_features_and_logits(backbones[model_name], heads[model_name], tf_pils, norm, to_tensor)
            m = compute_metrics(test_labels, logits)
            m["consistency"] = compute_prediction_consistency(clean_preds_dict[model_name], m["predictions"])
            results["color_bias"].setdefault(color_name, {})[model_name] = m

            if color_name == "grayscale":
                gray_feats_dict[model_name] = feats.numpy()
            else:
                hue_feats_dict[model_name] = feats.numpy()

            print(f"     [{model_name}] Acc: {m['accuracy']:.4f} | Consistency: {m['consistency']:.4f}")

    # -------------------------------------------------------------
    # STEP 3: Shape vs. Texture Bias (Cue Conflicts >= 200)
    # -------------------------------------------------------------
    print("\n[Step 3/6] Generating & Evaluating Cue Conflicts (Target >= 200 Valid Images)...")
    conflict_metadata = generate_cue_conflicts(test_wrapper, split_info["test_subset_indices"], target_per_pair=21)
    print(f"  -> Total accepted cue-conflict images: {len(conflict_metadata)}")

    cue_conflict_preds = {}
    cue_feats_dict = {}

    for model_name in backbones:
        norm = ImageNormalizer(backbones[model_name].mean, backbones[model_name].std)
        conf_pils = [Image.open(item["filepath"]).convert('RGB') for item in conflict_metadata]
        feats, logits = extract_features_and_logits(backbones[model_name], heads[model_name], conf_pils, norm, to_tensor)
        cue_feats_dict[model_name] = feats.numpy()

        preds = logits.argmax(dim=-1).tolist()
        cue_conflict_preds[model_name] = preds

        def predict_fn(path, _m=model_name, _norm=norm):
            img_t = _norm(to_tensor(Image.open(path).convert('RGB'))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                l = heads[_m](backbones[_m](img_t))
            return l.argmax(dim=-1).item(), torch.softmax(l, dim=-1).max().item()

        eval_res = evaluate_cue_conflicts(predict_fn, conflict_metadata, classes)
        results["cue_conflict"][model_name] = eval_res
        print(f"  -> {model_name}: Shape Bias: {eval_res['shape_bias_percent']:.2f}% | Coverage: {eval_res['coverage_percent']:.2f}% (Shape: {eval_res['N_shape']}, Texture: {eval_res['N_texture']}, Other: {eval_res['N_other']})")

    # -------------------------------------------------------------
    # STEP 4: Translation Sensitivity (Displacements 0, 8, 16, 32 px)
    # -------------------------------------------------------------
    print("\n[Step 4/6] Evaluating Translation Sensitivity...")
    deltas = [0, 8, 16, 32]
    trans32_feats_dict = {}

    for model_name in backbones:
        norm = ImageNormalizer(backbones[model_name].mean, backbones[model_name].std)
        results["translation"][model_name] = {}

        # Delta = 0 (Clean baseline)
        results["translation"][model_name]["0"] = {
            "accuracy": results["clean_baseline"][model_name]["accuracy"],
            "consistency": 1.0
        }

        for delta in [8, 16, 32]:
            directions = [
                ("up", 0, -delta),
                ("down", 0, delta),
                ("left", -delta, 0),
                ("right", delta, 0)
            ]
            dir_accs = []
            dir_cons = []
            sample_feats = None

            for d_name, dx, dy in directions:
                shifted_pils = [apply_translation(img, dx, dy) for img in clean_test_pils]
                feats, logits = extract_features_and_logits(backbones[model_name], heads[model_name], shifted_pils, norm, to_tensor)
                m = compute_metrics(test_labels, logits)
                c = compute_prediction_consistency(clean_preds_dict[model_name], m["predictions"])
                dir_accs.append(m["accuracy"])
                dir_cons.append(c)
                if delta == 32 and sample_feats is None:
                    sample_feats = feats.numpy()

            avg_acc = float(np.mean(dir_accs))
            avg_cons = float(np.mean(dir_cons))
            results["translation"][model_name][str(delta)] = {
                "accuracy": avg_acc,
                "consistency": avg_cons
            }
            if delta == 32:
                trans32_feats_dict[model_name] = sample_feats

            print(f"  -> {model_name} (delta={delta}px): Avg Acc: {avg_acc:.4f} | Consistency: {avg_cons:.4f}")

    # -------------------------------------------------------------
    # STEP 5: Patch Shuffling & Representation Stability
    # -------------------------------------------------------------
    print("\n[Step 5/6] Evaluating Patch Shuffling (4x4 Grid)...")
    shuff_pils = [get_patch_shuffled_image(img, grid_size=4, seed=SEED) for img in clean_test_pils]
    shuff_feats_dict = {}

    for model_name in backbones:
        norm = ImageNormalizer(backbones[model_name].mean, backbones[model_name].std)
        feats, logits = extract_features_and_logits(backbones[model_name], heads[model_name], shuff_pils, norm, to_tensor)
        m = compute_metrics(test_labels, logits)
        m["consistency"] = compute_prediction_consistency(clean_preds_dict[model_name], m["predictions"])
        results["patch_shuffle"][model_name] = m
        shuff_feats_dict[model_name] = feats.numpy()
        print(f"  -> {model_name} Acc: {m['accuracy']:.4f} | Consistency: {m['consistency']:.4f}")

    # -------------------------------------------------------------
    # STEP 6: Representation Analysis Across All Interventions
    # -------------------------------------------------------------
    print("\n[Step 6/6] Computing Cosine Stability (I_T) & Generating t-SNE Projections...")

    # Map content index for cue conflict pairs
    conflict_content_indices = [item["content_idx"] for item in conflict_metadata]
    test_indices_list = split_info["test_subset_indices"]
    idx_to_subset_pos = {idx: pos for pos, idx in enumerate(test_indices_list)}
    matched_clean_indices = [idx_to_subset_pos.get(c_idx, 0) for c_idx in conflict_content_indices]
    conflict_labels = np.array([classes.index(item["content_class"]) for item in conflict_metadata])

    for model_name in backbones:
        results["representation_stability"].setdefault("grayscale", {})[model_name] = compute_cosine_stability(
            clean_feats_dict[model_name], gray_feats_dict[model_name]
        )
        results["representation_stability"].setdefault("hue_rotation", {})[model_name] = compute_cosine_stability(
            clean_feats_dict[model_name], hue_feats_dict[model_name]
        )
        results["representation_stability"].setdefault("translation", {})[model_name] = compute_cosine_stability(
            clean_feats_dict[model_name], trans32_feats_dict[model_name]
        )
        results["representation_stability"].setdefault("patch_shuffle", {})[model_name] = compute_cosine_stability(
            clean_feats_dict[model_name], shuff_feats_dict[model_name]
        )

        matched_clean_feats = clean_feats_dict[model_name][matched_clean_indices]
        results["representation_stability"].setdefault("cue_conflict", {})[model_name] = compute_cosine_stability(
            matched_clean_feats, cue_feats_dict[model_name]
        )

        print(f"  -> {model_name} Cosine Stability:")
        print(f"       Grayscale:      {results['representation_stability']['grayscale'][model_name]['cosine_stability_mean']:.4f}")
        print(f"       Hue Rotation:   {results['representation_stability']['hue_rotation'][model_name]['cosine_stability_mean']:.4f}")
        print(f"       Translation:    {results['representation_stability']['translation'][model_name]['cosine_stability_mean']:.4f}")
        print(f"       Patch Shuffle:  {results['representation_stability']['patch_shuffle'][model_name]['cosine_stability_mean']:.4f}")
        print(f"       Cue Conflict:   {results['representation_stability']['cue_conflict'][model_name]['cosine_stability_mean']:.4f}")

        # Generate t-SNE visualizations with legends
        print(f"  -> Rendering t-SNE embeddings for {model_name}...")
        visualize_combined_representation(clean_feats_dict[model_name], shuff_feats_dict[model_name], test_labels, classes, model_name, "patch_shuffle")
        visualize_combined_representation(clean_feats_dict[model_name], gray_feats_dict[model_name], test_labels, classes, model_name, "grayscale")
        visualize_combined_representation(clean_feats_dict[model_name], trans32_feats_dict[model_name], test_labels, classes, model_name, "translation")

    # -------------------------------------------------------------
    # STEP 7: Generate Publication-Quality Report Figures
    # -------------------------------------------------------------
    print("\n[Step 7/7] Generating Comprehensive Evidence Plots for Report...")
    plot_translation_curves(results["translation"], save_path=os.path.join(plots_dir, "translation_curves.png"))
    plot_interventions_benchmark(results, save_path=os.path.join(plots_dir, "interventions_benchmark.png"))
    plot_shape_bias_analysis(results["cue_conflict"], save_path=os.path.join(plots_dir, "shape_bias_analysis.png"))
    plot_cosine_stability_summary(results["representation_stability"], save_path=os.path.join(plots_dir, "representation_cosine_stability.png"))
    plot_cue_conflict_examples(conflict_metadata, cue_conflict_preds, classes, save_path=os.path.join(plots_dir, "cue_conflict_examples.png"), num_examples=6)

    # Save final results JSON
    final_json_path = "./task1/results/task1_final_results.json"
    with open(final_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n============================================================")
    print(f"All Task 1 Experiments & Plots Completed Successfully!")
    print(f"Results written to: {final_json_path}")
    print(f"Plots written to:   {plots_dir}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
