# Task 2: Unsupervised Domain Adaptation (UDA) — Requirements & Checklist

This document contains the complete specification, rules, architecture, and verification checklist for **Task 2 (Unsupervised Domain Adaptation)** from the official **EE-5102/CS-6304 Advanced Topics in Machine Learning (PA1)** document.

---

## 1. Overview & Motivation
Standard empirical risk minimization assumes independent and identically distributed (IID) data. In Unsupervised Domain Adaptation (UDA):
- **Labeled source domain data** is available during training.
- **Unlabeled target domain data** is available during training (transductive UDA protocol).
- **Objective**: Use unlabeled target structure to align representations and close the domain gap without sacrificing discriminative class information learned from the sources.

### Key Concepts & Terminology
- **Domain Gap**: The performance drop when evaluating a model trained on source domains on an unseen/different target domain.
- **Marginal Alignment**: Aligning the global feature distributions $P(F(X_s))$ and $P(F(X_t))$ without accounting for class identities (e.g., DAN, DANN).
- **Class-Conditional Alignment**: Aligning features while conditioning on predicted class distributions (e.g., CDAN) to prevent mixing different classes across domains.
- **Negative Transfer**: When adaptation performs *worse* on the target than the simple Source-only baseline.
- **Target Leakage**: Using target class labels for training, checkpoint selection, or hyperparameter tuning. **Target class labels are strictly forbidden until final evaluation!**

---

## 2. Dataset, Splits & Protocol

### Dataset: PACS
- **Classes (7)**: `dog`, `elephant`, `giraffe`, `guitar`, `horse`, `house`, `person`
- **Domains (4)**:
  - `Photo` (P) — **Source Domain 1**
  - `Art Painting` (A) — **Source Domain 2**
  - `Cartoon` (C) — **Source Domain 3**
  - `Sketch` (S) — **Target Domain** (Used for both Task 2 and Task 3)

### Splitting Protocol
- **Source Splits**: Within each source domain (Photo, Art Painting, Cartoon), construct a stratified **80/20 train/validation split** using **Seed 6304**.
- **Shared Split File**: Save split indices to `shared/splits/pacs_sketch_seed6304.json` (must be shared with Task 3).
- **Target Adaptation Set**: All Sketch images serve as the unlabeled adaptation set in Task 2.
- **Model Selection Rule**: Select all checkpoints using the **mean macro-F1 across the three source validation splits**. Target labels must **never** be consulted for model selection!

---

## 3. Architecture & Training Details

### Backbone Network
- **Model**: `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
- **Classifier Head**: Replace the 1000-class ImageNet linear layer with a **7-class linear layer** ($512 \to 7$).
- **Fine-Tuning**: Fine-tune the complete network (backbone + head) for every method.

### Strict BatchNorm Policy (Crucial Rule)
- **Problem**: Updating running mean and variance on source+target mixed batches introduces uncontrolled implicit adaptation.
- **Rule**: **Freeze all BatchNorm running means and running variances at their pretrained ImageNet values.**
- **Trainable Parameters**: The BatchNorm affine scale ($\gamma$) and bias ($\beta$) remain **trainable**.
- **PyTorch Implementation**:
  ```python
  model.train()
  for m in model.modules():
      if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
          m.eval()  # Freeze running stats, keep parameters trainable
  ```

### Preprocessing & Augmentations
- **Training**:
  1. Resize to $256 \times 256$
  2. Random crop to $224 \times 224$
  3. Random horizontal flip ($p=0.5$)
  4. Pretrained ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Validation & Evaluation**:
  1. Resize to $256 \times 256$
  2. Center crop to $224 \times 224$
  3. Pretrained ImageNet normalization

### Optimization & Budget
- **Optimizer**: AdamW ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$)
- **Epoch Budget**: At most 30 source epochs.
- **Early Stopping**: Stop after **5 epochs** without improvement in mean source-validation macro-F1.
- **Seed**: Fixed seed **6304** across all runs.
- **Batch Composition**:
  - Each adaptation mini-batch must contain:
    - 8 Photo + 8 Art Painting + 8 Cartoon = **24 source examples** (domain-balanced).
    - **24 target examples** (Sketch).
    - Total: 24 source and 24 target (cycle loaders when necessary).

---

## 4. The Four Methods to Implement

### 1. Source-Only ERM Baseline
- Train ResNet-18 using cross-entropy on domain-balanced source batches ($8 \times 3 = 24$).
- Establishes the unadapted baseline performance and measures the raw domain gap.
- **Note**: Save this checkpoint. It will be reused unchanged as the ERM baseline for Task 3!

### 2. DAN (Deep Adaptation Networks — MMD Alignment)
- **Objective**:
  $$\mathcal{L}_{\text{DAN}} = \mathcal{L}_{\text{cls}}(X_s, Y_s) + \lambda_{\text{MMD}} \cdot \text{MMD}^2(F(X_s), F(X_t))$$
- **Feature Space**: Applied to the 512-d penultimate features before the classifier head.
- **Hyperparameters**: $\lambda_{\text{MMD}} = 1.0$.
- **Kernel Construction**: Sum of 3 Gaussian RBF kernels with bandwidths $\sigma \in \{0.5, 1.0, 2.0\} \times \text{median pairwise squared distance}$ of the combined batch.

### 3. DANN (Domain-Adversarial Neural Networks — Adversarial GRL)
- **Objective**: Source cross-entropy loss + binary domain classification loss (unit weight = 1.0).
- **Discriminator Architecture**:
  - Attached to the 512-d feature via a **Gradient Reversal Layer (GRL)**.
  - Linear($512 \to 256$) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear($256 \to 2$).
- **GRL Adaptation Schedule**:
  $$\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1, \quad p = \frac{\text{current\_epoch}}{\text{max\_epochs}} \in [0, 1]$$
- Source examples are labeled as domain 0; Target examples as domain 1.

### 4. CDAN (Conditional Adversarial Domain Adaptation)
- **Multilinear Conditioning**:
  $$g(x) = \text{vec}(f \otimes p)$$
  where $f = F(x) \in \mathbb{R}^{512}$ and $p = \text{Softmax}(C(f)) \in \mathbb{R}^7$.
  $g(x) \in \mathbb{R}^{3584}$ captures joint feature-prediction structure.
- **Discriminator Architecture**:
  - Attached to $g(x)$ via GRL with the exact same schedule $\alpha(p)$ as DANN.
  - Linear($3584 \to 256$) $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear($256 \to 2$).
- **Loss Weight**: Unit weight (1.0).
- **Restrictions**: Do **NOT** use entropy conditioning; do **NOT** detach $f$ or $p$.

---

## 5. Evaluation & Diagnostics

### 1. Main Performance Metrics
- **Source Validation**: Accuracy and Macro-F1 for each individual source domain (Photo, Art Painting, Cartoon) + Mean Source.
- **Target Evaluation**: Final Accuracy and Macro-F1 on the complete Sketch domain.
- **Target Accuracy Change**: $\Delta \text{Acc} = \text{Acc}_{\text{target}}^{\text{method}} - \text{Acc}_{\text{target}}^{\text{source-only}}$.

### 2. Domain Separability Diagnostic
- Measure how distinguishable source and target representations are:
  1. Freeze the backbone.
  2. Extract balanced features from source validation and target sets.
  3. Create a 70/30 train/test split using seed 6304.
  4. Train a balanced Logistic Regression classifier with $C=1.0$ to classify Source vs. Target.
  5. The test accuracy is the **Domain Separability Score** (50% = chance / ideal alignment; 100% = completely separable).

### 3. Per-Class Transfer & Failure Analysis
- Compute per-class accuracy on Sketch for all methods.
- Identify the classes with the greatest positive transfer (improvement) and greatest negative transfer (degradation).
- Generate confusion matrices to inspect dominant error patterns.

---

## 6. Controlled Design Study
Choose one bounded study:
- **Option A (Recommended — DAN)**: Vary $\lambda_{\text{MMD}} \in \{0.1, 1.0, 10.0\}$ while keeping all other settings fixed.
- **Option B (DANN)**: Vary maximum GRL strength over $\{0.25, 0.5, 1.0\}$.
- **Goal**: Analyze the trade-off between alignment pressure, domain separability, source accuracy, and target recognition.

---

## 7. Required Evidence for Report (PDF Page 8)

1. [ ] **Comprehensive Results Table**:
   - Comparing Source-only, DAN, DANN, and CDAN.
   - Columns: Photo Acc/F1, Art Acc/F1, Cartoon Acc/F1, Mean Source Acc/F1, Target (Sketch) Acc/F1, Target Gain ($\Delta \text{Acc}$), and Domain Separability Score.
2. [ ] **Training & Loss Curves**:
   - Classification loss and alignment/domain loss curves across epochs for all methods.
3. [ ] **Per-Class Transfer Analysis & Confusions**:
   - Bar chart of per-class accuracy change relative to Source-only.
   - Selected confusion matrices showing where adaptation helped or hurt.
4. [ ] **Controlled Study Table / Plot**:
   - Performance and separability curves across varying alignment strengths ($\lambda_{\text{MMD}}$ or GRL $\alpha_{\text{max}}$).

---

## 8. Research Questions to Answer in Report

1. **How large is the source-to-target domain gap for Source-only ERM, and which classes account for the most important failures?**
2. **Across the four methods, does lower domain separability correspond to better target recognition? Use aggregate and class-level evidence to identify successful adaptation or negative transfer.**
3. **How does class-conditional alignment compare with the marginal alignment used by DAN and DANN? Do CDAN’s gains or failures support the claim that conditioning helps preserve semantic structure?**
4. **How does increasing alignment strength change source performance, domain separability, and target performance? What trade-off does your controlled study reveal, and what setting could have been selected without consulting target labels?**

---

## 9. Comprehensive Implementation Checklist

### Setup & Shared Protocol
- [ ] Download PACS dataset and structure under `shared/data/PACS/` or `task2/data/PACS/`.
- [ ] Implement `shared/pacs.py` dataset loader with domain tags and labels.
- [ ] Implement `shared/pacs_protocol.py` generating stratified 80/20 splits for Photo, Art, Cartoon (seed 6304).
- [ ] Save splits to `shared/splits/pacs_sketch_seed6304.json`.

### Models & BatchNorm
- [ ] Implement `task2/models/backbone.py` wrapping ResNet-18 with ImageNet weights and custom 7-class linear head.
- [ ] Implement custom training loop enforcing the **frozen ImageNet BatchNorm running statistics** rule (`m.eval()` for BatchNorm layers).
- [ ] Implement `task2/models/domain_discriminator.py` with standard GRL and 256-hidden units.

### Methods
- [ ] Implement `task2/methods/source_only.py` (Domain-balanced batches: 8 Photo, 8 Art, 8 Cartoon).
- [ ] Implement `task2/methods/dan.py` with 3-kernel RBF MMD loss ($\lambda = 1.0$).
- [ ] Implement `task2/methods/dann.py` with binary discriminator and dynamic schedule $\alpha(p)$.
- [ ] Implement `task2/methods/cdan.py` with multilinear conditioning $g(x) = \text{vec}(f \otimes p)$.

### Diagnostics & Evaluation
- [ ] Implement `task2/evaluation/metrics.py` (Accuracy, Macro-F1, per-class accuracy).
- [ ] Implement `task2/evaluation/domain_separability.py` (Logistic Regression, $C=1$, 70/30 split).
- [ ] Implement `task2/evaluation/class_analysis.py` (Per-class transfer $\Delta \text{Acc}$, confusion matrices).
- [ ] Implement Controlled Study script for $\lambda_{\text{MMD}} \in \{0.1, 1.0, 10.0\}$.

### Execution & Deliverables
- [ ] Run all 4 methods with early stopping (patience 5, AdamW lr $10^{-4}$, seed 6304).
- [ ] Save checkpoints to `task2/results/checkpoints/`.
- [ ] Generate all required plots in `task2/results/plots/`.
- [ ] Save full numerical summary in `task2/results/task2_final_results.json`.
- [ ] Verify zero target leakage (no target labels used prior to final evaluation).
