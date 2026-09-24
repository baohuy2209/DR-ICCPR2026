# DRDG: Dual-Branch Multi-Instance Learning with Cumulative Link Models for Cross-Cohort Domain Generalization in Diabetic Retinopathy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **DRDG**, a dual-stream architecture engineered for robust out-of-distribution (OOD) generalization across multi-center retinal fundus photography cohorts without target domain adaptation or retraining.

---

## 🏛️ Proposed Framework Architecture

The end-to-end framework reflects the exact mathematical, architectural, and data-flow specifications implemented in [`notebooks/03_main_model.ipynb`](notebooks/03_main_model.ipynb), [`notebooks/04_head_ablation.ipynb`](notebooks/04_head_ablation.ipynb), and [`notebooks/05_component_analysis_lodo.ipynb`](notebooks/05_component_analysis_lodo.ipynb).

> 📄 **High-Resolution Vector Document**: [Download Proposed Framework Architecture (PDF)](documents/proposed-framework.pdf)

![Proposed Framework Architecture](documents/proposed-framework.png)

---

## 🔬 Five-Stage End-to-End Methodology

The framework decomposes the complex cross-center Diabetic Retinopathy grading challenge into five modular, reproducible stages:

### 1. Data Ingestion & Partitioning
- **6 Multi-Center Cohorts**: APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, and EyePACS ($N = 55{,}570$ images across $35{,}474$ patients).
- **Harmonization & Patient Isolation**: Robust patient extraction regexes eliminate cross-view and bilateral data leakage ($\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) = \emptyset$).
- **Partitioning Protocols**:
  - **Standard Pooled Split**: 70% Train, 15% Validation, 15% Test.
  - **Leave-One-Dataset-Out (LODO) Protocol**: 5 source cohorts (85% train / 15% validation with quota balancing) and 1 held-out cohort (100% OOD test), producing exactly 18 patient-isolated CSV files (`fold_{0-5}_{cohort}_{train|val|test}.csv`).

### 2. Multi-Scale Preprocessing Pipeline
- **Auto-Crop**: Removes redundant uninformative black borders surrounding the circular retina.
- **Robust 4-Step FOV Segmentation**: Otsu-based aperture mask segmentation isolating the retinal disk.
- **Green-Channel Extraction**: Isolates the green spectral band (~540–577 nm), where hemoglobin maximally absorbs light, suppressing choroidal background noise.
- **Ben Graham / CLAHE Preprocessing**: Standardizes local retinal contrast and color variations across disparate camera sensors.
- **Dual-Branch Feeding**:
  - **Global Branch**: Resized to $512 \times 512$ with spatial and photometric data augmentations.
  - **Local Branch**: High-resolution patches extracted at native scale from valid fundus regions (FOV ratio $\ge 0.5$).

### 3. Dual-Branch Multi-Scale MIL Backbone
- **Global Context Branch** (`ResNet-50` + `NonLocal` + `CBAM` + `Quadrant Tokens`):
  - Truncated at `layer4` ($2048 \times 16 \times 16$).
  - $1 \times 1$ Conv channel projection ($2048 \rightarrow 512$).
  - **CBAM**: Sequential channel and spatial attention modules.
  - **NonLocalBlock2D**: Bilateral vascular self-attention capturing hemispheric symmetry.
  - **Quadrant Tokens**: 4 anatomical quadrants ($8 \times 8$ each) pooling mean & max ($4 \times 512 = 2048$) concatenated with Global GAP ($512$) $\to$ **Global Feature: 4096-d**.
- **Local MIL Branch** (`EfficientNet-B0` + `Dual Pooling`):
  - Sliding-window patch extraction ($128 \times 128$, stride 96) filtered by FOV ratio $> 0.5$.
  - EfficientNet-B0 instance encoder projecting patches to $1280$-d embeddings.
  - **Dual Pooling**: Gated Attention Pooling (Ilse et al., 1280-d) + Top-$k$ Instance Max Pooling (1280-d) $\to$ **Local Feature: 2560-d**.
- **Multimodal Fusion**:
  - Concatenation: $\mathbf{f}_{\text{fused}} = [\mathbf{f}_{\text{global}} \,\|\, \mathbf{f}_{\text{local}}] \in \mathbb{R}^{B \times 6656}$.
  - LayerNorm MLP projection ($6656 \rightarrow 512$) with Mish activation and Dropout ($p=0.3$), preventing batch-size-1 instability $\to$ **Latent Severity Embedding: 512-d**.

### 4. Ordinal Severity Classification & Head Ablation
- **Champion Head**: Cumulative Link Model (CLM) with complementary log-log (`clog-log`) link function:
  $$F(u) = 1 - \exp(-\exp(u))$$
  - Latent linear severity score: $z = \mathbf{w}^T \mathbf{h}_{\text{latent}} \in \mathbb{R}$.
  - Parameterized strictly monotonic cutpoints via softplus: $b_0 = \theta_0, \; b_k = b_{k-1} + \text{softplus}(\Delta_k)$ guaranteeing $b_0 < b_1 < b_2 < b_3$.
- **Continuous QWK Hybrid Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CLM}} + \lambda_{\text{qwk}} \mathcal{L}_{\text{QWK}}$$
  where $\lambda_{\text{qwk}} = 1.0$ and $\mathcal{L}_{\text{QWK}}$ continuously penalizes off-by-grade mistakes quadratically.
- **5-Head Controlled Ablation Suite**:
  1. `Softmax_CE`: Nominal categorical cross-entropy baseline.
  2. `Softmax_QWK`: Categorical Softmax with Continuous QWK loss.
  3. `CORAL`: Consistent Rank Logits with binary ordinal sub-tasks.
  4. `CORN`: Conditional Ordinal Regression Neural Network with conditional binary probabilities.
  5. `CLM_QWK`: Cumulative Link Model with clog-log link and Continuous QWK loss.

### 5. Output & Clinical Decision Support
- **Discrete Severity Staging**: ICDRS Grades 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative DR).
- **Referable DR (RDR) Screening**: Post-hoc operating threshold $\tau^*$ calibrated strictly on source validation data at target specificity ($\ge 80\%$ or $\ge 95\%$), tested on held-out target cohorts with frozen threshold.
- **Explainable AI (XAI)**: Global Grad-CAM and Local MIL attention saliency projection objectively validated against expert lesion pixel ground truth from IDRiD (Pointing Game, Area-Matched Saliency Recall, Pixel AUROC).

---

## 📊 Model Variants Progression

| Variant | Global Branch | Local Branch | Output Head | Research Hypothesis |
| :--- | :--- | :--- | :--- | :--- |
| **V0** | ResNet-50 Baseline | None | Softmax CE | Nominal global CNN baseline |
| **V1** | ResNet-50 Global | EfficientNet-B0 + Gated Attention | Softmax CE | **H1**: Local MIL captures small focal lesions (microaneurysms) |
| **V2** | ResNet-50 + CBAM + NonLocal + Quadrants | EfficientNet-B0 + Gated Attention | Softmax CE | **H2**: Global structural attention captures vascular symmetry & optic disc |
| **V3** (Proposed) | ResNet-50 + CBAM + NonLocal + Quadrants | EfficientNet-B0 + Gated Attention | CLM + QWK Loss | **H3**: Ordinal CLM respects disease progression and penalizes severe errors |

---

## 📂 Modular Package Architecture (`src/drdg`)

```
src/drdg/
├── config.py                 # Typed @dataclass configurations (strictly zero YAML)
├── paths.py                  # Portable directory and path resolution utilities
├── data/
│   ├── balancing.py          # Quota balancing (2,000 images per class pool)
│   ├── collate.py            # Variable-length patch batch collator
│   ├── datasets.py           # FundusDualBranchDataset with fast O(1) in-memory indexing
│   ├── fov.py                # Robust 4-step retinal FOV segmentation & aperture masks
│   ├── lodo.py               # 6-Fold LODO generator (18 CSVs) & zero-leakage verifier
│   ├── metadata.py           # MasterFundusRecord unified dataclass schema
│   ├── patches.py            # Sliding-window patch extraction (128x128, stride 96)
│   ├── patient_ids.py        # Patient ID extraction regexes & multi-view isolation
│   ├── preprocessing.py      # Circular ROI crop, Green Channel CLAHE & Ben Graham
│   ├── splits.py             # Patient-stratified split generator (GroupShuffleSplit)
│   ├── transforms.py         # Albumentations global photometric/spatial augmentations
│   ├── downloaders/          # Automated HTTP/Kaggle dataset downloaders
│   └── preparation/          # Per-cohort raw ingestion & standardization scripts
├── documents/
│   ├── proposed-framework.pdf# High-resolution architectural framework diagram (PDF)
│   └── proposed-framework.png# High-resolution architectural framework diagram (PNG)
├── models/
│   ├── attention/            # CBAM, NonLocalBlock2D, and QuadrantTokenAggregator
│   ├── streams/              # GlobalContextStream (ResNet-50) & LocalMILBranch (EffNet-B0)
│   ├── heads/                # SoftmaxCE, SoftmaxQWK, CORAL, CORN, CLM
│   ├── variants/             # Model variant builders (V0, V1, V2, V3)
│   ├── fusion.py             # DualBranchFusion: LayerNorm MLP (6656 -> 512)
│   ├── full_model.py         # End-to-end multi-scale dual-branch model wrapper
│   └── factory.py            # Dynamic model factory builder
├── losses/
│   ├── qwk.py                # Continuous Quadratic Weighted Kappa loss
│   └── ordinal.py            # CORAL, CORN, and Ordinal NLL loss functions
├── training/
│   ├── trainer.py            # DRTrainer engine with mixed precision (AMP) & gradient clipping
│   ├── head_trainer.py       # DRHeadTrainer for controlled head ablation benchmarks
│   ├── optimization.py       # AdamW with two-tier LR parameter groups & Cosine Annealing
│   └── checkpoint.py         # PyTorch 2.6 safe checkpoint serialization & resume
├── evaluation/
│   ├── lodo.py               # Bipartite LODO fold evaluation (in_domain_val + ood_test)
│   ├── rdr.py                # Referable DR threshold calibration (tau*) & frozen evaluation
│   ├── result_schema.py      # Formal LODOFoldResult serialization schema (Zero fabrication)
│   ├── metrics.py            # QWK, Within-1 accuracy, per-grade sensitivity, accuracy
│   ├── ordinal_metrics.py    # Mean Absolute Error (MAE), Off-by-2+ severe error rates
│   └── statistics.py         # Wilcoxon signed-rank tests with Bonferroni correction
├── explainability/
│   ├── gradcam.py            # Global branch Gradient-weighted Class Activation Mapping
│   ├── mil_saliency.py       # Local patch attention 2D spatial projection & smoothing
│   ├── idrid_dataset.py      # Composite binary mask loader (MA, HE, EX, SE)
│   └── metrics.py            # Quantitative Pointing Game, Area-Matched Recall, Pixel AUROC
└── visualization/
    └── paper_figures.py      # Publication-quality vector figures (300 DPI) & LaTeX tables
```

---

## 🔬 Multi-Center Clinical Benchmark (6 Cohorts)

| Cohort | Origin Country | Optical Device / FOV | Total Images (Patients) | Grade Distribution (%) [$G_0 / G_1 / G_2 / G_3 / G_4$] |
| :--- | :--- | :--- | :--- | :--- |
| **APTOS 2019** | India | Zeiss, Topcon ($45^\circ$) | 3,662 (3,662) | 49.3 / 10.1 / 27.3 / 5.3 / 8.1 |
| **DDR** | China | Canon CR-2, Topcon ($45^\circ$) | 12,522 (12,522) | 50.0 / 5.0 / 35.8 / 1.9 / 7.3 |
| **DeepDRiD** | China | Topcon TRC-NW400 ($45^\circ$) | 2,000 (500) | 34.0 / 17.8 / 19.6 / 19.6 / 9.0 |
| **IDRiD** | India | Kowa VX-10$\alpha$ ($50^\circ$) | 516 (516) | 32.6 / 6.2 / 41.3 / 17.8 / 2.1 |
| **Messidor-2** | France | Topcon TRC-NW6 ($45^\circ$) | 1,748 (874) | 58.1 / 15.4 / 19.9 / 4.3 / 2.3 |
| **EyePACS** | USA | Multiple telemedical ($45^\circ$) | 35,122 (17,561) | 73.5 / 6.9 / 15.0 / 2.5 / 2.1 |

---

## 🚀 Execution & Reproduction Guide

### 1. Installation

```bash
git clone https://github.com/baohuy2209/deep-learning-for-computer-vision.git
cd deep-learning-for-computer-vision
pip install -e .
```

### 2. Dual-Layer Execution Protocol
- **Self-Contained Research Notebooks**: The 5 reference Jupyter notebooks in [`notebooks/`](notebooks/) remain complete and 100% self-contained for interactive experimentation and provenance without requiring `drdg` package imports.
- **Modular Python CLI**: `drdg` provides high-throughput, cluster-ready CLI entry points for automated multi-fold benchmarks, ablations, and CI testing.

### 3. Automated Notebook & Parity Verification

```bash
# Verify all 5 canonical research notebooks remain complete and untampered
python scripts/verify_notebook_integrity.py

# Execute full numerical and architectural parity test suite
pytest tests/parity -v
```

### 4. Running 6-Fold LODO Cross-Domain Benchmark

```bash
# Run proposed variant V3 on held-out target cohort (e.g., APTOS 2019)
python scripts/run_lodo.py --variant v3 --held-out aptos --device cuda

# Dry-run validation (CPU verification without full image dataset)
python scripts/run_lodo.py --variant v3 --held-out aptos --dry-run --device cpu
```

### 5. Running 5-Head Controlled Ablation Suite

```bash
# Benchmark all 5 prediction heads on standardized ResNet-50 features
python scripts/run_head_ablation.py --all-heads --device cuda

# Benchmark individual champion CLM head
python scripts/run_head_ablation.py --head clm_qwk --device cuda
```

### 6. Quantitative XAI Evaluation against IDRiD Pixel Masks

```bash
python scripts/run_xai.py --variant v3 --checkpoint checkpoints/v3_best.pt --device cuda
```

---

## 📜 Traceability & Documentation Index

- [`NOTEBOOK_CODE_MAP.md`](../../docs/NOTEBOOK_CODE_MAP.md): 100% symbol mapping between canonical notebooks and `drdg`.
- [`DATA_FLOW_AUDIT.md`](../../docs/DATA_FLOW_AUDIT.md): Producer-consumer data flow matrix across all 6 research stages.
- [`REFACTOR_AUDIT.md`](../../docs/REFACTOR_AUDIT.md): Discrepancy audit log (AUD-001 through AUD-013) with certified status **`VERIFIED`**.
- [`ARCHITECTURE.md`](../../docs/ARCHITECTURE.md): Mathematical derivation of Cumulative Link Models, attention mechanisms, and fusion.
- [`DATA.md`](../../docs/DATA.md): Detailed multi-cohort documentation, 18 LODO fold CSVs, and FOV extraction.
- [`RESULTS_SCHEMA.md`](../../docs/RESULTS_SCHEMA.md): Bipartite evaluation JSON serialization contract.
