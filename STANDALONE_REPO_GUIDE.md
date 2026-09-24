# Standalone Repository Extraction & Notebook Strategy Guide

This guide provides concrete, step-by-step instructions for extracting the `drdg` package and its accompanying research notebooks into an independent, public GitHub repository.

---

## 1. Why Self-Contained Notebooks Matter

In academic machine learning and medical imaging publications:
- **Reviewers and researchers** frequently test code by uploading a single notebook directly into Google Colab or Kaggle.
- If a notebook is merely a thin wrapper containing:
  ```python
  from drdg.models import DRFullModel  # Fails in Colab unless repository is cloned & installed
  ```
  it creates an unnecessary barrier to immediate replication.
- **The Golden Strategy**:
  1. The **Python package (`src/drdg`)** provides clean, modular, tested, and maintainable software engineering architecture suitable for production, cluster training, and CI/CD.
  2. The **Research Notebooks (`notebooks/`)** are fully self-contained with complete executable definitions of classes, functions, losses, and data pipelines directly in code cells. Anyone can open the notebook, execute the cells from top to bottom, and reproduce the paper's findings without dependency on local module imports.

---

## 2. Directory Layout for the Standalone Repository

When creating the new standalone repository (e.g. `github.com/baohuy2209/drdg`), structure it as follows:

```
baohuy2209/drdg/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated pytest, ruff lint, and parity tests
├── audit_artifacts/               # Parity audit logs, symbol mappings, config diffs
│   ├── config_parity.csv
│   ├── logic_integrity_report.json
│   ├── randomness_map.md
│   └── symbol_mapping.csv
├── docs/                          # Academic documentation & audit reports
│   ├── ARCHITECTURE.md
│   ├── DATASET_HARMONIZATION.md
│   ├── EVALUATION_PROTOCOL.md
│   └── LOGIC_INTEGRITY_AUDIT.md
├── notebooks/                     # FULL SELF-CONTAINED JUPYTER NOTEBOOKS
│   ├── 01_merge_dataset_lodo.ipynb        # Complete data harmonization & 6-fold splitting
│   ├── 02_main_model_lodo.ipynb           # Complete Champion V3 training & LODO evaluation
│   ├── 03_main_model_head_ablation.ipynb  # Complete 5-head comparative ablation study
│   └── 04_visualize_and_xai.ipynb         # Complete Grad-CAM & MIL Attention Saliency maps
├── src/
│   └── drdg/                      # Modular Python package
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       ├── models/
│       ├── losses/
│       ├── training/
│       ├── evaluation/
│       ├── explainability/
│       └── visualization/
├── tests/
│   ├── unit/                      # Standard unit tests
│   └── parity/                    # Golden reference parity test suite
├── pyproject.toml                 # Package build & dependency specifications
├── requirements.txt
├── LICENSE
└── README.md                      # Academic presentation & quickstart
```

---

## 3. Checklist for Splitting into Standalone Repo

1. [ ] **Initialize Git Repo**:
   ```bash
   mkdir drdg-standalone && cd drdg-standalone
   git init
   ```
2. [ ] **Copy Modular Core**:
   Copy `src/drdg/` to `src/drdg/` in the new repo.
3. [ ] **Copy Primary Self-Contained Notebooks**:
   - `notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb` $\rightarrow$ `notebooks/01_merge_dataset_lodo.ipynb`
   - `notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb` $\rightarrow$ `notebooks/02_main_model_lodo.ipynb`
   - `notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb` $\rightarrow$ `notebooks/03_main_model_head_ablation.ipynb`
   - `notebooks/diabetic_retinopathy/pipeline/model/visualize_main_model.ipynb` $\rightarrow$ `notebooks/04_visualize_and_xai.ipynb`
4. [ ] **Copy Documentation & Parity Artifacts**:
   Copy `docs/LOGIC_INTEGRITY_AUDIT.md`, `audit_artifacts/`, and `tests/parity/`.
5. [ ] **Verify Parity Tests**:
   Run `pytest tests/parity -v` in the new repo. Ensure all 25 parity tests pass.
6. [ ] **Publish & Link in Paper**:
   Provide the GitHub repository URL in the "Code and Data Availability" section of the manuscript.
