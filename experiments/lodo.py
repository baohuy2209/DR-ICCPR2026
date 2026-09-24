"""Leave-One-Dataset-Out (LODO) benchmark experiment orchestration.

Golden reference:
  - merge_dataset_lodo.ipynb  (fold generation, CSV naming)
  - main_model_lodo.ipynb     (Cell 17 — fold loop, CSV loading, evaluation)

Semantics (from notebook):
  - 6 folds, each holding out one cohort as OOD test.
  - 3 CSV files per fold: train, val, test.
  - Naming: fold_{fold_idx}_{fold_slug}_{split}.csv
  - Evaluation is bipartite: in-domain val + OOD test (held-out cohort).
  - No separate "source_test" partition exists in the research design.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drdg.config import DataConfig, LODOConfig, ModelConfig, TrainingConfig
from drdg.data.balancing import balance_training_records
from drdg.data.collate import trainer_compatible_collate_fn
from drdg.data.datasets import FundusDualBranchDataset
from drdg.data.lodo import LODOFold, generate_and_save_lodo_splits
from drdg.evaluation.lodo import evaluate_lodo_fold
from drdg.evaluation.result_schema import LODOFoldResult, MetricBundle
from drdg.models.factory import build_variant
from drdg.paths import CHECKPOINTS_DIR, LODO_FOLDS_DIR, LODO_RESULTS_DIR
from drdg.training.checkpoint import load_checkpoint
from drdg.training.trainer import DRTrainer
from drdg.utils.device import get_device_and_amp
from drdg.utils.logging import get_logger
from drdg.utils.seed import set_seed

logger = get_logger(__name__)

# Canonical cohort ordering matching DATASET_CONFIG in merge_dataset_lodo.ipynb
# This defines fold_idx -> cohort_key mapping:
#   fold_0 = aptos, fold_1 = idrid, fold_2 = messidor2,
#   fold_3 = ddr,   fold_4 = eyepacs, fold_5 = deepdrid
CANONICAL_COHORTS = [
    "aptos",
    "idrid",
    "messidor2",
    "ddr",
    "eyepacs",
    "deepdrid",
]


class SyntheticDualBranchDataset(torch.utils.data.Dataset):
    """Lightweight in-memory dataset for dry-run verification without image files."""

    def __init__(
        self, n_samples: int = 16, num_patches: int = 8, patch_size: int = 128
    ) -> None:
        self.n_samples = n_samples
        self.num_patches = num_patches
        self.patch_size = patch_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patches = torch.randn(
            self.num_patches,
            3,
            self.patch_size,
            self.patch_size,
            dtype=torch.float32,
        )
        return {
            "image_id": f"syn_{idx}",
            "label": torch.tensor(idx % 5, dtype=torch.long),
            "global_img": torch.randn(3, 512, 512, dtype=torch.float32),
            "local_patches": patches,
            "patch_imgs": patches,
            "patch_mask": torch.ones(self.num_patches, dtype=torch.float32),
            "patient_id": f"syn::patient_{idx // 2}",
        }


class LODOExperimentRunner:
    """Orchestrates 6-fold LODO cross-domain benchmark across model variants.

    Each fold holds out one cohort as OOD test, trains on the remaining 5
    cohorts (patient-level 85/15 train/val split with quota balancing on train).

    CSV naming convention (from merge_dataset_lodo.ipynb):
        fold_{fold_idx}_{fold_slug}_train.csv
        fold_{fold_idx}_{fold_slug}_val.csv
        fold_{fold_idx}_{fold_slug}_test.csv

    Evaluation is bipartite (from main_model_lodo.ipynb Cell 17):
        1. In-domain validation (source val) — calibrate threshold tau*
        2. OOD test (held-out cohort) — evaluate with frozen tau*
    """

    def __init__(
        self,
        variant: str = "v3",
        held_out: Optional[str] = None,
        all_folds: bool = False,
        splits_dir: Union[str, Path] = LODO_FOLDS_DIR,
        results_dir: Union[str, Path] = LODO_RESULTS_DIR,
        checkpoints_dir: Union[str, Path] = CHECKPOINTS_DIR,
        resume: bool = False,
        dry_run: bool = False,
        batch_size: int = 4,
        epochs: int = 30,
        lr: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
        amp: str = "auto",
    ) -> None:
        self.variant = variant.strip().lower()
        self.held_out = held_out.strip().lower() if held_out else None
        self.all_folds = all_folds
        self.splits_dir = Path(splits_dir)
        self.results_dir = Path(results_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.resume = resume
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.epochs = epochs if not dry_run else 1
        self.lr = lr
        self.seed = seed

        self.torch_device, self.amp_mode = get_device_and_amp(device, amp)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.seed)

    def get_target_folds(self) -> List[Tuple[int, str]]:
        """Determines which folds to execute based on CLI configuration."""
        if self.all_folds:
            return [(idx, c) for idx, c in enumerate(CANONICAL_COHORTS)]
        if self.held_out:
            if self.held_out not in CANONICAL_COHORTS:
                raise ValueError(
                    f"Held out cohort '{self.held_out}' not in {CANONICAL_COHORTS}"
                )
            idx = CANONICAL_COHORTS.index(self.held_out)
            return [(idx, self.held_out)]
        return [(0, CANONICAL_COHORTS[0])]

    def _get_fold_csv_path(self, fold_idx: int, cohort: str, split: str) -> Path:
        """Constructs fold CSV path matching merge_dataset_lodo.ipynb naming.

        Pattern: fold_{fold_idx}_{fold_slug}_{split}.csv
        Example: fold_0_aptos_train.csv
        """
        return self.splits_dir / f"fold_{fold_idx}_{cohort}_{split}.csv"

    def execute_fold(self, fold_idx: int, target_cohort: str) -> LODOFoldResult:
        """Executes training and bipartite evaluation for a single LODO fold.

        Matches main_model_lodo.ipynb Cell 17 semantics:
          - Train on balanced source train split
          - Validate on source val split (calibrate RDR threshold)
          - Test on held-out OOD cohort (evaluate with frozen threshold)
        """
        logger.info(
            f"=== Starting LODO Fold {fold_idx}: Held-Out OOD = '{target_cohort}' "
            f"(Variant = {self.variant}) ==="
        )
        out_json_path = (
            self.results_dir
            / f"{self.variant}_fold{fold_idx}_{target_cohort}.json"
        )

        # Check if fold already completed and resume is enabled
        if self.resume and out_json_path.exists():
            logger.info(
                f"Found existing fold result at {out_json_path}. Resuming from cached result."
            )
            return LODOFoldResult.load_json(out_json_path)

        # Build model variant
        model = build_variant(self.variant).to(self.torch_device)

        if self.dry_run:
            logger.info(
                "Dry-run mode active: using lightweight synthetic tensors."
            )
            train_loader = DataLoader(
                SyntheticDualBranchDataset(12),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                SyntheticDualBranchDataset(8),
                batch_size=self.batch_size,
                shuffle=False,
            )
            test_loader = DataLoader(
                SyntheticDualBranchDataset(8),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            # Load real fold CSVs — exact naming from merge_dataset_lodo.ipynb
            # Pattern: fold_{fold_idx}_{cohort}_{split}.csv
            p_train = self._get_fold_csv_path(fold_idx, target_cohort, "train")
            p_val = self._get_fold_csv_path(fold_idx, target_cohort, "val")
            p_test = self._get_fold_csv_path(fold_idx, target_cohort, "test")

            for p in (p_train, p_val, p_test):
                if not p.exists():
                    raise FileNotFoundError(
                        f"LODO fold file missing: {p}. "
                        f"Please ensure CSV files are uploaded to "
                        f"'{self.splits_dir}/fold_{fold_idx}_{target_cohort}_{{train,val,test}}.csv'. "
                        f"Generate splits via merge_dataset_lodo.ipynb first."
                    )

            train_df = pd.read_csv(p_train)
            val_df = pd.read_csv(p_val)
            test_df = pd.read_csv(p_test)

            logger.info(
                f"  [DATA] Fold {fold_idx} ({target_cohort}) Loading CSVs:\n"
                f"    - Train: {p_train.name} ({len(train_df):,d} rows)\n"
                f"    - Val:   {p_val.name} ({len(val_df):,d} rows)\n"
                f"    - Test:  {p_test.name} ({len(test_df):,d} rows)"
            )

            # Note: train CSVs are already quota-balanced by merge_dataset_lodo.ipynb.
            # The notebook applies create_balanced_train_pool() during fold generation,
            # so the saved train CSV is the final balanced version.
            train_dataset = FundusDualBranchDataset(train_df, split="train")
            val_dataset = FundusDualBranchDataset(val_df, split="val")
            test_dataset = FundusDualBranchDataset(test_df, split="test")

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=trainer_compatible_collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=trainer_compatible_collate_fn,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=trainer_compatible_collate_fn,
            )

        # Setup training components
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=1e-2
        )
        fold_ckpt_dir = (
            self.checkpoints_dir
            / f"{self.variant}_fold{fold_idx}_{target_cohort}"
        )

        trainer = DRTrainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=None,
            device=self.torch_device,
            use_amp=(
                self.amp_mode != "none" and str(self.torch_device) != "cpu"
            ),
            checkpoint_dir=str(fold_ckpt_dir),
            qwk_warmup_epochs=1 if self.dry_run else 5,
        )

        # Training loop — matches main_model_lodo.ipynb Cell 17 Section 6
        fold_model_name = f"lodo_{self.variant}_fold_{fold_idx}_{target_cohort}"
        logger.info(f"Training fold {fold_idx} ({self.epochs} epochs)...")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=self.epochs,
            model_name=fold_model_name,
            patience_early_stopping=12,
        )

        # Load best checkpoint if saved — matches Cell 17 Section 7
        best_ckpt = fold_ckpt_dir / f"{fold_model_name}_best.pth"
        if not best_ckpt.exists():
            best_ckpt = fold_ckpt_dir / "best_model.pt"
        if best_ckpt.exists():
            load_checkpoint(
                str(best_ckpt), model, device=str(self.torch_device)
            )

        # Bipartite evaluation — matches main_model_lodo.ipynb Cell 17 Section 7
        # Step 1: Evaluate source val → calibrate threshold
        # Step 2: Evaluate OOD test with frozen threshold
        fold_result = evaluate_lodo_fold(
            model=model,
            source_val_loader=val_loader,
            ood_test_loader=test_loader,
            device=self.torch_device,
            variant=self.variant,
            held_out_dataset=target_cohort,
            fold=fold_idx,
            seed=self.seed,
        )

        fold_result.save_json(out_json_path)
        logger.info(f"Saved LODO fold result to: {out_json_path}")
        return fold_result

    def run(self) -> Dict[str, Any]:
        """Runs configured LODO folds and compiles overall summary."""
        target_folds = self.get_target_folds()
        logger.info(
            f"Executing {len(target_folds)} LODO folds for variant '{self.variant}'."
        )

        fold_results: Dict[str, LODOFoldResult] = {}
        summary_rows: List[Dict[str, Any]] = []

        for fold_idx, target_cohort in target_folds:
            res = self.execute_fold(fold_idx, target_cohort)
            fold_results[target_cohort] = res
            summary_rows.append(
                {
                    "variant": res.variant,
                    "fold": res.fold,
                    "held_out": res.held_out_dataset,
                    "in_domain_val_qwk": res.in_domain_val.qwk,
                    "in_domain_val_acc": res.in_domain_val.accuracy,
                    "in_domain_val_rdr_auc": res.in_domain_val.rdr_auc,
                    "ood_test_qwk": res.ood_test.qwk,
                    "ood_test_acc": res.ood_test.accuracy,
                    "ood_test_rdr_auc": res.ood_test.rdr_auc,
                    "ood_test_rdr_sens95": res.ood_test.rdr_sensitivity_at_95_specificity,
                    "delta_lodo": res.in_domain_val.qwk - res.ood_test.qwk,
                    "threshold_spec95": res.threshold_rdr_spec95,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = self.results_dir / f"{self.variant}_lodo_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        summary_json = self.results_dir / f"{self.variant}_lodo_summary.json"
        summary_dict = {
            "variant": self.variant,
            "seed": self.seed,
            "n_folds": len(fold_results),
            "ood_test_mean_qwk": (
                float(summary_df["ood_test_qwk"].mean())
                if not summary_df.empty
                else 0.0
            ),
            "ood_test_std_qwk": (
                float(summary_df["ood_test_qwk"].std())
                if len(summary_df) > 1
                else 0.0
            ),
            "folds": [r.to_dict() for r in fold_results.values()],
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(
            f"LODO benchmark completed. Summary saved to {summary_csv} and {summary_json}"
        )
        return summary_dict
