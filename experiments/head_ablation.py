"""Controlled output-head ablation benchmark on standardized ResNet-50 representations."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from drdg.data.datasets import FundusDualBranchDataset
from drdg.data.splits import split_patient_stratified
from drdg.evaluation.grade_metrics import calculate_per_grade_metrics
from drdg.evaluation.metrics import calculate_dr_metrics
from drdg.models.factory import build_ablation_model
from drdg.paths import (
    CHECKPOINTS_DIR,
    HEAD_ABLATION_RESULTS_DIR,
    MASTER_METADATA_CSV,
)
from drdg.training.head_trainer import DRHeadTrainer
from drdg.utils.device import get_device_and_amp
from drdg.utils.logging import get_logger
from drdg.utils.seed import set_seed

logger = get_logger(__name__)

CANDIDATE_HEADS = ["softmax_ce", "softmax_qwk", "coral", "corn", "clm_qwk"]


class SyntheticSingleBranchDataset(Dataset):
    """Synthetic dataset with global fundus image tensors for dry-run testing."""

    def __init__(self, n_samples: int = 16, img_size: int = 512) -> None:
        self.n_samples = n_samples
        self.img_size = img_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "image": torch.randn(
                3, self.img_size, self.img_size, dtype=torch.float32
            ),
            "label": torch.tensor(idx % 5, dtype=torch.long),
            "patient_id": f"syn::patient_{idx // 2}",
        }


class HeadAblationRunner:
    """Manages training and evaluation across the 5 candidate classification and ordinal heads."""

    def __init__(
        self,
        head: Optional[str] = None,
        all_heads: bool = False,
        results_dir: Union[str, Path] = HEAD_ABLATION_RESULTS_DIR,
        checkpoints_dir: Union[str, Path] = CHECKPOINTS_DIR / "head_ablation",
        metadata_csv: Union[str, Path] = MASTER_METADATA_CSV,
        dry_run: bool = False,
        batch_size: int = 4,
        epochs: int = 30,
        lr: float = 1e-4,
        seed: int = 42,
        device: str = "auto",
        amp: str = "auto",
    ) -> None:
        self.head = head.strip().lower() if head else None
        self.all_heads = all_heads
        self.results_dir = Path(results_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.metadata_csv = Path(metadata_csv)
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.epochs = epochs if not dry_run else 1
        self.lr = lr
        self.seed = seed

        self.torch_device, self.amp_mode = get_device_and_amp(device, amp)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.seed)

    def get_heads_to_run(self) -> List[str]:
        if self.all_heads:
            return list(CANDIDATE_HEADS)
        if self.head:
            if self.head not in CANDIDATE_HEADS:
                raise ValueError(
                    f"Unknown head '{self.head}'. Supported: {CANDIDATE_HEADS}"
                )
            return [self.head]
        return ["softmax_ce"]

    def train_and_eval_head(self, head_name: str) -> Dict[str, Any]:
        """Trains a single head on ResNet-50 features and evaluates test performance."""
        logger.info(f"=== Evaluating Head Ablation: [{head_name}] ===")
        model = build_ablation_model(head_name, pretrained=False).to(
            self.torch_device
        )

        if self.dry_run:
            logger.info("Using dry-run synthetic dataset.")
            train_loader = DataLoader(
                SyntheticSingleBranchDataset(12),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                SyntheticSingleBranchDataset(8),
                batch_size=self.batch_size,
                shuffle=False,
            )
            test_loader = DataLoader(
                SyntheticSingleBranchDataset(8),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            if not self.metadata_csv.exists():
                raise FileNotFoundError(
                    f"Master metadata CSV not found at: {self.metadata_csv}"
                )
            df = pd.read_csv(self.metadata_csv)
            split = split_patient_stratified(
                df,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=self.seed,
            )

            train_dataset = FundusDualBranchDataset(
                split.train_df, split="train"
            )
            val_dataset = FundusDualBranchDataset(split.val_df, split="val")
            test_dataset = FundusDualBranchDataset(split.test_df, split="test")

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=1e-2
        )

        trainer = DRHeadTrainer(
            model=model,
            optimizer=optimizer,
            lr_scheduler=None,
            device=self.torch_device,
            use_amp=(
                self.amp_mode != "none" and str(self.torch_device) != "cpu"
            ),
            checkpoint_dir=str(self.checkpoints_dir),
            head_name=head_name,
            qwk_warmup_epochs=1 if self.dry_run else 5,
        )

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=self.epochs,
            patience_early_stopping=12,
        )

        test_metrics, y_true, y_pred, y_probs = trainer.evaluate(
            test_loader, desc="Test"
        )
        grade_info = calculate_per_grade_metrics(y_true, y_pred)
        per_grade_sens = {
            str(c): float(grade_info["per_grade"][str(c)]["sensitivity"])
            for c in range(5)
        }

        result = {
            "test_qwk": float(test_metrics.get("QWK", 0.0)),
            "test_accuracy": float(test_metrics.get("Accuracy", 0.0)),
            "within_1_accuracy": float(
                test_metrics.get(
                    "Within_1_Accuracy",
                    test_metrics.get("Within_1_Grade_Acc", 0.0),
                )
            ),
            "rdr_auc": (
                float(test_metrics.get("Referable_AUC", 0.0))
                if not np.isnan(test_metrics.get("Referable_AUC", 0.0))
                else 0.0
            ),
            "per_grade_sensitivity": per_grade_sens,
        }

        logger.info(
            f"Head [{head_name}] Results: QWK={result['test_qwk']:.4f}, "
            f"Accuracy={result['test_accuracy']:.4f}, Within-1={result['within_1_accuracy']:.4f}, "
            f"RDR AUC={result['rdr_auc']:.4f}"
        )
        return result

    def run(self) -> Dict[str, Any]:
        heads_to_run = self.get_heads_to_run()
        summary_path = self.results_dir / "summary.json"

        # Load existing results if present
        existing_heads = {}
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existing_heads = data.get("heads", {})
            except Exception:
                pass

        for head_name in heads_to_run:
            res = self.train_and_eval_head(head_name)
            existing_heads[head_name] = res

        summary = {
            "experiment": "controlled_head_ablation",
            "backbone": "resnet50_standardized",
            "dataset_split": "in_domain_70_15_15",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "heads": existing_heads,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Also write summary CSV
        rows = []
        for h_name, h_metrics in existing_heads.items():
            row = {
                "head": h_name,
                "test_qwk": h_metrics["test_qwk"],
                "test_accuracy": h_metrics["test_accuracy"],
                "within_1_accuracy": h_metrics["within_1_accuracy"],
                "rdr_auc": h_metrics["rdr_auc"],
            }
            for g in range(5):
                row[f"sens_g{g}"] = h_metrics["per_grade_sensitivity"].get(
                    str(g), 0.0
                )
            rows.append(row)

        csv_path = self.results_dir / "summary.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Summary written to {summary_path} and {csv_path}")

        return summary
