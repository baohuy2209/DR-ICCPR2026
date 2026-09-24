"""High-performance PyTorch training engine for diabetic retinopathy grading."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from drdg.evaluation.metrics import (
    calculate_dr_metrics,
    extract_rank_probs,
    rank_monotonicity_violations,
)
from drdg.training.checkpoint import load_checkpoint, save_checkpoint
from drdg.training.optimization import update_qwk_warmup
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


class DRTrainer:
    """High-Performance PyTorch Training & Evaluation Engine:

    - Native bfloat16 / float16 Mixed Precision via torch.amp
    - Checkpointing: Saves both '_best.pth' (Val QWK) and '_last.pth' (per-epoch recovery)
    - Full Resume Capability: Restores optimizer, scheduler, scaler, epoch, and history
    - Dynamic Learning Rate Control & Early Stopping
    - GPU-side ImageNet tensor normalization
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
        qwk_warmup_epochs: int = 5,
    ) -> None:
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.qwk_warmup_epochs = qwk_warmup_epochs
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.use_amp = use_amp and (self.device_type == "cuda")

        # GPU-side ImageNet Normalization Constants
        self.mean_global = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std_global = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.mean_local = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.std_local = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)

        # Precision configuration
        if self.device_type == "cuda" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False
        elif self.device_type == "cuda":
            self.amp_dtype = torch.float16
            self.use_scaler = self.use_amp
        else:
            self.amp_dtype = torch.float32
            self.use_scaler = False

        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_scaler)

    def _normalize_batch(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies GPU-side ImageNet standardization if tensors are uint8."""
        if global_img.dtype == torch.uint8:
            global_img = global_img.float().div_(255.0).sub_(self.mean_global).div_(self.std_global)

        if local_patches is not None and local_patches.dtype == torch.uint8:
            if local_patches.ndim == 5:
                local_patches = local_patches.float().div_(255.0).sub_(self.mean_local).div_(self.std_local)
            else:
                local_patches = local_patches.float().div_(255.0).sub_(self.mean_global).div_(self.std_global)

        return global_img, local_patches

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs one training epoch with mixed precision and gradient clipping."""
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []

        pbar = tqdm(dataloader, desc="Training", leave=False, unit="batch")
        for batch in pbar:
            global_img = batch["global_img"].to(self.device, non_blocking=True)
            local_patches = batch.get("local_patches")
            if local_patches is None:
                local_patches = batch.get("patch_imgs")
            if local_patches is not None:
                local_patches = local_patches.to(self.device, non_blocking=True)

            targets = batch["label"].to(self.device, non_blocking=True)
            patch_mask = batch.get("patch_mask")
            if patch_mask is not None:
                patch_mask = patch_mask.to(self.device, non_blocking=True)

            global_img, local_patches = self._normalize_batch(global_img, local_patches)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                if patch_mask is not None and local_patches is not None:
                    try:
                        outputs = self.model(global_img, local_patches, patch_mask=patch_mask)
                    except TypeError:
                        outputs = self.model(global_img, local_patches)
                elif local_patches is not None:
                    outputs = self.model(global_img, local_patches)
                else:
                    outputs = self.model(global_img)

                if hasattr(self.model, "compute_loss"):
                    loss = self.model.compute_loss(outputs, targets, class_weights=self.class_weights)
                elif hasattr(self.model, "head") and hasattr(self.model.head, "loss_fn"):
                    loss = self.model.head.loss_fn(outputs, targets, class_weights=self.class_weights)
                else:
                    loss = outputs["loss"]

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs["preds"].detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            if "probs" in outputs:
                all_probs.extend(outputs["probs"].detach().cpu().numpy())

        n_samples = len(all_targets)
        probs_np = np.array(all_probs) if all_probs else None
        metrics = calculate_dr_metrics(np.array(all_targets), np.array(all_preds), y_probs=probs_np)
        metrics["Loss"] = total_loss / max(1, n_samples)
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Evaluates model performance on validation or test sets."""
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []
        all_rank_probs: List[np.ndarray] = []

        pbar = tqdm(dataloader, desc="Eval", leave=False, unit="batch")
        for batch in pbar:
            global_img = batch["global_img"].to(self.device, non_blocking=True)
            local_patches = batch.get("local_patches")
            if local_patches is None:
                local_patches = batch.get("patch_imgs")
            if local_patches is not None:
                local_patches = local_patches.to(self.device, non_blocking=True)

            targets = batch["label"].to(self.device, non_blocking=True)
            patch_mask = batch.get("patch_mask")
            if patch_mask is not None:
                patch_mask = patch_mask.to(self.device, non_blocking=True)

            global_img, local_patches = self._normalize_batch(global_img, local_patches)

            with torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                if patch_mask is not None and local_patches is not None:
                    try:
                        outputs = self.model(global_img, local_patches, patch_mask=patch_mask)
                    except TypeError:
                        outputs = self.model(global_img, local_patches)
                elif local_patches is not None:
                    outputs = self.model(global_img, local_patches)
                else:
                    outputs = self.model(global_img)

                if hasattr(self.model, "compute_loss"):
                    loss = self.model.compute_loss(outputs, targets, class_weights=self.class_weights)
                elif hasattr(self.model, "head") and hasattr(self.model.head, "loss_fn"):
                    loss = self.model.head.loss_fn(outputs, targets, class_weights=self.class_weights)
                else:
                    loss = outputs["loss"]

            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs["preds"].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if "probs" in outputs:
                all_probs.extend(outputs["probs"].cpu().numpy())

            rank_p = extract_rank_probs(outputs)
            if rank_p is not None:
                all_rank_probs.extend(rank_p)

        n_samples = len(all_targets)
        probs_np = np.array(all_probs) if all_probs else None
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        metrics = calculate_dr_metrics(y_true, y_pred, y_probs=probs_np)
        metrics["Loss"] = total_loss / max(1, n_samples)
        metrics["probs"] = probs_np

        if all_rank_probs:
            metrics["Monotonicity_Violations"] = rank_monotonicity_violations(np.array(all_rank_probs))
        else:
            metrics["Monotonicity_Violations"] = None

        return metrics, y_true, y_pred

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        model_name: str = "dr_model",
        patience_early_stopping: int = 12,
        resume_checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes full training with ModelCheckpoint, LR scheduler, and Early Stopping."""
        best_val_qwk = -1.0
        best_val_loss = float("inf")
        start_epoch = 1
        epochs_no_improve = 0

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_qwk": [], "val_qwk": [],
            "val_within_1": [], "val_ref_sens": [],
            "val_ref_auc": [],
        }

        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pth")
        last_checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_last.pth")

        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            ckpt = load_checkpoint(
                resume_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler if self.use_scaler else None,
                scheduler=self.lr_scheduler,
                device=self.device,
            )
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_qwk = ckpt.get("best_val_qwk", -1.0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", history)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            logger.info(f"Resumed from epoch {start_epoch} with best val QWK: {best_val_qwk:.4f}")

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            active_qwk_weight = update_qwk_warmup(self.model, epoch, self.qwk_warmup_epochs)

            train_metrics = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            elapsed = time.time() - t0

            val_loss = val_metrics["Loss"]
            val_qwk = val_metrics["QWK"]

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_qwk)
                else:
                    self.lr_scheduler.step()

            history["train_loss"].append(train_metrics["Loss"])
            history["val_loss"].append(val_loss)
            history["train_qwk"].append(train_metrics["QWK"])
            history["val_qwk"].append(val_qwk)
            history["val_within_1"].append(val_metrics["Within_1_Grade_Acc"])
            history["val_ref_sens"].append(val_metrics["Referable_Sensitivity"])
            history["val_ref_auc"].append(val_metrics.get("Referable_AUC", float("nan")))

            is_best = val_qwk > best_val_qwk
            if is_best:
                best_val_qwk = val_qwk
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_scaler else None,
                "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "best_val_qwk": best_val_qwk,
                "best_val_loss": best_val_loss,
                "history": history,
                "epochs_no_improve": epochs_no_improve,
                "val_metrics": val_metrics,
            }
            save_checkpoint(
                state=state,
                checkpoint_path=last_checkpoint_path,
                is_best=is_best,
                best_path=best_checkpoint_path,
            )

            logger.info(
                f"[Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s)] "
                f"Train Loss: {train_metrics['Loss']:.4f} | QWK: {train_metrics['QWK']:.4f} || "
                f"Val Loss: {val_loss:.4f} | QWK: {val_qwk:.4f} | "
                f"Within-1: {val_metrics['Within_1_Grade_Acc']*100:.1f}% | "
                f"{'*BEST*' if is_best else ''}"
            )

            if epochs_no_improve >= patience_early_stopping:
                logger.info(f"Early stopping triggered after {epoch} epochs ({patience_early_stopping} epochs without improvement).")
                break

        # Optional test set evaluation using the best checkpoint
        if test_loader is not None and os.path.exists(best_checkpoint_path):
            logger.info(f"Evaluating best checkpoint on held-out test set: {best_checkpoint_path}")
            load_checkpoint(best_checkpoint_path, model=self.model, device=self.device)
            test_metrics, _, _ = self.evaluate(test_loader)
            history["test_metrics"] = test_metrics
            logger.info(f"Test QWK: {test_metrics['QWK']:.4f} | Test Loss: {test_metrics['Loss']:.4f}")

        return history
