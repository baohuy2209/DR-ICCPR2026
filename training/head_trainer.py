"""Trainer engine for controlled head ablation experiments on standardized backbones."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_fundus_tensor(
    tensor: torch.Tensor,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """Standardizes uint8 fundus image tensor on GPU with ImageNet mean and std."""
    tensor = tensor.to(device, non_blocking=True)
    if tensor.dtype == torch.uint8:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        return tensor.float().div_(255.0).sub_(mean).div_(std)
    return tensor


class DRHeadTrainer:
    """High-throughput PyTorch training and evaluation engine for head ablation experiments."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: Union[str, torch.device] = "cuda",
        use_amp: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "./checkpoints_head",
        head_name: str = "CLM_QWK",
        qwk_warmup_epochs: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpoint_dir = checkpoint_dir
        self.head_name = head_name
        self.qwk_warmup_epochs = qwk_warmup_epochs
        self.grad_clip = grad_clip
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.use_amp = use_amp and (self.device_type == "cuda")
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

    def train_epoch(self, dataloader: DataLoader, epoch: int, epochs: int) -> Tuple[float, float]:
        """Runs one full training epoch with step-level dynamic QWK warmup and mixed precision."""
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        total_warmup_steps = max(1, self.qwk_warmup_epochs * len(dataloader))

        head = getattr(self.model, "head", None)

        pbar = tqdm(dataloader, desc=f"Training [{self.head_name}] ({epoch}/{epochs})", leave=False, unit="batch")
        for batch_idx, batch in enumerate(pbar):

            image_key = "image" if "image" in batch else "global_img"
            images = normalize_fundus_tensor(batch[image_key], device=self.device)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(images, targets=labels, class_weights=self.class_weights, return_loss=True)
                loss = outputs["loss"]

            if self.scaler.is_enabled():
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

            total_loss += loss.item() * len(labels)
            all_preds.extend(outputs["preds"].detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())

        n_samples = len(all_targets)
        avg_loss = float(total_loss / max(1, n_samples))
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        qwk = calculate_dr_metrics(y_true, y_pred)["QWK"]
        return avg_loss, qwk

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, desc: str = "Eval") -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates model performance on validation or test sets without gradient tracking."""
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []
        all_rank_probs: List[np.ndarray] = []

        pbar = tqdm(dataloader, desc=f"{desc} [{self.head_name}]", leave=False, unit="batch")
        for batch in pbar:
            image_key = "image" if "image" in batch else "global_img"
            images = normalize_fundus_tensor(batch[image_key], device=self.device)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(images, targets=labels, class_weights=self.class_weights, return_loss=True)
                loss = outputs.get("loss", torch.tensor(0.0))

            total_loss += loss.item() * len(labels)
            probs = outputs["probs"].cpu().numpy()
            preds = outputs["preds"].cpu().numpy()
            rp = extract_rank_probs(outputs)

            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            if rp is not None:
                all_rank_probs.extend(rp)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)
        rank_probs_np = np.array(all_rank_probs) if all_rank_probs else None

        metrics = calculate_dr_metrics(y_true, y_pred, y_probs=y_probs)
        metrics["Loss"] = float(total_loss / max(1, len(y_true)))
        if rank_probs_np is not None:
            metrics["Monotonicity_Violations"] = rank_monotonicity_violations(rank_probs_np)
        else:
            metrics["Monotonicity_Violations"] = None

        return metrics, y_true, y_pred, y_probs

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience_early_stopping: int = 12,
        resume_checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes head ablation training loop with dual checkpointing."""
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

        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"resnet50_{self.head_name}_best.pth")
        last_checkpoint_path = os.path.join(self.checkpoint_dir, f"resnet50_{self.head_name}_last.pth")

        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            ckpt = load_checkpoint(
                resume_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler if self.use_scaler else None,
                scheduler=self.lr_scheduler,
                device=str(self.device),
            )
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_qwk = ckpt.get("best_val_qwk", -1.0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", history)
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            logger.info(f"Resumed [{self.head_name}] from epoch {start_epoch} (best QWK: {best_val_qwk:.4f})")

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            head = getattr(self.model, "head", self.model)
            if hasattr(head, "set_qwk_weight") and hasattr(head, "target_qwk_weight"):
                progress = min(1.0, epoch / max(1, self.qwk_warmup_epochs))
                head.set_qwk_weight(progress * head.target_qwk_weight)

            train_loss, train_qwk = self.train_epoch(train_loader, epoch, epochs)
            val_metrics, _, _, _ = self.evaluate(val_loader, desc="Val")
            elapsed = time.time() - t0

            val_loss = val_metrics["Loss"]
            val_qwk = val_metrics["QWK"]

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_qwk)
                else:
                    self.lr_scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_qwk"].append(train_qwk)
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
            save_checkpoint(state, last_checkpoint_path, is_best=is_best, best_path=best_checkpoint_path)

            logger.info(
                f"[{self.head_name} | Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s)] "
                f"Train Loss: {train_loss:.4f} | QWK: {train_qwk:.4f} || "
                f"Val Loss: {val_loss:.4f} | QWK: {val_qwk:.4f} | "
                f"{'*BEST*' if is_best else ''}"
            )

            if epochs_no_improve >= patience_early_stopping:
                logger.info(f"Early stopping reached after {epoch} epochs.")
                break

        history["best_val_qwk"] = best_val_qwk
        if test_loader is not None and os.path.exists(best_checkpoint_path):
            load_checkpoint(best_checkpoint_path, model=self.model, device=str(self.device))
            test_metrics, _, _, _ = self.evaluate(test_loader, desc="Test")
            history["test_metrics"] = test_metrics

        return history
