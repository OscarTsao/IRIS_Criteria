"""Unified trainer for IRIS and generic binary classifiers.

Supports:
- IRIS architecture with retrieval-based inputs
- Token-based models with tensor inputs
- Gradient accumulation for large effective batch sizes
- Mixed precision training (bf16/fp16)
- Early stopping with patience
- Model checkpointing
- Multiple loss functions (BCE, weighted BCE, focal)
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable
from pathlib import Path
import logging
from tqdm import tqdm

from .losses import create_loss_function

logger = logging.getLogger(__name__)


class Trainer:
    """Unified trainer for binary classification models.

    Supports IRIS and other binary classifiers with configurable training options.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[nn.Module] = None,
        loss_type: str = "bce",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        amp_dtype: str = "bfloat16",
        early_stopping_patience: int = 5,
        checkpoint_dir: Optional[str] = None,
        save_best_only: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model (IRIS or tensor-based classifier)
            optimizer: Optimizer instance
            loss_fn: Loss function (if None, created from loss_type)
            loss_type: Type of loss ('bce', 'weighted_bce', 'focal')
            scheduler: Learning rate scheduler (optional)
            device: Device for training
            gradient_accumulation_steps: Accumulate gradients over N steps
            max_grad_norm: Max gradient norm for clipping (0 = no clipping)
            use_amp: Use automatic mixed precision
            amp_dtype: AMP dtype ('bfloat16' or 'float16')
            early_stopping_patience: Patience for early stopping (0 = disabled)
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save best model checkpoint
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_best_only = save_best_only

        # Create checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup loss function
        if loss_fn is None:
            loss_fn = create_loss_function(loss_type)
        self.loss_fn = loss_fn

        # Setup AMP scaler
        self.scaler = None
        if use_amp:
            if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                # BF16 doesn't need gradient scaling
                self.scaler = None
            elif amp_dtype == "float16":
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                logger.warning(
                    f"AMP dtype {amp_dtype} not supported, disabling AMP"
                )
                self.use_amp = False

        # Training state
        self.best_val_loss = float("inf")
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.current_epoch = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            metrics: Dict with 'loss' and 'accuracy'
        """
        self.model.train()
        self.current_epoch = epoch

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Reset gradients
        self.optimizer.zero_grad()

        # Progress bar
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for batch_idx, batch in pbar:
            # Forward pass with gradient accumulation
            loss, correct, batch_size = self._train_step(batch, batch_idx)

            # Update metrics
            total_loss += loss * batch_size
            total_correct += correct
            total_samples += batch_size

            # Update progress bar
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})

        # Final metrics
        metrics = {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

        return metrics

    def _train_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Tuple[float, int, int]:
        """
        Single training step with gradient accumulation.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            loss: Loss value (float)
            correct: Number of correct predictions
            batch_size: Batch size
        """
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass
        if self.use_amp and self.amp_dtype:
            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                logits = self._forward(batch)
                labels = batch["label"]
                loss = self.loss_fn(logits, labels)
        else:
            logits = self._forward(batch)
            labels = batch["label"]
            loss = self.loss_fn(logits, labels)

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every N accumulation steps
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step (if per-step scheduler)
            if self.scheduler and hasattr(self.scheduler, "step_every_batch"):
                self.scheduler.step()

            # Reset gradients
            self.optimizer.zero_grad()
            self.global_step += 1

        # Compute accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            correct = (preds == labels.long()).sum().item()

        return loss.item() * self.gradient_accumulation_steps, correct, len(labels)

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass (handles both IRIS and tensor-based inputs).

        Args:
            batch: Batch dictionary with model inputs

        Returns:
            logits: [batch_size] or [batch_size, num_classes]
        """
        # Check if IRIS-style model (has build_retriever method)
        if hasattr(self.model, "build_retriever"):
            # IRIS model
            outputs = self.model(
                post_texts=batch.get("post_text"),
                criterion_texts=batch.get("criterion_text"),
            )
            logits = outputs.get("logits", outputs)
        else:
            # Generic token-based model
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )

        return logits.view(-1)  # Ensure [batch_size] shape

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        metric_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader
            metric_fn: Optional metric function (predictions, labels) -> score

        Returns:
            metrics: Dict with 'loss', 'accuracy', and optional custom metric
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            logits = self._forward(batch)
            labels = batch["label"]

            # Compute loss
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)

            # Compute predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_samples += len(labels)

            # Store for custom metric
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        # Compute metrics
        metrics = {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

        # Custom metric
        if metric_fn:
            custom_score = metric_fn(all_preds, all_labels)
            metrics["custom_metric"] = custom_score

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        metric_fn: Optional[Callable] = None,
        metric_name: str = "custom_metric",
        higher_is_better: bool = True,
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            metric_fn: Optional metric function for early stopping
            metric_name: Name of metric to monitor ('loss' or 'custom_metric')
            higher_is_better: True if higher metric is better

        Returns:
            history: Dict with training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        if metric_fn:
            history[metric_name] = []

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])

            # Validation
            val_metrics = self.evaluate(val_loader, metric_fn)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            if metric_fn:
                history[metric_name].append(val_metrics["custom_metric"])

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"train_loss: {train_metrics['loss']:.4f}, "
                f"val_loss: {val_metrics['loss']:.4f}, "
                f"val_acc: {val_metrics['accuracy']:.4f}"
            )

            if metric_fn:
                logger.info(f"  {metric_name}: {val_metrics['custom_metric']:.4f}")

            # Learning rate scheduling (if per-epoch scheduler)
            if self.scheduler and not hasattr(self.scheduler, "step_every_batch"):
                self.scheduler.step()

            # Early stopping and checkpointing
            should_stop = self._check_early_stopping(
                val_metrics, metric_name, higher_is_better, epoch
            )

            if should_stop:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"({self.early_stopping_patience} epochs without improvement)"
                )
                break

        return history

    def _check_early_stopping(
        self,
        val_metrics: Dict[str, float],
        metric_name: str,
        higher_is_better: bool,
        epoch: int,
    ) -> bool:
        """
        Check early stopping condition and save checkpoints.

        Args:
            val_metrics: Validation metrics
            metric_name: Metric to monitor
            higher_is_better: True if higher is better
            epoch: Current epoch

        Returns:
            should_stop: True if training should stop
        """
        # Get metric to monitor
        if metric_name == "loss":
            current_metric = val_metrics["loss"]
        else:
            current_metric = val_metrics.get(metric_name, val_metrics["loss"])

        # Check if improved
        if higher_is_better:
            improved = current_metric > self.best_val_metric
        else:
            improved = current_metric < self.best_val_metric

        if improved:
            self.best_val_metric = current_metric
            self.epochs_without_improvement = 0

            # Save checkpoint
            if self.checkpoint_dir:
                self.save_checkpoint(epoch, "best")
                logger.info(f"  New best {metric_name}: {current_metric:.4f}")
        else:
            self.epochs_without_improvement += 1

            # Save checkpoint (if not save_best_only)
            if self.checkpoint_dir and not self.save_best_only:
                self.save_checkpoint(epoch, f"epoch_{epoch}")

        # Check early stopping
        if (
            self.early_stopping_patience > 0
            and self.epochs_without_improvement >= self.early_stopping_patience
        ):
            return True

        return False

    def save_checkpoint(self, epoch: int, name: str = "checkpoint"):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            name: Checkpoint name
        """
        if not self.checkpoint_dir:
            return

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "global_step": self.global_step,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_metric = checkpoint.get("best_val_metric", 0.0)
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
