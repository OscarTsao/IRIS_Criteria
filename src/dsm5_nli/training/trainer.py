"""Training loop with GPU optimizations."""

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def create_optimizer_and_scheduler(
    model, train_loader, num_epochs, learning_rate, weight_decay,
    warmup_ratio, use_fused
):
    """Create optimizer and scheduler.

    Args:
        model: Model to optimize
        train_loader: Training dataloader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        use_fused: Use fused AdamW

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Separate parameters: no weight decay for bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Create optimizer
    if use_fused and torch.cuda.is_available():
        try:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                fused=True,
            )
        except:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
            )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
        )

    # Create scheduler
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


class Trainer:
    """Trainer with GPU optimizations."""

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 device, use_bf16=False, use_compile=False,
                 gradient_accumulation_steps=1, max_grad_norm=1.0,
                 mlflow_enabled=True, early_stopping_patience=None):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_bf16 = use_bf16 and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mlflow_enabled = mlflow_enabled
        self.early_stopping_patience = early_stopping_patience
        self.best_epoch = 0

        # Apply torch.compile if requested
        if use_compile and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except:
                pass

        self.model.to(self.device)
        self.best_val_f1 = 0

    def train(self, num_epochs, fold):
        """Train for num_epochs."""
        epochs_without_improvement = 0
        patience = self.early_stopping_patience

        for epoch in range(1, num_epochs + 1):
            # Train epoch
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward
                with torch.amp.autocast('cuda', enabled=self.use_bf16, dtype=torch.bfloat16):
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.gradient_accumulation_steps

                # Backward
                loss.backward()
                total_loss += loss.item() * self.gradient_accumulation_steps

                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{total_loss / (step+1):.4f}"})

            # Evaluate
            val_f1 = self._evaluate()
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                patience is not None
                and patience > 0
                and epochs_without_improvement >= patience
            ):
                tqdm.write(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best F1 {self.best_val_f1:.4f} at epoch {self.best_epoch})"
                )
                break

    @torch.no_grad()
    def _evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', enabled=self.use_bf16, dtype=torch.bfloat16):
                outputs = self.model(**batch)

            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate F1
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average="binary")
        return f1
