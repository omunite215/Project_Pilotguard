"""Train classification heads on DINOv2 features.

Supports three modes:
    1. Linear probe: 384 -> N classes (simple, fast)
    2. MLP probe: 384 -> 128 -> N classes (more expressive)
    3. Fusion: (384 + 7 geometric) -> 256 -> 128 -> N classes

All modes use:
    - Linear warmup → cosine annealing LR schedule
    - Mixed-precision training (bf16/fp16)
    - Early stopping on validation weighted F1
    - TensorBoard logging
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Result of a training run.

    Attributes:
        best_val_f1: Best validation F1 score achieved.
        best_epoch: Epoch at which best F1 was achieved.
        test_metrics: Dict of test set metrics.
        model: Trained PyTorch model.
        history: Per-epoch training history.
    """

    best_val_f1: float
    best_epoch: int
    test_metrics: dict[str, float]
    model: nn.Module
    history: list[dict[str, float]] = field(default_factory=list)


class LinearProbe(nn.Module):
    """Linear classification head with input normalization."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbe(nn.Module):
    """MLP classification head with two hidden layers and BatchNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    """Fusion classification head combining DINOv2 + geometric features."""

    def __init__(
        self,
        dinov2_dim: int = 384,
        geometric_dim: int = 7,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        total_dim = dinov2_dim + geometric_dim
        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Build a linear-warmup + cosine-annealing LR scheduler.

    Phase 1 (warmup): LR rises linearly from lr/25 to lr over warmup steps.
    Phase 2 (cosine): LR decays from lr to lr * min_lr_ratio following cosine curve.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.
        min_lr_ratio: Minimum LR as fraction of peak LR.

    Returns:
        LambdaLR scheduler (step-level granularity).
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: from 1/25 to 1.0
            return (1.0 / 25.0) + (1.0 - 1.0 / 25.0) * (current_step / max(warmup_steps, 1))
        # Cosine annealing
        progress = (current_step - warmup_steps) / max(cosine_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_classification_head(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    warmup_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    patience: int = 12,
    gradient_clip: float = 1.0,
    min_lr_ratio: float = 0.01,
    label_smoothing: float = 0.1,
    device: str = "cpu",
    tb_writer: object | None = None,
    run_name: str = "",
) -> TrainingResult:
    """Train a classification head with warmup + cosine schedule.

    Args:
        model: PyTorch model to train.
        x_train: Training features.
        y_train: Training labels (integers).
        x_val: Validation features.
        y_val: Validation labels.
        epochs: Maximum training epochs.
        warmup_epochs: Linear warmup epochs before cosine decay.
        batch_size: Batch size.
        lr: Peak learning rate (reached at end of warmup).
        weight_decay: AdamW weight decay.
        patience: Early stopping patience (epochs without improvement).
        gradient_clip: Max gradient norm for clipping.
        min_lr_ratio: Minimum LR as fraction of peak LR for cosine floor.
        device: Torch device string.
        tb_writer: Optional TensorBoard SummaryWriter.
        run_name: Prefix for TensorBoard tags.

    Returns:
        TrainingResult with best model, metrics, and training history.
    """
    dev = torch.device(device)
    model = model.to(dev)

    # Data loaders
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(dev.type == "cuda"),
    )

    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(dev)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(dev)

    # Loss with class weight balancing + label smoothing
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8,
    )

    # Warmup + Cosine schedule
    steps_per_epoch = len(train_loader)
    scheduler = _build_warmup_cosine_scheduler(
        optimizer, warmup_epochs, epochs, steps_per_epoch, min_lr_ratio,
    )

    # Mixed precision (bf16 on H200/A100, fp16 on older GPUs)
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # Training state
    best_val_f1 = 0.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: list[dict[str, float]] = []
    global_step = 0

    logger.info("=" * 60)
    logger.info("Training config:")
    logger.info("  Model params:    %d", sum(p.numel() for p in model.parameters()))
    logger.info("  Train samples:   %d", len(x_train))
    logger.info("  Val samples:     %d", len(x_val))
    logger.info("  Epochs:          %d (warmup: %d)", epochs, warmup_epochs)
    logger.info("  Batch size:      %d", batch_size)
    logger.info("  Steps/epoch:     %d", steps_per_epoch)
    logger.info("  Peak LR:         %.2e", lr)
    logger.info("  Device:          %s", dev)
    logger.info("  Mixed precision: %s (%s)", use_amp, amp_dtype if use_amp else "N/A")
    logger.info("=" * 60)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(dev, non_blocking=True), y_batch.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * x_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += x_batch.size(0)
            global_step += 1

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validate ----
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            val_logits = model(x_val_t)
            val_loss = nn.functional.cross_entropy(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_true = y_val_t.cpu().numpy()

        val_f1 = float(f1_score(val_true, val_preds, average="weighted"))
        val_acc = float(accuracy_score(val_true, val_preds))

        # Record history
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        history.append(epoch_stats)

        # TensorBoard logging
        if tb_writer is not None:
            tag = f"{run_name}/" if run_name else ""
            tb_writer.add_scalar(f"{tag}train/loss", train_loss, epoch + 1)
            tb_writer.add_scalar(f"{tag}train/acc", train_acc, epoch + 1)
            tb_writer.add_scalar(f"{tag}val/loss", val_loss, epoch + 1)
            tb_writer.add_scalar(f"{tag}val/f1", val_f1, epoch + 1)
            tb_writer.add_scalar(f"{tag}val/acc", val_acc, epoch + 1)
            tb_writer.add_scalar(f"{tag}lr", current_lr, epoch + 1)

        # Phase indicator
        phase = "warmup" if (epoch + 1) <= warmup_epochs else "cosine"

        # Log every epoch during warmup, every 5 epochs during cosine
        if (epoch + 1) <= warmup_epochs or (epoch + 1) % 5 == 0 or val_f1 > best_val_f1:
            logger.info(
                "[%s] Epoch %d/%d: loss=%.4f, val_f1=%.4f, val_acc=%.4f, lr=%.2e",
                phase, epoch + 1, epochs, train_loss, val_f1, val_acc, current_lr,
            )

        # Early stopping check (only after warmup completes)
        if (epoch + 1) > warmup_epochs:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs after warmup)",
                        epoch + 1, patience,
                    )
                    break
        elif val_f1 > best_val_f1:
            # Track best during warmup too (but don't count toward patience)
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(dev)

    logger.info("Best val F1: %.4f at epoch %d", best_val_f1, best_epoch)

    return TrainingResult(
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        test_metrics={},
        model=model,
        history=history,
    )


def evaluate_on_test(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate a trained model on the test set.

    Args:
        model: Trained PyTorch model.
        x_test: Test features.
        y_test: Test labels.
        label_names: Class label names for the report.
        device: Torch device.

    Returns:
        Dict of metric name to value.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    x_t = torch.tensor(x_test, dtype=torch.float32).to(dev)
    y_t = torch.tensor(y_test, dtype=torch.long).to(dev)

    with torch.no_grad():
        logits = model(x_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        true = y_t.cpu().numpy()

    metrics = {
        "accuracy": float(accuracy_score(true, preds)),
        "f1": float(f1_score(true, preds, average="weighted")),
        "precision": float(precision_score(true, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(true, preds, average="weighted", zero_division=0)),
    }

    logger.info("Test Results:")
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  F1:        %.4f", metrics["f1"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("\n%s", classification_report(true, preds, target_names=label_names, zero_division=0))

    return metrics
