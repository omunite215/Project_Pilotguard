"""Unified HPC training script for all PilotGuard ML models.

Runs the full training pipeline on GPU:
    1. Pre-compute DINOv2 features for all datasets (GPU-accelerated)
    2. Train XGBoost geometric classifier (CPU, fast)
    3. Train DINOv2 classification heads per dataset (GPU)
    4. Train HMM on labeled sequences (CPU)
    5. Export best models to ONNX

Usage (on HPC):
    python -m scripts.train_hpc --profile nthu_drowsiness
    python -m scripts.train_hpc --profile all
    python -m scripts.train_hpc --stage features  # Only pre-compute features
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = BACKEND_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"


def load_config() -> dict:
    """Load the global training config."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_manifest(manifest_path: Path) -> dict[str, list[dict[str, str]]]:
    """Load manifest and group rows by split."""
    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split in splits:
                splits[split].append(row)
    return splits


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        logger.info("GPU detected: %s (%.1f GB)", name, mem / (1024**3))
        return "cuda"
    logger.warning("No GPU detected — training will be slow")
    return "cpu"


# ---------------------------------------------------------------------------
# Stage 1: Pre-compute DINOv2 features
# ---------------------------------------------------------------------------

def precompute_dinov2_features(
    profile_name: str,
    profile: dict,
    device: str,
) -> Path:
    """Pre-compute DINOv2 features for a dataset and cache to disk.

    Args:
        profile_name: Name of the training profile.
        profile: Profile config dict.
        device: Torch device.

    Returns:
        Path to the saved .npz feature cache.
    """
    from src.ml.dinov2_features import precompute_features

    manifest_path = PROJECT_ROOT / profile["manifest"]
    cache_dir = DATA_ROOT / "features" / profile["dataset"]
    cache_path = cache_dir / "dinov2_features.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        logger.info(
            "DINOv2 features already cached: %s (%d samples)",
            cache_path, len(data["features"]),
        )
        return cache_path

    logger.info("Pre-computing DINOv2 features for %s...", profile_name)
    splits = load_manifest(manifest_path)

    # Collect all image paths
    all_paths: list[str] = []
    for split_name in ["train", "val", "test"]:
        for row in splits[split_name]:
            all_paths.append(str(DATA_ROOT / row["path"]))

    t0 = time.time()
    precompute_features(
        image_paths=all_paths,
        output_path=cache_path,
        device=device,
        batch_size=64,
    )
    elapsed = time.time() - t0
    logger.info("DINOv2 feature extraction: %.1fs (%.1f img/s)", elapsed, len(all_paths) / elapsed)

    return cache_path


# ---------------------------------------------------------------------------
# Stage 2: Train XGBoost geometric classifier
# ---------------------------------------------------------------------------

def train_xgboost_model(config: dict) -> None:
    """Train XGBoost on NTHU-DDD geometric features."""
    from src.cv.face_detector import FaceDetector
    from src.ml.geometric_features import FEATURE_NAMES
    from src.ml.train_geometric import evaluate_model, load_manifest, train_xgboost

    seed = config["seed"]
    manifest_path = DATA_ROOT / "processed" / "nthu_ddd_manifest.csv"

    if not manifest_path.exists():
        logger.error("NTHU manifest not found: %s", manifest_path)
        return

    logger.info("=" * 60)
    logger.info("Training XGBoost drowsiness classifier")
    logger.info("=" * 60)

    splits = load_manifest(manifest_path)
    label_map = {"alert": 0, "drowsy": 1}
    label_names = ["alert", "drowsy"]

    detector = FaceDetector()
    try:
        from src.ml.train_geometric import extract_split_features

        logger.info("Extracting geometric features (this takes a while on CPU)...")
        t0 = time.time()
        x_train, y_train = extract_split_features(splits["train"], detector, label_map)
        logger.info("  Train: %d samples (%.1fs)", len(x_train), time.time() - t0)

        t0 = time.time()
        x_val, y_val = extract_split_features(splits["val"], detector, label_map)
        logger.info("  Val: %d samples (%.1fs)", len(x_val), time.time() - t0)

        t0 = time.time()
        x_test, y_test = extract_split_features(splits["test"], detector, label_map)
        logger.info("  Test: %d samples (%.1fs)", len(x_test), time.time() - t0)
    finally:
        detector.close()

    if len(x_train) == 0:
        logger.error("No training data extracted")
        return

    model = train_xgboost(x_train, y_train, x_val, y_val, seed=seed)
    metrics = evaluate_model(model, x_test, y_test, label_names)

    # Save
    out_dir = MODELS_DIR / "drowsiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "geometric_xgb_v1.json"))

    meta = {
        "model_type": "xgboost",
        "features": FEATURE_NAMES,
        "label_map": label_map,
        "metrics": metrics,
        "train_samples": len(x_train),
        "seed": seed,
    }
    with open(out_dir / "geometric_xgb_v1_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("XGBoost model saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Stage 3: Train DINOv2 classification heads
# ---------------------------------------------------------------------------

def train_dinov2_profile(
    profile_name: str,
    profile: dict,
    config: dict,
    device: str,
) -> None:
    """Train a DINOv2 classification head for a specific dataset profile.

    Args:
        profile_name: e.g. "nthu_drowsiness".
        profile: The profile dict from training_config.yaml.
        config: Full config dict.
        device: Torch device.
    """
    from torch.utils.tensorboard import SummaryWriter

    from src.ml.train_dinov2_head import (
        FusionHead,
        LinearProbe,
        MLPProbe,
        evaluate_on_test,
        train_classification_head,
    )

    manifest_path = PROJECT_ROOT / profile["manifest"]
    if not manifest_path.exists():
        logger.error("Manifest not found: %s — skipping %s", manifest_path, profile_name)
        return

    logger.info("=" * 60)
    logger.info("Training DINOv2 head: %s", profile_name)
    logger.info("  Dataset:  %s", profile["dataset"])
    logger.info("  Task:     %s (%d classes)", profile["task"], profile["num_classes"])
    logger.info("  Head:     %s", profile["head"])
    logger.info("  Epochs:   %d (warmup: %d)", profile["epochs"], profile["warmup_epochs"])
    logger.info("  Batch:    %d, LR: %.2e", profile["batch_size"], profile["lr"])
    logger.info("=" * 60)

    # Load features from cache
    cache_path = DATA_ROOT / "features" / profile["dataset"] / "dinov2_features.npz"
    if not cache_path.exists():
        logger.error("Feature cache not found: %s — run --stage features first", cache_path)
        return

    feature_data = np.load(cache_path)
    all_features = feature_data["features"]
    valid_indices = set(feature_data["valid_indices"].tolist())

    # Load manifest and match features to splits
    splits = load_manifest(manifest_path)
    all_rows: list[dict[str, str]] = []
    for split_name in ["train", "val", "test"]:
        for row in splits[split_name]:
            all_rows.append(row)

    # Build label mapping
    labels_set = sorted({row["label"] for row in all_rows})
    label_map = {label: idx for idx, label in enumerate(labels_set)}
    label_names = list(label_map.keys())
    logger.info("Label map: %s", label_map)

    # Split features by train/val/test (matching valid_indices order)
    split_features: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    feature_idx = 0
    for row_idx, row in enumerate(all_rows):
        if row_idx in valid_indices:
            split_name = row["split"]
            split_features[split_name].append(all_features[feature_idx])
            split_labels[split_name].append(label_map[row["label"]])
            feature_idx += 1

    x_train = np.stack(split_features["train"]) if split_features["train"] else np.empty((0, 384))
    y_train = np.array(split_labels["train"], dtype=np.int64)
    x_val = np.stack(split_features["val"]) if split_features["val"] else np.empty((0, 384))
    y_val = np.array(split_labels["val"], dtype=np.int64)
    x_test = np.stack(split_features["test"]) if split_features["test"] else np.empty((0, 384))
    y_test = np.array(split_labels["test"], dtype=np.int64)

    logger.info("Features loaded — train: %d, val: %d, test: %d", len(x_train), len(x_val), len(x_test))

    if len(x_train) == 0 or len(x_val) == 0:
        logger.error("Insufficient data for training — skipping %s", profile_name)
        return

    # Build model
    head_type = profile["head"]
    num_classes = profile["num_classes"]
    input_dim = 384  # DINOv2 ViT-S/14

    if head_type == "linear":
        model = LinearProbe(input_dim, num_classes)
    elif head_type == "mlp":
        model = MLPProbe(input_dim, 256, num_classes)
    elif head_type == "fusion":
        # Fusion needs geometric features; fall back to MLP with larger hidden dim
        logger.info("Fusion head selected — using MLP with DINOv2-only features (384-dim)")
        model = MLPProbe(input_dim, 256, num_classes)
    else:
        logger.error("Unknown head type: %s", head_type)
        return

    # TensorBoard
    tb_dir = BACKEND_ROOT / "runs" / profile_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Train
    early_stopping_cfg = config["training"]["early_stopping"]
    min_lr = config["training"]["scheduler"].get("min_lr", 1e-6)
    peak_lr = profile["lr"]
    min_lr_ratio = min_lr / peak_lr if peak_lr > 0 else 0.01

    result = train_classification_head(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=profile["epochs"],
        warmup_epochs=profile["warmup_epochs"],
        batch_size=profile["batch_size"],
        lr=peak_lr,
        weight_decay=profile["weight_decay"],
        patience=early_stopping_cfg["patience"],
        gradient_clip=config["training"]["gradient_clip"],
        min_lr_ratio=min_lr_ratio,
        label_smoothing=0.1,
        device=device,
        tb_writer=writer,
        run_name=profile_name,
    )

    writer.close()

    # Evaluate on test set
    if len(x_test) > 0:
        test_metrics = evaluate_on_test(result.model, x_test, y_test, label_names, device)
    else:
        test_metrics = {}
        logger.warning("No test data — skipping evaluation")

    # Save model
    out_dir = MODELS_DIR / profile["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{profile_name}_best.pt"
    torch.save({
        "model_state_dict": result.model.state_dict(),
        "config": profile,
        "label_map": label_map,
        "best_epoch": result.best_epoch,
        "best_val_f1": result.best_val_f1,
        "test_metrics": test_metrics,
        "history": result.history,
    }, model_path)
    logger.info("Model saved: %s", model_path)

    # Save metadata as JSON for easy inspection
    meta_path = out_dir / f"{profile_name}_metadata.json"
    meta = {
        "profile": profile_name,
        "dataset": profile["dataset"],
        "task": profile["task"],
        "head": profile["head"],
        "num_classes": num_classes,
        "label_map": label_map,
        "epochs_trained": result.best_epoch,
        "best_val_f1": result.best_val_f1,
        "test_metrics": test_metrics,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "test_samples": len(x_test),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved: %s", meta_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_PROFILES = ["nthu_drowsiness", "uta_drowsiness", "affectnet_emotion", "disfa_au"]


def main() -> None:
    """Run HPC training pipeline."""
    parser = argparse.ArgumentParser(description="PilotGuard HPC Training Pipeline")
    parser.add_argument(
        "--profile", type=str, default="all",
        choices=[*ALL_PROFILES, "all"],
        help="Training profile to run (default: all)",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["features", "xgboost", "dinov2", "all"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--no-xgboost", action="store_true",
        help="Skip XGBoost training (useful when only running DINOv2 heads)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()
    set_seed(config["seed"])
    device = get_device()

    profiles_to_run = ALL_PROFILES if args.profile == "all" else [args.profile]

    t_start = time.time()

    # Stage 1: Pre-compute DINOv2 features
    if args.stage in ("features", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Pre-computing DINOv2 features")
        logger.info("=" * 60)
        for name in profiles_to_run:
            profile = config["training"]["profiles"][name]
            precompute_dinov2_features(name, profile, device)

    # Stage 2: XGBoost (CPU)
    if args.stage in ("xgboost", "all") and not args.no_xgboost:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Training XGBoost geometric classifier")
        logger.info("=" * 60)
        train_xgboost_model(config)

    # Stage 3: DINOv2 classification heads (GPU)
    if args.stage in ("dinov2", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Training DINOv2 classification heads")
        logger.info("=" * 60)
        for name in profiles_to_run:
            profile = config["training"]["profiles"][name]
            train_dinov2_profile(name, profile, config, device)

    total = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("ALL TRAINING COMPLETE — Total time: %.1f min", total / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
