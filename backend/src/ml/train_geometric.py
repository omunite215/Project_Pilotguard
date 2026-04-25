"""Train XGBoost drowsiness classifier on geometric features.

Uses the 7 geometric features extracted from facial landmarks to classify
alert vs. drowsy states. This is the lightweight, interpretable baseline
model that runs on CPU with very fast inference.

Usage:
    cd backend
    python -m src.ml.train_geometric --manifest data/processed/nthu_ddd_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.cv.face_detector import FaceDetector
from src.ml.geometric_features import FEATURE_NAMES, extract_features_from_image

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "drowsiness"


def load_manifest(manifest_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load manifest CSV and group by split.

    Args:
        manifest_path: Path to the manifest CSV.

    Returns:
        Dict mapping split name to list of (image_path, label) tuples.
    """
    splits: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}

    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split in splits:
                # Resolve path relative to data root
                img_path = str(DATA_ROOT / row["path"])
                splits[split].append((img_path, row["label"]))

    return splits


def extract_split_features(
    samples: list[tuple[str, str]],
    detector: FaceDetector,
    label_map: dict[str, int],
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract geometric features for a split.

    Args:
        samples: List of (image_path, label) tuples.
        detector: FaceDetector instance.
        label_map: Mapping from label string to integer.
        max_samples: Optional cap on number of samples to process.

    Returns:
        Tuple of (features, labels) arrays.
    """
    import cv2

    features: list[np.ndarray] = []
    labels: list[int] = []

    if max_samples is not None:
        samples = samples[:max_samples]

    for i, (path, label) in enumerate(samples):
        if label not in label_map:
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        feat = extract_features_from_image(img, detector)
        if feat is None:
            continue

        features.append(feat)
        labels.append(label_map[label])

        if (i + 1) % 500 == 0:
            logger.info("  Processed %d/%d images...", i + 1, len(samples))

    if not features:
        return np.empty((0, 7), dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.stack(features), np.array(labels, dtype=np.int32)


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with early stopping.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features.
        y_val: Validation labels.
        seed: Random seed.

    Returns:
        Trained XGBClassifier.
    """
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=seed,
        eval_metric="logloss",
        early_stopping_rounds=30,
        use_label_encoder=False,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=True,
    )

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> dict[str, float]:
    """Evaluate the model and return metrics.

    Args:
        model: Trained classifier.
        x_test: Test features.
        y_test: Test labels.
        label_names: Class label names.

    Returns:
        Dict of metric name to value.
    """
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
    }

    logger.info("Test Results:")
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  F1:        %.4f", metrics["f1"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred, target_names=label_names))

    # Feature importance
    importances = model.feature_importances_
    logger.info("Feature Importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances, strict=False), key=lambda x: -x[1]):
        logger.info("  %s: %.4f", name, imp)

    return metrics


def main() -> None:
    """Train geometric drowsiness classifier."""
    parser = argparse.ArgumentParser(description="Train XGBoost drowsiness classifier")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DATA_ROOT / "processed" / "nthu_ddd_manifest.csv"),
        help="Path to the cleaned manifest CSV",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        logger.error("Run: python -m scripts.clean_data --dataset nthu")
        return

    logger.info("Loading manifest: %s", manifest_path)
    splits = load_manifest(manifest_path)
    logger.info("  Train: %d, Val: %d, Test: %d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    label_map = {"alert": 0, "drowsy": 1}
    label_names = ["alert", "drowsy"]

    logger.info("Initializing face detector...")
    detector = FaceDetector()

    try:
        logger.info("Extracting training features...")
        t0 = time.time()
        x_train, y_train = extract_split_features(splits["train"], detector, label_map, args.max_samples)
        logger.info("  Train: %d samples in %.1fs", len(x_train), time.time() - t0)

        logger.info("Extracting validation features...")
        t0 = time.time()
        x_val, y_val = extract_split_features(splits["val"], detector, label_map, args.max_samples)
        logger.info("  Val: %d samples in %.1fs", len(x_val), time.time() - t0)

        logger.info("Extracting test features...")
        t0 = time.time()
        x_test, y_test = extract_split_features(splits["test"], detector, label_map, args.max_samples)
        logger.info("  Test: %d samples in %.1fs", len(x_test), time.time() - t0)
    finally:
        detector.close()

    if len(x_train) == 0 or len(x_val) == 0:
        logger.error("Not enough data to train. Check manifest and image paths.")
        return

    logger.info("Training XGBoost...")
    model = train_xgboost(x_train, y_train, x_val, y_val, seed=args.seed)

    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, x_test, y_test, label_names)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "geometric_xgb_v1.json"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Save metadata
    metadata = {
        "model_type": "xgboost",
        "features": FEATURE_NAMES,
        "label_map": label_map,
        "metrics": metrics,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "test_samples": len(x_test),
        "seed": args.seed,
    }
    meta_path = MODELS_DIR / "geometric_xgb_v1_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
