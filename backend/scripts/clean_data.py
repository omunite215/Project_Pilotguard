"""Run data cleaning pipeline for all PilotGuard datasets.

Usage:
    cd backend
    python -m scripts.clean_data [--dataset DATASET] [--sample-rate N]

Options:
    --dataset: Clean a specific dataset (nthu, uta, affectnet, disfa, all).
               Default: all
    --sample-rate: For NTHU-DDD, keep every Nth frame (default 1 = keep all).
    --uta-fps: Frame extraction rate for UTA-RLDD videos (default 2.0).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.data.cleaning import (
    AffectNetCleaner,
    DISFACleaner,
    ImageRecord,
    NTHUDDDCleaner,
    UTARLDDCleaner,
    stratified_split_by_subject,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"


def clean_nthu(sample_rate: int = 1) -> None:
    """Clean NTHU-DDD dataset."""
    logger.info("=" * 60)
    logger.info("Cleaning NTHU-DDD (sample_rate=%d)", sample_rate)
    logger.info("=" * 60)

    t0 = time.time()
    cleaner = NTHUDDDCleaner(
        data_dir=RAW_DIR / "nthu-ddd",
        sample_rate=sample_rate,
    )
    records, stats = cleaner.clean()

    # Apply splits
    records = stratified_split_by_subject(records)

    # Write manifest
    manifest_path = PROCESSED_DIR / "nthu_ddd_manifest.csv"
    write_manifest(records, manifest_path)

    elapsed = time.time() - t0
    logger.info("NTHU-DDD cleaning complete in %.1fs", elapsed)
    logger.info("\n%s", stats.summary())
    _log_split_distribution(records)


def clean_uta(extract_fps: float = 2.0) -> None:
    """Clean UTA-RLDD dataset (extract frames from videos)."""
    logger.info("=" * 60)
    logger.info("Cleaning UTA-RLDD (extract_fps=%.1f)", extract_fps)
    logger.info("=" * 60)

    t0 = time.time()
    output_dir = PROCESSED_DIR / "uta_frames"
    cleaner = UTARLDDCleaner(
        data_dir=RAW_DIR / "uta-rldd",
        output_dir=output_dir,
        extract_fps=extract_fps,
    )
    records, stats = cleaner.clean()

    # Apply splits
    records = stratified_split_by_subject(records)

    # Write manifest
    manifest_path = PROCESSED_DIR / "uta_rldd_manifest.csv"
    write_manifest(records, manifest_path)

    elapsed = time.time() - t0
    logger.info("UTA-RLDD cleaning complete in %.1fs", elapsed)
    logger.info("\n%s", stats.summary())
    _log_split_distribution(records)


def clean_affectnet() -> None:
    """Clean AffectNet dataset (filter to aviation emotions)."""
    logger.info("=" * 60)
    logger.info("Cleaning AffectNet")
    logger.info("=" * 60)

    t0 = time.time()
    cleaner = AffectNetCleaner(data_dir=RAW_DIR / "affectnet")
    records, stats = cleaner.clean()

    # Apply splits
    records = stratified_split_by_subject(records)

    # Write manifest
    manifest_path = PROCESSED_DIR / "affectnet_manifest.csv"
    write_manifest(records, manifest_path)

    elapsed = time.time() - t0
    logger.info("AffectNet cleaning complete in %.1fs", elapsed)
    logger.info("\n%s", stats.summary())
    _log_split_distribution(records)


def clean_disfa() -> None:
    """Clean DISFA dataset (pair images with AU labels)."""
    logger.info("=" * 60)
    logger.info("Cleaning DISFA")
    logger.info("=" * 60)

    t0 = time.time()
    cleaner = DISFACleaner(data_dir=RAW_DIR / "disfa")
    records, stats = cleaner.clean()

    # Apply splits
    records = stratified_split_by_subject(records)

    # Write manifest
    manifest_path = PROCESSED_DIR / "disfa_manifest.csv"
    write_manifest(records, manifest_path)

    elapsed = time.time() - t0
    logger.info("DISFA cleaning complete in %.1fs", elapsed)
    logger.info("\n%s", stats.summary())
    _log_split_distribution(records)


def _log_split_distribution(records: list[ImageRecord]) -> None:
    """Log the train/val/test split distribution."""
    splits: dict[str, dict[str, int]] = {}
    for r in records:
        splits.setdefault(r.split, {})
        splits[r.split][r.label] = splits[r.split].get(r.label, 0) + 1

    for split_name in ("train", "val", "test"):
        if split_name in splits:
            total = sum(splits[split_name].values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(splits[split_name].items()))
            logger.info("  %s: %d (%s)", split_name, total, detail)


def main() -> None:
    """Run the cleaning pipeline."""
    parser = argparse.ArgumentParser(description="PilotGuard data cleaning pipeline")
    parser.add_argument(
        "--dataset",
        choices=["nthu", "uta", "affectnet", "disfa", "all"],
        default="all",
        help="Which dataset to clean (default: all)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="NTHU-DDD: keep every Nth frame (default: 1 = all)",
    )
    parser.add_argument(
        "--uta-fps",
        type=float,
        default=2.0,
        help="UTA-RLDD: frame extraction rate (default: 2.0)",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("nthu", "all"):
        clean_nthu(sample_rate=args.sample_rate)

    if args.dataset in ("uta", "all"):
        clean_uta(extract_fps=args.uta_fps)

    if args.dataset in ("affectnet", "all"):
        clean_affectnet()

    if args.dataset in ("disfa", "all"):
        clean_disfa()

    logger.info("All cleaning complete!")


if __name__ == "__main__":
    main()
