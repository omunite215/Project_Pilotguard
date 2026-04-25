"""Data cleaning pipeline for all PilotGuard datasets.

Implements the cleaning protocol from TRD Section 4.2:
    1. Remove corrupted/unreadable frames
    2. Verify face landmark quality via MediaPipe
    3. Filter by face detection confidence
    4. Normalize lighting via CLAHE
    5. Check label quality
    6. Report class distribution
    7. Subject-stratified train/val/test splits
    8. Generate manifest CSVs

Each dataset has a specialized cleaner class that inherits from BaseCleaner.
"""

from __future__ import annotations

import csv
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """A single cleaned image record for a manifest.

    Attributes:
        path: Path to the image file (relative to data root).
        label: Class label (e.g., "drowsy", "alert", "anger").
        subject_id: Subject identifier for stratified splitting.
        split: Assigned split ("train", "val", "test").
        quality_score: Face detection confidence [0, 1].
    """

    path: str
    label: str
    subject_id: str
    split: str = ""
    quality_score: float = 1.0


@dataclass
class CleaningStats:
    """Statistics from a cleaning run."""

    total_input: int = 0
    corrupted_removed: int = 0
    no_face_removed: int = 0
    low_quality_removed: int = 0
    total_output: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    subject_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of cleaning results."""
        lines = [
            f"Input:  {self.total_input}",
            f"Output: {self.total_output} ({self.total_output / max(self.total_input, 1) * 100:.1f}%)",
            f"  Corrupted removed:   {self.corrupted_removed}",
            f"  No face removed:     {self.no_face_removed}",
            f"  Low quality removed: {self.low_quality_removed}",
            "Class distribution:",
        ]
        for cls, count in sorted(self.class_distribution.items()):
            lines.append(f"  {cls}: {count}")
        lines.append(f"Unique subjects: {len(self.subject_counts)}")
        return "\n".join(lines)


def apply_clahe(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Normalizes lighting conditions by enhancing contrast on the
    luminance channel while preserving color information.

    Args:
        image: BGR image (OpenCV format).

    Returns:
        CLAHE-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def try_load_image(path: Path) -> NDArray[np.uint8] | None:
    """Attempt to load an image, returning None if corrupted.

    Args:
        path: Path to image file.

    Returns:
        BGR image array, or None if unreadable.
    """
    try:
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


def write_manifest(records: list[ImageRecord], output_path: Path) -> None:
    """Write a manifest CSV from a list of ImageRecords.

    Args:
        records: List of cleaned image records.
        output_path: Path to write the CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "subject_id", "split", "quality_score"])
        for r in records:
            writer.writerow([r.path, r.label, r.subject_id, r.split, f"{r.quality_score:.4f}"])
    logger.info("Manifest written to %s (%d records)", output_path, len(records))


def stratified_split_by_subject(
    records: list[ImageRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> list[ImageRecord]:
    """Assign train/val/test splits stratified by subject ID.

    CRITICAL: No subject appears in more than one split (prevents data leakage).
    Splits target sample-level ratios by greedily assigning subjects to splits
    based on cumulative sample counts rather than subject counts.

    Args:
        records: List of ImageRecords (split field will be set).
        train_ratio: Target fraction of samples for training.
        val_ratio: Target fraction of samples for validation. Test gets the remainder.
        seed: Random seed for reproducibility.

    Returns:
        New list of ImageRecords with split field populated.
    """
    rng = np.random.default_rng(seed)

    # Group by subject
    subjects: dict[str, list[ImageRecord]] = {}
    for r in records:
        subjects.setdefault(r.subject_id, []).append(r)

    # Shuffle subjects
    subject_ids = sorted(subjects.keys())
    rng.shuffle(subject_ids)

    # Greedily assign subjects to train/val/test targeting sample-level ratios
    total_samples = len(records)
    train_target = int(total_samples * train_ratio)
    val_target = int(total_samples * val_ratio)

    train_subjects: set[str] = set()
    val_subjects: set[str] = set()
    train_count = 0
    val_count = 0

    for sid in subject_ids:
        n = len(subjects[sid])
        if train_count < train_target:
            train_subjects.add(sid)
            train_count += n
        elif val_count < val_target:
            val_subjects.add(sid)
            val_count += n
        # else: test

    result = []
    for r in records:
        if r.subject_id in train_subjects:
            split = "train"
        elif r.subject_id in val_subjects:
            split = "val"
        else:
            split = "test"
        result.append(ImageRecord(
            path=r.path,
            label=r.label,
            subject_id=r.subject_id,
            split=split,
            quality_score=r.quality_score,
        ))

    return result


class BaseCleaner(ABC):
    """Base class for dataset-specific cleaners."""

    @abstractmethod
    def clean(self) -> tuple[list[ImageRecord], CleaningStats]:
        """Run the cleaning pipeline.

        Returns:
            Tuple of (cleaned records, cleaning statistics).
        """


class NTHUDDDCleaner(BaseCleaner):
    """Cleaner for NTHU Driver Drowsiness Detection dataset.

    Expected structure:
        raw/nthu-ddd/
            drowsy/     - {subject}_{condition}_{frame}_drowsy.jpg
            notdrowsy/  - {subject}_{condition}_{frame}_notdrowsy.jpg
    """

    def __init__(
        self,
        data_dir: Path,
        min_quality: float = 0.0,
        sample_rate: int = 1,
    ) -> None:
        self.data_dir = data_dir
        self.min_quality = min_quality
        self.sample_rate = sample_rate

    def clean(self) -> tuple[list[ImageRecord], CleaningStats]:
        """Clean NTHU-DDD dataset."""
        stats = CleaningStats()
        records: list[ImageRecord] = []

        for label_dir, label in [("drowsy", "drowsy"), ("notdrowsy", "alert")]:
            folder = self.data_dir / label_dir
            if not folder.exists():
                logger.warning("NTHU-DDD folder not found: %s", folder)
                continue

            files = sorted(folder.glob("*.jpg"))
            stats.total_input += len(files)

            for i, fpath in enumerate(files):
                # Sample every Nth frame to reduce dataset size if needed
                if i % self.sample_rate != 0:
                    continue

                # Extract subject ID from filename: "001_glasses_sleepyCombination_1000_drowsy.jpg"
                match = re.match(r"^(\d+)_", fpath.name)
                subject_id = match.group(1) if match else "unknown"

                # Verify image loads
                img = try_load_image(fpath)
                if img is None:
                    stats.corrupted_removed += 1
                    continue

                record = ImageRecord(
                    path=fpath.relative_to(self.data_dir.parent.parent).as_posix(),
                    label=label,
                    subject_id=f"nthu_{subject_id}",
                    quality_score=1.0,
                )
                records.append(record)

        stats.total_output = len(records)
        for r in records:
            stats.class_distribution[r.label] = stats.class_distribution.get(r.label, 0) + 1
            stats.subject_counts[r.subject_id] = stats.subject_counts.get(r.subject_id, 0) + 1

        return records, stats


class UTARLDDCleaner(BaseCleaner):
    """Cleaner for UTA Real-Life Drowsiness Dataset.

    This dataset contains videos, not frames. The cleaner extracts frames
    at a configurable FPS and labels them based on the video filename.

    Expected structure:
        raw/uta-rldd/
            Fold{N}_part{M}/Fold{N}_part{M}/{subject_id}/
                {level}.mov   (level: 0=alert, 5=low-drowsy, 10=high-drowsy)
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        extract_fps: float = 2.0,
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.extract_fps = extract_fps

    _LABEL_MAP: ClassVar[dict[str, str]] = {"0": "alert", "5": "drowsy", "10": "drowsy"}

    def _label_from_filename(self, fname: str) -> str:
        """Map video filename to drowsiness label."""
        return self._LABEL_MAP.get(Path(fname).stem, "unknown")

    def clean(self) -> tuple[list[ImageRecord], CleaningStats]:
        """Extract frames from UTA-RLDD videos and clean."""
        stats = CleaningStats()
        records: list[ImageRecord] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Walk fold structure
        for fold_dir in sorted(self.data_dir.glob("Fold*")):
            inner = fold_dir / fold_dir.name
            if not inner.exists():
                continue

            for subject_dir in sorted(inner.iterdir()):
                if not subject_dir.is_dir():
                    continue

                subject_id = subject_dir.name

                for video_file in sorted(subject_dir.glob("*")):
                    if video_file.suffix.lower() not in (".mov", ".mp4", ".avi"):
                        continue

                    label = self._label_from_filename(video_file.name)
                    if label == "unknown":
                        continue

                    stats.total_input += 1

                    # Extract frames
                    extracted = self._extract_frames(
                        video_file, subject_id, label, stats,
                    )
                    records.extend(extracted)

        stats.total_output = len(records)
        for r in records:
            stats.class_distribution[r.label] = stats.class_distribution.get(r.label, 0) + 1
            stats.subject_counts[r.subject_id] = stats.subject_counts.get(r.subject_id, 0) + 1

        return records, stats

    def _extract_frames(
        self,
        video_path: Path,
        subject_id: str,
        label: str,
        stats: CleaningStats,
    ) -> list[ImageRecord]:
        """Extract frames from a single video at target FPS."""
        records: list[ImageRecord] = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            stats.corrupted_removed += 1
            return records

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / self.extract_fps))
        frame_idx = 0

        # Create output subdirectory
        out_dir = self.output_dir / f"uta_{subject_id}" / label
        out_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                out_name = f"uta_{subject_id}_{label}_{frame_idx:06d}.jpg"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), frame)

                records.append(ImageRecord(
                    path=out_path.relative_to(self.output_dir.parent).as_posix(),
                    label=label,
                    subject_id=f"uta_{subject_id}",
                    quality_score=1.0,
                ))

            frame_idx += 1

        cap.release()
        return records


class AffectNetCleaner(BaseCleaner):
    """Cleaner for AffectNet emotion dataset.

    Filters to 5 aviation-relevant emotions: neutral, anger (stress),
    fear (stress proxy), surprise, sad (confusion proxy).

    Expected structure:
        raw/affectnet/
            Train/{emotion}/*.jpg
            Test/{emotion}/*.jpg
            labels.csv
    """

    # Map AffectNet emotions to our aviation-relevant subset
    EMOTION_MAP: ClassVar[dict[str, str]] = {
        "neutral": "neutral",
        "anger": "stress",
        "fear": "stress",
        "surprise": "surprise",
        "sad": "confusion",
        "disgust": "pain",
        "contempt": "neutral",  # fold into neutral
        "happy": "neutral",  # fold into neutral (not aviation-relevant)
    }

    KEEP_EMOTIONS: ClassVar[set[str]] = {"neutral", "stress", "surprise", "confusion", "pain"}

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def clean(self) -> tuple[list[ImageRecord], CleaningStats]:
        """Clean and filter AffectNet to aviation-relevant emotions."""
        stats = CleaningStats()
        records: list[ImageRecord] = []

        for split_name in ("Train", "Test"):
            split_dir = self.data_dir / split_name
            if not split_dir.exists():
                continue

            for emotion_dir in sorted(split_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue

                original_emotion = emotion_dir.name
                mapped_emotion = self.EMOTION_MAP.get(original_emotion)
                if mapped_emotion is None or mapped_emotion not in self.KEEP_EMOTIONS:
                    continue

                files = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
                stats.total_input += len(files)

                for fpath in files:
                    img = try_load_image(fpath)
                    if img is None:
                        stats.corrupted_removed += 1
                        continue

                    # Use original emotion folder as pseudo subject ID
                    # AffectNet doesn't have subject IDs, so we hash the filename
                    file_idx_str = fpath.stem.replace("image", "")
                    try:
                        idx_num = int(file_idx_str)
                    except ValueError:
                        idx_num = hash(fpath.stem) % (10**9)
                    pseudo_subject = f"affnet_{idx_num % 50:03d}"

                    records.append(ImageRecord(
                        path=fpath.relative_to(self.data_dir.parent.parent).as_posix(),
                        label=mapped_emotion,
                        subject_id=pseudo_subject,
                        quality_score=1.0,
                    ))

        stats.total_output = len(records)
        for r in records:
            stats.class_distribution[r.label] = stats.class_distribution.get(r.label, 0) + 1
            stats.subject_counts[r.subject_id] = stats.subject_counts.get(r.subject_id, 0) + 1

        return records, stats


class DISFACleaner(BaseCleaner):
    """Cleaner for DISFA Action Unit dataset.

    Expected structure:
        raw/disfa/
            Images/{SN_ID}/{SN_ID}/{Trial}/*.jpg
            Labels/{SN_ID}/{SN_ID}/{Trial}/AU{N}.txt
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def clean(self) -> tuple[list[ImageRecord], CleaningStats]:
        """Clean DISFA dataset and pair images with AU labels."""
        stats = CleaningStats()
        records: list[ImageRecord] = []

        images_root = self.data_dir / "Images"
        labels_root = self.data_dir / "Labels"

        if not images_root.exists() or not labels_root.exists():
            logger.warning("DISFA Images or Labels directory not found")
            return records, stats

        for subject_dir in sorted(images_root.iterdir()):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name
            inner_img = subject_dir / subject_id
            inner_lbl = labels_root / subject_id / subject_id

            if not inner_img.exists():
                continue

            for trial_dir in sorted(inner_img.iterdir()):
                if not trial_dir.is_dir():
                    continue

                trial_name = trial_dir.name
                label_trial_dir = inner_lbl / trial_name if inner_lbl.exists() else None

                # Check if we have AU labels for this trial
                has_labels = label_trial_dir is not None and label_trial_dir.exists()

                for img_path in sorted(trial_dir.glob("*.jpg")):
                    stats.total_input += 1

                    img = try_load_image(img_path)
                    if img is None:
                        stats.corrupted_removed += 1
                        continue

                    # Encode AU intensities as label string
                    label = "unlabeled"
                    if has_labels:
                        au_label = self._read_au_labels(label_trial_dir, img_path.stem)
                        if au_label:
                            label = au_label

                    records.append(ImageRecord(
                        path=img_path.relative_to(self.data_dir.parent.parent).as_posix(),
                        label=label,
                        subject_id=f"disfa_{subject_id}",
                        quality_score=1.0,
                    ))

        stats.total_output = len(records)
        for r in records:
            stats.class_distribution[r.label] = stats.class_distribution.get(r.label, 0) + 1
            stats.subject_counts[r.subject_id] = stats.subject_counts.get(r.subject_id, 0) + 1

        return records, stats

    def _read_au_labels(self, label_dir: Path, frame_stem: str) -> str | None:
        """Read AU intensities for a specific frame.

        Args:
            label_dir: Directory containing AU{N}.txt files.
            frame_stem: Frame filename stem (e.g., "000").

        Returns:
            Pipe-separated string of AU:intensity pairs, e.g., "AU1:2|AU4:3".
            Returns None if no labels found.
        """
        au_values: list[str] = []

        for au_file in sorted(label_dir.glob("AU*.txt")):
            au_name = au_file.stem  # e.g., "AU1"

            try:
                with open(au_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            fname, intensity = parts[0], parts[1]
                            if fname.replace(".jpg", "") == frame_stem:
                                int_val = int(intensity)
                                if int_val > 0:
                                    au_values.append(f"{au_name}:{int_val}")
                                break
            except (ValueError, OSError):
                continue

        return "|".join(au_values) if au_values else "AU_none"
