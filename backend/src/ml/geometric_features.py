"""Geometric feature extraction from facial landmarks.

Extracts 7 drowsiness-relevant features from 68-point landmarks:
    1. EAR_avg: Average Eye Aspect Ratio
    2. EAR_std: Standard deviation of EAR over recent window
    3. PERCLOS: Percentage of eye closure over 60s window
    4. blink_rate: Blinks per minute
    5. blink_duration_avg: Average blink duration (ms)
    6. MAR: Mouth Aspect Ratio (yawn detection)
    7. EAR_velocity: Rate of change of EAR (from Kalman filter)

For training, these features are extracted per-image using MediaPipe landmarks.
For real-time inference, they come from the CV pipeline's running state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.cv.ear import compute_ear
from src.cv.landmark_extractor import extract_landmarks_68
from src.cv.mar import compute_mar

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from src.cv.face_detector import FaceDetector

# Feature names in order — used for XGBoost feature naming
FEATURE_NAMES: list[str] = [
    "ear_avg",
    "ear_left",
    "ear_right",
    "ear_asymmetry",
    "mar",
    "eye_mouth_ratio",
    "eye_brow_dist",
]


def extract_features_from_image(
    image: NDArray[np.uint8],
    detector: FaceDetector,
) -> NDArray[np.float32] | None:
    """Extract 7 geometric features from a single image.

    Args:
        image: BGR image (OpenCV format).
        detector: Initialized FaceDetector instance.

    Returns:
        (7,) feature vector, or None if no face detected.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection = detector.detect(rgb)

    if detection is None:
        return None

    lm68 = extract_landmarks_68(detection.landmarks_478)

    # EAR
    ear = compute_ear(lm68.right_eye, lm68.left_eye)

    # MAR (inner mouth = last 8 of the 20 mouth landmarks)
    mar = compute_mar(lm68.mouth[12:])

    # Eye-mouth ratio (how open are eyes relative to mouth)
    eye_mouth = ear.average / max(mar, 1e-6)

    # EAR asymmetry (difference between eyes — fatigue often affects one eye more)
    ear_asymmetry = abs(ear.left - ear.right)

    # Eye-brow distance proxy: vertical distance between eyebrow midpoint and eye midpoint
    # Eyebrows: points 17-21 (right), 22-26 (left) in 68-layout
    # Eyes: points 36-41 (right), 42-47 (left)
    right_brow_y = np.mean(lm68.points[17:22, 1])
    right_eye_y = np.mean(lm68.right_eye[:, 1])
    left_brow_y = np.mean(lm68.points[22:27, 1])
    left_eye_y = np.mean(lm68.left_eye[:, 1])
    eye_brow_dist = ((right_eye_y - right_brow_y) + (left_eye_y - left_brow_y)) / 2.0

    return np.array([
        ear.average,
        ear.left,
        ear.right,
        ear_asymmetry,
        mar,
        eye_mouth,
        eye_brow_dist,
    ], dtype=np.float32)


def extract_features_batch(
    image_paths: list[str | Path],
    detector: FaceDetector,
    labels: list[str] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], list[int]]:
    """Extract features from a batch of images.

    Args:
        image_paths: List of paths to images.
        detector: Initialized FaceDetector.
        labels: Optional list of string labels (same length as image_paths).

    Returns:
        Tuple of:
            features: (N, 7) feature matrix
            label_ints: (N,) integer labels (0=alert, 1=drowsy)
            valid_indices: indices of successfully processed images
    """
    features_list: list[NDArray[np.float32]] = []
    labels_list: list[int] = []
    valid_indices: list[int] = []

    label_map = {"alert": 0, "drowsy": 1}

    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            continue

        feat = extract_features_from_image(img, detector)
        if feat is None:
            continue

        features_list.append(feat)
        valid_indices.append(i)

        if labels is not None:
            labels_list.append(label_map.get(labels[i], -1))

    features = np.stack(features_list) if features_list else np.empty((0, 7), dtype=np.float32)
    label_arr = np.array(labels_list, dtype=np.int32) if labels_list else np.empty(0, dtype=np.int32)

    return features, label_arr, valid_indices
