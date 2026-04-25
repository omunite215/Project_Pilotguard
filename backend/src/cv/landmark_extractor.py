"""Extract dlib-compatible 68 landmarks from MediaPipe's 478-point FaceMesh.

MediaPipe FaceMesh outputs 478 3D landmarks. The classical EAR/MAR formulas
and most facial analysis literature use the dlib 68-point layout. This module
maps the closest MediaPipe indices to each of the 68 dlib points.

Reference mapping sourced from MediaPipe documentation and empirical alignment
with dlib's shape_predictor_68_face_landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ── MediaPipe 478 → dlib 68 index mapping ──────────────────────────────────
# Groups follow the standard dlib 68-point annotation:
#   0-16:  Jaw contour (17 points)
#   17-21: Right eyebrow (5 points)
#   22-26: Left eyebrow (5 points)
#   27-35: Nose (9 points)
#   36-41: Right eye (6 points)
#   42-47: Left eye (6 points)
#   48-67: Mouth (20 points)

MEDIAPIPE_TO_DLIB_68: list[int] = [
    # Jaw contour (0-16) — 17 points tracing the jawline
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378, 400,
    # Right eyebrow (17-21)
    285, 295, 282, 283, 276,
    # Left eyebrow (22-26)
    46, 53, 52, 65, 55,
    # Nose bridge (27-30)
    6, 197, 195, 5,
    # Nose bottom (31-35)
    48, 115, 220, 45, 4,
    # Right eye (36-41) — 6 points clockwise from outer corner
    33, 160, 158, 133, 153, 144,
    # Left eye (42-47) — 6 points clockwise from outer corner
    362, 385, 387, 263, 373, 380,
    # Outer mouth (48-59) — 12 points clockwise
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner mouth (60-67) — 8 points clockwise
    78, 82, 13, 312, 308, 317, 14, 87,
]

# Convenience: eye landmark indices within the 68-point layout
RIGHT_EYE_IDX: list[int] = [36, 37, 38, 39, 40, 41]
LEFT_EYE_IDX: list[int] = [42, 43, 44, 45, 46, 47]
MOUTH_IDX: list[int] = list(range(48, 68))
OUTER_MOUTH_IDX: list[int] = list(range(48, 60))
INNER_MOUTH_IDX: list[int] = list(range(60, 68))


@dataclass(frozen=True, slots=True)
class Landmarks68:
    """Extracted 68-point landmarks in dlib-compatible layout.

    Attributes:
        points: (68, 2) array of (x, y) coordinates normalized to [0, 1].
        right_eye: (6, 2) array — right eye landmarks (points 36-41).
        left_eye: (6, 2) array — left eye landmarks (points 42-47).
        mouth: (20, 2) array — mouth landmarks (points 48-67).
    """

    points: NDArray[np.float32]
    right_eye: NDArray[np.float32]
    left_eye: NDArray[np.float32]
    mouth: NDArray[np.float32]


def extract_landmarks_68(landmarks_478: NDArray[np.float32]) -> Landmarks68:
    """Map MediaPipe 478 landmarks to dlib-compatible 68-point layout.

    Args:
        landmarks_478: (478, 3) array from MediaPipe FaceMesh.
            Only x, y columns are used (z depth is discarded).

    Returns:
        Landmarks68 with all coordinates normalized to [0, 1].
    """
    # Extract only x, y (discard z depth) and select the 68 mapped points
    points_68 = landmarks_478[MEDIAPIPE_TO_DLIB_68, :2].copy()

    return Landmarks68(
        points=points_68,
        right_eye=points_68[RIGHT_EYE_IDX],
        left_eye=points_68[LEFT_EYE_IDX],
        mouth=points_68[MOUTH_IDX],
    )
