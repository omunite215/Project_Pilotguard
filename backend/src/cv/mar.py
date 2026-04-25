"""Mouth Aspect Ratio (MAR) calculation.

MAR detects yawning — a strong fatigue indicator. It mirrors the EAR concept
applied to the mouth using inner lip landmarks.

Formula:
    MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 · ||p1 - p5||)

Where p1-p8 are the 8 inner mouth landmarks (dlib points 60-67) in clockwise order:
    p1: left corner, p2: upper-left, p3: upper-center, p4: upper-right,
    p5: right corner, p6: lower-right, p7: lower-center, p8: lower-left

Typical MAR values:
    - Mouth closed: ~0.0 - 0.3
    - Normal speech: ~0.3 - 0.6
    - Yawning:       ~0.6 - 1.0+
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_mar(inner_mouth: NDArray[np.float32]) -> float:
    """Compute Mouth Aspect Ratio from inner mouth landmarks.

    Args:
        inner_mouth: (8, 2) array of inner mouth landmarks
            (dlib points 60-67 in clockwise order).

    Returns:
        MAR as a float. Returns 0.0 if horizontal distance is zero.
    """
    # Vertical distances (3 pairs)
    v1 = np.linalg.norm(inner_mouth[1] - inner_mouth[7])  # ||p2 - p8||
    v2 = np.linalg.norm(inner_mouth[2] - inner_mouth[6])  # ||p3 - p7||
    v3 = np.linalg.norm(inner_mouth[3] - inner_mouth[5])  # ||p4 - p6||

    # Horizontal distance
    h = np.linalg.norm(inner_mouth[0] - inner_mouth[4])  # ||p1 - p5||

    if h < 1e-8:
        return 0.0

    return float((v1 + v2 + v3) / (2.0 * h))
