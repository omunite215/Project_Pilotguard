"""Eye Aspect Ratio (EAR) calculation.

EAR is the primary metric for detecting eye closure, blinks, and microsleeps.
Formula from Soukupová & Čech (2016):

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)

Where p1-p6 are the 6 landmark points of each eye in clockwise order:
    p1: outer corner, p2: upper-outer, p3: upper-inner,
    p4: inner corner, p5: lower-inner, p6: lower-outer

Typical EAR values:
    - Eyes open:   ~0.25 - 0.35
    - Eyes closed:  ~0.05 - 0.15
    - Blink:       brief dip below threshold and return
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EARResult:
    """Eye Aspect Ratio computation result.

    Attributes:
        left: EAR for the left eye.
        right: EAR for the right eye.
        average: Mean of left and right EAR.
    """

    left: float
    right: float
    average: float


def _compute_ear(eye: NDArray[np.float32]) -> float:
    """Compute EAR for a single eye.

    Args:
        eye: (6, 2) array of eye landmarks in clockwise order.
            p1=outer corner, p2=upper-outer, p3=upper-inner,
            p4=inner corner, p5=lower-inner, p6=lower-outer.

    Returns:
        Eye Aspect Ratio as a float. Returns 0.0 if the horizontal
        distance is zero (degenerate case).
    """
    # Vertical distances
    v1 = np.linalg.norm(eye[1] - eye[5])  # ||p2 - p6||
    v2 = np.linalg.norm(eye[2] - eye[4])  # ||p3 - p5||

    # Horizontal distance
    h = np.linalg.norm(eye[0] - eye[3])  # ||p1 - p4||

    if h < 1e-8:
        return 0.0

    return float((v1 + v2) / (2.0 * h))


def compute_ear(
    right_eye: NDArray[np.float32],
    left_eye: NDArray[np.float32],
) -> EARResult:
    """Compute Eye Aspect Ratio for both eyes.

    Args:
        right_eye: (6, 2) array of right eye landmarks (dlib points 36-41).
        left_eye: (6, 2) array of left eye landmarks (dlib points 42-47).

    Returns:
        EARResult with left, right, and average EAR values.
    """
    ear_right = _compute_ear(right_eye)
    ear_left = _compute_ear(left_eye)
    ear_avg = (ear_left + ear_right) / 2.0

    return EARResult(left=ear_left, right=ear_right, average=ear_avg)
