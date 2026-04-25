"""Kalman filter for EAR smoothing.

Raw EAR values are noisy due to landmark jitter, lighting changes, and
partial occlusions. A constant-velocity Kalman filter smooths the signal
while preserving genuine state transitions (blinks, eye closures).

State vector: [EAR, dEAR/dt]
    - EAR: current smoothed eye aspect ratio
    - dEAR/dt: rate of change (velocity)

The filter balances responsiveness (tracking real blinks) vs. noise
rejection via the process_noise (Q) and measurement_noise (R) parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from filterpy.kalman import KalmanFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class KalmanState:
    """Current Kalman filter output.

    Attributes:
        ear_smoothed: Filtered EAR value.
        ear_velocity: Estimated rate of change of EAR.
    """

    ear_smoothed: float
    ear_velocity: float


class EARKalmanFilter:
    """Constant-velocity Kalman filter for EAR time series.

    Args:
        process_noise: Process noise covariance scalar (Q).
            Higher = more responsive to changes, less smoothing.
        measurement_noise: Measurement noise covariance scalar (R).
            Higher = more smoothing, slower response.
        initial_ear: Initial EAR estimate (default 0.3 = open eyes).
        initial_velocity: Initial velocity estimate (default 0.0 = stable).
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_ear: float = 0.3,
        initial_velocity: float = 0.0,
    ) -> None:
        self._kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition: constant velocity model
        # [EAR_t+1]     = [1, dt] [EAR_t]     (dt=1 for discrete steps)
        # [dEAR/dt_t+1]   [0,  1] [dEAR/dt_t]
        self._kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])

        # Observation matrix: we only observe EAR, not velocity
        self._kf.H = np.array([[1.0, 0.0]])

        # Process noise covariance
        self._kf.Q = np.array([
            [process_noise, 0.0],
            [0.0, process_noise * 0.5],
        ])

        # Measurement noise covariance
        self._kf.R = np.array([[measurement_noise]])

        # Initial state
        self._kf.x = np.array([[initial_ear], [initial_velocity]])

        # Initial covariance (moderate uncertainty)
        self._kf.P = np.eye(2) * 1.0

    def update(self, ear_raw: float) -> KalmanState:
        """Predict and update with a new raw EAR measurement.

        Args:
            ear_raw: Raw (noisy) EAR value from current frame.

        Returns:
            KalmanState with smoothed EAR and estimated velocity.
        """
        self._kf.predict()
        self._kf.update(np.array([[ear_raw]]))

        state: NDArray[np.float64] = self._kf.x.flatten()
        return KalmanState(
            ear_smoothed=float(state[0]),
            ear_velocity=float(state[1]),
        )

    def reset(self, ear: float = 0.3) -> None:
        """Reset filter state (e.g., after calibration).

        Args:
            ear: New initial EAR estimate.
        """
        self._kf.x = np.array([[ear], [0.0]])
        self._kf.P = np.eye(2) * 1.0
