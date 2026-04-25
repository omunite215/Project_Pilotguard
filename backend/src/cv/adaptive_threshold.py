"""Adaptive EAR thresholding via per-session calibration.

Individual EAR baselines vary significantly due to eye shape, glasses,
camera angle, and lighting. A fixed global threshold causes high false
positive rates. Instead, we calibrate during the first N seconds of each
session to establish a personal baseline.

Calibration protocol:
    1. Collect EAR values for calibration_duration seconds (eyes open, looking forward)
    2. Compute baseline = median(collected EAR values)
    3. Set threshold = baseline x multiplier (default 0.75)

This threshold is then used by the blink detector and PERCLOS calculator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class CalibrationState(Enum):
    """State of the calibration process."""

    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETE = auto()


@dataclass
class AdaptiveThreshold:
    """Per-session adaptive EAR threshold via calibration.

    Args:
        calibration_duration: Seconds of EAR collection for calibration.
        threshold_multiplier: Fraction of baseline EAR used as the
            closed-eye threshold (default 0.75).
        fallback_threshold: Used if calibration fails or hasn't started.
    """

    calibration_duration: float = 30.0
    threshold_multiplier: float = 0.75
    fallback_threshold: float = 0.20

    _state: CalibrationState = field(default=CalibrationState.NOT_STARTED, init=False)
    _calibration_ears: list[float] = field(default_factory=list, init=False)
    _calibration_start: float = field(default=0.0, init=False)
    _baseline_ear: float = field(default=0.0, init=False)
    _threshold: float = field(default=0.0, init=False)

    @property
    def state(self) -> CalibrationState:
        """Current calibration state."""
        return self._state

    @property
    def threshold(self) -> float:
        """Current eye-closure threshold."""
        if self._state == CalibrationState.COMPLETE:
            return self._threshold
        return self.fallback_threshold

    @property
    def baseline_ear(self) -> float:
        """Calibrated baseline EAR (0.0 if not yet calibrated)."""
        return self._baseline_ear

    def start_calibration(self, timestamp: float) -> None:
        """Begin the calibration phase.

        Args:
            timestamp: Current time in seconds.
        """
        self._state = CalibrationState.IN_PROGRESS
        self._calibration_start = timestamp
        self._calibration_ears.clear()

    def update(self, ear: float, timestamp: float) -> bool:
        """Feed an EAR value during calibration.

        Args:
            ear: Current raw EAR value.
            timestamp: Current time in seconds.

        Returns:
            True if calibration just completed on this call.
        """
        if self._state != CalibrationState.IN_PROGRESS:
            return False

        self._calibration_ears.append(ear)

        elapsed = timestamp - self._calibration_start
        if elapsed >= self.calibration_duration:
            return self._finalize()

        return False

    def _finalize(self) -> bool:
        """Compute baseline and threshold from collected samples.

        Returns:
            True if calibration succeeded, False if not enough data.
        """
        if len(self._calibration_ears) < 10:
            # Not enough samples — keep fallback
            self._state = CalibrationState.COMPLETE
            self._threshold = self.fallback_threshold
            return True

        ears = np.array(self._calibration_ears, dtype=np.float32)

        # Use median for robustness against outliers (brief blinks during calibration)
        self._baseline_ear = float(np.median(ears))
        self._threshold = self._baseline_ear * self.threshold_multiplier

        # Sanity clamp — threshold shouldn't be absurdly low or high
        self._threshold = max(0.10, min(0.30, self._threshold))

        self._state = CalibrationState.COMPLETE
        return True

    @property
    def calibration_progress(self) -> float:
        """Calibration progress as a fraction [0.0, 1.0]."""
        if self._state == CalibrationState.NOT_STARTED:
            return 0.0
        if self._state == CalibrationState.COMPLETE:
            return 1.0

        if not self._calibration_ears:
            return 0.0

        # Estimate based on sample count vs expected
        # At 30 FPS for 30s, expect ~900 samples
        expected = self.calibration_duration * 30.0
        return min(1.0, len(self._calibration_ears) / expected)

    def reset(self) -> None:
        """Reset to pre-calibration state."""
        self._state = CalibrationState.NOT_STARTED
        self._calibration_ears.clear()
        self._calibration_start = 0.0
        self._baseline_ear = 0.0
        self._threshold = 0.0
