"""Tests for adaptive EAR thresholding."""

import pytest

from src.cv.adaptive_threshold import AdaptiveThreshold, CalibrationState


class TestAdaptiveThreshold:
    """Tests for per-session EAR calibration."""

    def test_initial_state(self) -> None:
        """Should start in NOT_STARTED state with fallback threshold."""
        at = AdaptiveThreshold(fallback_threshold=0.20)
        assert at.state == CalibrationState.NOT_STARTED
        assert at.threshold == 0.20
        assert at.baseline_ear == 0.0

    def test_calibration_flow(self) -> None:
        """Full calibration: start → feed values → complete."""
        at = AdaptiveThreshold(
            calibration_duration=2.0,
            threshold_multiplier=0.75,
        )

        at.start_calibration(timestamp=0.0)
        assert at.state == CalibrationState.IN_PROGRESS

        # Feed 30fps for >2 seconds with EAR ~0.30
        for i in range(66):
            finished = at.update(ear=0.30, timestamp=i / 30.0)
            if finished:
                break

        assert at.state == CalibrationState.COMPLETE
        assert at.baseline_ear == pytest.approx(0.30, abs=0.01)
        assert at.threshold == pytest.approx(0.30 * 0.75, abs=0.02)

    def test_threshold_clamped(self) -> None:
        """Threshold should be clamped to [0.10, 0.30]."""
        # Very high EAR (unusual) — threshold should not exceed 0.30
        at = AdaptiveThreshold(
            calibration_duration=1.0,
            threshold_multiplier=0.75,
        )
        at.start_calibration(0.0)
        for i in range(60):
            at.update(ear=0.50, timestamp=i / 30.0)

        assert at.threshold <= 0.30

        # Very low EAR — threshold should not go below 0.10
        at2 = AdaptiveThreshold(
            calibration_duration=1.0,
            threshold_multiplier=0.75,
        )
        at2.start_calibration(0.0)
        for i in range(60):
            at2.update(ear=0.10, timestamp=i / 30.0)

        assert at2.threshold >= 0.10

    def test_median_robust_to_blinks(self) -> None:
        """Calibration should use median, robust to occasional blinks."""
        at = AdaptiveThreshold(
            calibration_duration=1.0,
            threshold_multiplier=0.75,
        )
        at.start_calibration(0.0)

        # Most frames open, a few blinks mixed in
        for i in range(36):
            ear = 0.30 if i % 10 != 0 else 0.10  # blink every 10th frame
            at.update(ear=ear, timestamp=i / 30.0)

        assert at.state == CalibrationState.COMPLETE
        # Median should be 0.30, not dragged down by blinks
        assert at.baseline_ear == pytest.approx(0.30, abs=0.02)

    def test_not_enough_samples_uses_fallback(self) -> None:
        """If calibration gets too few samples, use fallback threshold."""
        at = AdaptiveThreshold(
            calibration_duration=1.0,
            fallback_threshold=0.20,
        )
        at.start_calibration(0.0)

        # Only feed 5 samples (less than minimum 10)
        for i in range(5):
            at.update(ear=0.30, timestamp=i / 30.0)

        # Force finalize by jumping past duration
        at.update(ear=0.30, timestamp=2.0)

        assert at.state == CalibrationState.COMPLETE
        assert at.threshold == 0.20  # fallback

    def test_progress_reporting(self) -> None:
        """Calibration progress should go from 0 to 1."""
        at = AdaptiveThreshold(calibration_duration=2.0)

        assert at.calibration_progress == 0.0

        at.start_calibration(0.0)

        # Feed some frames
        for i in range(30):
            at.update(ear=0.30, timestamp=i / 30.0)

        progress = at.calibration_progress
        assert 0.0 < progress < 1.0

    def test_reset(self) -> None:
        """Reset should return to NOT_STARTED."""
        at = AdaptiveThreshold(calibration_duration=1.0)
        at.start_calibration(0.0)
        for i in range(60):
            at.update(ear=0.30, timestamp=i / 30.0)

        at.reset()
        assert at.state == CalibrationState.NOT_STARTED
        assert at.baseline_ear == 0.0

    def test_update_before_start_is_noop(self) -> None:
        """Calling update before starting calibration should be a no-op."""
        at = AdaptiveThreshold()
        result = at.update(ear=0.30, timestamp=0.0)
        assert result is False
        assert at.state == CalibrationState.NOT_STARTED
