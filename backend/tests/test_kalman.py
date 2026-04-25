"""Tests for Kalman filter EAR smoothing."""

import pytest

from src.cv.kalman import EARKalmanFilter, KalmanState


class TestEARKalmanFilter:
    """Tests for the Kalman filter."""

    def test_initial_output_near_initial_state(self) -> None:
        """First update should return a value close to the measurement."""
        kf = EARKalmanFilter(initial_ear=0.3)
        state = kf.update(0.3)

        assert isinstance(state, KalmanState)
        assert state.ear_smoothed == pytest.approx(0.3, abs=0.05)

    def test_smoothing_reduces_noise(self) -> None:
        """Noisy input around 0.3 should produce smooth output near 0.3."""
        kf = EARKalmanFilter(initial_ear=0.3)

        # Feed noisy values centered on 0.3
        noisy_values = [0.28, 0.35, 0.27, 0.33, 0.29, 0.31, 0.26, 0.34, 0.30, 0.32]
        outputs = [kf.update(v).ear_smoothed for v in noisy_values]

        # Variance of output should be less than variance of input
        input_var = sum((v - 0.3) ** 2 for v in noisy_values) / len(noisy_values)
        output_var = sum((v - 0.3) ** 2 for v in outputs) / len(outputs)

        assert output_var < input_var

    def test_tracks_step_change(self) -> None:
        """Filter should eventually track a step change (blink)."""
        kf = EARKalmanFilter(initial_ear=0.3)

        # Simulate eyes open then closed
        for _ in range(10):
            kf.update(0.30)

        # Step to closed
        for _ in range(10):
            state = kf.update(0.10)

        # Should have converged toward 0.10
        assert state.ear_smoothed < 0.20

    def test_velocity_negative_on_closure(self) -> None:
        """Velocity should be negative when EAR is dropping (eye closing)."""
        kf = EARKalmanFilter(initial_ear=0.30)

        for _ in range(5):
            kf.update(0.30)

        # Rapid drop
        state = kf.update(0.10)
        assert state.ear_velocity < 0

    def test_reset_restores_initial_state(self) -> None:
        """After reset, filter should behave like a fresh instance."""
        kf = EARKalmanFilter(initial_ear=0.3)

        # Drive to a different state
        for _ in range(20):
            kf.update(0.10)

        kf.reset(0.3)
        state = kf.update(0.30)

        assert state.ear_smoothed == pytest.approx(0.3, abs=0.05)
