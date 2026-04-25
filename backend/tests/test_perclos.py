"""Tests for PERCLOS calculation."""

import pytest

from src.cv.perclos import PERCLOSCalculator


class TestPERCLOSCalculator:
    """Tests for rolling PERCLOS computation."""

    def test_all_open_is_zero(self) -> None:
        """When all frames have open eyes, PERCLOS should be 0."""
        calc = PERCLOSCalculator(window_seconds=10.0)

        for i in range(100):
            perclos = calc.update(timestamp=i * 0.033, ear=0.30, threshold=0.20)

        assert perclos == 0.0

    def test_all_closed_is_100(self) -> None:
        """When all frames have closed eyes, PERCLOS should be 100."""
        calc = PERCLOSCalculator(window_seconds=10.0)

        for i in range(100):
            perclos = calc.update(timestamp=i * 0.033, ear=0.10, threshold=0.20)

        assert perclos == 100.0

    def test_half_closed_is_50(self) -> None:
        """50% open, 50% closed should give PERCLOS of 50."""
        calc = PERCLOSCalculator(window_seconds=10.0)
        threshold = 0.20

        # 50 open frames
        for i in range(50):
            calc.update(timestamp=i * 0.033, ear=0.30, threshold=threshold)

        # 50 closed frames
        for i in range(50, 100):
            perclos = calc.update(timestamp=i * 0.033, ear=0.10, threshold=threshold)

        assert perclos == pytest.approx(50.0)

    def test_window_eviction(self) -> None:
        """Old frames should be evicted from the window."""
        calc = PERCLOSCalculator(window_seconds=2.0)
        threshold = 0.20

        # First 2 seconds: all closed
        for i in range(60):  # 30fps for 2 seconds
            calc.update(timestamp=i / 30.0, ear=0.10, threshold=threshold)

        # Next 2 seconds: all open (evicts the closed frames)
        for i in range(60, 120):
            perclos = calc.update(timestamp=i / 30.0, ear=0.30, threshold=threshold)

        # After eviction, only open frames remain
        assert perclos == pytest.approx(0.0, abs=5.0)

    def test_empty_returns_zero(self) -> None:
        """Empty calculator should return 0."""
        calc = PERCLOSCalculator()
        assert calc.frame_count == 0

    def test_reset_clears_history(self) -> None:
        """Reset should clear all stored frames."""
        calc = PERCLOSCalculator(window_seconds=10.0)

        for i in range(50):
            calc.update(timestamp=i * 0.033, ear=0.10, threshold=0.20)

        assert calc.frame_count > 0

        calc.reset()
        assert calc.frame_count == 0
