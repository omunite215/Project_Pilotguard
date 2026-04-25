"""Tests for blink detection and microsleep identification."""

from src.cv.blink_detector import BlinkDetector, BlinkEvent


class TestBlinkDetector:
    """Tests for the blink detector state machine."""

    def test_initial_state_is_open(self) -> None:
        """Detector starts with eyes open."""
        bd = BlinkDetector()
        assert not bd.is_eyes_closed
        assert bd.total_blinks == 0

    def test_detects_normal_blink(self) -> None:
        """A brief dip below threshold and return should register as a blink."""
        bd = BlinkDetector(consecutive_frames_threshold=2)
        threshold = 0.20
        t = 0.0
        dt = 1 / 30  # 30 FPS

        # Eyes open
        for _ in range(10):
            bd.update(0.30, threshold, t)
            t += dt

        # Eyes close for ~200ms (6 frames at 30fps)
        for _ in range(6):
            event = bd.update(0.10, threshold, t)
            t += dt

        # Eyes reopen — this should complete the blink
        event = bd.update(0.30, threshold, t)
        t += dt

        assert event is not None
        assert isinstance(event, BlinkEvent)
        assert not event.is_microsleep
        assert event.duration_ms < 400
        assert bd.total_blinks == 1

    def test_detects_microsleep(self) -> None:
        """Closure >500ms should be classified as microsleep."""
        bd = BlinkDetector(
            microsleep_threshold_ms=500.0,
            consecutive_frames_threshold=2,
        )
        threshold = 0.20
        t = 0.0
        dt = 1 / 30

        # Eyes open
        for _ in range(5):
            bd.update(0.30, threshold, t)
            t += dt

        # Eyes close for ~600ms (18 frames at 30fps)
        for _ in range(18):
            bd.update(0.10, threshold, t)
            t += dt

        # Reopen
        event = bd.update(0.30, threshold, t)

        assert event is not None
        assert event.is_microsleep
        assert event.duration_ms > 500

    def test_consecutive_frames_threshold(self) -> None:
        """Single frame below threshold should not trigger closure."""
        bd = BlinkDetector(consecutive_frames_threshold=3)
        threshold = 0.20

        # Single frame dip
        bd.update(0.10, threshold, 0.0)
        assert not bd.is_eyes_closed

        # Two frames dip — still not enough
        bd.update(0.10, threshold, 0.033)
        assert not bd.is_eyes_closed

        # Third frame — now it triggers
        bd.update(0.10, threshold, 0.066)
        assert bd.is_eyes_closed

    def test_blink_rate_calculation(self) -> None:
        """Blink rate should correctly count blinks in time window."""
        bd = BlinkDetector(consecutive_frames_threshold=1)
        threshold = 0.20
        t = 0.0

        # Generate 3 blinks over ~3 seconds
        for _ in range(3):
            # Close
            for _ in range(3):
                bd.update(0.10, threshold, t)
                t += 1 / 30
            # Open
            bd.update(0.30, threshold, t)
            t += 0.8  # gap between blinks

        assert bd.total_blinks == 3

    def test_no_event_while_open(self) -> None:
        """No blink event should fire while eyes remain open."""
        bd = BlinkDetector()
        threshold = 0.20

        for i in range(20):
            event = bd.update(0.30, threshold, i * 0.033)
            assert event is None

    def test_min_ear_tracked(self) -> None:
        """BlinkEvent should record the minimum EAR during closure."""
        bd = BlinkDetector(consecutive_frames_threshold=1)
        threshold = 0.20
        t = 0.0

        # Close with varying EAR
        for ear in [0.15, 0.08, 0.12, 0.09, 0.14]:
            bd.update(ear, threshold, t)
            t += 0.033

        # Reopen
        event = bd.update(0.30, threshold, t)
        assert event is not None
        assert event.min_ear == 0.08

    def test_reset_clears_state(self) -> None:
        """Reset should clear all history and state."""
        bd = BlinkDetector(consecutive_frames_threshold=1)
        threshold = 0.20

        # Generate a blink
        bd.update(0.10, threshold, 0.0)
        bd.update(0.10, threshold, 0.033)
        bd.update(0.30, threshold, 0.066)

        assert bd.total_blinks == 1

        bd.reset()
        assert bd.total_blinks == 0
        assert not bd.is_eyes_closed
        assert bd.recent_events == []
