"""PERCLOS (Percentage of Eye Closure) calculation.

PERCLOS measures the proportion of time the eyes are closed over a rolling
time window. It is one of the most validated physiological indicators of
drowsiness in the fatigue research literature.

    PERCLOS = (frames where EAR < threshold) / (total frames in window) x 100

Fatigue thresholds (from PRD):
    - PERCLOS > 40%: Warning-level fatigue
    - PERCLOS > 25%: Caution-level fatigue
    - Normal alert:  ~5-15% (natural blinking)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Single frame's eye-closure status for PERCLOS tracking.

    Attributes:
        timestamp: Frame timestamp in seconds.
        is_closed: Whether the EAR was below the closure threshold.
    """

    timestamp: float
    is_closed: bool


@dataclass
class PERCLOSCalculator:
    """Rolling-window PERCLOS computation.

    Args:
        window_seconds: Duration of the rolling window in seconds (default 60s).
    """

    window_seconds: float = 60.0

    _frames: deque[FrameRecord] = field(default_factory=deque, init=False)

    def update(self, timestamp: float, ear: float, threshold: float) -> float:
        """Record a frame and compute current PERCLOS.

        Args:
            timestamp: Current frame timestamp in seconds.
            ear: Current (smoothed) EAR value.
            threshold: Adaptive eye-closure threshold.

        Returns:
            PERCLOS as a percentage (0-100).
        """
        is_closed = ear < threshold
        self._frames.append(FrameRecord(timestamp=timestamp, is_closed=is_closed))

        # Evict frames outside the window
        cutoff = timestamp - self.window_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

        if not self._frames:
            return 0.0

        closed_count = sum(1 for f in self._frames if f.is_closed)
        return (closed_count / len(self._frames)) * 100.0

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the window."""
        return len(self._frames)

    def reset(self) -> None:
        """Clear all frame history."""
        self._frames.clear()
