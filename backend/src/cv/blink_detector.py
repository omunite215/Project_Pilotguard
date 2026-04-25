"""Blink detection and microsleep identification.

Detects blink events by tracking EAR transitions across an adaptive threshold.
Classifies each closure event by duration:
    - Normal blink:  < 400ms
    - Long blink:    400-500ms (fatigue indicator)
    - Microsleep:    > 500ms (danger — immediate alert)

Uses a simple state machine: OPEN → CLOSING → CLOSED → OPENING → OPEN
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class EyeState(Enum):
    """Current eye state in the blink state machine."""

    OPEN = auto()
    CLOSED = auto()


@dataclass(frozen=True, slots=True)
class BlinkEvent:
    """A completed blink or closure event.

    Attributes:
        start_time: Timestamp when eyes closed (seconds).
        end_time: Timestamp when eyes reopened (seconds).
        duration_ms: Duration of closure in milliseconds.
        is_microsleep: True if duration exceeds microsleep threshold.
        min_ear: Lowest EAR observed during the closure.
    """

    start_time: float
    end_time: float
    duration_ms: float
    is_microsleep: bool
    min_ear: float


@dataclass
class BlinkDetector:
    """Detects blinks and microsleeps from a stream of EAR values.

    Args:
        microsleep_threshold_ms: Closure duration above which the event
            is classified as a microsleep (default 500ms per PRD).
        consecutive_frames_threshold: Number of consecutive below-threshold
            frames required to confirm eye closure (reduces false triggers
            from single-frame noise).
    """

    microsleep_threshold_ms: float = 500.0
    consecutive_frames_threshold: int = 2

    # Internal state
    _state: EyeState = field(default=EyeState.OPEN, init=False)
    _closure_start: float = field(default=0.0, init=False)
    _min_ear_in_closure: float = field(default=1.0, init=False)
    _below_threshold_count: int = field(default=0, init=False)
    _blink_events: list[BlinkEvent] = field(default_factory=list, init=False)
    _total_blinks: int = field(default=0, init=False)
    _session_start: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._session_start = time.monotonic()

    def update(self, ear: float, threshold: float, timestamp: float | None = None) -> BlinkEvent | None:
        """Process a new EAR value and detect blink/microsleep events.

        Args:
            ear: Current (smoothed) EAR value.
            threshold: Adaptive eye-closure threshold for this session.
            timestamp: Frame timestamp in seconds. Uses monotonic clock if None.

        Returns:
            A BlinkEvent if an eye-reopening just completed a closure, else None.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        event: BlinkEvent | None = None

        if ear < threshold:
            # Eyes are below threshold
            self._below_threshold_count += 1

            if self._state == EyeState.OPEN and self._below_threshold_count >= self.consecutive_frames_threshold:
                # Transition: OPEN → CLOSED
                self._state = EyeState.CLOSED
                self._closure_start = now
                self._min_ear_in_closure = ear

            if self._state == EyeState.CLOSED:
                self._min_ear_in_closure = min(self._min_ear_in_closure, ear)

        else:
            # Eyes are above threshold
            if self._state == EyeState.CLOSED:
                # Transition: CLOSED → OPEN — blink/microsleep complete
                duration_ms = (now - self._closure_start) * 1000.0
                event = BlinkEvent(
                    start_time=self._closure_start,
                    end_time=now,
                    duration_ms=duration_ms,
                    is_microsleep=duration_ms > self.microsleep_threshold_ms,
                    min_ear=self._min_ear_in_closure,
                )
                self._blink_events.append(event)
                self._total_blinks += 1
                self._state = EyeState.OPEN

            self._below_threshold_count = 0
            self._min_ear_in_closure = 1.0

        return event

    @property
    def is_eyes_closed(self) -> bool:
        """Whether eyes are currently detected as closed."""
        return self._state == EyeState.CLOSED

    @property
    def current_closure_duration_ms(self) -> float:
        """Duration of current eye closure in ms, or 0.0 if eyes are open."""
        if self._state != EyeState.CLOSED:
            return 0.0
        return (time.monotonic() - self._closure_start) * 1000.0

    @property
    def total_blinks(self) -> int:
        """Total number of completed blink events in this session."""
        return self._total_blinks

    def blink_rate_per_minute(self, window_seconds: float = 60.0) -> float:
        """Calculate blink rate over a recent time window.

        Args:
            window_seconds: Lookback window in seconds (default 60s).

        Returns:
            Blinks per minute within the window.
        """
        now = time.monotonic()
        cutoff = now - window_seconds
        recent = sum(1 for e in self._blink_events if e.end_time >= cutoff)
        elapsed = min(now - self._session_start, window_seconds)

        if elapsed < 1.0:
            return 0.0

        return (recent / elapsed) * 60.0

    @property
    def recent_events(self) -> list[BlinkEvent]:
        """All blink events (newest last)."""
        return list(self._blink_events)

    def reset(self) -> None:
        """Reset detector state for a new session."""
        self._state = EyeState.OPEN
        self._closure_start = 0.0
        self._min_ear_in_closure = 1.0
        self._below_threshold_count = 0
        self._blink_events.clear()
        self._total_blinks = 0
        self._session_start = time.monotonic()
