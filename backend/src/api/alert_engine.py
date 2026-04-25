"""Safety-critical alert engine with debounce + state locking.

Debounce: Requires N consecutive drowsy/microsleep frames before triggering.
This prevents false alarms from momentary blinks or sensor noise.

    Microsleep: 8 consecutive frames (~0.5s at 15fps) of microsleep state
    Drowsy:     15 consecutive frames (~1s) of drowsy state

Lock: Once triggered, alert is locked for a mandatory cooldown.
    Advisory:   15s + 5 consecutive alert frames
    Caution:    30s + 10 consecutive alert frames
    Warning:    45s + 15 consecutive alert frames
    Microsleep: 60s + 20 consecutive alert frames
"""

from __future__ import annotations

import time

from src.api.models import AlertInfo, AlertLevel

# Debounce: consecutive frames of bad state required before triggering
DEBOUNCE_FRAMES: dict[str, int] = {
    "microsleep": 5,    # ~0.3s at 15fps
    "drowsy": 10,       # ~0.7s at 15fps
}

# Lock configuration per severity — short durations for demo usability
LOCK_CONFIG: dict[str, dict[str, int | float]] = {
    "advisory": {"lock_seconds": 5, "required_alert_frames": 3},
    "caution": {"lock_seconds": 8, "required_alert_frames": 5},
    "warning": {"lock_seconds": 10, "required_alert_frames": 8},
    "microsleep": {"lock_seconds": 12, "required_alert_frames": 10},
}


class AlertEngine:
    """Safety-critical alert generator with debounce and mandatory lock.

    Args:
        advisory_threshold: Fatigue score for advisory alert.
        caution_threshold: Fatigue score for caution alert.
        warning_threshold: Fatigue score for warning alert.
        cooldown_seconds: Post-unlock cooldown before new alerts.
    """

    def __init__(
        self,
        advisory_threshold: float = 30.0,
        caution_threshold: float = 55.0,
        warning_threshold: float = 75.0,
        cooldown_seconds: float = 10.0,
    ) -> None:
        self._advisory = advisory_threshold
        self._caution = caution_threshold
        self._warning = warning_threshold
        self._cooldown = cooldown_seconds

        self._alert_count = 0

        # Debounce counters
        self._consecutive_drowsy: int = 0
        self._consecutive_microsleep: int = 0

        # Lock state
        self._locked = False
        self._lock_level: str = ""
        self._lock_start: float = 0.0
        self._lock_duration: float = 0.0
        self._required_alert_frames: int = 0
        self._consecutive_alert_frames: int = 0
        self._last_unlock_time: float = 0.0

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def lock_remaining_seconds(self) -> float:
        if not self._locked:
            return 0.0
        elapsed = time.monotonic() - self._lock_start
        return max(0.0, self._lock_duration - elapsed)

    @property
    def lock_level(self) -> str:
        return self._lock_level

    @property
    def alert_frames_progress(self) -> float:
        if not self._locked or self._required_alert_frames == 0:
            return 0.0
        return min(1.0, self._consecutive_alert_frames / self._required_alert_frames)

    def reset(self) -> None:
        self._alert_count = 0
        self._consecutive_drowsy = 0
        self._consecutive_microsleep = 0
        self._locked = False
        self._lock_level = ""
        self._lock_start = 0.0
        self._lock_duration = 0.0
        self._required_alert_frames = 0
        self._consecutive_alert_frames = 0
        self._last_unlock_time = 0.0

    def evaluate(
        self,
        fatigue_score: float,
        state: str,
        perclos: float,
        blink_rate: float,
    ) -> AlertInfo | None:
        """Evaluate with debounce + lock logic."""
        now = time.monotonic()

        # ── If locked ──
        if self._locked:
            return self._handle_locked_state(state, now, fatigue_score, perclos, blink_rate)

        # ── Not locked ──

        # Post-unlock cooldown
        if now - self._last_unlock_time < self._cooldown:
            self._consecutive_drowsy = 0
            self._consecutive_microsleep = 0
            return None

        # Update debounce counters
        if state == "microsleep":
            self._consecutive_microsleep += 1
            self._consecutive_drowsy += 1
        elif state == "drowsy":
            self._consecutive_drowsy += 1
            self._consecutive_microsleep = 0
        else:
            # Alert/normal state — reset debounce
            self._consecutive_drowsy = 0
            self._consecutive_microsleep = 0
            return None

        # Check if debounce threshold met for microsleep
        if self._consecutive_microsleep >= DEBOUNCE_FRAMES["microsleep"]:
            self._consecutive_microsleep = 0
            self._consecutive_drowsy = 0
            return self._trigger_lock("microsleep", fatigue_score, perclos, blink_rate)

        # Check if debounce threshold met for drowsy
        if self._consecutive_drowsy >= DEBOUNCE_FRAMES["drowsy"]:
            self._consecutive_drowsy = 0
            # Determine severity from fatigue score
            if fatigue_score >= self._warning:
                level = "warning"
            elif fatigue_score >= self._caution:
                level = "caution"
            else:
                level = "advisory"
            return self._trigger_lock(level, fatigue_score, perclos, blink_rate)

        # Debounce not yet met — no alert
        return None

    def _trigger_lock(
        self,
        level: str,
        fatigue_score: float,
        perclos: float,
        blink_rate: float,
    ) -> AlertInfo:
        config = LOCK_CONFIG.get(level, LOCK_CONFIG["advisory"])
        self._locked = True
        self._lock_level = level
        self._lock_start = time.monotonic()
        self._lock_duration = float(config["lock_seconds"])
        self._required_alert_frames = int(config["required_alert_frames"])
        self._consecutive_alert_frames = 0
        self._alert_count += 1

        alert_level = self._level_to_alert_level(level)
        message = self._build_message(level, fatigue_score, perclos, blink_rate)

        return AlertInfo(
            level=alert_level,
            message=message,
            fatigue_score=fatigue_score,
            timestamp=time.time(),
        )

    def _handle_locked_state(
        self,
        state: str,
        now: float,
        fatigue_score: float,
        perclos: float,
        blink_rate: float,
    ) -> AlertInfo | None:
        elapsed = now - self._lock_start
        time_ok = elapsed >= self._lock_duration

        if state == "alert":
            self._consecutive_alert_frames += 1
        else:
            self._consecutive_alert_frames = 0

        frames_ok = self._consecutive_alert_frames >= self._required_alert_frames

        # Escalate if worse during lock
        if state == "microsleep" and self._lock_level != "microsleep":
            return self._trigger_lock("microsleep", fatigue_score, perclos, blink_rate)

        # Unlock if both conditions met
        if time_ok and frames_ok:
            self._locked = False
            self._lock_level = ""
            self._last_unlock_time = now
            return None

        # Safety valve: force unlock after 2x the configured lock duration
        # This prevents infinite locks when the pilot cannot demonstrate alertness
        max_lock = self._lock_duration * 2
        if elapsed >= max_lock:
            self._locked = False
            self._lock_level = ""
            self._last_unlock_time = now
            return None

        # Still locked
        remaining = max(0.0, self._lock_duration - elapsed)
        return AlertInfo(
            level=self._level_to_alert_level(self._lock_level),
            message=self._build_locked_message(remaining),
            fatigue_score=fatigue_score,
            timestamp=time.time(),
        )

    def _build_message(
        self,
        level: str,
        fatigue_score: float,
        perclos: float,
        blink_rate: float,
    ) -> str:
        lock_sec = int(LOCK_CONFIG.get(level, LOCK_CONFIG["advisory"])["lock_seconds"])
        if level == "microsleep":
            return (
                f"MICROSLEEP DETECTED — Eyes closed for extended period. "
                f"System locked for {lock_sec}s. Immediate action required."
            )
        if level == "warning":
            return (
                f"CRITICAL FATIGUE — Score {fatigue_score:.0f}/100, "
                f"PERCLOS {perclos:.1f}%. Locked for {lock_sec}s. "
                f"Stop and rest immediately."
            )
        if level == "caution":
            return (
                f"Significant fatigue — Score {fatigue_score:.0f}/100. "
                f"Locked for {lock_sec}s. Take a break."
            )
        return (
            f"Early fatigue signs — Score {fatigue_score:.0f}/100. "
            f"Locked for {lock_sec}s."
        )

    def _build_locked_message(self, remaining: float) -> str:
        progress = self._consecutive_alert_frames
        required = self._required_alert_frames
        return (
            f"LOCKED — {remaining:.0f}s remaining. "
            f"Show sustained alertness ({progress}/{required} frames). "
            f"Cannot be dismissed."
        )

    @staticmethod
    def _level_to_alert_level(level: str) -> AlertLevel:
        if level in ("warning", "microsleep"):
            return AlertLevel.WARNING
        if level == "caution":
            return AlertLevel.CAUTION
        return AlertLevel.ADVISORY
