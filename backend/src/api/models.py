"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ──


class AlertLevel(str, Enum):
    """Alert severity levels."""

    NORMAL = "normal"
    ADVISORY = "advisory"
    CAUTION = "caution"
    WARNING = "warning"


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    COMPLETED = "completed"


# ── WebSocket Messages ──


class WSConfig(BaseModel):
    """Client-sent WebSocket configuration message."""

    type: str = "config"
    fps: int = Field(default=30, ge=1, le=60)
    resolution: tuple[int, int] = (640, 480)


class AlertInfo(BaseModel):
    """Alert generated during monitoring."""

    level: AlertLevel
    message: str
    fatigue_score: float
    timestamp: float


class FrameResponse(BaseModel):
    """Per-frame result sent back over WebSocket."""

    frame_id: int
    timestamp: float
    face_detected: bool
    landmarks: list[list[float]] | None = None
    ear_left: float | None = None
    ear_right: float | None = None
    ear_avg: float | None = None
    ear_smoothed: float | None = None
    mar: float | None = None
    state: str = "unknown"
    fatigue_score: float = 0.0
    emotion: str | None = None
    emotion_confidence: float | None = None
    confidence: float = 0.0
    perclos_60s: float = 0.0
    blink_rate_pm: float = 0.0
    is_calibrating: bool = False
    calibration_progress: float = 0.0
    processing_time_ms: float = 0.0
    alert: AlertInfo | None = None
    is_locked: bool = False
    lock_remaining_seconds: float = 0.0
    lock_level: str = ""
    lock_progress: float = 0.0
    pilot_message: str | None = None
    auto_stop: bool = False
    session_summary: str | None = None


# ── REST Models ──


class SessionStartRequest(BaseModel):
    """Request to start a new monitoring session."""

    calibration_duration: float = Field(default=30.0, ge=5.0, le=120.0)


class SessionStartResponse(BaseModel):
    """Response after starting a session."""

    session_id: str
    started_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE


class SessionStopResponse(BaseModel):
    """Response after stopping a session."""

    session_id: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    total_frames: int
    avg_fatigue_score: float
    max_fatigue_score: float
    alert_count: int
    status: SessionStatus = SessionStatus.COMPLETED


class SessionInfo(BaseModel):
    """Session summary for listing."""

    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float = 0.0
    total_frames: int = 0
    avg_fatigue_score: float = 0.0
    max_fatigue_score: float = 0.0
    alert_count: int = 0
    status: SessionStatus = SessionStatus.ACTIVE


class SessionListResponse(BaseModel):
    """Paginated session list."""

    sessions: list[SessionInfo]
    total: int
    page: int
    page_size: int


class CalibrationRequest(BaseModel):
    """Request to run calibration."""

    num_frames: int = Field(default=90, ge=10, le=300)


class CalibrationResponse(BaseModel):
    """Calibration result."""

    baseline_ear: float
    threshold: float
    frames_used: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    active_sessions: int = 0
