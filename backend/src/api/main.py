"""PilotGuard FastAPI application.

Endpoints:
    GET  /api/health              — Health check + model status
    POST /api/session/start       — Create a new monitoring session
    POST /api/session/{id}/stop   — End a session, get summary stats
    GET  /api/session/{id}        — Get session details
    GET  /api/sessions            — List sessions (paginated)
    GET  /api/alerts/{session_id} — Get alerts for a session
    WS   /api/ws/{session_id}     — Real-time frame processing
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import InferenceService
from src.api.models import (
    HealthResponse,
    SessionInfo,
    SessionListResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionStatus,
    SessionStopResponse,
)
from src.api.session_store import SessionStore
from src.api.ws_handler import handle_ws_session
from src.utils.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Shared services (initialized in lifespan)
_inference: InferenceService | None = None
_store: SessionStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models and resources on startup, clean up on shutdown."""
    global _inference, _store  # noqa: PLW0603

    # Initialize session store
    db_path = settings.data_dir / "pilotguard.db"
    _store = SessionStore(db_path)
    await _store.initialize()

    # Initialize inference service — auto-detect GPU
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA GPU detected: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("No CUDA GPU — running on CPU")
    _inference = InferenceService(
        models_dir=settings.models_dir,
        device=device,
        dinov2_every_n=3,
    )
    _inference.load_models()

    logger.info("PilotGuard API ready — models: %s", _inference.models_loaded)

    yield

    # Cleanup
    if _inference:
        _inference.close()
    if _store:
        await _store.close()
    logger.info("PilotGuard API shutdown complete")


app = FastAPI(
    title="PilotGuard API",
    version="0.1.0",
    description="Real-time pilot cognitive state monitoring",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──


@app.get("/api/health")
async def health() -> HealthResponse:
    """Health check with model status."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        models_loaded=_inference.models_loaded if _inference else {},
        active_sessions=len(_inference._sessions) if _inference else 0,
    )


# ── Sessions ──


@app.post("/api/session/start")
async def session_start(
    body: SessionStartRequest | None = None,
) -> SessionStartResponse:
    """Create a new monitoring session."""
    assert _store is not None
    req = body or SessionStartRequest()
    session_id = await _store.create_session(req.calibration_duration)
    info = await _store.get_session(session_id)
    return SessionStartResponse(
        session_id=info.session_id,
        started_at=info.started_at,
        status=SessionStatus.ACTIVE,
    )


@app.post("/api/session/{session_id}/stop")
async def session_stop(session_id: str) -> SessionStopResponse:
    """Stop an active session and get summary statistics."""
    assert _store is not None and _inference is not None

    # Get inference stats and end session
    frames, avg_f, max_f, alerts = _inference.end_session(session_id)
    info = await _store.end_session(session_id, frames, avg_f, max_f, alerts)

    return SessionStopResponse(
        session_id=info.session_id,
        started_at=info.started_at,
        ended_at=info.ended_at,
        duration_seconds=info.duration_seconds,
        total_frames=info.total_frames,
        avg_fatigue_score=info.avg_fatigue_score,
        max_fatigue_score=info.max_fatigue_score,
        alert_count=info.alert_count,
        status=SessionStatus.COMPLETED,
    )


@app.get("/api/session/{session_id}")
async def session_detail(session_id: str) -> SessionInfo:
    """Get session details by ID."""
    assert _store is not None
    return await _store.get_session(session_id)


@app.get("/api/sessions")
async def session_list(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> SessionListResponse:
    """List sessions with pagination."""
    assert _store is not None
    sessions, total = await _store.list_sessions(page, page_size)
    return SessionListResponse(
        sessions=sessions,
        total=total,
        page=page,
        page_size=page_size,
    )


# ── Alerts ──


@app.get("/api/alerts/{session_id}")
async def alerts_for_session(session_id: str) -> list:
    """Get all alerts for a session."""
    assert _store is not None
    return await _store.get_alerts(session_id)


# ── WebSocket ──


@app.websocket("/api/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time frame processing.

    Client sends binary JPEG frames. Server responds with JSON FrameResponse.
    """
    assert _inference is not None and _store is not None
    await handle_ws_session(ws, session_id, _inference, _store)
