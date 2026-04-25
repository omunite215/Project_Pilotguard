"""Integration tests for the PilotGuard FastAPI backend.

Tests REST endpoints and WebSocket frame processing.
Uses httpx for async HTTP and FastAPI's TestClient for WebSocket.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.alert_engine import AlertEngine
from src.api.models import AlertLevel, FrameResponse, SessionStatus
from src.api.session_store import SessionStore


# ── Alert Engine Tests ──


class TestAlertEngine:
    """Unit tests for the AlertEngine with debounce."""

    @staticmethod
    def _fire_drowsy(engine: AlertEngine, n: int = 15, score: float = 35.0) -> AlertInfo | None:
        """Send N consecutive drowsy frames (default=15 to meet debounce)."""
        result = None
        for _ in range(n):
            result = engine.evaluate(fatigue_score=score, state="drowsy", perclos=40.0, blink_rate=10.0)
        return result

    @staticmethod
    def _fire_microsleep(engine: AlertEngine, n: int = 8) -> AlertInfo | None:
        """Send N consecutive microsleep frames (default=8 to meet debounce)."""
        result = None
        for _ in range(n):
            result = engine.evaluate(fatigue_score=50.0, state="microsleep", perclos=50.0, blink_rate=5.0)
        return result

    def test_normal_score_no_alert(self) -> None:
        engine = AlertEngine()
        result = engine.evaluate(fatigue_score=10.0, state="alert", perclos=5.0, blink_rate=15.0)
        assert result is None

    def test_debounce_prevents_single_frame_alert(self) -> None:
        engine = AlertEngine()
        result = engine.evaluate(fatigue_score=35.0, state="drowsy", perclos=40.0, blink_rate=10.0)
        assert result is None  # Single frame — debounce blocks it

    def test_advisory_alert_after_debounce(self) -> None:
        engine = AlertEngine()
        result = self._fire_drowsy(engine, 15, score=35.0)
        assert result is not None
        assert result.level == AlertLevel.ADVISORY

    def test_caution_alert(self) -> None:
        engine = AlertEngine()
        result = self._fire_drowsy(engine, 15, score=60.0)
        assert result is not None
        assert result.level == AlertLevel.CAUTION

    def test_warning_alert(self) -> None:
        engine = AlertEngine()
        result = self._fire_drowsy(engine, 15, score=80.0)
        assert result is not None
        assert result.level == AlertLevel.WARNING

    def test_microsleep_after_debounce(self) -> None:
        engine = AlertEngine()
        result = self._fire_microsleep(engine, 8)
        assert result is not None
        assert result.level == AlertLevel.WARNING
        assert "MICROSLEEP" in result.message

    def test_debounce_resets_on_alert_frame(self) -> None:
        engine = AlertEngine()
        # 10 drowsy frames, then 1 alert, then 5 more drowsy = no trigger
        self._fire_drowsy(engine, 10)
        engine.evaluate(fatigue_score=10.0, state="alert", perclos=5.0, blink_rate=15.0)
        result = self._fire_drowsy(engine, 5)
        assert result is None
        assert not engine.is_locked

    def test_lock_prevents_dismiss(self) -> None:
        engine = AlertEngine()
        self._fire_drowsy(engine, 15)
        assert engine.is_locked
        # Alert frame during lock — still locked
        r2 = engine.evaluate(fatigue_score=10.0, state="alert", perclos=5.0, blink_rate=15.0)
        assert r2 is not None
        assert engine.is_locked

    def test_lock_escalation(self) -> None:
        engine = AlertEngine()
        self._fire_drowsy(engine, 15, score=35.0)
        assert engine.lock_level == "advisory"
        # Microsleep during lock escalates
        r = engine.evaluate(fatigue_score=80.0, state="microsleep", perclos=50.0, blink_rate=3.0)
        assert r is not None
        assert r.level == AlertLevel.WARNING
        assert engine.lock_level == "microsleep"

    def test_alert_count(self) -> None:
        engine = AlertEngine()
        self._fire_drowsy(engine, 15)
        assert engine.alert_count == 1
        # Escalation during lock
        engine.evaluate(fatigue_score=80.0, state="microsleep", perclos=50.0, blink_rate=3.0)
        assert engine.alert_count == 2

    def test_reset_clears_state(self) -> None:
        engine = AlertEngine(cooldown_seconds=0.0)
        engine.evaluate(fatigue_score=35.0, state="alert", perclos=20.0, blink_rate=12.0)
        engine.reset()
        assert engine.alert_count == 0


# ── Session Store Tests ──


class TestSessionStore:
    """Test async SQLite session storage."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        s = SessionStore(tmp_path / "test.db")
        await s.initialize()
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_create_and_get_session(self, store: SessionStore) -> None:
        sid = await store.create_session()
        info = await store.get_session(sid)
        assert info.session_id == sid
        assert info.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_end_session(self, store: SessionStore) -> None:
        sid = await store.create_session()
        info = await store.end_session(sid, 100, 25.5, 60.0, 3)
        assert info.status == SessionStatus.COMPLETED
        assert info.total_frames == 100
        assert info.avg_fatigue_score == 25.5
        assert info.alert_count == 3

    @pytest.mark.asyncio
    async def test_list_sessions(self, store: SessionStore) -> None:
        await store.create_session()
        await store.create_session()
        sessions, total = await store.list_sessions(page=1, page_size=10)
        assert total == 2
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_save_and_get_alerts(self, store: SessionStore) -> None:
        from src.api.models import AlertInfo

        sid = await store.create_session()
        alert = AlertInfo(
            level=AlertLevel.CAUTION,
            message="Test alert",
            fatigue_score=55.0,
            timestamp=1000.0,
        )
        await store.save_alert(sid, alert)
        alerts = await store.get_alerts(sid)
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.CAUTION

    @pytest.mark.asyncio
    async def test_get_missing_session_raises(self, store: SessionStore) -> None:
        with pytest.raises(KeyError):
            await store.get_session("nonexistent")


# ── REST Endpoint Tests ──


class TestRESTEndpoints:
    """Test FastAPI REST endpoints with mocked inference."""

    @pytest_asyncio.fixture
    async def client(self, tmp_path):
        """Create test client with mocked inference service."""
        # Mock the inference service to avoid loading real models
        mock_inference = MagicMock()
        mock_inference.models_loaded = {
            "cv_pipeline": True, "dinov2_backbone": False,
            "emotion_head": False, "drowsiness_head": False,
        }
        mock_inference._sessions = {}
        mock_inference.end_session.return_value = (100, 25.0, 60.0, 2)

        with (
            patch("src.api.main._inference", mock_inference),
            patch("src.api.main.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.models_dir = tmp_path / "models"

            from src.api.main import app

            # Initialize store manually for tests
            import src.api.main as main_module

            store = SessionStore(tmp_path / "test.db")
            await store.initialize()
            main_module._store = store

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                yield c

            await store.close()

    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient) -> None:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, client: AsyncClient) -> None:
        # Start session
        resp = await client.post("/api/session/start")
        assert resp.status_code == 200
        data = resp.json()
        session_id = data["session_id"]
        assert data["status"] == "active"

        # Get session
        resp = await client.get(f"/api/session/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

        # Stop session
        resp = await client.post(f"/api/session/{session_id}/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_sessions(self, client: AsyncClient) -> None:
        await client.post("/api/session/start")
        await client.post("/api/session/start")
        resp = await client.get("/api/sessions?page=1&page_size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2

    @pytest.mark.asyncio
    async def test_alerts_empty_session(self, client: AsyncClient) -> None:
        resp = await client.post("/api/session/start")
        sid = resp.json()["session_id"]
        resp = await client.get(f"/api/alerts/{sid}")
        assert resp.status_code == 200
        assert resp.json() == []


# ── Frame Response Model Tests ──


class TestFrameResponse:
    """Test FrameResponse serialization."""

    def test_minimal_frame_response(self) -> None:
        resp = FrameResponse(
            frame_id=1,
            timestamp=1000.0,
            face_detected=False,
        )
        data = resp.model_dump()
        assert data["frame_id"] == 1
        assert data["face_detected"] is False
        assert data["state"] == "unknown"
        assert data["alert"] is None

    def test_full_frame_response(self) -> None:
        from src.api.models import AlertInfo

        resp = FrameResponse(
            frame_id=42,
            timestamp=1000.0,
            face_detected=True,
            landmarks=[[0.1, 0.2]] * 68,
            ear_left=0.31,
            ear_right=0.29,
            ear_avg=0.30,
            ear_smoothed=0.305,
            mar=0.15,
            state="alert",
            fatigue_score=23.5,
            emotion="neutral",
            emotion_confidence=0.87,
            confidence=0.95,
            perclos_60s=12.3,
            blink_rate_pm=18.0,
            alert=AlertInfo(
                level=AlertLevel.ADVISORY,
                message="test",
                fatigue_score=23.5,
                timestamp=1000.0,
            ),
        )
        data = resp.model_dump()
        assert data["state"] == "alert"
        assert data["alert"]["level"] == "advisory"
        assert len(data["landmarks"]) == 68
