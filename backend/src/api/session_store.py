"""Session and alert storage using async SQLite.

Sessions table stores session lifecycle data.
Alerts table stores per-session alert events.
Frame results are kept in-memory during active sessions only
(too high-volume for persistent storage).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from pathlib import Path

from src.api.models import AlertInfo, AlertLevel, SessionInfo, SessionStatus

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    total_frames INTEGER DEFAULT 0,
    avg_fatigue_score REAL DEFAULT 0.0,
    max_fatigue_score REAL DEFAULT 0.0,
    alert_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    fatigue_score REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


class SessionStore:
    """Async SQLite-backed session and alert storage.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database and tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("Session store initialized: %s", self._db_path)

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def create_session(self, calibration_duration: float = 30.0) -> str:
        """Create a new monitoring session.

        Returns:
            New session ID.
        """
        assert self._db is not None
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            "INSERT INTO sessions (session_id, started_at, status) VALUES (?, ?, ?)",
            (session_id, now, SessionStatus.ACTIVE.value),
        )
        await self._db.commit()
        logger.info("Session created: %s", session_id)
        return session_id

    async def end_session(
        self,
        session_id: str,
        total_frames: int,
        avg_fatigue: float,
        max_fatigue: float,
        alert_count: int,
    ) -> SessionInfo:
        """End an active session with final statistics.

        Returns:
            Updated SessionInfo.
        """
        assert self._db is not None
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """UPDATE sessions SET
                ended_at = ?, total_frames = ?, avg_fatigue_score = ?,
                max_fatigue_score = ?, alert_count = ?, status = ?
            WHERE session_id = ?""",
            (now, total_frames, avg_fatigue, max_fatigue, alert_count,
             SessionStatus.COMPLETED.value, session_id),
        )
        await self._db.commit()
        return await self.get_session(session_id)

    async def get_session(self, session_id: str) -> SessionInfo:
        """Get session info by ID."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                msg = f"Session not found: {session_id}"
                raise KeyError(msg)
            return self._row_to_session_info(row)

    async def list_sessions(
        self, page: int = 1, page_size: int = 20,
    ) -> tuple[list[SessionInfo], int]:
        """List sessions with pagination.

        Returns:
            Tuple of (sessions, total_count).
        """
        assert self._db is not None
        async with self._db.execute("SELECT COUNT(*) FROM sessions") as cursor:
            total = (await cursor.fetchone())[0]

        offset = (page - 1) * page_size
        async with self._db.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (page_size, offset),
        ) as cursor:
            rows = await cursor.fetchall()
            sessions = [self._row_to_session_info(r) for r in rows]

        return sessions, total

    async def save_alert(self, session_id: str, alert: AlertInfo) -> None:
        """Save an alert event."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO alerts (session_id, timestamp, level, message, fatigue_score) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, alert.timestamp, alert.level.value, alert.message, alert.fatigue_score),
        )
        await self._db.commit()

    async def get_alerts(self, session_id: str) -> list[AlertInfo]:
        """Get all alerts for a session."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM alerts WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                AlertInfo(
                    level=AlertLevel(row["level"]),
                    message=row["message"],
                    fatigue_score=row["fatigue_score"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    @staticmethod
    def _row_to_session_info(row: aiosqlite.Row) -> SessionInfo:
        """Convert a database row to SessionInfo."""
        started = datetime.fromisoformat(row["started_at"])
        ended = datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None
        duration = (ended - started).total_seconds() if ended else 0.0
        return SessionInfo(
            session_id=row["session_id"],
            started_at=started,
            ended_at=ended,
            duration_seconds=duration,
            total_frames=row["total_frames"],
            avg_fatigue_score=row["avg_fatigue_score"],
            max_fatigue_score=row["max_fatigue_score"],
            alert_count=row["alert_count"],
            status=SessionStatus(row["status"]),
        )
