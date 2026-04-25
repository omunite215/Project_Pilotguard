"""WebSocket handler for real-time frame processing.

Protocol:
    Client sends: binary JPEG/PNG frames OR JSON config messages.
    Server responds: JSON FrameResponse per frame.

Uses a simple receive-process loop (no queue) to avoid complexity.
"""

from __future__ import annotations

import logging
import time
import traceback

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.api.inference import InferenceService
from src.api.session_store import SessionStore

logger = logging.getLogger(__name__)


async def handle_ws_session(
    ws: WebSocket,
    session_id: str,
    inference: InferenceService,
    store: SessionStore,
) -> None:
    """Handle a WebSocket monitoring session.

    Accepts binary JPEG frames from the client, processes them through
    the inference pipeline, and sends back JSON results.
    """
    await ws.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    min_frame_interval = 1.0 / 15  # 15 FPS max
    last_frame_time = 0.0

    # Start inference session
    inference.start_session(session_id)

    try:
        while True:
            # Receive data from client
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                logger.info("Client disconnected: session=%s", session_id)
                break
            except Exception:
                logger.debug("Receive error", exc_info=True)
                break

            # Check for disconnect message
            msg_type = message.get("type", "")
            if msg_type == "websocket.disconnect":
                break

            # Get binary data
            frame_bytes = message.get("bytes")
            if not frame_bytes:
                continue

            # Rate limiting
            now = time.monotonic()
            if now - last_frame_time < min_frame_interval:
                continue
            last_frame_time = now

            try:
                # Decode JPEG/PNG
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Run inference
                result = inference.process_frame(session_id, frame)

                # Save alert if fired
                if result.alert:
                    await store.save_alert(session_id, result.alert)

                # Send result
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(result.model_dump())

                # Break loop on auto-stop so the session cleanly terminates
                if result.auto_stop:
                    logger.info("Auto-stop triggered: session=%s", session_id)
                    break

            except Exception:
                logger.warning("Frame processing error: %s", traceback.format_exc())
                # Send a minimal error response so the client knows we're alive
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({
                            "frame_id": 0,
                            "timestamp": time.time(),
                            "face_detected": False,
                            "state": "unknown",
                            "fatigue_score": 0,
                            "confidence": 0,
                            "perclos_60s": 0,
                            "blink_rate_pm": 0,
                            "is_calibrating": False,
                            "calibration_progress": 0,
                            "processing_time_ms": 0,
                            "alert": None,
                        })
                except Exception:
                    break

    finally:
        # End session and persist stats
        frames, avg_f, max_f, alerts = inference.end_session(session_id)
        try:
            await store.end_session(session_id, frames, avg_f, max_f, alerts)
        except Exception:
            logger.debug("Failed to persist session end", exc_info=True)
        logger.info(
            "Session ended: %s — %d frames, avg_fatigue=%.1f, alerts=%d",
            session_id, frames, avg_f, alerts,
        )
