"""Inference service with strict session state machine.

Session phases:
    CALIBRATING → MONITORING → ENDED

    CALIBRATING: CV pipeline calibrates EAR baseline. No fatigue, no alerts.
    MONITORING:  Active detection with countdown timer. Alerts fire.
    ENDED:       Session complete. auto_stop=True sent until WS closes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from src.api.alert_engine import AlertEngine
from src.api.models import AlertInfo, FrameResponse
from src.cv.pipeline import CVPipeline, FrameResult
from src.ml.fatigue_scorer import FatigueScorer, FatigueScoreResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ── Session configuration ──
MONITORING_DURATION_S = 45   # Seconds of active monitoring after calibration
HARD_LIMIT_FRAMES = 15 * 150  # 150s absolute max (~2.5 min)


class SessionPhase(str, Enum):
    """Strict session phases — no overlap allowed."""
    CALIBRATING = "calibrating"
    MONITORING = "monitoring"
    ENDED = "ended"


@dataclass
class SessionState:
    """Per-session runtime state."""

    phase: SessionPhase = SessionPhase.CALIBRATING
    frame_count: int = 0
    fatigue_scores: list[float] = field(default_factory=list)
    alert_engine: AlertEngine = field(default_factory=AlertEngine)
    last_emotion: str | None = None
    last_emotion_confidence: float = 0.0

    # Monitoring phase tracking
    monitoring_start_time: float = 0.0  # wall-clock time when monitoring started
    fatigue_detected_frames: int = 0
    was_locked: bool = False

    # Summary (set once when entering ENDED)
    end_summary: str = ""


class InferenceService:
    """End-to-end inference with strict phase enforcement."""

    def __init__(
        self,
        models_dir: Path,
        device: str = "cpu",
        dinov2_every_n: int = 3,
    ) -> None:
        self._models_dir = models_dir
        self._device = torch.device(device)
        self._dinov2_every_n = dinov2_every_n

        self._cv_pipeline: CVPipeline | None = None
        self._dinov2_model: nn.Module | None = None
        self._emotion_head: nn.Module | None = None
        self._emotion_labels: list[str] = []
        self._drowsiness_head: nn.Module | None = None
        self._drowsiness_labels: list[str] = []
        self._fatigue_scorer = FatigueScorer()

        self._sessions: dict[str, SessionState] = {}

    @property
    def models_loaded(self) -> dict[str, bool]:
        return {
            "cv_pipeline": self._cv_pipeline is not None,
            "dinov2_backbone": self._dinov2_model is not None,
            "emotion_head": self._emotion_head is not None,
            "drowsiness_head": self._drowsiness_head is not None,
        }

    def load_models(self) -> None:
        """Load all models into memory."""
        logger.info("Loading models from %s", self._models_dir)
        self._cv_pipeline = CVPipeline()
        logger.info("CV pipeline initialized")
        self._load_dinov2_backbone()
        self._load_head("affectnet", "affectnet_emotion", "_emotion_head", "_emotion_labels")
        self._load_head("nthu_ddd", "nthu_drowsiness", "_drowsiness_head", "_drowsiness_labels")
        logger.info("All models loaded: %s", self.models_loaded)

    def _load_dinov2_backbone(self) -> None:
        try:
            self._dinov2_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True,
            )
            self._dinov2_model = self._dinov2_model.to(self._device)
            self._dinov2_model.eval()
            logger.info("DINOv2 backbone loaded on %s", self._device)
        except Exception:
            logger.warning("Failed to load DINOv2 — emotion/drowsiness ML disabled")
            self._dinov2_model = None

    def _load_head(self, subdir: str, prefix: str, attr_head: str, attr_labels: str) -> None:
        model_path = self._models_dir / subdir / f"{prefix}_best.pt"
        meta_path = self._models_dir / subdir / f"{prefix}_metadata.json"
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            return
        try:
            checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                label_map = meta.get("label_map", {})
                labels = [""] * len(label_map)
                for name, idx in label_map.items():
                    labels[idx] = name
                setattr(self, attr_labels, labels)
            head = self._build_head_from_checkpoint(checkpoint)
            if head is not None:
                head = head.to(self._device)
                head.eval()
                setattr(self, attr_head, head)
                logger.info("Loaded %s (%d classes)", prefix, len(getattr(self, attr_labels)))
        except Exception:
            logger.exception("Failed to load %s", prefix)

    def _build_head_from_checkpoint(self, checkpoint: dict) -> nn.Module | None:
        from src.ml.train_dinov2_head import LinearProbe, MLPProbe
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        keys = list(state_dict.keys())
        last_weight_key = [k for k in keys if "weight" in k][-1]
        num_classes = state_dict[last_weight_key].shape[0]
        input_dim = 384
        if len(keys) <= 4:
            head = LinearProbe(input_dim, num_classes)
        else:
            first_linear = [k for k in keys if ".1.weight" in k or "net.1.weight" in k]
            hidden_dim = state_dict[first_linear[0]].shape[0] if first_linear else 128
            head = MLPProbe(input_dim, hidden_dim, num_classes)
        head.load_state_dict(state_dict)
        return head

    # ── Session lifecycle ──

    def start_session(self, session_id: str) -> None:
        state = SessionState()
        state.alert_engine = AlertEngine()
        self._sessions[session_id] = state
        if self._cv_pipeline:
            self._cv_pipeline.start_session()
        logger.info("Session started: %s (phase=CALIBRATING)", session_id)

    def end_session(self, session_id: str) -> tuple[int, float, float, int]:
        state = self._sessions.pop(session_id, None)
        if state is None:
            return 0, 0.0, 0.0, 0
        avg = sum(state.fatigue_scores) / len(state.fatigue_scores) if state.fatigue_scores else 0.0
        max_f = max(state.fatigue_scores) if state.fatigue_scores else 0.0
        return state.frame_count, avg, max_f, state.alert_engine.alert_count

    # ── Frame processing with strict phase enforcement ──

    def process_frame(
        self,
        session_id: str,
        frame: NDArray[np.uint8],
        frame_rgb: NDArray[np.uint8] | None = None,
    ) -> FrameResponse:
        state = self._sessions.get(session_id)
        if state is None:
            self.start_session(session_id)
            state = self._sessions[session_id]

        state.frame_count += 1

        if self._cv_pipeline is None:
            msg = "CV pipeline not loaded"
            raise RuntimeError(msg)

        # ── Always run CV pipeline (needed for calibration) ──
        cv_result: FrameResult = self._cv_pipeline.process_frame(frame)

        # ── Hard limit: force ENDED ──
        if state.frame_count >= HARD_LIMIT_FRAMES and state.phase != SessionPhase.ENDED:
            self._transition_to_ended(state, "Maximum session duration reached.")

        # ── Phase: CALIBRATING ──
        if state.phase == SessionPhase.CALIBRATING:
            return self._handle_calibrating(state, cv_result)

        # ── Phase: MONITORING ──
        if state.phase == SessionPhase.MONITORING:
            return self._handle_monitoring(state, cv_result, frame, frame_rgb)

        # ── Phase: ENDED ──
        return self._handle_ended(state, cv_result)

    def _handle_calibrating(self, state: SessionState, cv: FrameResult) -> FrameResponse:
        """CALIBRATING phase: only CV pipeline runs. Zero fatigue. No alerts."""

        # Check if calibration just finished
        if not cv.is_calibrating and cv.face_detected and state.frame_count > 5:
            # Transition to MONITORING
            state.phase = SessionPhase.MONITORING
            state.monitoring_start_time = time.monotonic()
            logger.info("Phase → MONITORING at frame %d", state.frame_count)

        landmarks = cv.landmarks_478[:, :2].tolist() if cv.landmarks_478 is not None else None

        return FrameResponse(
            frame_id=state.frame_count,
            timestamp=cv.timestamp,
            face_detected=cv.face_detected,
            landmarks=landmarks,
            ear_left=cv.ear.left if cv.ear else None,
            ear_right=cv.ear.right if cv.ear else None,
            ear_avg=cv.ear.average if cv.ear else None,
            ear_smoothed=cv.ear_smoothed.ear_smoothed if cv.ear_smoothed else None,
            mar=cv.mar,
            state="alert",  # Always "alert" during calibration
            fatigue_score=0.0,  # Always zero during calibration
            confidence=cv.confidence,
            perclos_60s=0.0,
            blink_rate_pm=0.0,
            is_calibrating=True,
            calibration_progress=cv.calibration_progress,
            processing_time_ms=cv.processing_time_ms,
            # No alerts, no auto_stop during calibration
            alert=None,
            is_locked=False,
            lock_remaining_seconds=0.0,
            lock_level="",
            lock_progress=0.0,
            pilot_message=None,
            auto_stop=False,
            session_summary=None,
        )

    def _handle_monitoring(
        self,
        state: SessionState,
        cv: FrameResult,
        frame: NDArray[np.uint8],
        frame_rgb: NDArray[np.uint8] | None,
    ) -> FrameResponse:
        """MONITORING phase: active fatigue detection, alerts, timer countdown."""

        elapsed = time.monotonic() - state.monitoring_start_time

        # ── DINOv2 emotion (every N frames) ──
        emotion = state.last_emotion
        emotion_conf = state.last_emotion_confidence
        if (
            cv.face_detected
            and self._dinov2_model is not None
            and self._emotion_head is not None
            and state.frame_count % self._dinov2_every_n == 0
        ):
            emotion, emotion_conf = self._run_emotion(frame_rgb or frame, cv)
            state.last_emotion = emotion
            state.last_emotion_confidence = emotion_conf

        # ── Fatigue scoring ──
        ear_dev = 0.0
        if cv.ear_smoothed and self._cv_pipeline._threshold.baseline_ear > 0:
            baseline = self._cv_pipeline._threshold.baseline_ear
            ear_dev = max(0.0, baseline - cv.ear_smoothed.ear_smoothed)

        fatigue = self._fatigue_scorer.compute(
            perclos=cv.perclos,
            blink_rate=cv.blink_rate_pm,
            ear_deviation=ear_dev,
            mar=cv.mar or 0.0,
        )
        state.fatigue_scores.append(fatigue.score)

        # ── Alert evaluation ──
        current_state = cv.state.name.lower()
        alert = state.alert_engine.evaluate(
            fatigue_score=fatigue.score,
            state=current_state,
            perclos=cv.perclos,
            blink_rate=cv.blink_rate_pm,
        )

        # Track fatigue detections
        if current_state in ("drowsy", "microsleep"):
            state.fatigue_detected_frames += 1

        # Track lock transitions
        currently_locked = state.alert_engine.is_locked
        if currently_locked:
            state.was_locked = True

        # ── Check end conditions ──
        should_end = False
        end_reason = ""

        # Condition 1: Lock was active and just released → end session
        if state.was_locked and not currently_locked:
            should_end = True
            end_reason = "Fatigue alert resolved. Session complete."

        # Condition 2: Timer expired
        if elapsed >= MONITORING_DURATION_S:
            should_end = True
            end_reason = "Monitoring period complete."

        if should_end:
            self._transition_to_ended(state, end_reason)

        landmarks = cv.landmarks_478[:, :2].tolist() if cv.landmarks_478 is not None else None

        return FrameResponse(
            frame_id=state.frame_count,
            timestamp=cv.timestamp,
            face_detected=cv.face_detected,
            landmarks=landmarks,
            ear_left=cv.ear.left if cv.ear else None,
            ear_right=cv.ear.right if cv.ear else None,
            ear_avg=cv.ear.average if cv.ear else None,
            ear_smoothed=cv.ear_smoothed.ear_smoothed if cv.ear_smoothed else None,
            mar=cv.mar,
            state=current_state,
            fatigue_score=fatigue.score,
            emotion=emotion,
            emotion_confidence=emotion_conf,
            confidence=cv.confidence,
            perclos_60s=cv.perclos,
            blink_rate_pm=cv.blink_rate_pm,
            is_calibrating=False,
            calibration_progress=1.0,
            processing_time_ms=cv.processing_time_ms,
            alert=alert,
            is_locked=state.alert_engine.is_locked,
            lock_remaining_seconds=state.alert_engine.lock_remaining_seconds,
            lock_level=state.alert_engine.lock_level,
            lock_progress=state.alert_engine.alert_frames_progress,
            pilot_message=None,
            auto_stop=state.phase == SessionPhase.ENDED,
            session_summary=state.end_summary if state.phase == SessionPhase.ENDED else None,
        )

    def _handle_ended(self, state: SessionState, cv: FrameResult) -> FrameResponse:
        """ENDED phase: keep sending auto_stop=True until WS closes."""
        landmarks = cv.landmarks_478[:, :2].tolist() if cv.landmarks_478 is not None else None

        return FrameResponse(
            frame_id=state.frame_count,
            timestamp=cv.timestamp,
            face_detected=cv.face_detected,
            landmarks=landmarks,
            state="alert",
            fatigue_score=0.0,
            is_calibrating=False,
            calibration_progress=1.0,
            processing_time_ms=cv.processing_time_ms,
            auto_stop=True,
            session_summary=state.end_summary,
        )

    def _transition_to_ended(self, state: SessionState, reason: str) -> None:
        """Transition to ENDED phase and build summary."""
        if state.phase == SessionPhase.ENDED:
            return

        state.phase = SessionPhase.ENDED
        elapsed = time.monotonic() - state.monitoring_start_time if state.monitoring_start_time > 0 else 0
        avg_f = sum(state.fatigue_scores) / len(state.fatigue_scores) if state.fatigue_scores else 0.0
        max_f = max(state.fatigue_scores) if state.fatigue_scores else 0.0
        total_monitoring_frames = max(1, len(state.fatigue_scores))
        fatigue_pct = state.fatigue_detected_frames / total_monitoring_frames * 100

        if state.fatigue_detected_frames > 0:
            state.end_summary = (
                f"Fatigue was detected during this session. "
                f"{reason} "
                f"Duration: {elapsed:.0f}s | "
                f"Fatigue in {fatigue_pct:.0f}% of frames | "
                f"Avg score: {avg_f:.1f}/100 | "
                f"Peak: {max_f:.1f}/100 | "
                f"Alerts: {state.alert_engine.alert_count}."
            )
        else:
            state.end_summary = (
                f"No significant fatigue detected. {reason} "
                f"Duration: {elapsed:.0f}s | "
                f"Avg score: {avg_f:.1f}/100 | "
                f"Peak: {max_f:.1f}/100."
            )

        logger.info("Phase → ENDED: %s", state.end_summary[:80])

    def _run_emotion(
        self,
        frame: NDArray[np.uint8],
        cv_result: FrameResult,
    ) -> tuple[str, float]:
        from src.ml.dinov2_features import DINOV2_TRANSFORM
        assert self._dinov2_model is not None
        assert self._emotion_head is not None
        try:
            if cv_result.landmarks_68 is None:
                return "unknown", 0.0
            pts = cv_result.landmarks_68.points
            h, w = frame.shape[:2]
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            pad = 0.3
            face_w = xs.max() - xs.min()
            face_h = ys.max() - ys.min()
            x1 = max(0, int(xs.min() - face_w * pad))
            y1 = max(0, int(ys.min() - face_h * pad))
            x2 = min(w, int(xs.max() + face_w * pad))
            y2 = min(h, int(ys.max() + face_h * pad))
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return "unknown", 0.0
            tensor = DINOV2_TRANSFORM(face_crop).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._dinov2_model(tensor)
                logits = self._emotion_head(features)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = probs.max(dim=1)
            label = self._emotion_labels[pred_idx.item()] if self._emotion_labels else "unknown"
            return label, confidence.item()
        except Exception:
            logger.debug("Emotion inference failed", exc_info=True)
            return "unknown", 0.0

    def close(self) -> None:
        if self._cv_pipeline:
            self._cv_pipeline.close()
            self._cv_pipeline = None
        self._dinov2_model = None
        self._emotion_head = None
        self._drowsiness_head = None
        self._sessions.clear()
