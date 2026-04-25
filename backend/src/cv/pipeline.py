"""Full per-frame computer vision pipeline.

Orchestrates all CV modules into a single process_frame() call:
    Frame → Preprocess → Face Detection → Landmarks → EAR/MAR →
    Kalman Filter → Blink Detection → PERCLOS → State Classification

Target latency: <30ms per frame on CPU (geometric pipeline, no DINOv2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import cv2
import numpy as np  # noqa: TC002

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.cv.adaptive_threshold import AdaptiveThreshold, CalibrationState
from src.cv.blink_detector import BlinkDetector, BlinkEvent
from src.cv.ear import EARResult, compute_ear
from src.cv.face_detector import FaceDetector
from src.cv.kalman import EARKalmanFilter, KalmanState
from src.cv.landmark_extractor import Landmarks68, extract_landmarks_68
from src.cv.mar import compute_mar
from src.cv.perclos import PERCLOSCalculator


class CognitiveState(Enum):
    """High-level cognitive state classification."""

    ALERT = auto()
    DROWSY = auto()
    MICROSLEEP = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Complete result of processing a single video frame.

    Attributes:
        timestamp: Processing timestamp in seconds.
        face_detected: Whether a face was found in the frame.
        landmarks_68: 68-point landmarks (None if no face).
        ear: EAR result (None if no face).
        ear_smoothed: Kalman-filtered EAR (None if no face).
        mar: Mouth aspect ratio (None if no face).
        state: Classified cognitive state.
        perclos: Current PERCLOS percentage.
        blink_rate_pm: Blinks per minute.
        blink_event: Completed blink event if one just ended, else None.
        is_eyes_closed: Whether eyes are currently closed.
        confidence: Detection confidence.
        calibration_progress: Calibration progress [0, 1].
        is_calibrating: Whether currently in calibration phase.
        processing_time_ms: Time taken to process this frame.
    """

    timestamp: float
    face_detected: bool
    landmarks_68: Landmarks68 | None
    landmarks_478: NDArray[np.float32] | None
    ear: EARResult | None
    ear_smoothed: KalmanState | None
    mar: float | None
    state: CognitiveState
    perclos: float
    blink_rate_pm: float
    blink_event: BlinkEvent | None
    is_eyes_closed: bool
    confidence: float
    calibration_progress: float
    is_calibrating: bool
    processing_time_ms: float


@dataclass
class CVPipeline:
    """Full computer vision pipeline for fatigue monitoring.

    Manages the lifecycle of all CV components and provides a single
    process_frame() entry point.

    Args:
        calibration_duration: Seconds for EAR calibration (default 30).
        ear_threshold_multiplier: Fraction of baseline EAR for threshold.
        perclos_window_seconds: PERCLOS rolling window duration.
        microsleep_threshold_ms: Minimum closure duration for microsleep.
        kalman_process_noise: Kalman filter process noise.
        kalman_measurement_noise: Kalman filter measurement noise.
        frame_width: Target frame width for preprocessing.
        frame_height: Target frame height for preprocessing.
    """

    calibration_duration: float = 10.0
    ear_threshold_multiplier: float = 0.75
    perclos_window_seconds: float = 60.0
    microsleep_threshold_ms: float = 500.0
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.1
    frame_width: int = 640
    frame_height: int = 480

    # Components (initialized in __post_init__)
    _detector: FaceDetector = field(init=False)
    _kalman: EARKalmanFilter = field(init=False)
    _blink_detector: BlinkDetector = field(init=False)
    _perclos: PERCLOSCalculator = field(init=False)
    _threshold: AdaptiveThreshold = field(init=False)
    _session_active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._detector = FaceDetector()
        self._kalman = EARKalmanFilter(
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise,
        )
        self._blink_detector = BlinkDetector(
            microsleep_threshold_ms=self.microsleep_threshold_ms,
        )
        self._perclos = PERCLOSCalculator(
            window_seconds=self.perclos_window_seconds,
        )
        self._threshold = AdaptiveThreshold(
            calibration_duration=self.calibration_duration,
            threshold_multiplier=self.ear_threshold_multiplier,
        )

    def start_session(self) -> None:
        """Initialize a new monitoring session with calibration."""
        self._kalman.reset()
        self._blink_detector.reset()
        self._perclos.reset()
        self._threshold.reset()
        self._threshold.start_calibration(time.monotonic())
        self._session_active = True

    def process_frame(self, frame: NDArray[np.uint8]) -> FrameResult:
        """Process a single video frame through the full CV pipeline.

        Args:
            frame: Input frame in BGR or RGB format, any resolution.
                Will be resized and converted as needed.

        Returns:
            FrameResult with all computed metrics.
        """
        t_start = time.perf_counter()
        now = time.monotonic()

        # Auto-start session on first frame
        if not self._session_active:
            self.start_session()

        # ── 1. Preprocess ──
        frame_rgb = self._preprocess(frame)

        # ── 2. Face Detection ──
        detection = self._detector.detect(frame_rgb)

        if detection is None:
            processing_ms = (time.perf_counter() - t_start) * 1000.0
            return FrameResult(
                timestamp=now,
                face_detected=False,
                landmarks_68=None,
                landmarks_478=None,
                ear=None,
                ear_smoothed=None,
                mar=None,
                state=CognitiveState.UNKNOWN,
                perclos=self._perclos.update(now, 1.0, self._threshold.threshold),
                blink_rate_pm=self._blink_detector.blink_rate_per_minute(),
                blink_event=None,
                is_eyes_closed=False,
                confidence=0.0,
                calibration_progress=self._threshold.calibration_progress,
                is_calibrating=self._threshold.state == CalibrationState.IN_PROGRESS,
                processing_time_ms=processing_ms,
            )

        # ── 3. Landmark Extraction ──
        landmarks = extract_landmarks_68(detection.landmarks_478)

        # ── 4. EAR & MAR ──
        ear_result = compute_ear(landmarks.right_eye, landmarks.left_eye)
        mar_value = compute_mar(landmarks.mouth[12:])  # Inner mouth = last 8 of 20

        # ── 5. Calibration ──
        if self._threshold.state == CalibrationState.IN_PROGRESS:
            just_finished = self._threshold.update(ear_result.average, now)
            if just_finished:
                self._kalman.reset(self._threshold.baseline_ear)

        # ── 6. Kalman Filter ──
        kalman_state = self._kalman.update(ear_result.average)

        # ── 7. Blink Detection ──
        threshold = self._threshold.threshold
        blink_event = self._blink_detector.update(
            kalman_state.ear_smoothed, threshold, now,
        )

        # ── 8. PERCLOS ──
        perclos = self._perclos.update(now, kalman_state.ear_smoothed, threshold)

        # ── 9. State Classification (geometric-only for Phase 2) ──
        # During calibration, always report ALERT — threshold is not calibrated yet
        if self._threshold.state == CalibrationState.IN_PROGRESS:
            state = CognitiveState.ALERT
        else:
            state = self._classify_state(kalman_state.ear_smoothed, threshold, perclos)

        processing_ms = (time.perf_counter() - t_start) * 1000.0

        return FrameResult(
            timestamp=now,
            face_detected=True,
            landmarks_68=landmarks,
            landmarks_478=detection.landmarks_478,
            ear=ear_result,
            ear_smoothed=kalman_state,
            mar=mar_value,
            state=state,
            perclos=perclos,
            blink_rate_pm=self._blink_detector.blink_rate_per_minute(),
            blink_event=blink_event,
            is_eyes_closed=self._blink_detector.is_eyes_closed,
            confidence=detection.confidence,
            calibration_progress=self._threshold.calibration_progress,
            is_calibrating=self._threshold.state == CalibrationState.IN_PROGRESS,
            processing_time_ms=processing_ms,
        )

    def _preprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Resize and convert frame for MediaPipe processing.

        Args:
            frame: Input frame (BGR from OpenCV or RGB).

        Returns:
            RGB frame resized to target dimensions.
        """
        h, w = frame.shape[:2]

        # Resize if larger than target
        if w != self.frame_width or h != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        # Convert BGR → RGB if needed (OpenCV default is BGR)
        # MediaPipe expects RGB
        frame_rgb: NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame_rgb

    def _classify_state(
        self,
        ear_smoothed: float,
        threshold: float,
        perclos: float,
    ) -> CognitiveState:
        """Classify cognitive state from geometric features.

        This is the Phase 2 geometric-only classifier. Phase 3 will add
        HMM temporal modeling and DINOv2 deep features.

        Args:
            ear_smoothed: Kalman-filtered EAR value.
            threshold: Adaptive eye-closure threshold.
            perclos: Current PERCLOS percentage.

        Returns:
            Classified CognitiveState.
        """
        # Microsleep: eyes closed for >1.5 seconds continuous
        # (normal blinks are 150-400ms, slow blinks up to 600ms)
        if self._blink_detector.current_closure_duration_ms > 1500:
            return CognitiveState.MICROSLEEP

        # Drowsy: requires SUSTAINED indicators
        # PERCLOS > 60% means eyes closed >60% of the last 60 seconds
        if perclos > 60.0:
            return CognitiveState.DROWSY
        if ear_smoothed < threshold * 0.8 and perclos > 40.0:
            return CognitiveState.DROWSY

        return CognitiveState.ALERT

    def close(self) -> None:
        """Release all resources."""
        self._detector.close()
        self._session_active = False

    def __enter__(self) -> CVPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
