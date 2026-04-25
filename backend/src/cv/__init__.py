"""PilotGuard computer vision pipeline.

Public API:
    - CVPipeline: Full per-frame processing orchestrator
    - FrameResult: Result of processing a single frame
    - CognitiveState: Alert/Drowsy/Microsleep classification
    - FaceDetector: MediaPipe face detection
    - Landmarks68: 68-point landmark data
    - EARResult: Eye Aspect Ratio result
    - BlinkEvent: Completed blink/microsleep event
    - CalibrationState: Calibration progress state

All imports are lazy — use direct submodule imports for better performance.
"""

__all__ = [
    "AdaptiveThreshold",
    "BlinkDetector",
    "BlinkEvent",
    "CVPipeline",
    "CalibrationState",
    "CognitiveState",
    "EARKalmanFilter",
    "EARResult",
    "EyeState",
    "FaceDetection",
    "FaceDetector",
    "FrameResult",
    "KalmanState",
    "Landmarks68",
    "PERCLOSCalculator",
    "compute_ear",
    "compute_mar",
    "extract_landmarks_68",
]


def __getattr__(name: str) -> object:
    """Lazy import to avoid eagerly loading heavy dependencies (scipy, torch)."""
    if name in ("AdaptiveThreshold", "CalibrationState"):
        from src.cv.adaptive_threshold import AdaptiveThreshold, CalibrationState

        return AdaptiveThreshold if name == "AdaptiveThreshold" else CalibrationState
    if name in ("BlinkDetector", "BlinkEvent", "EyeState"):
        from src.cv.blink_detector import BlinkDetector, BlinkEvent, EyeState

        m = {"BlinkDetector": BlinkDetector, "BlinkEvent": BlinkEvent, "EyeState": EyeState}
        return m[name]
    if name in ("EARResult", "compute_ear"):
        from src.cv.ear import EARResult, compute_ear

        return EARResult if name == "EARResult" else compute_ear
    if name in ("FaceDetection", "FaceDetector"):
        from src.cv.face_detector import FaceDetection, FaceDetector

        return FaceDetection if name == "FaceDetection" else FaceDetector
    if name in ("EARKalmanFilter", "KalmanState"):
        from src.cv.kalman import EARKalmanFilter, KalmanState

        return EARKalmanFilter if name == "EARKalmanFilter" else KalmanState
    if name in ("Landmarks68", "extract_landmarks_68"):
        from src.cv.landmark_extractor import Landmarks68, extract_landmarks_68

        return Landmarks68 if name == "Landmarks68" else extract_landmarks_68
    if name == "compute_mar":
        from src.cv.mar import compute_mar

        return compute_mar
    if name == "PERCLOSCalculator":
        from src.cv.perclos import PERCLOSCalculator

        return PERCLOSCalculator
    if name in ("CVPipeline", "CognitiveState", "FrameResult"):
        from src.cv.pipeline import CognitiveState, CVPipeline, FrameResult

        m = {"CVPipeline": CVPipeline, "CognitiveState": CognitiveState, "FrameResult": FrameResult}
        return m[name]
    raise AttributeError(f"module 'src.cv' has no attribute {name!r}")
