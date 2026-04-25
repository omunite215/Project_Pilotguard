"""Face detection using MediaPipe FaceLandmarker (Tasks API).

Provides real-time face detection and raw landmark extraction from video frames.
MediaPipe FaceLandmarker returns 478 3D landmarks per face.

Uses the new MediaPipe Tasks API (0.10.9+), which replaces the legacy
mp.solutions.face_mesh interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mediapipe as mp
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Default model path relative to backend/ directory
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """Result of face detection on a single frame.

    Attributes:
        landmarks_478: Raw MediaPipe 478 landmarks as (478, 3) array,
            normalized to [0, 1] relative to frame dimensions.
        confidence: Detection confidence score in [0, 1].
    """

    landmarks_478: NDArray[np.float32]
    confidence: float


class FaceDetector:
    """MediaPipe FaceLandmarker-based face detector.

    Uses the Tasks API for face landmark detection. Requires the
    face_landmarker.task model file (downloaded separately).

    Args:
        model_path: Path to face_landmarker.task model file.
        num_faces: Maximum number of faces to detect (default 1).
        min_detection_confidence: Minimum confidence for initial detection.
        min_tracking_confidence: Minimum confidence for landmark tracking.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        resolved_path = str(model_path or _DEFAULT_MODEL_PATH)

        if not Path(resolved_path).exists():
            msg = (
                f"FaceLandmarker model not found at {resolved_path}. "
                "Download it with: curl -L -o backend/models/face_landmarker.task "
                '"https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/1/face_landmarker.task"'
            )
            raise FileNotFoundError(msg)

        base_options = mp.tasks.BaseOptions(model_asset_path=resolved_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_rgb: NDArray[np.uint8]) -> FaceDetection | None:
        """Detect face and extract 478 landmarks from an RGB frame.

        Args:
            frame_rgb: Input frame in RGB format, shape (H, W, 3), dtype uint8.

        Returns:
            FaceDetection with normalized landmarks, or None if no face found.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks],
            dtype=np.float32,
        )

        # Use average visibility/presence as confidence proxy
        # NormalizedLandmark has visibility and presence fields
        presences = [lm.presence for lm in face_landmarks if lm.presence is not None and lm.presence > 0]
        confidence = float(np.mean(presences)) if presences else 1.0

        return FaceDetection(landmarks_478=landmarks, confidence=confidence)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> FaceDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
