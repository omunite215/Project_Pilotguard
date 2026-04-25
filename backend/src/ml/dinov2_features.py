"""DINOv2 feature extraction for face analysis.

Extracts 384-dimensional CLS token embeddings from DINOv2 ViT-S/14.
These self-supervised features capture subtle visual cues beyond geometry:
facial texture, muscle tension, skin coloring changes under fatigue.

Features are computed per face crop and cached for efficiency during training.
At inference time, features are computed every Nth frame (configurable).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# DINOv2 ViT-S/14 preprocessing
DINOV2_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DINOv2FeatureExtractor:
    """Extract features from DINOv2 ViT-S/14 for face analysis.

    Args:
        device: Torch device ("cuda" or "cpu").
        model_name: DINOv2 variant name.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "dinov2_vits14",
    ) -> None:
        self.device = torch.device(device)
        logger.info("Loading DINOv2 model: %s on %s", model_name, device)

        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.feature_dim = 384  # ViT-S/14 CLS token dimension

    @torch.no_grad()
    def extract(self, face_crop_rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Extract CLS token features from a face crop.

        Args:
            face_crop_rgb: RGB face crop, any size (will be resized to 224x224).

        Returns:
            (384,) feature vector.
        """
        tensor = DINOV2_TRANSFORM(face_crop_rgb).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return features.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def extract_batch(
        self,
        face_crops: list[NDArray[np.uint8]],
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """Extract features from multiple face crops.

        Args:
            face_crops: List of RGB face crop images.
            batch_size: Processing batch size.

        Returns:
            (N, 384) feature matrix.
        """
        all_features: list[NDArray[np.float32]] = []

        for i in range(0, len(face_crops), batch_size):
            batch = face_crops[i : i + batch_size]
            tensors = torch.stack([DINOV2_TRANSFORM(img) for img in batch])
            tensors = tensors.to(self.device)

            features = self.model(tensors)
            all_features.append(features.cpu().numpy().astype(np.float32))

        return np.concatenate(all_features, axis=0) if all_features else np.empty((0, self.feature_dim), dtype=np.float32)


def crop_face_from_landmarks(
    image: NDArray[np.uint8],
    landmarks_478: NDArray[np.float32],
    padding: float = 0.3,
) -> NDArray[np.uint8]:
    """Crop face region from image using landmark bounding box.

    Args:
        image: Full frame (RGB or BGR).
        landmarks_478: (478, 3) normalized landmarks.
        padding: Fraction of face size to add as padding.

    Returns:
        Cropped face region (RGB).
    """
    h, w = image.shape[:2]
    xs = landmarks_478[:, 0] * w
    ys = landmarks_478[:, 1] * h

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_x = face_w * padding
    pad_y = face_h * padding

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    return image[y1:y2, x1:x2]


def precompute_features(
    image_paths: list[str | Path],
    output_path: Path,
    device: str = "cpu",
    batch_size: int = 32,
) -> None:
    """Pre-compute DINOv2 features for a dataset and save to disk.

    Args:
        image_paths: List of image file paths.
        output_path: Path to save .npz file with features.
        device: Torch device.
        batch_size: Processing batch size.
    """
    from src.cv.face_detector import FaceDetector

    extractor = DINOv2FeatureExtractor(device=device)
    detector = FaceDetector()

    features: list[NDArray[np.float32]] = []
    valid_indices: list[int] = []
    batch_crops: list[NDArray[np.uint8]] = []
    batch_indices: list[int] = []

    try:
        for i, path in enumerate(image_paths):
            img = cv2.imread(str(path))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detection = detector.detect(rgb)

            if detection is None:
                continue

            crop = crop_face_from_landmarks(rgb, detection.landmarks_478)
            if crop.size == 0:
                continue

            batch_crops.append(crop)
            batch_indices.append(i)

            if len(batch_crops) >= batch_size:
                batch_feats = extractor.extract_batch(batch_crops, batch_size)
                features.append(batch_feats)
                valid_indices.extend(batch_indices)
                batch_crops.clear()
                batch_indices.clear()

            if (i + 1) % 500 == 0:
                logger.info("Processed %d/%d images", i + 1, len(image_paths))

        # Process remaining
        if batch_crops:
            batch_feats = extractor.extract_batch(batch_crops, batch_size)
            features.append(batch_feats)
            valid_indices.extend(batch_indices)

    finally:
        detector.close()

    if features:
        all_features = np.concatenate(features, axis=0)
    else:
        all_features = np.empty((0, 384), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=all_features,
        valid_indices=np.array(valid_indices, dtype=np.int32),
    )
    logger.info("Saved %d features to %s", len(all_features), output_path)
