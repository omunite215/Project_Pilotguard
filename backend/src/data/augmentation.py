"""Data augmentation pipeline using Albumentations.

Implements cockpit-specific augmentations from TRD Section 4.3:
    - Spatial: flips, rotation, shift-scale
    - Photometric: brightness/contrast, noise, CLAHE, shadows, color jitter
    - Occlusion: coarse dropout (simulating partial face occlusion)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import albumentations as albu

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_training_augmentation() -> albu.Compose:
    """Create the full training-time augmentation pipeline.

    Simulates cockpit lighting conditions: instrument panel glow,
    sunlight glare, shadows, and partial occlusions.

    Returns:
        Albumentations Compose pipeline.
    """
    return albu.Compose([
        # Spatial
        albu.HorizontalFlip(p=0.5),
        albu.Rotate(limit=15, p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),

        # Photometric (cockpit conditions)
        albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        albu.GaussNoise(std_range=(10.0 / 255.0, 50.0 / 255.0), p=0.3),
        albu.CLAHE(clip_limit=4.0, p=0.3),
        albu.RandomShadow(p=0.2),
        albu.ColorJitter(hue=0.05, p=0.2),
        albu.GaussianBlur(blur_limit=(3, 3), p=0.1),

        # Occlusion (simulating partial face blockage)
        albu.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(10, 30),
            hole_width_range=(10, 30),
            p=0.2,
        ),
    ])


def get_validation_augmentation() -> albu.Compose:
    """Create validation/test augmentation (no augmentation, just normalization).

    Returns:
        Minimal Albumentations pipeline.
    """
    return albu.Compose([])


def augment_image(
    image: NDArray,
    pipeline: albu.Compose | None = None,
) -> NDArray:
    """Apply augmentation to a single image.

    Args:
        image: BGR or RGB image array (H, W, 3).
        pipeline: Albumentations pipeline. Uses training pipeline if None.

    Returns:
        Augmented image.
    """
    if pipeline is None:
        pipeline = get_training_augmentation()

    result = pipeline(image=image)
    return result["image"]
