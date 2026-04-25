"""Tests for landmark extraction (478 → 68 mapping)."""

import numpy as np

from src.cv.landmark_extractor import (
    LEFT_EYE_IDX,
    MEDIAPIPE_TO_DLIB_68,
    RIGHT_EYE_IDX,
    Landmarks68,
    extract_landmarks_68,
)


class TestExtractLandmarks68:
    """Tests for the MediaPipe → dlib landmark mapping."""

    def test_output_shape(self) -> None:
        """Output should have exactly 68 points with 2 coordinates each."""
        fake_478 = np.random.rand(478, 3).astype(np.float32)
        result = extract_landmarks_68(fake_478)

        assert result.points.shape == (68, 2)

    def test_returns_landmarks68_type(self) -> None:
        """Should return a Landmarks68 dataclass."""
        fake_478 = np.random.rand(478, 3).astype(np.float32)
        result = extract_landmarks_68(fake_478)
        assert isinstance(result, Landmarks68)

    def test_eye_landmarks_shape(self) -> None:
        """Eye arrays should have 6 points each."""
        fake_478 = np.random.rand(478, 3).astype(np.float32)
        result = extract_landmarks_68(fake_478)

        assert result.right_eye.shape == (6, 2)
        assert result.left_eye.shape == (6, 2)

    def test_mouth_landmarks_shape(self) -> None:
        """Mouth array should have 20 points."""
        fake_478 = np.random.rand(478, 3).astype(np.float32)
        result = extract_landmarks_68(fake_478)

        assert result.mouth.shape == (20, 2)

    def test_z_coordinate_discarded(self) -> None:
        """Only x, y should be kept; z should be discarded."""
        fake_478 = np.zeros((478, 3), dtype=np.float32)
        fake_478[:, 2] = 99.0  # Set z to a distinctive value

        result = extract_landmarks_68(fake_478)

        # Points should only have x, y columns (both 0)
        assert result.points.shape[1] == 2

    def test_mapping_indices_valid(self) -> None:
        """All mapping indices should be within [0, 477]."""
        for idx in MEDIAPIPE_TO_DLIB_68:
            assert 0 <= idx <= 477

    def test_mapping_has_68_entries(self) -> None:
        """Mapping should have exactly 68 entries."""
        assert len(MEDIAPIPE_TO_DLIB_68) == 68

    def test_eye_indices_correct(self) -> None:
        """Eye indices should point to correct positions in the 68 layout."""
        assert RIGHT_EYE_IDX == [36, 37, 38, 39, 40, 41]
        assert LEFT_EYE_IDX == [42, 43, 44, 45, 46, 47]

    def test_values_preserved_from_source(self) -> None:
        """Extracted values should exactly match the source at mapped indices."""
        fake_478 = np.random.rand(478, 3).astype(np.float32)
        result = extract_landmarks_68(fake_478)

        for i, mp_idx in enumerate(MEDIAPIPE_TO_DLIB_68):
            np.testing.assert_array_almost_equal(
                result.points[i],
                fake_478[mp_idx, :2],
            )

    def test_normalized_range(self) -> None:
        """If input is in [0, 1], output should also be in [0, 1]."""
        fake_478 = np.random.rand(478, 3).astype(np.float32)
        result = extract_landmarks_68(fake_478)

        assert result.points.min() >= 0.0
        assert result.points.max() <= 1.0
