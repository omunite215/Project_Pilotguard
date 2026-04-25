"""Tests for Eye Aspect Ratio (EAR) calculation."""

import numpy as np
import pytest

from src.cv.ear import EARResult, compute_ear


def _make_eye(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
    p5: tuple[float, float],
    p6: tuple[float, float],
) -> np.ndarray:
    """Create a (6, 2) eye landmark array from 6 points."""
    return np.array([p1, p2, p3, p4, p5, p6], dtype=np.float32)


class TestComputeEAR:
    """Tests for the EAR formula."""

    def test_wide_open_eye(self) -> None:
        """A wide-open eye should have EAR ~0.3-0.4."""
        eye = _make_eye(
            p1=(0.0, 0.5),  # outer corner
            p2=(0.2, 0.3),  # upper-outer
            p3=(0.4, 0.3),  # upper-inner
            p4=(0.6, 0.5),  # inner corner
            p5=(0.4, 0.7),  # lower-inner
            p6=(0.2, 0.7),  # lower-outer
        )
        result = compute_ear(eye, eye)

        assert result.left > 0.3
        assert result.right > 0.3
        assert result.average > 0.3

    def test_closed_eye(self) -> None:
        """A closed eye should have EAR close to 0."""
        eye = _make_eye(
            p1=(0.0, 0.5),
            p2=(0.2, 0.49),  # nearly same as lower
            p3=(0.4, 0.49),
            p4=(0.6, 0.5),
            p5=(0.4, 0.51),
            p6=(0.2, 0.51),
        )
        result = compute_ear(eye, eye)

        assert result.average < 0.1

    def test_symmetric_eyes_equal(self) -> None:
        """Left and right EAR should be equal when given identical landmarks."""
        eye = _make_eye(
            (0.0, 0.5), (0.2, 0.3), (0.4, 0.3),
            (0.6, 0.5), (0.4, 0.7), (0.2, 0.7),
        )
        result = compute_ear(eye, eye)

        assert result.left == pytest.approx(result.right)
        assert result.average == pytest.approx(result.left)

    def test_known_ear_value(self) -> None:
        """Test with known geometric values.

        With vertical distances = 0.4 each, horizontal = 0.6:
        EAR = (0.4 + 0.4) / (2 * 0.6) = 0.6667
        """
        eye = _make_eye(
            p1=(0.0, 0.5),
            p2=(0.2, 0.3),  # v1 with p6: sqrt((0)^2 + (0.4)^2) = 0.4
            p3=(0.4, 0.3),  # v2 with p5: sqrt((0)^2 + (0.4)^2) = 0.4
            p4=(0.6, 0.5),  # h: sqrt((0.6)^2 + 0) = 0.6
            p5=(0.4, 0.7),
            p6=(0.2, 0.7),
        )
        result = compute_ear(eye, eye)

        expected = (0.4 + 0.4) / (2.0 * 0.6)
        assert result.average == pytest.approx(expected, abs=1e-4)

    def test_degenerate_horizontal_zero(self) -> None:
        """If horizontal distance is zero, EAR should be 0."""
        eye = _make_eye(
            p1=(0.3, 0.5),
            p2=(0.3, 0.3),
            p3=(0.3, 0.3),
            p4=(0.3, 0.5),  # same x as p1
            p5=(0.3, 0.7),
            p6=(0.3, 0.7),
        )
        result = compute_ear(eye, eye)
        assert result.average == 0.0

    def test_returns_ear_result_type(self) -> None:
        """Result should be an EARResult dataclass."""
        eye = _make_eye(
            (0.0, 0.5), (0.2, 0.3), (0.4, 0.3),
            (0.6, 0.5), (0.4, 0.7), (0.2, 0.7),
        )
        result = compute_ear(eye, eye)
        assert isinstance(result, EARResult)

    def test_asymmetric_eyes(self) -> None:
        """Different left and right eyes should produce different EARs."""
        right_eye = _make_eye(
            (0.0, 0.5), (0.2, 0.3), (0.4, 0.3),
            (0.6, 0.5), (0.4, 0.7), (0.2, 0.7),
        )
        left_eye = _make_eye(
            (0.0, 0.5), (0.2, 0.45), (0.4, 0.45),
            (0.6, 0.5), (0.4, 0.55), (0.2, 0.55),
        )
        result = compute_ear(right_eye, left_eye)

        assert result.right > result.left
        assert result.average == pytest.approx((result.left + result.right) / 2.0)
