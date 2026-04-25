"""Tests for Mouth Aspect Ratio (MAR) calculation."""

import numpy as np
import pytest

from src.cv.mar import compute_mar


def _make_inner_mouth(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
    p5: tuple[float, float],
    p6: tuple[float, float],
    p7: tuple[float, float],
    p8: tuple[float, float],
) -> np.ndarray:
    """Create (8, 2) inner mouth landmark array."""
    return np.array([p1, p2, p3, p4, p5, p6, p7, p8], dtype=np.float32)


class TestComputeMAR:
    """Tests for the MAR formula."""

    def test_closed_mouth(self) -> None:
        """Closed mouth should have MAR near 0."""
        mouth = _make_inner_mouth(
            p1=(0.3, 0.5),
            p2=(0.35, 0.499),
            p3=(0.4, 0.499),
            p4=(0.45, 0.499),
            p5=(0.5, 0.5),
            p6=(0.45, 0.501),
            p7=(0.4, 0.501),
            p8=(0.35, 0.501),
        )
        mar = compute_mar(mouth)
        assert mar < 0.05

    def test_open_mouth_yawn(self) -> None:
        """Wide-open mouth (yawn) should have high MAR."""
        mouth = _make_inner_mouth(
            p1=(0.3, 0.5),
            p2=(0.35, 0.3),
            p3=(0.4, 0.25),
            p4=(0.45, 0.3),
            p5=(0.5, 0.5),
            p6=(0.45, 0.7),
            p7=(0.4, 0.75),
            p8=(0.35, 0.7),
        )
        mar = compute_mar(mouth)
        assert mar > 0.5

    def test_known_value(self) -> None:
        """Test with calculable geometry.

        Horizontal = 0.2
        Vertical distances: v1=0.4, v2=0.5, v3=0.4
        MAR = (0.4 + 0.5 + 0.4) / (2 * 0.2) = 1.3 / 0.4 = 3.25
        """
        mouth = _make_inner_mouth(
            p1=(0.3, 0.5),
            p2=(0.35, 0.3),   # v1 with p8: |0.3-0.7| = 0.4
            p3=(0.4, 0.25),   # v2 with p7: |0.25-0.75| = 0.5
            p4=(0.45, 0.3),   # v3 with p6: |0.3-0.7| = 0.4
            p5=(0.5, 0.5),    # h: |0.3-0.5| = 0.2
            p6=(0.45, 0.7),
            p7=(0.4, 0.75),
            p8=(0.35, 0.7),
        )
        mar = compute_mar(mouth)
        expected = (0.4 + 0.5 + 0.4) / (2.0 * 0.2)
        assert mar == pytest.approx(expected, abs=1e-4)

    def test_degenerate_horizontal_zero(self) -> None:
        """If horizontal distance is zero, MAR should be 0."""
        mouth = _make_inner_mouth(
            p1=(0.4, 0.5),
            p2=(0.4, 0.3),
            p3=(0.4, 0.25),
            p4=(0.4, 0.3),
            p5=(0.4, 0.5),  # same as p1
            p6=(0.4, 0.7),
            p7=(0.4, 0.75),
            p8=(0.4, 0.7),
        )
        mar = compute_mar(mouth)
        assert mar == 0.0

    def test_returns_float(self) -> None:
        """MAR should be a Python float."""
        mouth = _make_inner_mouth(
            (0.3, 0.5), (0.35, 0.4), (0.4, 0.4), (0.45, 0.4),
            (0.5, 0.5), (0.45, 0.6), (0.4, 0.6), (0.35, 0.6),
        )
        assert isinstance(compute_mar(mouth), float)
