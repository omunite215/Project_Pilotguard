"""Tests for composite fatigue scorer."""

import pytest

from src.ml.fatigue_scorer import FatigueScorer, FatigueScoreResult


class TestFatigueScorer:
    """Tests for the fatigue scoring system."""

    def test_alert_state_low_score(self) -> None:
        """Normal values should produce a low fatigue score."""
        scorer = FatigueScorer()
        result = scorer.compute(
            perclos=5.0,
            blink_rate=17.0,
            ear_deviation=0.01,
            mar=0.1,
        )
        assert result.score < 30
        assert result.alert_level == "normal"

    def test_drowsy_state_high_score(self) -> None:
        """Fatigued values should produce a high score."""
        scorer = FatigueScorer()
        result = scorer.compute(
            perclos=60.0,
            blink_rate=35.0,
            ear_deviation=0.12,
            mar=0.7,
            micro_expression_energy=0.8,
        )
        assert result.score > 55
        assert result.alert_level in ("caution", "warning")

    def test_score_range(self) -> None:
        """Score should always be in [0, 100]."""
        scorer = FatigueScorer()

        # Minimum
        result = scorer.compute(0.0, 17.0, 0.0, 0.0)
        assert 0.0 <= result.score <= 100.0

        # Maximum
        result = scorer.compute(100.0, 50.0, 0.3, 1.0, 1.0)
        assert 0.0 <= result.score <= 100.0

    def test_returns_correct_type(self) -> None:
        """Should return FatigueScoreResult."""
        scorer = FatigueScorer()
        result = scorer.compute(10.0, 15.0, 0.02, 0.1)
        assert isinstance(result, FatigueScoreResult)
        assert isinstance(result.components, dict)
        assert len(result.components) == 5

    def test_alert_levels(self) -> None:
        """Alert levels should correspond to thresholds."""
        scorer = FatigueScorer(
            advisory_threshold=30.0,
            caution_threshold=55.0,
            warning_threshold=75.0,
        )

        # Low
        result = scorer.compute(5.0, 17.0, 0.01, 0.1)
        assert result.alert_level == "normal"

    def test_custom_weights(self) -> None:
        """Custom weights should affect the score."""
        # All weight on PERCLOS
        scorer = FatigueScorer(weights={
            "perclos": 1.0, "blink_rate": 0.0,
            "ear_deviation": 0.0, "mar": 0.0, "micro_expression": 0.0,
        })
        result = scorer.compute(perclos=80.0, blink_rate=17.0, ear_deviation=0.0, mar=0.0)
        assert result.score == pytest.approx(100.0, abs=1.0)
