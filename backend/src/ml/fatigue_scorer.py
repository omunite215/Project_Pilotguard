"""Composite fatigue score computation.

Combines multiple fatigue indicators into a single 0-100 score:
    FatigueScore = w1*PERCLOS + w2*BlinkRate + w3*EAR_deviation
                 + w4*MAR_score + w5*MicroExpr_energy

Weights can be:
    1. Hand-tuned initial values (from config)
    2. Learned via logistic regression on labeled data

Alert thresholds:
    Advisory:  > 30
    Caution:   > 55
    Warning:   > 75
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FatigueScoreResult:
    """Result of fatigue score computation.

    Attributes:
        score: Composite fatigue score [0, 100].
        components: Individual normalized component values.
        alert_level: "normal", "advisory", "caution", or "warning".
    """

    score: float
    components: dict[str, float]
    alert_level: str


class FatigueScorer:
    """Compute composite fatigue score from multiple indicators.

    Args:
        weights: Dict of component name to weight. Must sum to ~1.0.
        advisory_threshold: Score above which advisory alert fires.
        caution_threshold: Score above which caution alert fires.
        warning_threshold: Score above which warning alert fires.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        advisory_threshold: float = 30.0,
        caution_threshold: float = 55.0,
        warning_threshold: float = 75.0,
    ) -> None:
        self.weights = weights or {
            "perclos": 0.30,
            "blink_rate": 0.20,
            "ear_deviation": 0.20,
            "mar": 0.15,
            "micro_expression": 0.15,
        }
        self.advisory_threshold = advisory_threshold
        self.caution_threshold = caution_threshold
        self.warning_threshold = warning_threshold

    def compute(
        self,
        perclos: float,
        blink_rate: float,
        ear_deviation: float,
        mar: float,
        micro_expression_energy: float = 0.0,
        baseline_blink_rate: float = 17.0,
    ) -> FatigueScoreResult:
        """Compute the composite fatigue score.

        Args:
            perclos: Current PERCLOS percentage [0, 100].
            blink_rate: Current blinks per minute.
            ear_deviation: How much current EAR deviates from baseline [0, 1].
            mar: Current MAR value [0, 1].
            micro_expression_energy: High-frequency wavelet energy [0, 1].
            baseline_blink_rate: Normal blink rate for deviation calculation.

        Returns:
            FatigueScoreResult with score, components, and alert level.
        """
        # Normalize each component to [0, 1]
        perclos_norm = min(1.0, perclos / 80.0)  # 80% PERCLOS = max fatigue
        blink_norm = min(1.0, abs(blink_rate - baseline_blink_rate) / 20.0)
        ear_dev_norm = min(1.0, ear_deviation / 0.15)  # 0.15 EAR drop = high fatigue
        mar_norm = min(1.0, max(0.0, (mar - 0.3)) / 0.5)  # yawn threshold at 0.3
        micro_norm = min(1.0, micro_expression_energy)

        components = {
            "perclos": perclos_norm,
            "blink_rate": blink_norm,
            "ear_deviation": ear_dev_norm,
            "mar": mar_norm,
            "micro_expression": micro_norm,
        }

        # Weighted sum -> scale to 0-100
        raw_score = sum(
            self.weights.get(name, 0.0) * value
            for name, value in components.items()
        )
        score = min(100.0, max(0.0, raw_score * 100.0))

        # Determine alert level
        if score >= self.warning_threshold:
            level = "warning"
        elif score >= self.caution_threshold:
            level = "caution"
        elif score >= self.advisory_threshold:
            level = "advisory"
        else:
            level = "normal"

        return FatigueScoreResult(score=score, components=components, alert_level=level)

    def save_weights(self, path: Path) -> None:
        """Save weights to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.weights, f, indent=2)

    @classmethod
    def load_weights(cls, path: Path, **kwargs: float) -> FatigueScorer:
        """Load weights from JSON."""
        with open(path) as f:
            weights = json.load(f)
        return cls(weights=weights, **kwargs)
