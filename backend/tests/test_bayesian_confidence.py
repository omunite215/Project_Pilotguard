"""Tests for Bayesian confidence estimation."""

import numpy as np
import pytest

from src.ml.bayesian_confidence import (
    compute_confidence,
    compute_entropy,
    compute_posterior,
    should_alert,
)


class TestBayesianConfidence:
    """Tests for confidence estimation."""

    def test_posterior_normalization(self) -> None:
        """Posterior should sum to 1."""
        likelihood = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        prior = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        posterior = compute_posterior(likelihood, prior)
        assert posterior.sum() == pytest.approx(1.0)

    def test_posterior_respects_likelihood(self) -> None:
        """State with highest likelihood*prior should have highest posterior."""
        likelihood = np.array([0.9, 0.05, 0.03, 0.01, 0.01])
        prior = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        posterior = compute_posterior(likelihood, prior)
        assert posterior.argmax() == 0

    def test_uniform_entropy(self) -> None:
        """Uniform distribution should have maximum entropy."""
        uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        entropy = compute_entropy(uniform)
        expected = np.log(5)
        assert entropy == pytest.approx(expected, abs=1e-6)

    def test_certain_entropy_zero(self) -> None:
        """Delta distribution should have zero entropy."""
        certain = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        entropy = compute_entropy(certain)
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_confidence_range(self) -> None:
        """Confidence should be in [0, 1]."""
        uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert 0.0 <= compute_confidence(uniform) <= 1.0

        certain = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert compute_confidence(certain) == pytest.approx(1.0, abs=1e-6)

    def test_high_confidence_alerts(self) -> None:
        """High confidence should allow alerting."""
        assert should_alert(0.9, threshold=0.6)
        assert not should_alert(0.3, threshold=0.6)

    def test_zero_inputs_handled(self) -> None:
        """Zero likelihood and prior should return uniform posterior."""
        likelihood = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        prior = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        posterior = compute_posterior(likelihood, prior)
        assert posterior.sum() == pytest.approx(1.0)
