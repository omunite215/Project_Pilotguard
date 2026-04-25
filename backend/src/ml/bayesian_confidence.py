"""Bayesian confidence estimation for cognitive state predictions.

Computes posterior P(State | Observations) using:
    - Prior: HMM transition probabilities
    - Likelihood: Classifier output probabilities

Confidence is measured by the entropy of the posterior distribution:
    - Low entropy = high confidence (one state dominates)
    - High entropy = low confidence (uncertain between states)

When confidence is low, the system should avoid triggering alerts
to prevent false positives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_posterior(
    likelihood: NDArray[np.float64],
    prior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Bayesian posterior from likelihood and prior.

    Args:
        likelihood: P(Observations | State) from classifier, shape (N_states,).
        prior: P(State) from HMM transition probabilities, shape (N_states,).

    Returns:
        Posterior distribution P(State | Observations), normalized.
    """
    unnormalized = likelihood * prior
    total = unnormalized.sum()

    if total < 1e-10:
        # Uniform if both are zero
        return np.ones_like(prior) / len(prior)

    return unnormalized / total


def compute_entropy(distribution: NDArray[np.float64]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        distribution: Probability distribution (sums to 1).

    Returns:
        Entropy in nats. Higher = more uncertain.
    """
    # Filter out zeros to avoid log(0)
    nonzero = distribution[distribution > 1e-10]
    return float(-np.sum(nonzero * np.log(nonzero)))


def compute_confidence(
    posterior: NDArray[np.float64],
    n_states: int = 5,
) -> float:
    """Convert posterior entropy to a confidence score [0, 1].

    Args:
        posterior: Posterior distribution over states.
        n_states: Number of possible states (for max entropy calculation).

    Returns:
        Confidence score in [0, 1]. 1 = fully confident, 0 = maximum uncertainty.
    """
    entropy = compute_entropy(posterior)
    max_entropy = np.log(n_states)  # Entropy of uniform distribution

    if max_entropy < 1e-10:
        return 1.0

    # Confidence = 1 - normalized_entropy
    return float(max(0.0, 1.0 - entropy / max_entropy))


def should_alert(confidence: float, threshold: float = 0.6) -> bool:
    """Determine if confidence is high enough to trigger an alert.

    Args:
        confidence: Confidence score [0, 1].
        threshold: Minimum confidence for alerting.

    Returns:
        True if confident enough to alert.
    """
    return confidence >= threshold
