"""Hidden Markov Model for cognitive state estimation.

Models temporal transitions between cognitive states:
    States: Alert -> Mild Fatigue -> Fatigued -> Microsleep -> Incapacitated

Uses discretized observations from the CV pipeline:
    [EAR_bin, blink_rate_bin, PERCLOS_bin, MAR_bin]

Training: Baum-Welch (EM) on labeled sequences.
Decoding: Viterbi algorithm for most likely state sequence.
"""

from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING

import numpy as np
from hmmlearn.hmm import CategoricalHMM

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# State definitions
STATES = ["alert", "mild_fatigue", "fatigued", "microsleep", "incapacitated"]
N_STATES = len(STATES)


def discretize_observations(
    ear: float,
    blink_rate: float,
    perclos: float,
    mar: float,
    n_bins: int = 4,
) -> int:
    """Discretize continuous observations into a single categorical value.

    Each feature is binned into n_bins levels, then combined into a single
    integer observation via mixed-radix encoding.

    Args:
        ear: Smoothed EAR value.
        blink_rate: Blinks per minute.
        perclos: PERCLOS percentage.
        mar: Mouth Aspect Ratio.
        n_bins: Number of bins per feature.

    Returns:
        Integer observation index in [0, n_bins^4 - 1].
    """
    def _bin(value: float, low: float, high: float) -> int:
        normalized = max(0.0, min(1.0, (value - low) / (high - low)))
        return min(int(normalized * n_bins), n_bins - 1)

    ear_bin = _bin(ear, 0.10, 0.40)
    blink_bin = _bin(blink_rate, 5.0, 40.0)
    perclos_bin = _bin(perclos, 0.0, 80.0)
    mar_bin = _bin(mar, 0.0, 0.8)

    # Mixed-radix encoding
    return ear_bin * n_bins**3 + blink_bin * n_bins**2 + perclos_bin * n_bins + mar_bin


class CognitiveStateHMM:
    """HMM for temporal cognitive state estimation.

    Args:
        n_obs_bins: Number of bins per observation dimension.
        n_iter: Maximum Baum-Welch iterations.
        seed: Random seed.
    """

    def __init__(
        self,
        n_obs_bins: int = 4,
        n_iter: int = 100,
        seed: int = 42,
    ) -> None:
        self.n_obs_bins = n_obs_bins
        n_features = n_obs_bins**4  # 4 observation dimensions

        self.model = CategoricalHMM(
            n_components=N_STATES,
            n_features=n_features,
            n_iter=n_iter,
            random_state=seed,
            init_params="ste",
        )

        # Set physically meaningful initial transition matrix
        # States tend to progress: alert -> mild -> fatigued -> microsleep
        # But can also recover: fatigued -> mild -> alert
        self.model.startprob_ = np.array([0.7, 0.15, 0.10, 0.04, 0.01])
        self._is_fitted = False

    def fit(self, sequences: list[NDArray[np.int32]], lengths: list[int] | None = None) -> None:
        """Train HMM on observation sequences.

        Args:
            sequences: List of observation sequences (each is 1D int array).
            lengths: Length of each sequence. If None, each array is one sequence.
        """
        if lengths is None:
            all_obs = np.concatenate(sequences).reshape(-1, 1)
            lengths = [len(s) for s in sequences]
        else:
            all_obs = np.concatenate(sequences).reshape(-1, 1)

        self.model.fit(all_obs, lengths)
        self._is_fitted = True
        logger.info("HMM fitted on %d sequences (%d total observations)",
                     len(lengths), len(all_obs))

    def decode(self, observations: NDArray[np.int32]) -> tuple[list[str], NDArray[np.float64]]:
        """Decode the most likely state sequence (Viterbi).

        Args:
            observations: 1D array of discretized observation indices.

        Returns:
            Tuple of (state_names, log_probability_per_step).
        """
        obs = observations.reshape(-1, 1)
        _log_prob, state_seq = self.model.decode(obs, algorithm="viterbi")
        state_names = [STATES[s] for s in state_seq]

        # Get posterior probabilities for confidence estimation
        posteriors = self.model.predict_proba(obs)

        return state_names, posteriors

    def predict_state(self, observation: int) -> tuple[str, NDArray[np.float64]]:
        """Predict state from a single observation.

        Args:
            observation: Single discretized observation index.

        Returns:
            Tuple of (most_likely_state, posterior_distribution).
        """
        obs = np.array([[observation]])
        posteriors = self.model.predict_proba(obs)
        state_idx = posteriors[0].argmax()
        return STATES[state_idx], posteriors[0]

    def save(self, path: Path) -> None:
        """Save fitted HMM to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "n_obs_bins": self.n_obs_bins}, f)
        logger.info("HMM saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> CognitiveStateHMM:
        """Load a fitted HMM from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        hmm = cls(n_obs_bins=data["n_obs_bins"])
        hmm.model = data["model"]
        hmm._is_fitted = True
        return hmm
