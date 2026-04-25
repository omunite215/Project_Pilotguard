"""Tests for wavelet feature extraction."""

import numpy as np
import pytest

from src.ml.wavelet_features import (
    WAVELET_FEATURE_NAMES,
    compute_wavelet_energy,
    extract_wavelet_features,
)


class TestWaveletFeatures:
    """Tests for wavelet decomposition."""

    def test_energy_sums_to_one(self) -> None:
        """Normalized energies should sum to 1."""
        signal = np.random.randn(128)
        energy = compute_wavelet_energy(signal)
        assert energy.sum() == pytest.approx(1.0, abs=1e-6)

    def test_energy_shape(self) -> None:
        """Energy should have (level + 1) components."""
        signal = np.random.randn(128)
        energy = compute_wavelet_energy(signal, level=4)
        assert energy.shape == (5,)  # cA4, cD4, cD3, cD2, cD1

    def test_short_signal_padded(self) -> None:
        """Short signals should be padded and still work."""
        signal = np.array([0.3, 0.3, 0.1, 0.3])
        energy = compute_wavelet_energy(signal, level=4)
        assert energy.shape == (5,)
        assert np.all(np.isfinite(energy))

    def test_extract_combined_features(self) -> None:
        """Combined EAR+MAR features should have 10 components."""
        ear = np.random.randn(64)
        mar = np.random.randn(64)
        features = extract_wavelet_features(ear, mar)
        assert features.shape == (10,)

    def test_feature_names_count(self) -> None:
        """Should have 10 feature names."""
        assert len(WAVELET_FEATURE_NAMES) == 10

    def test_constant_signal_energy(self) -> None:
        """Constant signal should have all energy in approximation coefficients."""
        signal = np.ones(128) * 0.3
        energy = compute_wavelet_energy(signal, level=4)
        # Most energy should be in cA4 (index 0) for a constant signal
        assert energy[0] > 0.9
