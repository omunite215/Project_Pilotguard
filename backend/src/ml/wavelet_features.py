"""Wavelet decomposition for micro-expression and fatigue detection.

Applies Discrete Wavelet Transform (DWT) to landmark displacement time series
to extract frequency-band energy features:
    - Low-frequency: gross head movement, posture shifts
    - Mid-frequency: normal blinks, expressions
    - High-frequency: micro-expressions, tremors (fatigue markers)

Uses Daubechies-4 (db4) wavelet with 4-level decomposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pywt

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_wavelet_energy(
    signal: NDArray[np.float64],
    wavelet: str = "db4",
    level: int = 4,
) -> NDArray[np.float64]:
    """Compute energy in each wavelet sub-band.

    Args:
        signal: 1D time series (e.g., EAR values over time).
        wavelet: Wavelet family (default: Daubechies-4).
        level: Decomposition level (default: 4).

    Returns:
        (level + 1,) array of energies: [cA4, cD4, cD3, cD2, cD1].
    """
    # Ensure signal is long enough for decomposition
    min_length = 2**level
    if len(signal) < min_length:
        signal = np.pad(signal, (0, min_length - len(signal)), mode="edge")

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Compute energy (sum of squares) for each sub-band
    energies = np.array([np.sum(c**2) for c in coeffs], dtype=np.float64)

    # Normalize by total energy
    total = energies.sum()
    if total > 1e-10:
        energies = energies / total

    return energies


def extract_wavelet_features(
    ear_history: NDArray[np.float64],
    mar_history: NDArray[np.float64],
    wavelet: str = "db4",
    level: int = 4,
) -> NDArray[np.float64]:
    """Extract wavelet energy features from EAR and MAR time series.

    Args:
        ear_history: Recent EAR values (e.g., last 2-5 seconds).
        mar_history: Recent MAR values (same length).
        wavelet: Wavelet family.
        level: Decomposition level.

    Returns:
        (2 * (level + 1),) feature vector combining EAR and MAR wavelet energies.
    """
    ear_energy = compute_wavelet_energy(ear_history, wavelet, level)
    mar_energy = compute_wavelet_energy(mar_history, wavelet, level)

    return np.concatenate([ear_energy, mar_energy])


# Feature names for the wavelet features
WAVELET_FEATURE_NAMES: list[str] = [
    "ear_wavelet_cA4", "ear_wavelet_cD4", "ear_wavelet_cD3",
    "ear_wavelet_cD2", "ear_wavelet_cD1",
    "mar_wavelet_cA4", "mar_wavelet_cD4", "mar_wavelet_cD3",
    "mar_wavelet_cD2", "mar_wavelet_cD1",
]
