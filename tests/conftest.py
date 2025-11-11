"""Shared test fixtures for PTE-ECG tests."""

import neurokit2 as nk
import numpy as np
import pytest


@pytest.fixture
def test_data() -> tuple[np.ndarray, int]:
    """Generate synthetic ECG data for testing.

    Returns:
        Tuple of (ecg_data, sfreq) where ecg_data has shape (n_samples, n_channels, n_timepoints)
    """
    n_samples = 2
    sfreq = 100
    duration = 10

    ecg_data = np.zeros((n_samples, duration * sfreq, 12))
    for i in range(n_samples):
        ecg_data[i] = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sfreq,
            noise=0.01,
            heart_rate=70,
            method="multileads",
            random_state=i,
        )
    # Transpose to (n_samples, n_channels, n_timepoints)
    ecg_data = np.transpose(ecg_data, (0, 2, 1))
    return ecg_data, sfreq


@pytest.fixture
def single_channel_data() -> tuple[np.ndarray, int]:
    """Generate single-channel synthetic ECG data for testing.

    Returns:
        Tuple of (ecg_data, sfreq) where ecg_data has shape (n_samples, 1, n_timepoints)
    """
    n_samples = 2
    sfreq = 100
    duration = 5

    ecg_data = np.zeros((n_samples, 1, duration * sfreq))
    for i in range(n_samples):
        ecg_signal = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sfreq,
            noise=0.01,
            heart_rate=70,
            random_state=i,
        )
        ecg_data[i, 0, :] = ecg_signal
    return ecg_data, sfreq


@pytest.fixture
def short_ecg_data() -> tuple[np.ndarray, int]:
    """Generate short ECG data for testing edge cases.

    Returns:
        Tuple of (ecg_data, sfreq) where ecg_data has shape (n_samples, n_channels, n_timepoints)
    """
    n_samples = 1
    n_channels = 2
    sfreq = 100
    duration = 2  # 2 seconds - short signal

    ecg_data = np.random.randn(n_samples, n_channels, duration * sfreq) * 0.5
    return ecg_data, sfreq
