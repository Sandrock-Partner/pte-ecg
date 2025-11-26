"""Unit tests for the ECG feature extraction pipeline."""

import numpy as np
import pandas as pd
import pytest

import pte_ecg


def test_get_features_basic(test_data: tuple[np.ndarray, int]):
    """Test basic functionality of get_features with default settings."""

    ecg_data, sfreq = test_data

    settings = pte_ecg.Settings()

    # Extract features with default settings (morphological enabled by default)
    features = pte_ecg.get_features(ecg_data, sfreq=sfreq, settings=settings)

    # Basic assertions
    assert isinstance(features, pd.DataFrame)
    assert len(features.columns) > 0  # At least one feature extracted

    # Check for common feature prefixes (morphological is enabled by default)
    feature_columns = set(features.columns)
    assert any(col.startswith("morphological_") for col in feature_columns)


def test_get_features_custom_settings(test_data: tuple[np.ndarray, int]):
    """Test get_features with custom settings."""
    ecg_data, sfreq = test_data

    # Create custom settings with only FFT features enabled
    settings = pte_ecg.Settings()
    settings.preprocessing.resample.enabled = True
    settings.preprocessing.resample.sfreq_new = sfreq / 2

    settings.preprocessing.bandpass.enabled = True
    settings.preprocessing.bandpass.l_freq = 0.5
    settings.preprocessing.bandpass.h_freq = sfreq / 5

    settings.preprocessing.notch.enabled = True
    settings.preprocessing.notch.freq = sfreq / 6

    settings.preprocessing.normalize.enabled = True
    settings.preprocessing.normalize.mode = "zscore"

    # Enable only FFT, disable all others
    settings.features.fft = {"enabled": True}
    settings.features.morphological = {"enabled": False}
    settings.features.nonlinear = {"enabled": False}
    settings.features.welch = {"enabled": False}
    settings.features.statistical = {"enabled": False}

    # Extract features
    features = pte_ecg.get_features(ecg=ecg_data, sfreq=sfreq, settings=settings)

    # Check that only FFT features were extracted
    assert all(col.startswith("fft_") for col in features.columns)


def test_get_features_invalid_input():
    """Test get_features with invalid input."""
    # Test with wrong dimensions
    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(100), sfreq=360)  # 1D input

    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(10, 10), sfreq=360)  # 2D input

    # Test with invalid sfreq
    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(5, 1, 1000), sfreq=0)  # Zero sfreq

    # Test with invalid settings type
    with pytest.raises(TypeError):
        pte_ecg.get_features(np.random.randn(5, 1, 1000), sfreq=360, settings=1)  # ty: ignore[invalid-argument-type]

    # Test with invalid settings value
    with pytest.raises(ValueError):
        pte_ecg.get_features(np.random.randn(5, 1, 1000), sfreq=360, settings="random")


def test_get_features_empty_features(test_data: tuple[np.ndarray, int]):
    """Test get_features with no features enabled."""
    ecg_data, sfreq = test_data

    # Should raise ValueError when no features are enabled
    with pytest.raises(ValueError, match="At least one feature extractor must be enabled"):
        pte_ecg.Settings(
            features=pte_ecg.FeaturesConfig(
                fft={"enabled": False},
                morphological={"enabled": False},
                nonlinear={"enabled": False},
                statistical={"enabled": False},
                welch={"enabled": False},
                waveshape={"enabled": False},
            )
        )


if __name__ == "__main__":
    pytest.main([__file__])
