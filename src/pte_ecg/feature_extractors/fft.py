"""FFT-based frequency domain feature extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats

from .base import BaseFeatureExtractor

if TYPE_CHECKING:
    from ..core import FeatureExtractor

# Small constant for numerical stability
EPS = 1e-10


class FFTExtractor(BaseFeatureExtractor):
    """Extract FFT-based frequency domain features from ECG data.

    This extractor calculates various FFT-based features including spectral
    characteristics, frequency band powers, and energy distributions.

    Available features (21 per channel):
        - sum_freq: Sum of FFT magnitudes
        - mean_freq: Mean of FFT magnitudes
        - variance_freq: Variance of FFT magnitudes
        - dominant_frequency: Frequency with maximum power
        - bandwidth: 95% cumulative energy bandwidth
        - spectral_entropy: Shannon entropy of normalized spectrum
        - spectral_flatness: Geometric mean / arithmetic mean ratio
        - hf_power: High frequency power (15-40 Hz)
        - lf_power: Low frequency power (0.5-15 Hz)
        - hf_lf_ratio: HF/LF power ratio
        - band_energy_0_10: Energy in 0-10 Hz band
        - band_ratio_0_10: Ratio of 0-10 Hz energy to total
        - band_energy_10_20: Energy in 10-20 Hz band
        - band_ratio_10_20: Ratio of 10-20 Hz energy to total
        - band_energy_20_30: Energy in 20-30 Hz band
        - band_ratio_20_30: Ratio of 20-30 Hz energy to total
        - band_energy_30_40: Energy in 30-40 Hz band
        - band_ratio_30_40: Ratio of 30-40 Hz energy to total
        - power_below_50Hz: Total power below 50 Hz
        - power_above_50Hz: Total power above 50 Hz
        - relative_power_below_50Hz: Ratio of power below 50 Hz to total

    Examples:
        # Extract all FFT features (via FeatureExtractor)
        extractor = FeatureExtractor(sfreq=1000)
        features = extractor.extract_features(ecg_data)
    """

    name = "fft"
    available_features = [
        "sum_freq",
        "mean_freq",
        "variance_freq",
        "dominant_frequency",
        "bandwidth",
        "spectral_entropy",
        "spectral_flatness",
        "hf_power",
        "lf_power",
        "hf_lf_ratio",
        "band_energy_0_10",
        "band_ratio_0_10",
        "band_energy_10_20",
        "band_ratio_10_20",
        "band_energy_20_30",
        "band_ratio_20_30",
        "band_energy_30_40",
        "band_ratio_30_40",
        "power_below_50Hz",
        "power_above_50Hz",
        "relative_power_below_50Hz",
    ]

    def __init__(self, parent: FeatureExtractor):
        """Initialize the FFT extractor.

        Args:
            parent: Parent FeatureExtractor instance for accessing sfreq, lead_order, etc.
            **kwargs: Additional config parameters (features, n_jobs, etc.)
        """
        self.parent = parent

    def get_features(
        self,
        ecg: np.ndarray,
    ) -> pd.DataFrame:
        """Extract FFT features from ECG data.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)

        Returns:
            DataFrame with shape (n_samples, n_features) containing FFT features.
            Column names follow pattern: fft_{feature_name}_{lead_name}

        Raises:
            ValueError: If ecg does not have 3 dimensions
        """
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), got shape {ecg.shape}"
            )

        n_samples, n_channels, n_timepoints = ecg.shape

        # Compute FFT
        xf = np.fft.rfftfreq(n_timepoints, 1 / self.sfreq)  # (freqs,)
        yf = np.abs(np.fft.rfft(ecg, axis=-1))  # (samples, channels, freqs)

        # Compute all features
        feature_dict = {}

        # Basic frequency statistics
        feature_dict["sum_freq"] = np.sum(yf, axis=-1)
        feature_dict["mean_freq"] = np.mean(yf, axis=-1)
        feature_dict["variance_freq"] = np.var(yf, axis=-1)

        # Dominant frequency
        dominant_freq_idx = np.argmax(yf, axis=-1)
        feature_dict["dominant_frequency"] = xf[dominant_freq_idx]

        # Normalize for spectral entropy and bandwidth
        yf_norm = yf / (np.sum(yf, axis=-1, keepdims=True) + EPS)

        # Bandwidth (95% cumulative energy)
        cumsum = np.cumsum(yf_norm, axis=-1)
        bandwidth_idx = (cumsum >= 0.95).argmax(axis=-1)
        feature_dict["bandwidth"] = xf[bandwidth_idx]

        # Spectral entropy
        feature_dict["spectral_entropy"] = -np.sum(yf_norm * np.log2(yf_norm + EPS), axis=-1)

        # Spectral flatness
        gmean = scipy.stats.gmean(yf + EPS, axis=-1)
        feature_dict["spectral_flatness"] = gmean / (np.mean(yf + EPS, axis=-1))

        # Frequency band masks
        def band_mask(low: float, high: float) -> np.ndarray:
            return (xf >= low) & (xf < high)

        def apply_band(mask: np.ndarray) -> np.ndarray:
            return np.sum(yf[..., mask], axis=-1)

        # HF/LF power
        hf_mask = band_mask(15, 40)
        lf_mask = band_mask(0.5, 15)
        feature_dict["hf_power"] = apply_band(hf_mask)
        feature_dict["lf_power"] = apply_band(lf_mask)
        feature_dict["hf_lf_ratio"] = feature_dict["hf_power"] / (feature_dict["lf_power"] + EPS)

        # Energy bands
        total_energy = feature_dict["sum_freq"]

        feature_dict["band_energy_0_10"] = apply_band(band_mask(0, 10))
        feature_dict["band_ratio_0_10"] = feature_dict["band_energy_0_10"] / (total_energy + EPS)

        feature_dict["band_energy_10_20"] = apply_band(band_mask(10, 20))
        feature_dict["band_ratio_10_20"] = feature_dict["band_energy_10_20"] / (total_energy + EPS)

        feature_dict["band_energy_20_30"] = apply_band(band_mask(20, 30))
        feature_dict["band_ratio_20_30"] = feature_dict["band_energy_20_30"] / (total_energy + EPS)

        feature_dict["band_energy_30_40"] = apply_band(band_mask(30, 40))
        feature_dict["band_ratio_30_40"] = feature_dict["band_energy_30_40"] / (total_energy + EPS)

        # Power distribution
        feature_dict["power_below_50Hz"] = apply_band(band_mask(0, 50))
        feature_dict["power_above_50Hz"] = apply_band(band_mask(50, xf[-1] + 1))
        feature_dict["relative_power_below_50Hz"] = feature_dict["power_below_50Hz"] / (total_energy + EPS)

        # Stack features in order
        feature_list = [feature_dict[feat_name] for feat_name in self.available_features]

        # Stack all features: shape -> (samples, channels, n_features)
        features_stacked = np.stack(feature_list, axis=-1)

        # Reshape to (samples, channels × features)
        features_reshaped = features_stacked.reshape(n_samples, -1)

        # Create column names using lead names
        column_names = [
            f"fft_{name}_{self.lead_order[ch]}" for ch in range(n_channels) for name in self.available_features
        ]

        return pd.DataFrame(features_reshaped, columns=column_names)
