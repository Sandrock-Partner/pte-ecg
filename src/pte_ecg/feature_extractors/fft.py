"""FFT-based frequency domain feature extractor."""

import numpy as np
import pandas as pd
import scipy.stats

from .base import BaseFeatureExtractor

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

    Args:
        selected_features: List of features to extract. If None, extract all.
        n_jobs: Not used for FFT (vectorized operations)

    Examples:
        # Extract all FFT features
        extractor = FFTExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)

        # Extract specific features only
        extractor = FFTExtractor(
            selected_features=["dominant_frequency", "spectral_entropy"]
        )
        features = extractor.get_features(ecg_data, sfreq=1000)
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

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract FFT features from ECG data.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_samples, n_features) containing FFT features.
            Column names follow pattern: fft_{feature_name}_ch{N}

        Raises:
            ValueError: If ecg does not have 3 dimensions
        """
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), "
                f"got shape {ecg.shape}"
            )

        n_samples, n_channels, n_timepoints = ecg.shape

        # Compute FFT
        xf = np.fft.rfftfreq(n_timepoints, 1 / sfreq)  # (freqs,)
        yf = np.abs(np.fft.rfft(ecg, axis=-1))  # (samples, channels, freqs)

        # Compute all features (we'll filter later)
        feature_dict = {}

        # Basic frequency statistics
        if self._should_extract_feature("sum_freq"):
            feature_dict["sum_freq"] = np.sum(yf, axis=-1)
        if self._should_extract_feature("mean_freq"):
            feature_dict["mean_freq"] = np.mean(yf, axis=-1)
        if self._should_extract_feature("variance_freq"):
            feature_dict["variance_freq"] = np.var(yf, axis=-1)

        # Dominant frequency
        if self._should_extract_feature("dominant_frequency"):
            dominant_freq_idx = np.argmax(yf, axis=-1)
            feature_dict["dominant_frequency"] = xf[dominant_freq_idx]

        # Normalize for spectral entropy and bandwidth
        yf_norm = yf / (np.sum(yf, axis=-1, keepdims=True) + EPS)

        # Bandwidth (95% cumulative energy)
        if self._should_extract_feature("bandwidth"):
            cumsum = np.cumsum(yf_norm, axis=-1)
            bandwidth_idx = (cumsum >= 0.95).argmax(axis=-1)
            feature_dict["bandwidth"] = xf[bandwidth_idx]

        # Spectral entropy
        if self._should_extract_feature("spectral_entropy"):
            feature_dict["spectral_entropy"] = -np.sum(
                yf_norm * np.log2(yf_norm + EPS), axis=-1
            )

        # Spectral flatness
        if self._should_extract_feature("spectral_flatness"):
            gmean = scipy.stats.gmean(yf + EPS, axis=-1)
            feature_dict["spectral_flatness"] = gmean / (np.mean(yf + EPS, axis=-1))

        # Frequency band masks
        def band_mask(low: float, high: float) -> np.ndarray:
            return (xf >= low) & (xf < high)

        def apply_band(mask: np.ndarray) -> np.ndarray:
            return np.sum(yf[..., mask], axis=-1)

        # HF/LF power
        if self._should_extract_feature("hf_power"):
            hf_mask = band_mask(15, 40)
            feature_dict["hf_power"] = apply_band(hf_mask)

        if self._should_extract_feature("lf_power"):
            lf_mask = band_mask(0.5, 15)
            feature_dict["lf_power"] = apply_band(lf_mask)

        if self._should_extract_feature("hf_lf_ratio"):
            if "hf_power" not in feature_dict:
                hf_mask = band_mask(15, 40)
                hf_power = apply_band(hf_mask)
            else:
                hf_power = feature_dict["hf_power"]

            if "lf_power" not in feature_dict:
                lf_mask = band_mask(0.5, 15)
                lf_power = apply_band(lf_mask)
            else:
                lf_power = feature_dict["lf_power"]

            feature_dict["hf_lf_ratio"] = hf_power / (lf_power + EPS)

        # Energy bands
        total_energy = np.sum(yf, axis=-1) if "sum_freq" not in feature_dict else feature_dict["sum_freq"]

        if self._should_extract_feature("band_energy_0_10"):
            feature_dict["band_energy_0_10"] = apply_band(band_mask(0, 10))
        if self._should_extract_feature("band_ratio_0_10"):
            if "band_energy_0_10" not in feature_dict:
                band_energy = apply_band(band_mask(0, 10))
            else:
                band_energy = feature_dict["band_energy_0_10"]
            feature_dict["band_ratio_0_10"] = band_energy / (total_energy + EPS)

        if self._should_extract_feature("band_energy_10_20"):
            feature_dict["band_energy_10_20"] = apply_band(band_mask(10, 20))
        if self._should_extract_feature("band_ratio_10_20"):
            if "band_energy_10_20" not in feature_dict:
                band_energy = apply_band(band_mask(10, 20))
            else:
                band_energy = feature_dict["band_energy_10_20"]
            feature_dict["band_ratio_10_20"] = band_energy / (total_energy + EPS)

        if self._should_extract_feature("band_energy_20_30"):
            feature_dict["band_energy_20_30"] = apply_band(band_mask(20, 30))
        if self._should_extract_feature("band_ratio_20_30"):
            if "band_energy_20_30" not in feature_dict:
                band_energy = apply_band(band_mask(20, 30))
            else:
                band_energy = feature_dict["band_energy_20_30"]
            feature_dict["band_ratio_20_30"] = band_energy / (total_energy + EPS)

        if self._should_extract_feature("band_energy_30_40"):
            feature_dict["band_energy_30_40"] = apply_band(band_mask(30, 40))
        if self._should_extract_feature("band_ratio_30_40"):
            if "band_energy_30_40" not in feature_dict:
                band_energy = apply_band(band_mask(30, 40))
            else:
                band_energy = feature_dict["band_energy_30_40"]
            feature_dict["band_ratio_30_40"] = band_energy / (total_energy + EPS)

        # Power distribution
        if self._should_extract_feature("power_below_50Hz"):
            feature_dict["power_below_50Hz"] = apply_band(band_mask(0, 50))
        if self._should_extract_feature("power_above_50Hz"):
            feature_dict["power_above_50Hz"] = apply_band(
                band_mask(50, xf[-1] + 1)
            )
        if self._should_extract_feature("relative_power_below_50Hz"):
            if "power_below_50Hz" not in feature_dict:
                power_below = apply_band(band_mask(0, 50))
            else:
                power_below = feature_dict["power_below_50Hz"]
            feature_dict["relative_power_below_50Hz"] = power_below / (
                total_energy + EPS
            )

        # Stack selected features in order
        feature_list = []
        feature_names_ordered = []
        for feat_name in self.available_features:
            if feat_name in feature_dict:
                feature_list.append(feature_dict[feat_name])
                feature_names_ordered.append(feat_name)

        # Stack all features: shape -> (samples, channels, n_selected_features)
        features_stacked = np.stack(feature_list, axis=-1)

        # Reshape to (samples, channels × features)
        features_reshaped = features_stacked.reshape(n_samples, -1)

        # Create column names
        column_names = [
            f"fft_{name}_ch{ch}"
            for ch in range(n_channels)
            for name in feature_names_ordered
        ]

        return pd.DataFrame(features_reshaped, columns=column_names)

    def _should_extract_feature(self, feature_name: str) -> bool:
        """Override to add more flexible feature checking."""
        return feature_name in self.selected_features
