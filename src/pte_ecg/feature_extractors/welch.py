"""Welch's method power spectral density feature extractor."""

import numpy as np
import pandas as pd
import scipy.signal

from .base import BaseFeatureExtractor

# Small constant for numerical stability
EPS = 1e-10


class WelchExtractor(BaseFeatureExtractor):
    """Extract Welch's method power spectral density features from ECG data.

    This extractor uses Welch's method to estimate the power spectral density
    and calculates various frequency domain features.

    Available features (19 per channel):
        - bin_0 to bin_9: Power in 10 equal-sized dynamic frequency bins
        - log_power_ratio: Log ratio of high (>15Hz) to low (≤15Hz) power
        - band_0_0_5: Power in 0-0.5 Hz band
        - band_0_5_4: Power in 0.5-4 Hz band
        - band_4_15: Power in 4-15 Hz band
        - band_15_40: Power in 15-40 Hz band
        - band_over_40: Power above 40 Hz
        - spectral_entropy: Shannon entropy of normalized spectrum
        - total_power: Total spectral power
        - peak_frequency: Frequency with maximum power

    Args:
        selected_features: List of features to extract. If None, extract all.
        n_jobs: Not used for Welch (vectorized operations)

    Examples:
        # Extract all Welch features
        extractor = WelchExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)

        # Extract specific features only
        extractor = WelchExtractor(
            selected_features=["total_power", "peak_frequency", "spectral_entropy"]
        )
        features = extractor.get_features(ecg_data, sfreq=1000)
    """

    name = "welch"

    # Note: bin features are generated dynamically
    available_features = [
        "bin_0",
        "bin_1",
        "bin_2",
        "bin_3",
        "bin_4",
        "bin_5",
        "bin_6",
        "bin_7",
        "bin_8",
        "bin_9",
        "log_power_ratio",
        "band_0_0_5",
        "band_0_5_4",
        "band_4_15",
        "band_15_40",
        "band_over_40",
        "spectral_entropy",
        "total_power",
        "peak_frequency",
    ]

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract Welch features from ECG data.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_samples, n_features) containing Welch features.
            Column names follow pattern: welch_{feature_name}_ch{N}

        Raises:
            ValueError: If ecg does not have 3 dimensions
        """
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), "
                f"got shape {ecg.shape}"
            )

        n_samples, n_channels, n_timepoints = ecg.shape

        # Flatten to (n_samples * n_channels, n_timepoints) for Welch computation
        flat_data = ecg.reshape(-1, n_timepoints)

        # Compute Welch spectra for all channels
        psd_list = []
        for channel_data in flat_data:
            freqs, pxx = scipy.signal.welch(
                channel_data,
                fs=sfreq,
                nperseg=int(sfreq),
                scaling="density",
            )
            psd_list.append(pxx)

        freqs = freqs  # Same for all channels
        psd_array = np.array(psd_list)  # Shape: (n_samples * n_channels, len(freqs))

        # Compute all features
        feature_dict = {}

        # Dynamic bins (split frequency range into 10 equal parts)
        n_bins = 10
        bin_features_needed = any(
            self._should_extract_feature(f"bin_{i}") for i in range(n_bins)
        )

        if bin_features_needed:
            bins = np.zeros((psd_array.shape[0], n_bins))
            bin_freqs = []
            for i, bin_idx in enumerate(
                np.array_split(np.arange(psd_array.shape[1]), n_bins)
            ):
                bins[:, i] = np.mean(psd_array[:, bin_idx], axis=1)
                bin_freqs.append((freqs[bin_idx[0]], freqs[bin_idx[-1]]))

            for i in range(n_bins):
                if self._should_extract_feature(f"bin_{i}"):
                    feature_dict[f"bin_{i}"] = bins[:, i]

        # Frequency band masks
        mask_low = freqs <= 15
        mask_high = freqs > 15
        mask_0_0_5 = (freqs >= 0) & (freqs <= 0.5)
        mask_0_5_4 = (freqs > 0.5) & (freqs <= 4)
        mask_4_15 = (freqs > 4) & (freqs <= 15)
        mask_15_40 = (freqs > 15) & (freqs <= 40)
        mask_over_40 = freqs > 40

        # Log power ratio
        if self._should_extract_feature("log_power_ratio"):
            low_power = np.sum(psd_array[:, mask_low], axis=1)
            high_power = np.sum(psd_array[:, mask_high], axis=1)
            feature_dict["log_power_ratio"] = np.log(
                high_power / (low_power + EPS) + EPS
            )

        # Frequency bands
        if self._should_extract_feature("band_0_0_5"):
            feature_dict["band_0_0_5"] = np.sum(psd_array[:, mask_0_0_5], axis=1)
        if self._should_extract_feature("band_0_5_4"):
            feature_dict["band_0_5_4"] = np.sum(psd_array[:, mask_0_5_4], axis=1)
        if self._should_extract_feature("band_4_15"):
            feature_dict["band_4_15"] = np.sum(psd_array[:, mask_4_15], axis=1)
        if self._should_extract_feature("band_15_40"):
            feature_dict["band_15_40"] = np.sum(psd_array[:, mask_15_40], axis=1)
        if self._should_extract_feature("band_over_40"):
            feature_dict["band_over_40"] = np.sum(psd_array[:, mask_over_40], axis=1)

        # Other features
        if self._should_extract_feature("total_power"):
            total_power = np.sum(psd_array, axis=1)
            feature_dict["total_power"] = total_power
        else:
            # Compute for normalization if needed
            total_power = None

        if self._should_extract_feature("spectral_entropy"):
            if total_power is None:
                total_power = np.sum(psd_array, axis=1)
            psd_norm = psd_array / (total_power[:, None] + EPS)
            feature_dict["spectral_entropy"] = -np.sum(
                psd_norm * np.log2(psd_norm + EPS), axis=1
            )

        if self._should_extract_feature("peak_frequency"):
            peak_indices = np.argmax(psd_array, axis=1)
            feature_dict["peak_frequency"] = freqs[peak_indices]

        # Stack selected features in order
        feature_list = []
        feature_names_ordered = []
        for feat_name in self.available_features:
            if feat_name in feature_dict:
                feature_list.append(feature_dict[feat_name])
                feature_names_ordered.append(feat_name)

        # Stack all features: shape -> (n_samples * n_channels, n_selected_features)
        features_stacked = np.column_stack(feature_list)

        # Reshape to (samples, channels * features)
        features_reshaped = features_stacked.reshape(
            n_samples, n_channels * len(feature_names_ordered)
        )

        # Create column names
        column_names = [
            f"welch_{name}_ch{ch}"
            for ch in range(n_channels)
            for name in feature_names_ordered
        ]

        return pd.DataFrame(features_reshaped, columns=column_names)

    def _should_extract_feature(self, feature_name: str) -> bool:
        """Check if a feature should be extracted."""
        return feature_name in self.selected_features
