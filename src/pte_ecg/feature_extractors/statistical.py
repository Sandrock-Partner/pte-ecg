"""Statistical feature extractor."""

import numpy as np
import pandas as pd
import scipy.stats

from .base import BaseFeatureExtractor


class StatisticalExtractor(BaseFeatureExtractor):
    """Extract statistical features from ECG data.

    This extractor calculates various statistical features that summarize
    the distribution and properties of the ECG signal.

    Available features (13 per channel):
        - sum: Sum of all values
        - mean: Arithmetic mean
        - median: Median value
        - mode: Most frequent value
        - var: Variance
        - range: Range (max - min)
        - min: Minimum value
        - max: Maximum value
        - iqr: Interquartile range (Q3 - Q1)
        - skew: Skewness (asymmetry of distribution)
        - kurt: Kurtosis (tailedness of distribution)
        - peak_to_peak: Peak-to-peak amplitude
        - autocorr: Lag-1 autocorrelation

    Args:
        selected_features: List of features to extract. If None, extract all.
        n_jobs: Number of parallel jobs (not used - vectorized operations)

    Examples:
        # Extract all statistical features
        extractor = StatisticalExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)

        # Extract specific features only
        extractor = StatisticalExtractor(
            selected_features=["mean", "var", "skew"]
        )
        features = extractor.get_features(ecg_data, sfreq=1000)
    """

    name = "statistical"
    available_features = [
        "sum",
        "mean",
        "median",
        "mode",
        "var",
        "range",
        "min",
        "max",
        "iqr",
        "skew",
        "kurt",
        "peak_to_peak",
        "autocorr",
    ]

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract statistical features from ECG data.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz (not used for statistical features)

        Returns:
            DataFrame with shape (n_samples, n_features) containing statistical features.
            Column names follow pattern: statistical_{feature_name}_{lead_name}

        Raises:
            ValueError: If ecg does not have 3 dimensions
        """
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), "
                f"got shape {ecg.shape}"
            )

        n_samples, n_channels, n_timepoints = ecg.shape

        # Compute all features (we'll filter later)
        feature_dict = {}

        # Compute features along the time axis (axis=-1)
        if self._should_extract_feature("sum"):
            feature_dict["sum"] = np.sum(ecg, axis=-1)
        if self._should_extract_feature("mean"):
            feature_dict["mean"] = np.mean(ecg, axis=-1)
        if self._should_extract_feature("median"):
            feature_dict["median"] = np.median(ecg, axis=-1)
        if self._should_extract_feature("mode"):
            # Mode computation is more expensive
            mode_values = np.zeros((n_samples, n_channels))
            for i in range(n_samples):
                for j in range(n_channels):
                    mode_values[i, j] = scipy.stats.mode(
                        ecg[i, j, :], keepdims=False
                    ).mode
            feature_dict["mode"] = mode_values
        if self._should_extract_feature("var"):
            feature_dict["var"] = np.var(ecg, axis=-1)
        if self._should_extract_feature("range"):
            feature_dict["range"] = np.ptp(ecg, axis=-1)
        if self._should_extract_feature("min"):
            feature_dict["min"] = np.min(ecg, axis=-1)
        if self._should_extract_feature("max"):
            feature_dict["max"] = np.max(ecg, axis=-1)
        if self._should_extract_feature("iqr"):
            q75 = np.percentile(ecg, 75, axis=-1)
            q25 = np.percentile(ecg, 25, axis=-1)
            feature_dict["iqr"] = q75 - q25
        if self._should_extract_feature("skew"):
            feature_dict["skew"] = scipy.stats.skew(ecg, axis=-1)
        if self._should_extract_feature("kurt"):
            feature_dict["kurt"] = scipy.stats.kurtosis(ecg, axis=-1)
        if self._should_extract_feature("peak_to_peak"):
            if "max" in feature_dict and "min" in feature_dict:
                feature_dict["peak_to_peak"] = feature_dict["max"] - feature_dict["min"]
            else:
                max_vals = np.max(ecg, axis=-1)
                min_vals = np.min(ecg, axis=-1)
                feature_dict["peak_to_peak"] = max_vals - min_vals
        if self._should_extract_feature("autocorr"):
            feature_dict["autocorr"] = self._autocorr_lag1(ecg)

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

        # Create column names using lead names
        column_names = [
            f"statistical_{name}_{self.lead_order[ch]}"
            for ch in range(n_channels)
            for name in feature_names_ordered
        ]

        return pd.DataFrame(features_reshaped, columns=column_names)

    def _autocorr_lag1(self, ecg: np.ndarray) -> np.ndarray:
        """Calculate lag-1 autocorrelation for all samples and channels.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)

        Returns:
            Autocorrelation values with shape (n_samples, n_channels)
        """
        x = ecg[:, :, :-1]  # All but last timepoint
        y = ecg[:, :, 1:]   # All but first timepoint

        x_mean = np.mean(x, axis=-1, keepdims=True)
        y_mean = np.mean(y, axis=-1, keepdims=True)

        numerator = np.sum((x - x_mean) * (y - y_mean), axis=-1)
        denominator = np.sqrt(
            np.sum((x - x_mean) ** 2, axis=-1) * np.sum((y - y_mean) ** 2, axis=-1)
        )

        return numerator / denominator

    def _should_extract_feature(self, feature_name: str) -> bool:
        """Check if a feature should be extracted."""
        return feature_name in self.selected_features
