"""Morphological ECG feature extractor.

This module wraps the existing morphological feature extraction from features.py.
TODO: Refactor to be fully self-contained in a future update.
"""

import numpy as np
import pandas as pd

from .. import features
from .._logging import logger
from .base import BaseFeatureExtractor


class MorphologicalExtractor(BaseFeatureExtractor):
    """Extract morphological features from ECG waveforms.

    This extractor performs comprehensive waveform analysis including:
    - P, Q, R, S, T wave detection and measurements
    - Interval calculations (QRS, QT, PR, etc.)
    - ST segment analysis
    - Heart rate variability metrics
    - QRS fragmentation
    - T-wave symmetry
    - Electrical axes (multi-lead)
    - Territory-specific markers (12-lead ECG)

    Note: Currently wraps the existing implementation from features.py.
    Feature selection is not yet fully implemented for morphological features.

    Args:
        selected_features: List of features to extract (not yet implemented for filtering)
        n_jobs: Number of parallel jobs (passed to underlying implementation)

    Examples:
        # Extract all morphological features
        extractor = MorphologicalExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)
    """

    name = "morphological"

    # Available features list (comprehensive but not exhaustive)
    available_features = [
        # Durations and dispersions
        "qrs_duration", "qrs_dispersion",
        "qt_interval", "qt_dispersion", "qtc_interval",
        "pq_interval", "pq_dispersion",
        "p_duration", "p_dispersion",
        "t_duration", "t_dispersion",
        "st_duration", "st_dispersion",
        "rt_duration", "rt_dispersion",
        # Amplitudes
        "p_amplitude", "q_amplitude", "r_amplitude", "s_amplitude", "t_amplitude",
        # Areas and slopes
        "p_area", "t_area",
        "r_slope", "t_slope",
        # ST segment
        "st_elevation", "st_depression", "j_point_elevation", "st_slope",
        # T-wave analysis
        "t_wave_inversion_depth", "t_symmetry",
        # RR intervals
        "rr_interval_mean", "rr_interval_std", "rr_interval_median",
        "rr_interval_iqr", "rr_interval_skewness", "rr_interval_kurtosis",
        "sd1", "sd2", "sd1_sd2_ratio",
        # Advanced
        "qrs_fragmentation",
        "qt_rr_ratio", "pr_rr_ratio", "t_qt_ratio",
        # Multi-lead (global features)
        "qrs_axis", "p_axis",
        # Territory-specific (12-lead only)
        "V1_V3_ST_elevation", "V1_V4_T_inversion",
        "V1_Q_amplitude", "V1_Q_to_R_ratio",
        "II_III_aVF_ST_elevation", "II_III_aVF_T_inversion",
        "III_Q_amplitude", "III_Q_to_R_ratio",
        "I_aVL_V5_V6_ST_elevation", "I_aVL_V5_V6_T_inversion",
        "V5_Q_amplitude", "V5_Q_to_R_ratio",
        "V6_Q_amplitude", "V6_Q_to_R_ratio",
        "aVR_ST_elevation",
    ]

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract morphological features from ECG data.

        Currently delegates to the existing implementation in features.py.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_samples, n_features) containing morphological features.
            Column names follow pattern: morphological_{feature_name}_ch{N} for per-channel
            features, and morphological_{feature_name} for global features.

        Raises:
            ValueError: If ecg does not have 3 dimensions
        """
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), "
                f"got shape {ecg.shape}"
            )

        logger.info("Extracting morphological features (using legacy implementation)")

        # Delegate to existing implementation
        # TODO: Implement feature selection when refactoring
        result = features.get_morphological_features(ecg, sfreq, n_jobs=self.n_jobs)

        return result
