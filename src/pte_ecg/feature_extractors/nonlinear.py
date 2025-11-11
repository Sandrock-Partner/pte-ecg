"""Nonlinear ECG feature extractor.

This module wraps the existing nonlinear feature extraction from features.py.
Requires the optional 'nolds' dependency.
"""

import numpy as np
import pandas as pd

from .. import features
from .._logging import logger
from .base import BaseFeatureExtractor


class NonlinearExtractor(BaseFeatureExtractor):
    """Extract nonlinear complexity and entropy features from ECG data.

    This extractor calculates various nonlinear metrics that capture complex
    dynamic properties of the ECG signal, including entropy measures, fractal
    dimensions, and detrended fluctuation analysis.

    Available features (30 per channel):
        - sample_entropy: Sample entropy (signal complexity)
        - hurst_exponent: Hurst exponent (long-term memory)
        - higuchi_fractal_dimension: Higuchi fractal dimension
        - recurrence_rate: Recurrence rate
        - dfa_alpha1, dfa_alpha2: Detrended fluctuation analysis parameters
        - change_dfa_alpha: Change in DFA alpha
        - embedding_dimension: Optimal embedding dimension
        - Various complexity, entropy, and information measures

    Note: Requires the 'nolds' package. Install with:
        pip install pte-ecg[nonlinear]
    or:
        pip install nolds>=0.6.2

    Args:
        selected_features: List of features to extract (not yet implemented for filtering)
        n_jobs: Number of parallel jobs

    Examples:
        # Extract all nonlinear features
        extractor = NonlinearExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)

    Raises:
        ImportError: If nolds is not installed
    """

    name = "nonlinear"

    # Comprehensive list of available features
    available_features = [
        "sample_entropy",
        "hurst_exponent",
        "dfa_alpha1",
        "dfa_alpha2",
        "change_dfa_alpha",
        "embedding_dimension",
        "higuchi_fractal_dimension",
        "recurrence_rate",
        # Plus ~20 more complexity features from neurokit2
    ]

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract nonlinear features from ECG data.

        Currently delegates to the existing implementation in features.py.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_samples, n_features) containing nonlinear features.
            Column names follow pattern: nonlinear_{feature_name}_ch{N}

        Raises:
            ValueError: If ecg does not have 3 dimensions
            ImportError: If nolds package is not installed
        """
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), "
                f"got shape {ecg.shape}"
            )

        logger.info("Extracting nonlinear features (using legacy implementation)")

        # Check for nolds dependency
        try:
            import nolds  # noqa: F401
        except ImportError:
            raise ImportError(
                "nolds is required for nonlinear features. "
                "Install with: pip install pte-ecg[nonlinear] "
                "or: pip install nolds>=0.6.2"
            ) from None

        # Delegate to existing implementation
        # TODO: Implement feature selection when refactoring
        result = features.get_nonlinear_features(ecg, sfreq, n_jobs=self.n_jobs)

        return result
