"""Waveshape (bispectrum) feature extractor.

This module wraps the existing waveshape feature extraction from features.py.
Requires the optional 'pybispectra' dependency.
"""

import numpy as np
import pandas as pd

from .. import features
from .._logging import logger
from .base import BaseFeatureExtractor


class WaveShapeExtractor(BaseFeatureExtractor):
    """Extract waveshape (bispectrum) features from ECG data.

    This extractor uses bispectral analysis to capture non-linear phase coupling
    and waveform characteristics in the frequency domain.

    Note: Requires the 'pybispectra' and 'numba' packages. Install with:
        pip install pte-ecg[bispectrum]
    or:
        pip install pybispectra>=1.2.1 numba>=0.61.2

    Args:
        selected_features: List of features to extract (not yet implemented for filtering)
        n_jobs: Number of parallel jobs

    Examples:
        # Extract all waveshape features
        extractor = WaveShapeExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)

    Raises:
        ImportError: If pybispectra is not installed
    """

    name = "waveshape"

    # Available features depend on bispectrum configuration
    available_features = [
        "bispectrum_abs",
        "bispectrum_real",
        "bispectrum_imag",
        "bispectrum_angle",
    ]

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract waveshape features from ECG data.

        Currently delegates to the existing implementation in features.py.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_samples, n_features) containing waveshape features.

        Raises:
            ValueError: If ecg does not have 3 dimensions
            ImportError: If pybispectra package is not installed
        """
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), "
                f"got shape {ecg.shape}"
            )

        logger.info("Extracting waveshape features (using legacy implementation)")

        # Check for pybispectra dependency
        try:
            import pybispectra  # noqa: F401
        except ImportError:
            raise ImportError(
                "pybispectra is required for waveshape features. "
                "Install with: pip install pte-ecg[bispectrum] "
                "or: pip install pybispectra>=1.2.1 numba>=0.61.2"
            ) from None

        # Delegate to existing implementation
        # TODO: Implement feature selection when refactoring
        result = features.get_waveshape_features(ecg, sfreq, n_jobs=self.n_jobs)

        return result
