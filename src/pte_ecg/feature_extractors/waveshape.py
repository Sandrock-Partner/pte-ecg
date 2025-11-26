"""Waveshape (bispectrum) feature extractor.

This extractor uses bispectral analysis to capture non-linear phase coupling
and waveform characteristics in the frequency domain.

Note: Requires the 'pybispectra' and 'numba' packages. Install with:
    pip install pte-ecg[bispectrum]
or:
    pip install pybispectra>=1.2.1 numba>=0.61.2
"""

import numpy as np
import pandas as pd

from .._logging import logger
from ..core import FeatureExtractor
from . import utils
from .base import BaseFeatureExtractor

try:
    import pybispectra

    HAS_PYBISPECTRA = True
except ImportError:
    HAS_PYBISPECTRA = False
    pybispectra = None  # type: ignore


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

    def __init__(self, parent: FeatureExtractor, n_jobs: int=-1):
        self.parent = parent
        self.n_jobs = n_jobs

    def get_features(self, ecg: np.ndarray) -> pd.DataFrame:
        """Extract waveshape features from ECG data.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_samples, n_features) containing waveshape features.

        Raises:
            ValueError: If ecg does not have 3 dimensions
            ImportError: If pybispectra package is not installed
        """
        utils.assert_3_dims(ecg)

        if not HAS_PYBISPECTRA:
            raise ImportError(
                "pybispectra is required for waveshape features. "
                "Install with: pip install pte-ecg[bispectrum] "
                "or: pip install pybispectra>=1.2.1 numba>=0.61.2"
            )

        logger.info("Extracting waveshape features")

        # Set precision for pybispectra
        pybispectra.set_precision("single")

        # Convert sfreq to int if needed
        sfreq = self.sfreq
        if isinstance(sfreq, float):
            sfreq = int(sfreq)

        # Get number of processes
        processes = utils.get_n_processes(self.n_jobs, ecg.shape[0])

        # Compute FFT coefficients
        fft_coeffs_all, freqs = pybispectra.compute_fft(
            data=ecg,
            sampling_freq=sfreq,
            n_points=int(sfreq // 3),
            window="hanning",
            n_jobs=processes,
            verbose=False,
        )

        results_all = []
        # TODO: Fix incomplete implementation - only processes first 2 samples
        for i, fft_coeffs in enumerate(fft_coeffs_all[:2]):
            fft_coeffs = fft_coeffs[:][np.newaxis, :]
            waveshape = pybispectra.WaveShape(
                data=fft_coeffs,
                freqs=freqs,
                sampling_freq=sfreq,
                verbose=False,
            )
            waveshape.compute(f1s=(1, 40), f2s=(1, 40))  # type: ignore  # compute waveshape
            results = waveshape.results.get_results(copy=False)

            # TODO: Fix results_all reassignment bug - this overwrites previous iterations
            results_all = []
            # Store original shape before reshape
            orig_shape = results.shape
            # transform results from (channels, f1s, f2s) to (channels*f1s*f2s)
            results = results.reshape(results.shape[0] * results.shape[1] * results.shape[2], -1)
            # [np.abs, np.real, np.imag, np.angle]
            results_all.append(results)

            # TODO: Decide if plotting code should remain in production extractor
            print(f"Waveshape results: [{orig_shape[0]} channels x {orig_shape[1]} f1s x {orig_shape[2]} f2s]")
            figs, axes = waveshape.results.plot(
                major_tick_intervals=10,
                minor_tick_intervals=2,
                # cbar_range_abs=(0, 1),
                # cbar_range_real=(-1, 1),
                # cbar_range_imag=(-1, 1),
                # cbar_range_phase=(0, 2),
                plot_absolute=False,
                show=False,
            )
            figs[0].show()

        # TODO: Complete implementation - currently has '...' placeholder
        # - Proper handling of all samples (not just first 2)
        # - Fix results_all accumulation logic
        # - Return proper DataFrame with named columns
        # - Remove or make plotting optional
        return pd.DataFrame(results_all)
