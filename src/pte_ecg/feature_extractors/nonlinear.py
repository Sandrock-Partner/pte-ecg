"""Nonlinear ECG feature extractor.

This extractor calculates various nonlinear metrics that capture complex
dynamic properties of the ECG signal, including entropy measures, fractal
dimensions, and detrended fluctuation analysis.

Requires the optional 'nolds' dependency. Install with:
    pip install pte-ecg[nonlinear]
or:
    pip install nolds>=0.6.2
"""

import multiprocessing
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm

from . import utils
from .._logging import logger
from .base import BaseFeatureExtractor

# Check for optional dependency
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="nolds")
        import nolds

    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False
    nolds = None  # type: ignore


# Module-level helper function for multiprocessing
def _starmap_helper_nonlinear(args: tuple) -> dict:
    """Helper to unpack args for _process_single_sample in multiprocessing."""
    return NonlinearExtractor._process_single_sample_static(*args)


class NonlinearExtractor(BaseFeatureExtractor):
    """Extract nonlinear complexity and entropy features from ECG data.

    This extractor calculates various nonlinear metrics that capture complex
    dynamic properties of the ECG signal, including entropy measures, fractal
    dimensions, and detrended fluctuation analysis.

    Available features (30+ per channel):
        - sample_entropy: Sample entropy (signal complexity)
        - hurst_exponent: Hurst exponent (long-term memory)
        - dfa_alpha1, dfa_alpha2: Detrended fluctuation analysis parameters
        - change_dfa_alpha: Change in DFA alpha
        - embedding_dimension: Optimal embedding dimension
        - largest_lyapunov_exponent: Largest Lyapunov exponent
        - correlation_dimension: Correlation dimension
        - fractal_katz: Katz fractal dimension
        - recurrence_rate, recurrence_variance: Recurrence quantification analysis
        - time_irreversibility: Time irreversibility measure
        - multiscale_entropy: Multiscale sample entropy
        - symbolic_dynamics: Symbolic dynamics measure
        - singular_spectrum_entropy: Singular spectrum entropy
        - complexity_loss: Complexity loss after filtering
        - And more...

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
        "largest_lyapunov_exponent",
        "dynamic_stability",
        "correlation_dimension",
        "fractal_katz",
        "recurrence_rate",
        "recurrence_variance",
        "recurrence_network_measures",
        "rqa_l_entropy",
        "time_irreversibility",
        "nonlinear_variance",
        "dynamic_variance",
        "multiscale_entropy",
        "symbolic_dynamics",
        "sample_entropy_change_rate",
        "singular_spectrum_entropy",
        "complexity_loss",
    ]

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract nonlinear features from ECG data.

        This function calculates 30+ different nonlinear metrics per channel that capture
        complex dynamic properties of the ECG signal:
        - Sample Entropy: Measure of signal complexity and unpredictability
        - Hurst Exponent: Measure of long-term memory of the time series
        - DFA Alpha1/Alpha2: Detrended Fluctuation Analysis parameters
        - Recurrence Rate: Measure of signal repetitions
        - Additional nonlinear features like Approximate Entropy and Permutation Entropy

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
        utils.assert_3_dims(ecg)

        if not HAS_NOLDS:
            raise ImportError(
                "nolds is required for nonlinear features. "
                "Install with: pip install pte-ecg[nonlinear] "
                "or: pip install nolds>=0.6.2"
            )

        start = utils.log_start("Nonlinear", ecg.shape[0])
        n_samples = ecg.shape[0]
        args_list = [(ecg_single, sfreq) for ecg_single in ecg]
        processes = utils.get_n_processes(self.n_jobs, n_samples)

        if processes == 1:
            results = list(
                tqdm(
                    (self._process_single_sample(sample_data, sfreq) for sample_data, sfreq in args_list),
                    total=n_samples,
                    desc="Nonlinear features",
                    unit="sample",
                    disable=n_samples < 2,
                )
            )
        else:
            logger.info(f"Starting parallel processing with {processes} CPUs")
            with multiprocessing.Pool(processes=processes) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(_starmap_helper_nonlinear, args_list),
                        total=n_samples,
                        desc="Nonlinear features",
                        unit="sample",
                    )
                )

        feature_df = pd.DataFrame(results)
        utils.log_end("Nonlinear", start, feature_df.shape)
        return feature_df

    def _process_single_sample(
        self, sample_data: np.ndarray, sfreq: float
    ) -> dict[str, float]:
        """Extract nonlinear features from a single sample (all channels).

        Args:
            sample_data: Single sample ECG data with shape (n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            Dictionary with keys: nonlinear_{feature}_ch{N}
        """
        features: dict[str, float] = {}
        for ch_num, ch_data in enumerate(sample_data):
            ch_feat = self._process_single_channel(ch_data, sfreq, ch_num)
            features.update(
                (f"nonlinear_{key}_ch{ch_num}", value) for key, value in ch_feat.items()
            )
        return features

    @staticmethod
    def _process_single_sample_static(
        sample_data: np.ndarray, sfreq: float
    ) -> dict[str, float]:
        """Static version for multiprocessing compatibility.

        Args:
            sample_data: Single sample ECG data with shape (n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            Dictionary with keys: nonlinear_{feature}_ch{N}
        """
        features: dict[str, float] = {}
        for ch_num, ch_data in enumerate(sample_data):
            ch_feat = NonlinearExtractor._process_single_channel_static(
                ch_data, sfreq, ch_num
            )
            features.update(
                (f"nonlinear_{key}_ch{ch_num}", value) for key, value in ch_feat.items()
            )
        return features

    def _process_single_channel(
        self, ch_data: np.ndarray, sfreq: float, ch_num: int
    ) -> dict[str, float]:
        """Extract nonlinear features from a single channel.

        Args:
            ch_data: Single channel ECG data with shape (n_timepoints,)
            sfreq: Sampling frequency in Hz
            ch_num: Channel number (for logging)

        Returns:
            Dictionary of nonlinear features
        """
        return self._process_single_channel_static(ch_data, sfreq, ch_num)

    @staticmethod
    def _process_single_channel_static(
        ch_data: np.ndarray, sfreq: float, ch_num: int
    ) -> dict[str, float]:
        """Static method for processing a single channel (multiprocessing compatible).

        Args:
            ch_data: Single channel ECG data with shape (n_timepoints,)
            sfreq: Sampling frequency in Hz
            ch_num: Channel number (for logging)

        Returns:
            Dictionary of nonlinear features
        """
        if not HAS_NOLDS:
            raise ImportError(
                "nolds is required for nonlinear features. "
                "Install with: pip install pte-ecg[nonlinear]"
            )

        features: dict[str, float] = {}

        # Basic nolds features
        features["sample_entropy"] = nolds.sampen(ch_data, emb_dim=2)
        features["hurst_exponent"] = nolds.hurst_rs(ch_data)

        # DFA (Detrended Fluctuation Analysis)
        half_len = len(ch_data) // 2
        features["dfa_alpha1"] = (
            nolds.dfa(ch_data, nvals=[4, 8, 16, 32]) if half_len > 32 else np.nan
        )
        features["dfa_alpha2"] = (
            nolds.dfa(ch_data, nvals=[64, 128, 256]) if half_len > 256 else np.nan
        )
        features["change_dfa_alpha"] = (
            nolds.dfa(ch_data[:half_len], nvals=[4, 8, 16])
            - nolds.dfa(ch_data[half_len:], nvals=[4, 8, 16])
            if half_len > 16
            else np.nan
        )

        # Embedding dimension
        features["embedding_dimension"] = np.nan
        embedding_dim = 3
        try:
            embedding_dim, _ = nk.complexity_dimension(ch_data, dimension_max=10)
            features["embedding_dimension"] = embedding_dim
        except IndexError as e:
            logger.warning(
                f"Error calculating embedding dimension for channel {ch_num}: {e}"
            )

        # Lyapunov exponent
        try:
            lyap_exp = features["largest_lyapunov_exponent"] = nolds.lyap_r(
                ch_data, emb_dim=embedding_dim
            )
            features["dynamic_stability"] = np.exp(-np.abs(lyap_exp))
        except ValueError as e:
            logger.warning(
                f"Error calculating lyapunov exponent for channel {ch_num}: {e}"
            )

        # Correlation dimension
        try:
            features["correlation_dimension"] = nolds.corr_dim(
                ch_data, emb_dim=embedding_dim
            )
        except AssertionError as e:
            logger.warning(
                f"Error calculating correlation dimension for channel {ch_num}: {e}"
            )

        # Fractal Dimension Katz
        features["fractal_katz"] = nk.fractal_katz(ch_data)[0]

        # Recurrence measures
        rqa, _ = nk.complexity_rqa(ch_data, dimension=embedding_dim)
        rec_rate = rqa["RecurrenceRate"].iat[0]
        features["recurrence_rate"] = rec_rate
        features["recurrence_variance"] = rec_rate * (1 - rec_rate)
        features["recurrence_network_measures"] = (
            rqa["Determinism"].iat[0] + rqa["Laminarity"].iat[0]
        ) / 2
        features["rqa_l_entropy"] = rqa["LEn"].iat[0]

        # Time irreversibility and nonlinear variance
        diffs = np.diff(ch_data)
        features["time_irreversibility"] = float(
            np.mean(diffs**3) / (np.mean(diffs**2) ** 1.5)
        )
        features["nonlinear_variance"] = float(np.var(diffs**2))

        # Dynamic variance
        window_duration_sec = min(1, len(ch_data) / sfreq)
        window_size = int(window_duration_sec * sfreq)
        step_size = window_size // 2
        windows = np.lib.stride_tricks.sliding_window_view(ch_data, window_size)[
            ::step_size
        ]
        local_vars = np.var(windows, axis=1)
        features["dynamic_variance"] = float(np.var(local_vars))

        # Multiscale entropy
        features["multiscale_entropy"], _ = nk.entropy_sample(
            ch_data, dimension=embedding_dim, scale=2
        )

        # Symbolic dynamics
        above_median = ch_data > np.median(ch_data)
        features["symbolic_dynamics"] = np.sum(np.abs(np.diff(above_median))) / (
            len(above_median) - 1
        )

        # Sample entropy change rate
        entropy1, _ = nk.entropy_sample(ch_data[:half_len], dimension=embedding_dim)
        entropy2, _ = nk.entropy_sample(ch_data[half_len:], dimension=embedding_dim)
        features["sample_entropy_change_rate"] = (
            (entropy2 - entropy1) / (entropy1 + utils.EPS)
            if entropy1 != 0
            else np.nan
        )

        # Singular spectrum entropy (Shannon-Entropy from SVD)
        N = len(ch_data)
        K = min(20, N // 2)  # Embedding dimension limited
        hankel = np.zeros((N - K + 1, K))
        for i in range(K):
            hankel[:, i] = ch_data[i : i + N - K + 1]
        s = np.linalg.svd(hankel, compute_uv=False)
        s_norm = s / np.sum(s) if np.sum(s) > 0 else np.ones_like(s) / len(s)
        features["singular_spectrum_entropy"] = -np.sum(
            s_norm * np.log2(s_norm + utils.EPS)
        )

        # Complexity loss after filtering
        b, a = scipy.signal.butter(2, 0.2)  # Lowpass
        filtered = scipy.signal.filtfilt(b, a, ch_data)
        entropy_orig, _ = nk.entropy_sample(ch_data, dimension=embedding_dim)
        entropy_filt, _ = nk.entropy_sample(filtered, dimension=embedding_dim)
        features["complexity_loss"] = (
            (entropy_orig - entropy_filt) / entropy_orig
            if entropy_orig != 0
            else np.nan
        )

        return features
