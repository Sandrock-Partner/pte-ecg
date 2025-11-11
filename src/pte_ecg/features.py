"""LEGACY: Feature extraction module for ECG signal analysis.

.. deprecated:: 0.4.0
    This module is deprecated and will be removed in version 1.0.0.
    Use the new plugin-based architecture instead:
    - Import extractors from `pte_ecg.feature_extractors`
    - Use `pte_ecg.get_features()` with Settings configuration

    Example::

        import pte_ecg
        settings = pte_ecg.Settings()
        settings.features.morphological.enabled = True
        features = pte_ecg.get_features(ecg_data, sfreq=1000, settings=settings)

This module provides functions to extract various types of features from ECG signals,
including statistical, morphological, and nonlinear features. It supports parallel
processing for efficient computation on multi-channel ECG data.

**NOTE**: This module is maintained for backward compatibility only. All new development
should use the extractors in `pte_ecg.feature_extractors` package.
"""

import multiprocessing
import os
import sys
import time
import warnings
from typing import Literal

import neurokit2 as nk
import numpy as np
import pandas as pd
import pydantic
import scipy.signal
import scipy.stats
from pydantic import Field
from tqdm import tqdm

try:
    import pybispectra

    HAS_PYBISPECTRA = True
except ImportError:
    HAS_PYBISPECTRA = False
    pybispectra = None

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="nolds")
        import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False
    nolds = None

from ._logging import logger

EPS = 1e-10  # Small constant for numerical stability


def _deprecation_warning(feature_type: str) -> None:
    """Issue a deprecation warning for legacy feature extraction functions.

    Args:
        feature_type: Type of feature being extracted (e.g., 'morphological', 'nonlinear')
    """
    warnings.warn(
        f"The get_{feature_type}_features() function is deprecated and will be removed "
        "in version 1.0.0. Use the new plugin-based architecture instead:\n"
        f"  from pte_ecg.feature_extractors.{feature_type} import "
        f"{feature_type.capitalize()}Extractor\n"
        "  extractor = "
        f"{feature_type.capitalize()}Extractor()\n"
        "  features = extractor.get_features(ecg_data, sfreq)\n"
        "Or use pte_ecg.get_features() with Settings configuration.",
        DeprecationWarning,
        stacklevel=2
    )


# Helper functions for multiprocessing with progress bars
def _starmap_helper_stat(args: tuple) -> dict:
    """Helper to unpack args for _stat_single_patient in multiprocessing."""
    return _stat_single_patient(*args)


def _starmap_helper_nonlinear(args: tuple) -> dict:
    """Helper to unpack args for _nonlinear_single_patient in multiprocessing."""
    return _nonlinear_single_patient(*args)


def _starmap_helper_morph(args: tuple) -> dict:
    """Helper to unpack args for _morph_single_patient in multiprocessing."""
    return _morph_single_patient(*args)


_METHODS_FINDPEAKS = [
    "neurokit",
    "pantompkins",
    "nabian",
    "slopesumfunction",
    "zong",
    "hamilton",
    "christov",
    "engzeemod",
    "elgendi",
    "kalidas",
    "rodrigues",
    "vg",
    "emrich2023",
    "promac",
    # "gamboa", # Does not currently work reliably
    # "manikandan", # Does not currently work reliably
    # "martinez",# Does not currently work reliably
]


class BaseFeature(pydantic.BaseModel):
    """Base class for feature extraction settings.

    Attributes:
        enabled: Whether this feature type should be extracted.
    """

    enabled: bool = True


class FFTArgs(BaseFeature):
    """Settings for Fast Fourier Transform (FFT) feature extraction.

    Attributes:
        enabled: Whether to compute FFT features.
    """


class WelchArgs(BaseFeature):
    """Settings for Welch's method power spectral density feature extraction.

    Attributes:
        enabled: Whether to compute Welch spectral features.
    """


class StatisticalArgs(BaseFeature):
    """Settings for statistical feature extraction.

    Attributes:
        enabled: Whether to compute statistical features.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    """

    n_jobs: int = -1


class MorphologicalArgs(BaseFeature):
    """Settings for morphological feature extraction.

    Attributes:
        enabled: Whether to compute morphological features.
        n_jobs: Number of parallel jobs to run. -1 means using all processors.
    """

    n_jobs: int = -1


class NonlinearArgs(BaseFeature):
    """Settings for nonlinear feature extraction.

    Attributes:
        enabled: Whether to compute nonlinear features.
    """

    enabled: bool = False
    n_jobs: int = -1


class WaveShapeArgs(BaseFeature):
    """Settings for wave shape feature extraction.

    Attributes:
        enabled: Whether to compute wave shape features.
    """

    enabled: bool = False


class FeatureSettings(pydantic.BaseModel):
    """Container for all feature extraction settings.

    Attributes:
        fft: Settings for FFT feature extraction.
        welch: Settings for Welch's method feature extraction.
        statistical: Settings for statistical feature extraction.
        morphological: Settings for morphological feature extraction.
        nonlinear: Settings for nonlinear feature extraction.
        waveshape: Settings for wave shape feature extraction.
    """

    fft: FFTArgs = Field(default_factory=FFTArgs)
    welch: WelchArgs = Field(default_factory=WelchArgs)
    statistical: StatisticalArgs = Field(default_factory=StatisticalArgs)
    morphological: MorphologicalArgs = Field(default_factory=MorphologicalArgs)
    nonlinear: NonlinearArgs = Field(default_factory=NonlinearArgs)
    waveshape: WaveShapeArgs = Field(default_factory=WaveShapeArgs)


class ECGDelineationError(Exception):
    """Raised when all ECG delineation methods fail to detect peaks properly."""

    pass


def assert_3_dims(ecg_data: np.ndarray) -> None:
    """Ensure the input array has 3 dimensions.

    Args:
        ecg_data: Input array to check.

    Raises:
        ValueError: If input array doesn't have exactly 3 dimensions.
    """
    if ecg_data.ndim != 3:
        raise ValueError("ECG data must be 3D (n_samples, n_channels, n_timepoints)")


def get_waveshape_features(
    ecg_data: np.ndarray, sfreq: float, n_jobs: int = -1
) -> pd.DataFrame:
    _deprecation_warning("waveshape")
    if not HAS_PYBISPECTRA:
        raise ImportError(
            "pybispectra is required for waveshape features. "
            "Install with: pip install pte-ecg[bispectrum]"
        )

    pybispectra.set_precision("single")
    if isinstance(sfreq, float):
        sfreq = int(sfreq)
    processes = _get_n_processes(n_jobs, ecg_data.shape[0])
    fft_coeffs_all, freqs = pybispectra.compute_fft(
        data=ecg_data,
        sampling_freq=sfreq,
        n_points=int(sfreq // 3),
        window="hanning",
        n_jobs=processes,
        verbose=False,
    )
    results_all = []
    for i, fft_coeffs in enumerate(fft_coeffs_all[:2]):
        fft_coeffs = fft_coeffs[:][np.newaxis, :]  #
        waveshape = pybispectra.WaveShape(
            data=fft_coeffs,
            freqs=freqs,
            sampling_freq=sfreq,
            verbose=False,
        )
        waveshape.compute(f1s=(1, 40), f2s=(1, 40))  # )  # compute waveshape
        results = waveshape.results.get_results(copy=False)

        results_all = []
        # transform results from (channels, f1s, f2s) to (channels*f1s*f2s)
        results = results.reshape(
            results.shape[0] * results.shape[1] * results.shape[2], -1
        )
        # [np.abs, np.real, np.imag, np.angle]
        results_all.append(results)

        print(
            f"Waveshape results: [{results.shape[0]} channels x "
            f"{results.shape[1]} f1s x {results.shape[2]} f2s]"
        )
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
    ...
    return pd.DataFrame(results_all)


def get_fft_features(ecg_data: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Extract FFT features from ECG data for each sample and channel.

    This function calculates various FFT-based features, including:
    - Sum of frequencies
    - Mean of frequencies
    - Variance of frequencies
    - Dominant frequency
    - Bandwidth (95% cumulative energy)
    - Spectral entropy
    - Spectral flatness
    - Frequency band masks (e.g., HF, LF, VLF)

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        DataFrame containing the extracted FFT features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    _deprecation_warning("fft")
    assert_3_dims(ecg_data)
    start = _log_start("FFT", ecg_data.shape[0])

    n_samples, n_channels, n_timepoints = ecg_data.shape
    xf = np.fft.rfftfreq(n_timepoints, 1 / sfreq)  # (freqs,)
    yf = np.abs(np.fft.rfft(ecg_data, axis=-1))  # (samples, channels, freqs)

    sum_freq = np.sum(yf, axis=-1)
    mean_freq = np.mean(yf, axis=-1)
    var_freq = np.var(yf, axis=-1)

    # Dominant frequency
    dominant_freq_idx = np.argmax(yf, axis=-1)
    dominant_freq = xf[dominant_freq_idx]

    # Normalize for spectral entropy and bandwidth
    yf_norm = yf / (np.sum(yf, axis=-1, keepdims=True) + EPS)

    # Bandwidth (95% cumulative energy)
    cumsum = np.cumsum(yf_norm, axis=-1)
    bandwidth_idx = (cumsum >= 0.95).argmax(axis=-1)
    bandwidth = xf[bandwidth_idx]

    # Spectral entropy
    spectral_entropy = -np.sum(yf_norm * np.log2(yf_norm + EPS), axis=-1)

    # Spectral flatness
    gmean = scipy.stats.gmean(yf + EPS, axis=-1)
    spectral_flatness = gmean / (np.mean(yf + EPS, axis=-1))

    # Frequency band masks
    def band_mask(low, high):
        return (xf >= low) & (xf < high)

    def apply_band(mask):
        return np.sum(yf[..., mask], axis=-1)

    hf_mask = band_mask(15, 40)
    lf_mask = band_mask(0.5, 15)
    b0_10 = band_mask(0, 10)
    b10_20 = band_mask(10, 20)
    b20_30 = band_mask(20, 30)
    b30_40 = band_mask(30, 40)
    below_50 = band_mask(0, 50)
    above_50 = band_mask(50, xf[-1] + 1)

    hf_power = apply_band(hf_mask)
    lf_power = apply_band(lf_mask)
    hf_lf_ratio = hf_power / (lf_power + EPS)

    band_energy_0_10 = apply_band(b0_10)
    band_energy_10_20 = apply_band(b10_20)
    band_energy_20_30 = apply_band(b20_30)
    band_energy_30_40 = apply_band(b30_40)

    total_energy = sum_freq
    band_ratio_0_10 = band_energy_0_10 / (total_energy + EPS)
    band_ratio_10_20 = band_energy_10_20 / (total_energy + EPS)
    band_ratio_20_30 = band_energy_20_30 / (total_energy + EPS)
    band_ratio_30_40 = band_energy_30_40 / (total_energy + EPS)

    power_below_50Hz = apply_band(below_50)
    power_above_50Hz = apply_band(above_50)
    relative_power_below_50Hz = power_below_50Hz / (total_energy + EPS)

    # Stack all features: shape -> (samples, channels, features)
    features = np.stack(
        [
            sum_freq,
            mean_freq,
            var_freq,
            dominant_freq,
            bandwidth,
            spectral_entropy,
            spectral_flatness,
            hf_power,
            lf_power,
            hf_lf_ratio,
            band_energy_0_10,
            band_ratio_0_10,
            band_energy_10_20,
            band_ratio_10_20,
            band_energy_20_30,
            band_ratio_20_30,
            band_energy_30_40,
            band_ratio_30_40,
            power_below_50Hz,
            power_above_50Hz,
            relative_power_below_50Hz,
        ],
        axis=-1,
    )

    # Reshape to (samples, channels × features)
    features_reshaped = features.reshape(n_samples, -1)

    # Create column names
    base_names = [
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
    column_names = [
        f"fft_{name}_ch{ch}" for ch in range(n_channels) for name in base_names
    ]

    feature_df = pd.DataFrame(features_reshaped, columns=column_names)
    _log_end("FFT", start, feature_df.shape)
    return feature_df


def get_statistical_features(
    ecg_data: np.ndarray,
    sfreq: float,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Extract statistical features from ECG data for each sample and channel.

    This function calculates various statistical features, including:
    - Sum
    - Mean
    - Median
    - Mode
    - Variance
    - Range
    - Min
    - Max
    - IQR
    - Skewness
    - Kurtosis
    - Peak-to-peak
    - Autocorrelation
    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz
        n_jobs: Number of parallel jobs to run. -1 means using all processors.

    Returns:
        DataFrame containing the extracted statistical features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    _deprecation_warning("statistical")
    assert_3_dims(ecg_data)
    start = _log_start("Statistical", ecg_data.shape[0])
    n_samples = ecg_data.shape[0]
    args_list = [(ecg_single, sfreq) for ecg_single in ecg_data]
    processes = _get_n_processes(n_jobs, n_samples)

    if processes == 1:
        results = list(
            tqdm(
                (_stat_single_patient(*args) for args in args_list),
                total=n_samples,
                desc="Statistical features",
                unit="sample",
                disable=n_samples < 2,
            )
        )
    else:
        logger.info(f"Starting parallel processing with {processes} CPUs")
        with multiprocessing.Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_starmap_helper_stat, args_list),
                    total=n_samples,
                    desc="Statistical features",
                    unit="sample",
                )
            )
    feature_df = pd.DataFrame(results)
    _log_end("Statistical", start, feature_df.shape)
    return feature_df


def _stat_single_patient(sample_data: np.ndarray, sfreq: float) -> dict[str, float]:
    """Extract statistical features from a single sample of ECG data.

    Args:
        sample_data: Single sample of ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        Dictionary containing the extracted statistical features
    """
    sum_ = np.sum(sample_data, axis=1)
    mean = np.mean(sample_data, axis=1)
    median = np.median(sample_data, axis=1)
    mode = scipy.stats.mode(sample_data, axis=1, keepdims=False).mode
    variance = np.var(sample_data, axis=1)
    range_ = np.ptp(sample_data, axis=1)
    min_ = np.min(sample_data, axis=1)
    max_ = np.max(sample_data, axis=1)
    iqr = np.percentile(sample_data, 75, axis=1) - np.percentile(
        sample_data, 25, axis=1
    )
    skewness = scipy.stats.skew(sample_data, axis=1)
    kurt = scipy.stats.kurtosis(sample_data, axis=1)
    peak_to_peak = max_ - min_
    autocorr = _autocorr_lag1(sample_data)
    feature_arr = np.stack(
        [
            sum_,
            mean,
            median,
            mode,
            variance,
            range_,
            min_,
            max_,
            iqr,
            skewness,
            kurt,
            peak_to_peak,
            autocorr,
        ],
        axis=1,
    )
    base_names = [
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
    column_names = [
        f"statistical_{name}_ch{ch}"
        for ch in range(sample_data.shape[0])
        for name in base_names
    ]
    feature_arr = feature_arr.flatten()
    features = {
        name: value for name, value in zip(column_names, feature_arr, strict=True)
    }
    return features


def _autocorr_lag1(sample_data: np.ndarray) -> np.ndarray:
    x = sample_data[:, :-1]
    y = sample_data[:, 1:]
    x_mean = np.mean(x, axis=1, keepdims=True)
    y_mean = np.mean(y, axis=1, keepdims=True)
    numerator = np.sum((x - x_mean) * (y - y_mean), axis=1)
    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2, axis=1) * np.sum((y - y_mean) ** 2, axis=1)
    )
    return numerator / denominator


def get_nonlinear_features(
    ecg_data: np.ndarray, sfreq: float, n_jobs: int = -1
) -> pd.DataFrame:
    """Extract nonlinear features from ECG data for each sample and channel.

    This function calculates 30 different nonlinear metrics per channel that capture
    complex dynamic properties of the ECG signal:
    - Sample Entropy: Measure of signal complexity and unpredictability
    - Hurst Exponent: Measure of long-term memory of the time series
    - Higuchi Fractal Dimension: Measure of the fractal dimension of the signal
    - Recurrence Rate: Measure of signal repetitions
    - DFA Alpha1/Alpha2: Detrended Fluctuation Analysis parameters
    - SD1/SD2: Poincaré plot parameters for heart rate variability
    - SD1/SD2 Ratio: Ratio of Poincaré plot parameters
    - Additional nonlinear features like Approximate Entropy and Permutation Entropy

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        DataFrame containing the extracted nonlinear features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    _deprecation_warning("nonlinear")
    assert_3_dims(ecg_data)
    start = _log_start("Nonlinear", ecg_data.shape[0])
    n_samples = ecg_data.shape[0]
    args_list = [(ecg_single, sfreq) for ecg_single in ecg_data]
    processes = _get_n_processes(n_jobs, n_samples)

    if processes == 1:
        results = list(
            tqdm(
                (_nonlinear_single_patient(*args) for args in args_list),
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
    _log_end("Nonlinear", start, feature_df.shape)
    return feature_df


def _nonlinear_single_patient(
    sample_data: np.ndarray, sfreq: float
) -> dict[str, float]:
    features: dict[str, float] = {}
    for ch_num, ch_data in enumerate(sample_data):
        ch_feat = _nonlinear_single_channel(ch_data, sfreq, ch_num)
        features.update(
            (f"nonlinear_{key}_ch{ch_num}", value) for key, value in ch_feat.items()
        )
    return features


def _nonlinear_single_channel(
    ch_data: np.ndarray, sfreq: float, ch_num: int
) -> dict[str, float]:
    """Extract nonlinear features from a single channel of ECG data.

    Args:
        ch_data: Single channel of ECG data with shape (n_timepoints,)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        dict containing the extracted nonlinear features
    """
    if not HAS_NOLDS:
        raise ImportError(
            "nolds is required for nonlinear features. "
            "Install with: pip install pte-ecg[nonlinear]"
        )

    features: dict[str, float] = {}
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

    features["embedding_dimension"] = np.nan
    embedding_dim = 3
    try:
        embedding_dim, _ = nk.complexity_dimension(ch_data, dimension_max=10)
        features["embedding_dimension"] = embedding_dim
    except IndexError as e:
        logger.warning(
            f"Error calculating embedding dimension for channel {ch_num}: {e}"
        )

    # Lyapunov
    try:
        lyap_exp = features["largest_lyapunov_exponent"] = nolds.lyap_r(
            ch_data, emb_dim=embedding_dim
        )
        features["dynamic_stability"] = np.exp(-np.abs(lyap_exp))
    except ValueError as e:
        logger.warning(f"Error calculating lyapunov exponent for channel {ch_num}: {e}")

    try:
        features["correlation_dimension"] = nolds.corr_dim(
            ch_data, emb_dim=embedding_dim
        )
    except AssertionError as e:
        logger.warning(
            f"Error calculating correlation dimension for channel {ch_num}: {e}"
        )
    # Too slow
    # # Fractal Dimension Higuchi
    # features["fractal_higuchi"] = nk.fractal_higuchi(ch_data, k_max="default")[
    #     0
    # ]
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

    diffs = np.diff(ch_data)
    features["time_irreversibility"] = np.mean(diffs**3) / (np.mean(diffs**2) ** 1.5)
    features["nonlinear_variance"] = np.var(diffs**2)

    window_duration_sec = min(1, len(ch_data) / sfreq)
    window_size = int(window_duration_sec * sfreq)
    step_size = window_size // 2
    windows = np.lib.stride_tricks.sliding_window_view(ch_data, window_size)[
        ::step_size
    ]
    local_vars = np.var(windows, axis=1)
    features["dynamic_variance"] = np.var(local_vars)

    features["multiscale_entropy"], _ = nk.entropy_sample(
        ch_data, dimension=embedding_dim, scale=2
    )

    above_median = ch_data > np.median(ch_data)
    features["symbolic_dynamics"] = np.sum(np.abs(np.diff(above_median))) / (
        len(above_median) - 1
    )
    entropy1, _ = nk.entropy_sample(ch_data[:half_len], dimension=embedding_dim)
    entropy2, _ = nk.entropy_sample(ch_data[half_len:], dimension=embedding_dim)
    features["sample_entropy_change_rate"] = (
        (entropy2 - entropy1) / (entropy1 + EPS) if entropy1 != 0 else np.nan
    )
    # Shannon-Entropy
    N = len(ch_data)
    K = min(20, N // 2)  # Einbettungsdimension begrenzen
    hankel = np.zeros((N - K + 1, K))
    for i in range(K):
        hankel[:, i] = ch_data[i : i + N - K + 1]
    s = np.linalg.svd(hankel, compute_uv=False)
    s_norm = s / np.sum(s) if np.sum(s) > 0 else np.ones_like(s) / len(s)
    features["singular_spectrum_entropy"] = -np.sum(s_norm * np.log2(s_norm + EPS))

    b, a = scipy.signal.butter(2, 0.2)  # Lowpass
    filtered = scipy.signal.filtfilt(b, a, ch_data)
    entropy_orig, _ = nk.entropy_sample(ch_data, dimension=embedding_dim)
    entropy_filt, _ = nk.entropy_sample(filtered, dimension=embedding_dim)
    features["complexity_loss"] = (
        (entropy_orig - entropy_filt) / entropy_orig if entropy_orig != 0 else np.nan
    )
    return features


def get_morphological_features(
    ecg_data: np.ndarray, sfreq: float, n_jobs: int | None = -1
) -> pd.DataFrame:
    """Extract morphological features from ECG data for each sample and channel.

    This function calculates various morphological features.

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz
        n_jobs: Number of parallel jobs to run. -1 means using all processors.

    Returns:
        DataFrame containing the extracted morphological features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    _deprecation_warning("morphological")
    assert_3_dims(ecg_data)
    start = _log_start("Morphological", ecg_data.shape[0])
    n_samples = ecg_data.shape[0]
    args_list = [(ecg_single, sfreq) for ecg_single in ecg_data]
    processes = _get_n_processes(n_jobs, n_samples)

    if processes == 1:
        results = list(
            tqdm(
                (_morph_single_patient(*args) for args in args_list),
                total=n_samples,
                desc="Morphological features",
                unit="sample",
                disable=n_samples < 2,
            )
        )
    else:
        logger.info(f"Starting parallel processing with {processes} CPUs")
        with multiprocessing.Pool(processes=processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_starmap_helper_morph, args_list),
                    total=n_samples,
                    desc="Morphological features",
                    unit="sample",
                )
            )
    feature_df = pd.DataFrame(results)
    _log_end("Morphological", start, feature_df.shape)
    return feature_df


def _morph_single_patient(sample_data: np.ndarray, sfreq: float) -> dict[str, float]:
    """Extract morphological features from a single sample of ECG data.

    Args:
        sample_data: Single sample of ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        dict containing the extracted morphological features
    """
    features: dict[str, float] = {}
    flat_chs = np.all(np.isclose(sample_data, sample_data[:, 0:1]), axis=1)
    if np.all(flat_chs):
        logger.warning("All channels are flat lines. Skipping morphological features.")
        return features
    for ch_num, (ch_data, is_flat) in enumerate(zip(sample_data, flat_chs)):
        if is_flat:
            logger.warning(
                f"Channel {ch_num} is a flat line. Skipping morphological features."
            )
            continue
        ch_feat = _morph_single_channel(ch_data, sfreq)
        features.update(
            (f"morphological_{key}_ch{ch_num}", value) for key, value in ch_feat.items()
        )

    # Calculate electrical axes (requires combining data from multiple channels)
    # Assumes standard 12-lead ECG ordering: I, II, III, aVR, aVL, aVF, V1-V6
    # Lead I = ch0, aVF = ch5
    # QRS axis from R-wave amplitudes
    r_amp_lead_i = features.get("morphological_r_amplitude_ch0")
    r_amp_lead_avf = features.get("morphological_r_amplitude_ch5")
    if (
        r_amp_lead_i is not None
        and r_amp_lead_avf is not None
        and (r_amp_lead_i != 0 or r_amp_lead_avf != 0)
    ):
        features["morphological_qrs_axis"] = float(
            np.arctan2(r_amp_lead_avf, r_amp_lead_i) * 180 / np.pi
        )

    # P axis from P-wave amplitudes
    p_amp_lead_i = features.get("morphological_p_amplitude_ch0")
    p_amp_lead_avf = features.get("morphological_p_amplitude_ch5")
    if (
        p_amp_lead_i is not None
        and p_amp_lead_avf is not None
        and (p_amp_lead_i != 0 or p_amp_lead_avf != 0)
    ):
        features["morphological_p_axis"] = float(
            np.arctan2(p_amp_lead_avf, p_amp_lead_i) * 180 / np.pi
        )

    # Territory-Specific Markers (requires 12-lead ECG)
    # Standard 12-lead ordering: I, II, III, aVR, aVL, aVF, V1-V6
    if sample_data.shape[0] >= 12:
        # ANTERIOR WALL (LAD Territory - V1-V4)
        v1_v3_leads = [6, 7, 8]
        v1_v4_leads = [6, 7, 8, 9]
        v1_v3_st_elev = np.mean(
            [
                features.get(f"morphological_st_elevation_ch{ch}", 0.0)
                for ch in v1_v3_leads
            ]
        )
        features["morphological_V1_V3_ST_elevation"] = float(v1_v3_st_elev)

        v1_v4_t_inv = np.mean(
            [
                features.get(f"morphological_t_wave_inversion_depth_ch{ch}", 0.0)
                for ch in v1_v4_leads
            ]
        )
        features["morphological_V1_V4_T_inversion"] = float(v1_v4_t_inv)

        q_v1 = abs(features.get("morphological_q_amplitude_ch6", 0.0))
        r_v1 = features.get("morphological_r_amplitude_ch6", 1.0)
        features["morphological_V1_Q_amplitude"] = float(q_v1)
        features["morphological_V1_Q_to_R_ratio"] = (
            float(q_v1 / r_v1) if r_v1 > 0 else 0.0
        )

        # INFERIOR WALL (RCA Territory - II, III, aVF)
        inferior_leads = [1, 2, 5]
        inf_st_elev = np.mean(
            [
                features.get(f"morphological_st_elevation_ch{ch}", 0.0)
                for ch in inferior_leads
            ]
        )
        features["morphological_II_III_aVF_ST_elevation"] = float(inf_st_elev)

        inf_t_inv = np.mean(
            [
                features.get(f"morphological_t_wave_inversion_depth_ch{ch}", 0.0)
                for ch in inferior_leads
            ]
        )
        features["morphological_II_III_aVF_T_inversion"] = float(inf_t_inv)

        q_iii = abs(features.get("morphological_q_amplitude_ch2", 0.0))
        r_iii = features.get("morphological_r_amplitude_ch2", 1.0)
        features["morphological_III_Q_amplitude"] = float(q_iii)
        features["morphological_III_Q_to_R_ratio"] = (
            float(q_iii / r_iii) if r_iii > 0 else 0.0
        )

        # LATERAL WALL (LCX Territory - I, aVL, V5, V6)
        lateral_leads = [0, 4, 10, 11]
        lat_st_elev = np.mean(
            [
                features.get(f"morphological_st_elevation_ch{ch}", 0.0)
                for ch in lateral_leads
            ]
        )
        features["morphological_I_aVL_V5_V6_ST_elevation"] = float(lat_st_elev)

        lat_t_inv = np.mean(
            [
                features.get(f"morphological_t_wave_inversion_depth_ch{ch}", 0.0)
                for ch in lateral_leads
            ]
        )
        features["morphological_I_aVL_V5_V6_T_inversion"] = float(lat_t_inv)

        q_v5 = abs(features.get("morphological_q_amplitude_ch10", 0.0))
        r_v5 = features.get("morphological_r_amplitude_ch10", 1.0)
        q_v6 = abs(features.get("morphological_q_amplitude_ch11", 0.0))
        r_v6 = features.get("morphological_r_amplitude_ch11", 1.0)

        features["morphological_V5_Q_amplitude"] = float(q_v5)
        features["morphological_V5_Q_to_R_ratio"] = (
            float(q_v5 / r_v5) if r_v5 > 0 else 0.0
        )
        features["morphological_V6_Q_amplitude"] = float(q_v6)
        features["morphological_V6_Q_to_R_ratio"] = (
            float(q_v6 / r_v6) if r_v6 > 0 else 0.0
        )

        # GLOBAL PATTERNS (aVR)
        features["morphological_aVR_ST_elevation"] = float(
            features.get("morphological_st_elevation_ch3", 0.0)
        )

    return features


def _get_r_peaks(
    ch_data: np.ndarray, sfreq: float
) -> tuple[np.ndarray | None, int, Literal[*_METHODS_FINDPEAKS]]:
    peaks_per_method: dict[Literal[*_METHODS_FINDPEAKS], np.ndarray] = {}
    max_n_peaks = 0
    for method in _METHODS_FINDPEAKS:
        _, peaks_info = nk.ecg_peaks(
            ch_data,
            sampling_rate=np.rint(sfreq).astype(int)
            if method in ["zong", "emrich2023"]
            else sfreq,
            method=method,
        )
        r_peaks: np.ndarray | None = peaks_info["ECG_R_Peaks"]
        n_r_peaks = len(r_peaks) if r_peaks is not None else 0
        if not n_r_peaks:
            logger.debug(f"No R-peaks detected for method '{method}'.")
            continue
        max_n_peaks = max(max_n_peaks, n_r_peaks)
        peaks_per_method[method] = r_peaks
        if n_r_peaks > 1:  # We need at least 2 R-peaks for some features
            return r_peaks, n_r_peaks, method
    if not max_n_peaks:
        return None, max_n_peaks, None
    for method, r_peaks in peaks_per_method.items():
        return r_peaks, max_n_peaks, method  # return first item


def _morph_single_channel(ch_data: np.ndarray, sfreq: float) -> dict[str, float]:
    """Extract morphological features from a single channel of ECG data.

    Args:
        ch_data: Single channel of ECG data with shape (n_timepoints,)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        dict containing the extracted morphological features
    """
    features: dict[str, float] = {}
    r_peaks, n_r_peaks, r_peak_method = _get_r_peaks(ch_data, sfreq)
    if not n_r_peaks:
        logger.warning("No R-peaks detected. Skipping morphological features.")
        return {}
    waves_dict: dict = {}

    # Optimized method selection based on profiling results:
    # - prominence is fastest (4x faster than dwt) and most reliable
    # - cwt performs poorly at low sampling rates (<100 Hz)
    # - dwt and peak are fallback options
    if sfreq < 100:
        # Low sampling rate: skip cwt as it detects significantly fewer features
        methods = ["prominence", "dwt", "peak"]
        logger.debug(f"Using low-frequency optimized methods for {sfreq} Hz: {methods}")
    else:
        # High sampling rate: use all methods with optimized order
        methods = ["prominence", "dwt", "cwt", "peak"]
        logger.debug(
            f"Using high-frequency optimized methods for {sfreq} Hz: {methods}"
        )

    for method in methods:
        if n_r_peaks < 2 and method in {"prominence", "cwt"}:
            logger.info(f"Not enough R-peaks ({n_r_peaks}) for {method} method.")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", nk.misc.NeuroKitWarning)
                warnings.simplefilter(
                    "ignore", scipy.signal._peak_finding_utils.PeakPropertyWarning
                )
                _, waves_dict = nk.ecg_delineate(
                    ch_data,
                    rpeaks=r_peaks,
                    sampling_rate=sfreq,
                    method=method,
                )
            logger.debug(f"ECG delineation successful with method: {method}")
            break
        except nk.misc.NeuroKitWarning as e:
            if "Too few peaks detected" in str(e):
                logger.warning(f"Peak detection failed with method '{method}': {e}")
            else:
                raise
    if not waves_dict:
        raise ECGDelineationError("ECG delineation failed with all available methods.")

    # Extrahiere Intervalle
    p_peaks = waves_dict["ECG_P_Peaks"]
    q_peaks = waves_dict["ECG_Q_Peaks"]
    s_peaks = waves_dict["ECG_S_Peaks"]
    t_peaks = waves_dict["ECG_T_Peaks"]

    p_onsets = waves_dict["ECG_P_Onsets"]
    p_offsets = waves_dict["ECG_P_Offsets"]
    t_onsets = waves_dict["ECG_T_Onsets"]
    t_offsets = waves_dict["ECG_T_Offsets"]

    n_p_peaks = len(p_peaks) if p_peaks is not None else 0
    n_q_peaks = len(q_peaks) if q_peaks is not None else 0
    n_s_peaks = len(s_peaks) if s_peaks is not None else 0
    n_t_peaks = len(t_peaks) if t_peaks is not None else 0
    n_p_onsets = len(p_onsets) if p_onsets is not None else 0
    n_p_offsets = len(p_offsets) if p_offsets is not None else 0
    n_t_onsets = len(t_onsets) if t_onsets is not None else 0
    n_t_offsets = len(t_offsets) if t_offsets is not None else 0

    # QRS-Dauer
    if n_q_peaks and n_s_peaks:
        # Berechne durchschnittliche QRS-Dauer
        qrs_durations: list[float] = []
        max_index = min(n_q_peaks, n_s_peaks)
        for q, s in zip(q_peaks[:max_index], s_peaks[:max_index]):
            if q >= s or np.isnan(q) or np.isnan(s):
                continue
            qrs_durations.append((s - q) / sfreq * 1000)  # in ms
        if qrs_durations:
            features["qrs_duration"] = np.mean(qrs_durations)
            features["qrs_dispersion"] = np.std(qrs_durations)

    # QT-Intervall
    if n_q_peaks and n_t_peaks:
        qt_intervals = []
        max_index = min(n_q_peaks, n_t_peaks)
        for q, t in zip(q_peaks[:max_index], t_peaks[:max_index]):
            if q >= t or np.isnan(q) or np.isnan(t):
                continue
            qt_intervals.append((t - q) / sfreq * 1000)  # in ms
        if qt_intervals:
            features["qt_interval"] = np.mean(qt_intervals)
            features["qt_dispersion"] = np.std(qt_intervals)

    # PQ-Intervall
    if n_p_peaks and n_q_peaks:
        pq_intervals = []
        max_index = min(n_p_peaks, n_q_peaks)
        for p, q in zip(p_peaks[:max_index], q_peaks[:max_index]):
            if p >= q or np.isnan(p) or np.isnan(q):
                continue
            pq_intervals.append((q - p) / sfreq * 1000)  # in ms
        if pq_intervals:
            features["pq_interval"] = np.mean(pq_intervals)
            features["pq_dispersion"] = np.std(pq_intervals)

    # P-Dauer
    if n_p_onsets and n_p_offsets:
        p_durations = []
        max_index = min(n_p_onsets, n_p_offsets)
        for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index]):
            if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                continue
            p_durations.append((p_off - p_on) / sfreq * 1000)
        if p_durations:
            features["p_duration"] = np.mean(p_durations)
            features["p_dispersion"] = np.std(p_durations)

    # T-Dauer
    if n_t_onsets and n_t_offsets:
        t_durations = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            t_durations.append((t_off - t_on) / sfreq * 1000)
        if t_durations:
            features["t_duration"] = np.mean(t_durations)
            features["t_dispersion"] = np.std(t_durations)

    # ST-Dauer
    if n_s_peaks and n_t_onsets:
        st_durations = []
        max_index = min(n_s_peaks, n_t_onsets)
        for s, t_on in zip(s_peaks[:max_index], t_onsets[:max_index]):
            if s >= t_on or np.isnan(s) or np.isnan(t_on):
                continue
            st_durations.append(t_on - s)
        if st_durations:
            features["st_duration"] = np.mean(st_durations) / sfreq * 1000
            features["st_dispersion"] = np.std(st_durations) / sfreq * 1000

    # RT-Dauer
    if n_r_peaks and n_t_onsets:
        rt_durations = []
        max_index = min(n_r_peaks, n_t_onsets)
        for r, t_on in zip(r_peaks[:max_index], t_onsets[:max_index]):
            if r >= t_on or np.isnan(r) or np.isnan(t_on):
                continue
            rt_durations.append((t_on - r) / sfreq * 1000)
        if rt_durations:
            features["rt_duration"] = np.mean(rt_durations)
            features["rt_dispersion"] = np.std(rt_durations)

    # Flächen (Integrale unter den Kurven)
    if n_p_onsets and n_p_offsets:
        p_areas = []
        max_index = min(n_p_onsets, n_p_offsets)
        for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index]):
            if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                continue
            p_areas.append(np.sum(np.abs(ch_data[p_on:p_off])))
        if p_areas:
            features["p_area"] = np.mean(p_areas)

    # T Area
    if n_t_onsets and n_t_offsets:
        t_areas = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            t_areas.append(np.sum(np.abs(ch_data[t_on:t_off])))
        if t_areas:
            features["t_area"] = np.mean(t_areas)

    # R Slope
    if n_r_peaks and n_q_peaks:
        r_slopes = []
        max_index = min(n_r_peaks, n_q_peaks)
        for r, q in zip(r_peaks[:max_index], q_peaks[:max_index]):
            if r < q or np.isnan(r) or np.isnan(q):
                continue
            delta_y = ch_data[r] - ch_data[q]
            delta_x = (r - q) / sfreq
            if delta_x > 0:
                r_slopes.append(delta_y / delta_x)
        if r_slopes:
            features["r_slope"] = np.mean(r_slopes)

    # T Slope
    if n_t_onsets and n_t_offsets:
        t_slopes = []
        max_index = min(n_t_onsets, n_t_offsets)
        for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
            if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                continue
            delta_y = ch_data[t_on] - ch_data[t_off]
            delta_x = (t_on - t_off) / sfreq
            if delta_x > 0:
                t_slopes.append(delta_y / delta_x)
        if t_slopes:
            features["t_slope"] = np.mean(t_slopes)

    # Amplituden
    if n_p_peaks:
        p_amplitudes = [ch_data[p] for p in p_peaks if not np.isnan(p)]
        if p_amplitudes:
            features["p_amplitude"] = np.mean(p_amplitudes)

    if n_q_peaks:
        q_amplitudes = [ch_data[q] for q in q_peaks if not np.isnan(q)]
        if q_amplitudes:
            features["q_amplitude"] = np.mean(q_amplitudes)

    if n_r_peaks:
        r_amplitudes = [ch_data[r] for r in r_peaks if not np.isnan(r)]
        if r_amplitudes:
            features["r_amplitude"] = np.mean(r_amplitudes)

    if n_r_peaks > 1:
        rr_intervals = np.diff(r_peaks) / sfreq
        rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        features["rr_interval_mean"] = mean_rr
        features["rr_interval_std"] = std_rr
        if len(rr_intervals) > 1:
            features["rr_interval_median"] = np.median(rr_intervals)
            features["rr_interval_iqr"] = np.percentile(
                rr_intervals, 75
            ) - np.percentile(rr_intervals, 25)
            cv = std_rr / (abs(mean_rr) + EPS)
            if cv > 0.01 and std_rr > 1e-6:  # Sufficient variance
                features["rr_interval_skewness"] = scipy.stats.skew(rr_intervals)
                features["rr_interval_kurtosis"] = scipy.stats.kurtosis(rr_intervals)
            else:
                # Data is nearly constant, skewness/kurtosis are meaningless
                features["rr_interval_skewness"] = 0.0  # No skew
                features["rr_interval_kurtosis"] = 0.0  # Normal kurtosis (excess=0)
                logger.debug(
                    f"RR intervals nearly constant (CV={cv:.6f}, std={std_rr:.6f}), "
                    "skipping skewness/kurtosis calculation"
                )
            # SD1: short-term variability
            diff_rr = np.diff(rr_intervals)
            sd1 = np.nanstd(diff_rr / np.sqrt(2))
            # SD2: long-term variability
            sdrr = np.nanstd(rr_intervals)  # overall HRV
            interm = 2 * sdrr**2 - sd1**2
            sd2 = np.sqrt(interm) if interm > 0 else np.nan
            features["sd1"] = sd1
            features["sd2"] = sd2
            features["sd1_sd2_ratio"] = (
                sd1 / (sd2 + EPS) if not np.isnan(sd2) else np.nan
            )

    if n_s_peaks:
        s_amplitudes = [ch_data[s] for s in s_peaks if not np.isnan(s)]
        if s_amplitudes:
            features["s_amplitude"] = np.mean(s_amplitudes)

    # QRS Fragmentation (count notches/direction changes in QRS complex)
    features["qrs_fragmentation"] = 0.0
    if n_q_peaks and n_s_peaks:
        fragmentations = []
        for q_idx, s_idx in zip(q_peaks, s_peaks):
            if np.isnan(q_idx) or np.isnan(s_idx):
                continue
            q_pos = int(q_idx)
            s_pos = int(s_idx)
            if s_pos <= q_pos:
                continue
            qrs_region = ch_data[q_pos : s_pos + 1]
            if len(qrs_region) < 3:
                continue
            diff = np.diff(qrs_region)
            sign_changes = np.diff(np.sign(diff))
            notch_count = int(np.sum(np.abs(sign_changes) > 0.1))
            fragmentations.append(notch_count)
        if fragmentations:
            features["qrs_fragmentation"] = float(np.mean(fragmentations))

    if n_t_peaks:
        t_amplitudes = [ch_data[t] for t in t_peaks if not np.isnan(t)]
        if t_amplitudes:
            features["t_amplitude"] = np.mean(t_amplitudes)

            # T-wave inversion depth (for negative T-waves)
            t_inversion_depths = [abs(amp) for amp in t_amplitudes if amp < 0]
            if t_inversion_depths:
                features["t_wave_inversion_depth"] = float(np.mean(t_inversion_depths))
            else:
                features["t_wave_inversion_depth"] = 0.0

    # T-wave symmetry (ratio of ascending to descending limb duration)
    features["t_symmetry"] = 1.0
    t_onsets = waves_dict.get("ECG_T_Onsets")
    t_offsets = waves_dict.get("ECG_T_Offsets")
    if t_onsets is not None and t_offsets is not None and n_t_peaks:
        symmetry_ratios = []
        for onset, peak, offset in zip(t_onsets, t_peaks, t_offsets):
            if np.isnan(onset) or np.isnan(peak) or np.isnan(offset):
                continue
            ascending_duration = peak - onset
            descending_duration = offset - peak
            if ascending_duration <= 0 or descending_duration <= 0:
                continue
            ratio = ascending_duration / descending_duration
            ratio = max(0.1, min(2.0, ratio))
            symmetry_ratios.append(ratio)
        if symmetry_ratios:
            features["t_symmetry"] = float(np.mean(symmetry_ratios))

    # ST Segment Features
    # Calculate global baseline (isoelectric line) from first 200ms of signal
    baseline_samples = int(0.2 * sfreq)  # 200ms
    global_baseline = np.mean(ch_data[: min(baseline_samples, len(ch_data))])

    if n_s_peaks and n_r_peaks:
        st_elevations = []
        st_depressions = []
        st_slopes = []

        # Process each S-peak to extract ST segment features
        for s_peak in s_peaks:
            if np.isnan(s_peak):
                continue

            s_idx = int(s_peak)

            # ST segment: J-point (S-peak) + 20ms to J-point + 80ms
            st_start = s_idx + int(0.02 * sfreq)  # J+20ms
            st_end = min(len(ch_data), s_idx + int(0.08 * sfreq))  # J+80ms

            if st_end > st_start and st_start < len(ch_data):
                st_segment = ch_data[st_start:st_end]

                # ST level relative to baseline
                st_level = np.mean(st_segment) - global_baseline

                # ST elevation (positive deviation)
                st_elevations.append(max(0, st_level))

                # ST depression (negative deviation)
                st_depressions.append(max(0, -st_level))

                # ST slope (trend across ST segment)
                if len(st_segment) > 1:
                    slope = (st_segment[-1] - st_segment[0]) / len(st_segment)
                    st_slopes.append(slope)

        # Store averaged ST segment features
        if st_elevations:
            features["st_elevation"] = float(np.mean(st_elevations))
        if st_depressions:
            features["st_depression"] = float(np.mean(st_depressions))
        if st_elevations and st_depressions:
            features["j_point_elevation"] = features.get(
                "st_elevation", 0.0
            ) - features.get("st_depression", 0.0)
        if st_slopes:
            features["st_slope"] = float(np.mean(st_slopes))

    # QTc (Corrected QT interval using Bazett's formula)
    # QTc = QT / √(RR) where QT is in ms and RR is in seconds
    if "qt_interval" in features and "rr_interval_mean" in features:
        qt_ms = features["qt_interval"]
        rr_sec = features["rr_interval_mean"]
        if rr_sec > 0:
            features["qtc_interval"] = qt_ms / np.sqrt(rr_sec)
        else:
            features["qtc_interval"] = qt_ms  # Fallback if RR is invalid

    # Interval Ratios
    # Convert RR interval from seconds to milliseconds for ratio calculations
    rr_ms = features.get("rr_interval_mean", 0.0) * 1000

    # QT/RR ratio
    if "qt_interval" in features and rr_ms > 0:
        features["qt_rr_ratio"] = features["qt_interval"] / rr_ms

    # PR/RR ratio (using pq_interval as PR interval)
    if "pq_interval" in features and rr_ms > 0:
        features["pr_rr_ratio"] = features["pq_interval"] / rr_ms

    # T/QT ratio
    if "t_duration" in features and "qt_interval" in features:
        qt_ms = features["qt_interval"]
        if qt_ms > 0:
            features["t_qt_ratio"] = features["t_duration"] / qt_ms

    return features


def _get_n_processes(n_jobs: int | None, n_tasks: int) -> int:
    """Get the number of processes to use for parallel processing.

    Args:
        n_jobs: Number of parallel jobs to run.
                - None or -1: Use all available CPUs
                - Positive int: Use exactly that many CPUs
                - Negative int (< -1): Use (total_cpus + n_jobs + 1) CPUs
        n_tasks: Number of tasks to process (used to cap the number of processes)

    Returns:
        Number of processes to use, capped by n_tasks
    """
    # Get total number of CPUs
    if sys.version_info >= (3, 13):
        total_cpus = os.process_cpu_count()
        logger.debug(f"Using os.process_cpu_count: {total_cpus} CPUs available")
    else:
        total_cpus = os.cpu_count()
        logger.debug(f"Using os.cpu_count: {total_cpus} CPUs available")

    # Handle None or fallback if cpu_count returns None
    if total_cpus is None:
        logger.warning("Could not determine CPU count, defaulting to 1")
        total_cpus = 1

    # Handle different n_jobs values
    if n_jobs is None or n_jobs == -1:
        # Use all available CPUs
        n_processes = total_cpus
    elif n_jobs > 0:
        # Use exactly n_jobs CPUs
        n_processes = n_jobs
    elif n_jobs < -1:
        # Use (total_cpus + n_jobs + 1) CPUs
        # e.g., n_jobs=-2 means use all CPUs except 1
        n_processes = max(1, total_cpus + n_jobs + 1)
    else:
        # n_jobs == 0, which doesn't make sense, default to 1
        logger.warning(f"Invalid n_jobs value: {n_jobs}, defaulting to 1")
        n_processes = 1

    # Cap by number of tasks (no point using more processes than tasks)
    n_processes = min(n_processes, n_tasks)

    # Ensure at least 1 process
    n_processes = max(1, n_processes)

    logger.debug(f"Using {n_processes} processes for {n_tasks} tasks")
    return n_processes


def _log_end(feature_name: str, start_time: float, shape: tuple[int, int]) -> None:
    """Log the end of feature extraction.

    Args:
        feature_name: Name of the feature type being extracted.
        start_time: Start time of the feature extraction.
        shape: Shape of the extracted features.
    """
    logger.info(
        "Completed %s feature extraction. Shape: %s. Time taken: %.1f s",
        feature_name,
        shape,
        time.time() - start_time,
    )


def _log_start(feature_name: str, n_samples: int) -> float:
    """Log the start of feature extraction and return the current time.

    Args:
        feature_name: Name of the feature type being extracted.
        n_samples: Number of samples.

    Returns:
        Current time.
    """
    logger.info(
        "Starting %s feature extraction for %s samples...", feature_name, n_samples
    )
    return time.time()


def get_welch_features(ecg_data: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Extract Welch's method power spectral density features from ECG data for each sample and channel.

    This function calculates various Welch's method features, including:
    - Log power ratio
    - Band 0-0.5 Hz
    - Band 0.5-4 Hz
    - Band 4-15 Hz
    - Band 15-40 Hz
    - Band over 40 Hz
    - Spectral entropy
    - Total power
    - Peak frequency

    Args:
        ecg_data: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency of the ECG data in Hz

    Returns:
        DataFrame containing the extracted Welch's method features

    Raises:
        ValueError: If input data has incorrect dimensions
    """
    _deprecation_warning("welch")
    assert_3_dims(ecg_data)
    start = _log_start("Welch", ecg_data.shape[0])
    n_samples, n_channels, n_timepoints = ecg_data.shape
    flat_data = ecg_data.reshape(
        -1, n_timepoints
    )  # Shape: (n_samples * n_channels, n_timepoints)

    # Compute Welch spectra for each channel
    psd_list = []
    freq_list = []
    for channel_data in flat_data:
        freqs, Pxx = scipy.signal.welch(
            channel_data,
            fs=sfreq,
            nperseg=sfreq,
            scaling="density",
        )
        psd_list.append(Pxx)
        freq_list.append(freqs)

    freqs = np.array(freq_list[0])
    psd_array = np.array(psd_list)  # Shape: (n_samples * n_channels, len(freqs))

    # Split into n_bins equal-sized dynamic bins (dependent on sampling frequency)
    n_bins = 10
    bins = np.zeros((psd_array.shape[0], n_bins))
    bin_freqs = np.zeros((n_bins, 2))
    for i, bin_idx in enumerate(np.array_split(np.arange(psd_array.shape[1]), n_bins)):
        bins[:, i] = np.mean(psd_array[:, bin_idx], axis=1)
        bin_freqs[i, 0] = freqs[bin_idx[0]]
        bin_freqs[i, 1] = freqs[bin_idx[-1]]

    # Create masks for hard-coded frequency bands
    mask_low = freqs <= 15
    mask_high = freqs > 15
    mask_0_0_5 = (freqs >= 0) & (freqs <= 0.5)
    mask_0_5_4 = (freqs > 0.5) & (freqs <= 4)
    mask_4_15 = (freqs > 4) & (freqs <= 15)
    mask_15_40 = (freqs > 15) & (freqs <= 40)
    mask_over_40 = freqs > 40

    low_power = np.sum(psd_array[:, mask_low], axis=1)
    high_power = np.sum(psd_array[:, mask_high], axis=1)
    log_power_ratio = np.log(high_power / (low_power + 1e-10) + 1e-10)

    band_0_0_5 = np.sum(psd_array[:, mask_0_0_5], axis=1)
    band_0_5_4 = np.sum(psd_array[:, mask_0_5_4], axis=1)
    band_4_15 = np.sum(psd_array[:, mask_4_15], axis=1)
    band_15_40 = np.sum(psd_array[:, mask_15_40], axis=1)
    band_over_40 = np.sum(psd_array[:, mask_over_40], axis=1)

    # Calculate other features
    total_power = np.sum(psd_array, axis=1)
    psd_norm = psd_array / (total_power[:, None] + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)
    peak_indices = np.argmax(psd_array, axis=1)
    peak_frequency = freqs[peak_indices]

    all_features = np.column_stack(
        [
            bins,
            log_power_ratio,
            band_0_0_5,
            band_0_5_4,
            band_4_15,
            band_15_40,
            band_over_40,
            spectral_entropy,
            total_power,
            peak_frequency,
        ]
    )
    all_features = all_features.reshape(n_samples, n_channels * all_features.shape[1])
    base_names = [
        *(f"bin_{low}_{high}" for low, high in bin_freqs),
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
    column_names = [
        f"welch_{name}_ch{ch}" for ch in range(ecg_data.shape[1]) for name in base_names
    ]
    feature_df = pd.DataFrame(all_features, columns=column_names)
    _log_end("Welch", start, feature_df.shape)
    return feature_df
