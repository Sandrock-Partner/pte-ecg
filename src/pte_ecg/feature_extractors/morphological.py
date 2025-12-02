"""Morphological ECG feature extractor.

This extractor performs comprehensive waveform analysis including peak detection,
interval calculations, ST segment analysis, and territory-specific markers.
"""

import math
import multiprocessing
import warnings
from typing import Literal, Self, get_args

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tqdm import tqdm

from .._logging import logger
from ..core import FeatureExtractor
from . import utils
from .base import BaseFeatureExtractor

PeakMethod = Literal[
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
]
PeakMethods = list[PeakMethod]
PEAKMETHODS: PeakMethods = list(get_args(PeakMethod))

WaveName = Literal[
    "P_Peaks",
    "Q_Peaks",
    "R_Peaks",
    "S_Peaks",
    "T_Peaks",
    "P_Onsets",
    "P_Offsets",
    "R_Onsets",
    "R_Offsets",
    "T_Onsets",
    "T_Offsets",
    "J_Points",
]
WaveNames = frozenset[WaveName]
WAVE_NAMES: WaveNames = frozenset(get_args(WaveName))


WAVE_INTERVALS: list[tuple[WaveName, WaveName, str, float, float]] = [
    ("P_Peaks", "R_Peaks", "pr_interval", 80, 300),  # PR interval: typically 120-200ms
    ("P_Peaks", "Q_Peaks", "pq_interval", 80, 300),  # PQ interval: similar to PR
    ("Q_Peaks", "R_Peaks", "qr_interval", 5, 100),  # Part of QRS: typically <40ms
    ("Q_Peaks", "S_Peaks", "qrs_duration", 40, 200),  # QRS duration: typically 60-100ms
    ("R_Peaks", "S_Peaks", "rs_interval", 15, 100),  # Part of QRS: typically 40-60ms
    ("S_Peaks", "T_Onsets", "st_duration", 40, 200),  # ST segment: typically 80-120ms
    ("Q_Peaks", "T_Offsets", "qt_interval", 200, 700),  # QT interval: typically 300-450ms (HR dependent)
    ("R_Peaks", "T_Onsets", "rt_duration", 100, 400),  # R to T onset: typically 200-300ms
    ("R_Peaks", "T_Peaks", "rt_interval", 150, 500),  # R to T peak: typically 250-350ms
    ("P_Peaks", "T_Peaks", "pt_interval", 200, 1000),  # P to T: entire cycle minus TP segment
    ("R_Onsets", "R_Offsets", "r_duration", 40, 200),  # R wave duration: typically 60-100ms
    ("P_Onsets", "P_Offsets", "p_duration", 40, 150),  # P wave duration: typically 80-120ms
    ("T_Onsets", "T_Offsets", "t_duration", 50, 250),  # T wave duration: typically 100-200ms
]


class Waves(BaseModel):
    """Pydantic model for ECG wave delineation results.

    Contains indices for P, Q, R, S, T wave peaks and their onsets/offsets.
    All lists have the same length (one entry per detected R-peak/beat).
    Values can be NaN (represented as float('nan')) when a particular wave
    feature was not detected for a beat.

    The model accepts both snake_case field names (p_peaks, q_peaks, etc.) and
    neurokit2-style aliases (ECG_P_Peaks, ECG_Q_Peaks, etc.) for construction.

    Attributes:
        r_peaks: R-wave peak indices (list of int or float for NaN)
        p_peaks: P-wave peak indices (list of int or float for NaN)
        q_peaks: Q-wave peak indices
        s_peaks: S-wave peak indices
        t_peaks: T-wave peak indices
        p_onsets: P-wave onset indices
        p_offsets: P-wave offset indices
        r_onsets: R-wave onset (Q-onset) indices
        r_offsets: R-wave offset indices
        t_onsets: T-wave onset indices
        t_offsets: T-wave offset indices
        j_points: J-point indices (R-offsets with optional offset applied)
    """

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    r_peaks: list[int | float] = Field(alias="ECG_R_Peaks")
    p_peaks: list[int | float] = Field(alias="ECG_P_Peaks")
    q_peaks: list[int | float] = Field(alias="ECG_Q_Peaks")
    s_peaks: list[int | float] = Field(alias="ECG_S_Peaks")
    t_peaks: list[int | float] = Field(alias="ECG_T_Peaks")
    p_onsets: list[int | float] = Field(alias="ECG_P_Onsets")
    p_offsets: list[int | float] = Field(alias="ECG_P_Offsets")
    r_onsets: list[int | float] = Field(alias="ECG_R_Onsets")
    r_offsets: list[int | float] = Field(alias="ECG_R_Offsets")
    t_onsets: list[int | float] = Field(alias="ECG_T_Onsets")
    t_offsets: list[int | float] = Field(alias="ECG_T_Offsets")
    j_points: list[int | float] = Field(alias="ECG_J_Points")

    @field_validator("*", mode="before")
    @classmethod
    def convert_to_list(cls, v: list | np.ndarray) -> list[int | float]:
        """Convert input to list, handle NaN as float('nan')."""
        return [np.nan if np.isnan(x) else int(x) for x in v]

    @model_validator(mode="after")
    def validate_consistent_lengths(self) -> Self:
        """Ensure all wave lists have consistent lengths."""
        lengths = {name: len(getattr(self, name)) for name in self.model_fields if len(getattr(self, name)) > 0}
        if lengths:
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                raise ValueError(f"Inconsistent wave lengths detected: {lengths}")
        return self

    @property
    def n_beats(self) -> int:
        """Return the number of beats (length of lists)."""
        for name in self.model_fields:
            lst = getattr(self, name)
            if len(lst) > 0:
                return len(lst)
        return 0

    def __len__(self) -> int:
        """Return the number of beats."""
        return self.n_beats

    def get_array(self, wave_name: str) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get wave indices as numpy array for a given wave name.

        Examples:
            >>> waves = Waves(...)
            >>> p_array = waves.get_array('P_Peaks')
            >>> r_array = waves.get_array('R_Peaks')
        """
        wave_list = getattr(self, wave_name.lower())
        return np.array(wave_list, dtype=float)


def _safe_nanmean(values: list | np.ndarray) -> float:
    """Compute nanmean only if at least one non-NaN value exists.

    This avoids RuntimeWarning: Mean of empty slice when all values are NaN.
    """
    arr = np.asarray(values)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


class ECGDelineationError(Exception):
    """Raised when all ECG delineation methods fail to detect peaks properly."""

    pass


def ecg_delineate(
    ch_data: np.ndarray,
    r_peaks: np.ndarray[tuple[int], np.dtype[np.int32]],
    sfreq: float,
    j_point_offset_ms: float | Literal["auto"] = "auto",
) -> Waves:
    """Delineate P, Q, S, T waves using neurokit2 with optimized method selection.

    This function uses the same optimized method selection logic as the morphological
    feature extractor, prioritizing methods based on sampling frequency and reliability.

    Args:
        ch_data: Single channel ECG data with shape (n_timepoints,)
        r_peaks: R-peak indices (must be valid, non-empty array)
        sfreq: Sampling frequency in Hz
        j_point_offset_ms: Additional offset in milliseconds to add to R-Offsets (J-points).
            Since neurokit2 often detects R-Offsets slightly before the true J-point,
            a small positive offset (e.g., 10-20ms) can improve accuracy.
            If "auto", applies method-specific default offset (20ms for all methods).
            Default: "auto"

    Returns:
        Waves: Pydantic model containing wave indices for each wave type

    Raises:
        ECGDelineationError: If all delineation methods fail
        ValueError: If r_peaks is empty or None

    Examples:
        >>> import numpy as np
        >>> import pte_ecg
        >>> # Detect R-peaks first
        >>> r_peaks = np.array([100, 300, 500])
        >>> # Delineate waves with automatic J-point offset
        >>> waves = pte_ecg.ecg_delineate(ecg_signal, r_peaks, sfreq=1000)
        >>> # Or with custom offset
        >>> waves = pte_ecg.ecg_delineate(ecg_signal, r_peaks, sfreq=1000, j_point_offset_ms=10.0)
        >>> p_peaks = waves.p_peaks
        >>> q_peaks = waves.q_peaks
    """
    if r_peaks is None or len(r_peaks) == 0:
        raise ValueError("r_peaks must be a non-empty array")

    n_r_peaks = len(r_peaks)
    waves_dict: dict = {}
    used_method: str | None = None

    # Method-specific default offsets (in milliseconds)
    METHOD_OFFSETS: dict[str, float] = {
        "prominence": 20.0,
        "dwt": 20.0,
        "cwt": 20.0,
    }

    # Optimized method selection based on profiling results:
    # - dwt might be best for detecting on- and offsets of waves
    # - prominence is fastest (4x faster than dwt)
    # - cwt performs poorly at low sampling rates (<100 Hz)
    # - peak is not used as it does not detect most on- and offsets of waves
    methods = ["prominence", "dwt"]
    if sfreq > 100:
        methods.append("cwt")
    logger.debug(f"Using methods for ECG delineation: {methods}")

    for method in methods:
        if n_r_peaks < 2 and method in {"prominence", "cwt"}:
            logger.info(f"Not enough R-peaks (got {n_r_peaks}) for {method} method.")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", nk.misc.NeuroKitWarning)  # type: ignore
                warnings.simplefilter(
                    "ignore",
                    scipy.signal._peak_finding_utils.PeakPropertyWarning,  # type: ignore
                )
                waves_dict = nk_ecg_delineate(
                    ch_data,
                    rpeaks=r_peaks,
                    sampling_rate=sfreq,
                    method=method,
                )
            logger.debug(f"ECG delineation successful with method: {method}")
            used_method = method
            break
        except nk.misc.NeuroKitWarning as e:  # type: ignore
            if "Too few peaks detected" in str(e):
                logger.warning(f"Peak detection failed with method '{method}': {e}")
            else:
                raise
        except Exception as e:
            logger.warning(f"Delineation failed with method '{method}': {e}")
            continue

    if not waves_dict:
        raise ECGDelineationError("ECG delineation failed with all available methods.")

    # Determine actual offset to use
    if j_point_offset_ms == "auto":
        if used_method and used_method in METHOD_OFFSETS:
            actual_offset_ms = METHOD_OFFSETS[used_method]
            logger.debug(f"Using auto J-point offset: {actual_offset_ms}ms for method '{used_method}'")
        else:
            # Fallback if method is unknown
            actual_offset_ms = 20.0
            logger.debug(f"Using fallback auto J-point offset: {actual_offset_ms}ms")
    else:
        actual_offset_ms = j_point_offset_ms

    # Calculate J-points from R-Offsets with optional offset
    if "ECG_R_Offsets" in waves_dict:
        r_offsets = waves_dict["ECG_R_Offsets"]
        if not r_offsets or actual_offset_ms == 0:
            waves_dict["ECG_J_Points"] = r_offsets
        else:
            r_offsets_arr = np.asarray(r_offsets, dtype=float)
            offset_samples = int(actual_offset_ms * sfreq / 1000.0)
            n_samples = len(ch_data)
            # Apply offset only to non-NaN values and clamp to valid range
            valid_mask = ~np.isnan(r_offsets_arr)
            j_points_arr = r_offsets_arr.copy()
            j_points_arr[valid_mask] = np.clip(j_points_arr[valid_mask] + offset_samples, 0, n_samples - 1)
            waves_dict["ECG_J_Points"] = [np.nan if np.isnan(j) else int(j) for j in j_points_arr]
            logger.debug(f"Applied J-point offset of {actual_offset_ms}ms ({offset_samples} samples)")
    waves_dict["ECG_R_Peaks"] = r_peaks
    return Waves(**waves_dict)


def nk_ecg_delineate(
    ecg_cleaned, rpeaks=None, sampling_rate=1000, method="dwt", show=False, show_type="peaks", check=False, **kwargs
):
    """**Delineate QRS complex**(MODIFIED FROM NEUROKIT2 version 0.2.12)

    This function is a modified version of the neurokit2.ecg.ecg_delineate function.
    It is modified to not remove NaN and None values, and to not return the signals dictionary.

    Function to delineate the QRS complex, i.e., the different waves of the cardiac cycles. A
    typical ECG heartbeat consists of a P wave, a QRS complex and a T wave. The P wave represents
    the wave of depolarization that spreads from the SA-node throughout the atria. The QRS complex
    reflects the rapid depolarization of the right and left ventricles. Since the ventricles are
    the largest part of the heart, in terms of mass, the QRS complex usually has a much larger
    amplitude than the P-wave. The T wave represents the ventricular repolarization of the
    ventricles.On rare occasions, a U wave can be seen following the T wave. The U wave is believed
    to be related to the last remnants of ventricular repolarization.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by ``ecg_clean()``.
    rpeaks : Union[list, np.array, pd.Series]
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info
        dictionary returned by ``ecg_findpeaks()``.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        Can be one of ``"peak"`` for a peak-based method, ``"prominence"`` for a peak-prominence-based
        method (Emrich et al., 2024), ``"cwt"`` for continuous wavelet transform or ``"dwt"`` (default)
        for discrete wavelet transform.
        The ``"prominence"`` method might be useful to detect the waves, allowing to set individual physiological
        limits (see kwargs), while the ``"dwt"`` method might be more precise for detecting the onsets and offsets
        of the waves (but might exhibit lower accuracy when there is significant variation in wave morphology).
        The ``"peak"`` method, which uses the zero-crossings of the signal derivatives, works best with very clean signals.
    show : bool
        If ``True``, will return a plot to visualizing the delineated waves information.
    show_type: str
        The type of delineated waves information showed in the plot.
        Can be ``"peaks"``, ``"bounds_R"``, ``"bounds_T"``, ``"bounds_P"`` or ``"all"``.
    check : bool
        Defaults to ``False``. If ``True``, replaces the delineated features with ``np.nan`` if its
        standardized distance from R-peaks is more than 3.
    **kwargs
        Other optional arguments:
        If using the ``"prominence"`` method, additional parameters (in milliseconds) can be passed to set
        individual physiological limits for the search boundaries:
        - ``max_qrs_interval``: The maximum allowable QRS complex interval. Defaults to 180 ms.
        - ``max_pr_interval``: The maximum PR interval duration. Defaults to 300 ms.
        - ``max_r_rise_time``: Maximum duration for the R-wave rise. Defaults to 120 ms.
        - ``typical_st_segment``: Typical duration of the ST segment. Defaults to 150 ms.
        - ``max_p_basepoint_interval``: The maximum interval between P-wave on- and offset. Defaults to 100 ms.
        - ``max_r_basepoint_interval``: The maximum interval between R-wave on- and offset. Defaults to 100 ms.
        - ``max_t_basepoint_interval``: The maximum interval between T-wave on- and offset. Defaults to 200 ms.

    Returns
    -------
    waves : dict
        A dictionary containing additional information.
        For derivative method, the dictionary contains the samples at which P-peaks, Q-peaks,
        S-peaks, T-peaks, P-onsets and T-offsets occur, accessible with the keys ``"ECG_P_Peaks"``,
        ``"ECG_Q_Peaks"``, ``"ECG_S_Peaks"``, ``"ECG_T_Peaks"``, ``"ECG_P_Onsets"``,
        ``"ECG_T_Offsets"``, respectively.

        For the wavelet and prominence methods, in addition to the above information, the dictionary contains the
        samples at which QRS-onsets and QRS-offsets occur, accessible with the key
        ``"ECG_P_Peaks"``, ``"ECG_T_Peaks"``, ``"ECG_P_Onsets"``, ``"ECG_P_Offsets"``,
        ``"ECG_Q_Peaks"``, ``"ECG_S_Peaks"``, ``"ECG_T_Onsets"``, ``"ECG_T_Offsets"``,
        ``"ECG_R_Onsets"``, ``"ECG_R_Offsets"``, respectively.

    See Also
    --------
    ecg_clean, .signal_fixpeaks, ecg_peaks, .signal_rate, ecg_process, ecg_plot

    Examples
    --------
    * Step 1. Delineate

    .. ipython:: python

      import neurokit2 as nk

      # Simulate ECG signal
      ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
      # Get R-peaks location
      _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=1000)
      # Delineate cardiac cycle
      waves = nk_ecg_delineate(ecg, rpeaks, sampling_rate=1000)

    * Step 2. Plot P-Peaks and T-Peaks

    .. ipython:: python

      @savefig p_ecg_delineate1.png scale=100%
      nk.events_plot([waves["ECG_P_Peaks"], waves["ECG_T_Peaks"]], ecg)
      @suppress
      plt.close()

    References
    --------------
    - Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based
      ECG delineator: evaluation on standard databases. IEEE Transactions on biomedical engineering,
      51(4), 570-581.
    - Emrich, J., Gargano, A., Koka, T., & Muma, M. (2024). Physiology-Informed ECG Delineation Based
      on Peak Prominence. 32nd European Signal Processing Conference (EUSIPCO), 1402-1406.

    """
    from neurokit2.ecg.ecg_delineate import (
        _dwt_ecg_delineator,
        _ecg_delineator_cwt,
        _ecg_delineator_peak,
        _prominence_ecg_delineator,
        ecg_peaks,
        epochs_to_df,
    )

    # Sanitize input for ecg_cleaned
    if isinstance(ecg_cleaned, pd.DataFrame):
        cols = [col for col in ecg_cleaned.columns if "ECG_Clean" in col]
        if cols:
            ecg_cleaned = ecg_cleaned[cols[0]].values
        else:
            raise ValueError("NeuroKit error: ecg_delineate(): Wrong input, we couldn't extractcleaned signal.")

    elif isinstance(ecg_cleaned, dict):
        for i in ecg_cleaned:
            cols = [col for col in ecg_cleaned[i].columns if "ECG_Clean" in col]
            if cols:
                signals = epochs_to_df(ecg_cleaned)
                ecg_cleaned = signals[cols[0]].values

            else:
                raise ValueError("NeuroKit error: ecg_delineate(): Wrong input, we couldn't extractcleaned signal.")

    elif isinstance(ecg_cleaned, pd.Series):
        ecg_cleaned = ecg_cleaned.values

    # Sanitize input for rpeaks
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks["ECG_R_Peaks"]

    if isinstance(rpeaks, dict):
        rpeaks = rpeaks["ECG_R_Peaks"]

    method = method.lower()  # remove capitalised letters
    if method in ["peak", "peaks", "derivative", "gradient"]:
        waves = _ecg_delineator_peak(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
    elif method in ["cwt", "continuous wavelet transform"]:
        waves = _ecg_delineator_cwt(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate)
    elif method in ["dwt", "discrete wavelet transform"]:
        waves = _dwt_ecg_delineator(
            ecg_cleaned, rpeaks, sampling_rate=sampling_rate, analysis_sampling_rate=sampling_rate
        )
    elif method in ["prominence", "peak-prominence", "emrich", "emrich2024"]:
        waves = _prominence_ecg_delineator(ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate, **kwargs)

    else:
        raise ValueError(
            "NeuroKit error: ecg_delineate(): 'method' should be one of 'peak', 'prominence','cwt' or 'dwt'."
        )

    for _, value in waves.items():
        if np.isnan(value[-1]):
            continue
        assert value[-1] < len(ecg_cleaned), (
            f"Wave index {value[-1]} is larger than ECG signal length {len(ecg_cleaned)}"
        )

    return waves


def _detect_r_peaks(
    ch_data: np.ndarray, sfreq: float
) -> tuple[np.ndarray[tuple[int], np.dtype[np.int32]], int, PeakMethod | None]:
    """Detect R-peaks using multiple methods with automatic fallback.

    Args:
        ch_data: Single channel ECG data with shape (n_timepoints,)
        sfreq: Sampling frequency in Hz

    Returns:
        Tuple of (r_peaks array, number of peaks, method used)
    """
    peaks_per_method: dict[PeakMethod, np.ndarray] = {}
    max_n_peaks = 0
    for method in PEAKMETHODS:
        _, peaks_info = nk.ecg_peaks(
            ch_data,
            sampling_rate=np.rint(sfreq).astype(int) if method in ["zong", "emrich2023"] else sfreq,
            method=method,
        )
        r_peaks: np.ndarray | None = peaks_info["ECG_R_Peaks"]
        if r_peaks is None:
            continue
        n_r_peaks = len(r_peaks)
        if not n_r_peaks:
            logger.debug(f"No R-peaks detected for method '{method}'.")
            continue
        max_n_peaks = max(max_n_peaks, n_r_peaks)
        peaks_per_method[method] = r_peaks
        if n_r_peaks > 1:  # We need at least 2 R-peaks for some features
            return r_peaks, n_r_peaks, method
    if not max_n_peaks:
        return np.ndarray([], dtype=np.int32), max_n_peaks, None
    for method, r_peaks in peaks_per_method.items():
        return r_peaks, max_n_peaks, method  # return first item
    return np.ndarray([], dtype=np.int32), 0, None


def detect_r_peaks(
    ecg_data: np.ndarray,
    sfreq: float,
    lead_order: list[str] | None = None,
) -> dict[int | str, tuple[np.ndarray[tuple[int], np.dtype[np.int32]], int, PeakMethod | None]]:
    """Detect R-peaks from ECG data independently for each channel.

    Args:
        ecg_data: ECG data. Can be:
            - 1D array: Single channel data with shape (n_timepoints,)
            - 2D array: Multi-channel data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        lead_order: List of channel names (e.g., ["I", "II", "III", ...]).
            Optional, but recommended for consistent naming of results.

    Returns:
        Dictionary mapping channel identifier to (r_peaks, n_peaks, method) tuple:
        - For 1D input: {0: (r_peaks, n_peaks, method)}
        - For 2D input: {ch_idx: (r_peaks, n_peaks, method), ...}
        where:
            - r_peaks: Array of R-peak indices (or None if detection failed)
            - n_peaks: Number of peaks detected
            - method: Peak detection method used (or None if failed)

    Raises:
        ValueError: If inputs are invalid (e.g., wrong dimensions, mismatched lead_order)

    Examples:
        >>> import numpy as np
        >>> import pte_ecg
        >>>
        >>> # Single channel detection
        >>> ecg_1d = np.random.randn(10000)
        >>> peaks = pte_ecg.detect_r_peaks(ecg_1d, sfreq=1000)
        >>> r_peaks, n_peaks, method = peaks[0]
        >>>
        >>> # Multi-channel detection
        >>> ecg_12lead = np.random.randn(12, 10000)
        >>> lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        >>> peaks = pte_ecg.detect_r_peaks(ecg_12lead, sfreq=1000, lead_order=lead_names)
        >>> r_peaks_lead_ii, n_peaks_ii, method_ii = peaks["II"]
    """
    if ecg_data.ndim == 1:
        # Single channel: convert to 2D for consistent processing
        ecg_data = ecg_data[np.newaxis, :]
    elif ecg_data.ndim != 2:
        raise ValueError(f"ecg_data must be 1D or 2D, got {ecg_data.ndim}D")

    n_channels, n_timepoints = ecg_data.shape

    # Validate lead_order length matches number of channels
    if lead_order is not None and len(lead_order) != n_channels:
        raise ValueError(
            f"lead_order length ({len(lead_order)}) must match number of channels ({n_channels}). "
            f"lead_order: {lead_order}"
        )

    # Check for flat channels
    flat_chs = np.all(np.isclose(ecg_data, ecg_data[:, 0:1]), axis=1)

    # Per-channel detection: detect independently for each channel
    results: dict[int | str, tuple[np.ndarray, int, PeakMethod | None]] = {}

    for ch_idx in range(n_channels):
        if flat_chs[ch_idx]:
            logger.debug(f"Channel {ch_idx} is a flat line. Skipping peak detection.")
            ch_key: int | str = lead_order[ch_idx] if (lead_order and ch_idx < len(lead_order)) else ch_idx
            results[ch_key] = (np.array([], dtype=np.int32), 0, None)
            continue

        ch_data = ecg_data[ch_idx]
        r_peaks, n_peaks, method = _detect_r_peaks(ch_data, sfreq)

        if lead_order and ch_idx < len(lead_order):
            ch_key = lead_order[ch_idx]
        else:
            ch_key = ch_idx

        results[ch_key] = (r_peaks, n_peaks, method)

    return results


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

    Available features (80+ per channel):
        - Intervals (mean, median, std, min, max): qrs_duration, qt_interval, pq_interval,
          pr_interval, qr_interval, rs_interval, p_duration, t_duration,
          st_duration, rt_duration, rt_interval, pt_interval
        - Amplitudes: p_amplitude, q_amplitude, r_amplitude, etc.
        - Areas: p_area, t_area
        - Slopes: r_slope, t_slope
        - ST segment: st_level, st_slope
        - T-wave: t_inversion_depth, t_symmetry
        - RR intervals: rr_interval_mean, rr_interval_std, etc.
        - Advanced: qrs_fragmentation, qtc_bazett, qtc_fridericia, qt_rr_ratio, etc.
        - Global (12-lead): qrs_axis, p_axis, territory markers
        - Early MI markers: aVL_t_inversion, precordial_t_imbalance

    Args:
        selected_features: List of features to extract (not yet implemented for filtering)
        n_jobs: Number of parallel jobs
        j_point_offset_ms: Additional offset in milliseconds to add to detected J-points (R-Offsets).
            Since neurokit2 often detects J-points slightly before the true position,
            a small positive offset (e.g., 10-20ms) can improve ST segment measurements.
            If "auto", applies method-specific default offset (20ms for all methods).
            Default: "auto"

    Examples:
        # Extract all morphological features
        extractor = MorphologicalExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)
    """

    name = "morphological"

    def __init__(self, parent: FeatureExtractor, n_jobs: int = -1, j_point_offset_ms: float | Literal["auto"] = "auto"):
        """Initialize the morphological extractor.

        Args:
            parent: Parent FeatureExtractor instance for accessing sfreq, lead_order, etc.
            n_jobs: Number of parallel jobs (-1 for auto)
            j_point_offset_ms: Additional offset in milliseconds to add to detected J-points.
                If "auto", applies method-specific default offset (20ms for all methods).
                Default: "auto"
        """
        self.parent = parent
        self.n_jobs = n_jobs
        self.j_point_offset_ms = j_point_offset_ms

    available_features = [
        # Interval means
        "p_duration_mean",
        "qrs_duration_mean",
        "t_duration_mean",
        "st_duration_mean",
        "rt_duration_mean",
        "pq_interval_mean",
        "pr_interval_mean",
        "qr_interval_mean",
        "rs_interval_mean",
        "rt_interval_mean",
        "pt_interval_mean",
        "qt_interval_mean",
        # Interval medians
        "p_duration_median",
        "qrs_duration_median",
        "t_duration_median",
        "st_duration_median",
        "rt_duration_median",
        "pq_interval_median",
        "pr_interval_median",
        "qr_interval_median",
        "rs_interval_median",
        "rt_interval_median",
        "pt_interval_median",
        "qt_interval_median",
        # Interval std deviations
        "p_duration_std",
        "qrs_duration_std",
        "t_duration_std",
        "st_duration_std",
        "rt_duration_std",
        "qt_interval_std",
        "pq_interval_std",
        "pr_interval_std",
        "qr_interval_std",
        "rs_interval_std",
        "rt_interval_std",
        "pt_interval_std",
        # Amplitudes (mean)
        "p_amplitude_mean",
        "q_amplitude_mean",
        "r_amplitude_mean",
        "s_amplitude_mean",
        "t_amplitude_mean",
        "j_amplitude_mean",
        # Amplitudes (std)
        "p_amplitude_std",
        "q_amplitude_std",
        "r_amplitude_std",
        "s_amplitude_std",
        "t_amplitude_std",
        "j_amplitude_std",
        # Ratios
        "q_to_r_ratio",
        # Areas and slopes
        "p_area",
        "t_area",
        "r_slope",
        "t_slope",
        # ST segment
        "st_level",
        "st_area",
        "st_slope",
        "st60_level",
        "st80_level",
        "baseline_median",
        # T-wave analysis
        "t_inversion_depth",
        "t_symmetry",
        "t_biphasic",
        "t_preterminal_negative",
        "t_peak_to_end_interval",
        # RR intervals
        "rr_interval_mean",
        "rr_interval_std",
        "rr_interval_median",
        "rr_interval_iqr",
        "rr_interval_skewness",
        "rr_interval_kurtosis",
        "rr_interval_sd1",
        "rr_interval_sd2",
        "rr_interval_sd1_sd2_ratio",
        # Advanced
        "qrs_fragmentation",
        "qt_rr_ratio",
        "pr_rr_ratio",
        "t_qt_ratio",
        "t_r_ratio",
        "qtc_bazett",
        "qtc_fridericia",
        # Multi-lead (global features)
        "qrs_axis",
        "p_axis",
        "r_progression",
        # Territory-specific (12-lead only)
        # Septal (V1-V2)
        "V1_V2_st_level",
        "V1_V2_j_amplitude",
        "V1_V2_t_r_ratio",
        "V1_V2_hyperacute_t",
        "V1_V2_t_inversion",
        # Anterior (V3-V4)
        "V3_V4_st_level",
        "V3_V4_j_amplitude",
        "V3_V4_t_r_ratio",
        "V3_V4_hyperacute_t",
        "V3_V4_t_inversion",
        # Anteroseptal (V1-V4)
        "V1_V4_st_level",
        "V1_V4_j_amplitude",
        "V1_V4_t_r_ratio",
        "V1_V4_hyperacute_t",
        "V1_V4_t_inversion",
        # Inferior (II, III, aVF)
        "II_III_aVF_st_level",
        "II_III_aVF_j_amplitude",
        "II_III_aVF_t_r_ratio",
        "II_III_aVF_hyperacute_t",
        "II_III_aVF_t_inversion",
        "III_II_st_level_ratio",
        # Lateral (I, aVL, V5, V6)
        "I_aVL_V5_V6_st_level",
        "I_aVL_V5_V6_j_amplitude",
        "I_aVL_V5_V6_t_r_ratio",
        "I_aVL_V5_V6_hyperacute_t",
        "I_aVL_V5_V6_t_inversion",
        # Even more specific territory features
        # Right Ventricular (V1 to V2 ratio)
        "V1_V2_st_level_ratio",
        # Posterior (reciprocal in V1-V3)
        "V1_V3_r_amplitude",
        # Phase 1: Early MI Markers
        "aVL_t_inversion",
        "precordial_t_balance_cv",
        "precordial_t_imbalance",
        "precordial_t_max_min_ratio",
        "precordial_t_imbalance_ratio",
    ]

    def get_features(self, ecg: np.ndarray) -> pd.DataFrame:
        """Extract morphological features from ECG data.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)

        Returns:
            DataFrame with shape (n_samples, n_features) containing morphological features.
            Column names follow pattern: morphological_{feature_name}_{lead_name} for per-channel
            features, and morphological_{feature_name} for global features.

        Raises:
            ValueError: If ecg does not have 3 dimensions
        """
        utils.assert_3_dims(ecg)

        start = utils.log_start("Morphological", ecg.shape[0])
        n_samples = ecg.shape[0]
        processes = utils.get_n_processes(self.n_jobs, n_samples)

        if processes == 1:
            results = list(
                tqdm(
                    (self._process_single_sample(ecg_single) for ecg_single in ecg),
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
                        pool.imap_unordered(self._process_single_sample, ecg),
                        total=n_samples,
                        desc="Morphological features",
                        unit="sample",
                    )
                )

        feature_df = pd.DataFrame(results)
        utils.log_end("Morphological", start, feature_df.shape)
        return feature_df

    def _process_single_sample(self, sample_data: np.ndarray) -> dict[str, float]:
        """Extract morphological features from a single sample (all channels).

        Args:
            sample_data: Single sample ECG data with shape (n_channels, n_timepoints)

        Returns:
            Dictionary with keys: morphological_{feature}_{lead_name} and morphological_{feature}
        """
        lead_order = self.lead_order
        features: dict[str, float] = {}
        flat_chs = np.all(np.isclose(sample_data, sample_data[:, 0:1]), axis=1)
        if np.all(flat_chs):
            logger.warning("All channels are flat lines. Skipping morphological features.")
            return features

        for ch_num, (ch_data, is_flat) in enumerate(zip(sample_data, flat_chs, strict=True)):
            if is_flat:
                lead_name = lead_order[ch_num] if ch_num < len(lead_order) else f"ch{ch_num}"
                logger.warning(f"Channel {ch_num} ({lead_name}) is a flat line. Skipping morphological features.")
                continue
            ch_feat = MorphologicalExtractor._process_single_channel(
                ch_data, self.sfreq, r_peaks=None, j_point_offset_ms=self.j_point_offset_ms
            )
            lead_name = lead_order[ch_num] if ch_num < len(lead_order) else f"ch{ch_num}"
            features.update((f"morphological_{key}_{lead_name}", value) for key, value in ch_feat.items())

        # Calculate electrical axes (requires combining data from multiple channels)
        # QRS axis from R-wave amplitudes (using Lead I and aVF)
        lead_i_idx = lead_order.index("I") if "I" in lead_order else None
        lead_avf_idx = lead_order.index("aVF") if "aVF" in lead_order else None

        if lead_i_idx is not None and lead_avf_idx is not None:
            lead_i_name = lead_order[lead_i_idx]
            lead_avf_name = lead_order[lead_avf_idx]
            r_amp_lead_i = features.get(f"morphological_r_amplitude_mean_{lead_i_name}")
            r_amp_lead_avf = features.get(f"morphological_r_amplitude_mean_{lead_avf_name}")
            if r_amp_lead_i is not None and r_amp_lead_avf is not None and (r_amp_lead_i != 0 or r_amp_lead_avf != 0):
                features["morphological_qrs_axis"] = float(np.arctan2(r_amp_lead_avf, r_amp_lead_i) * 180 / np.pi)

        # P axis from P-wave amplitudes (using Lead I and aVF)
        if lead_i_idx is not None and lead_avf_idx is not None:
            lead_i_name = lead_order[lead_i_idx]
            lead_avf_name = lead_order[lead_avf_idx]
            p_amp_lead_i = features.get(f"morphological_p_amplitude_mean_{lead_i_name}")
            p_amp_lead_avf = features.get(f"morphological_p_amplitude_mean_{lead_avf_name}")
            if p_amp_lead_i is not None and p_amp_lead_avf is not None and (p_amp_lead_i != 0 or p_amp_lead_avf != 0):
                features["morphological_p_axis"] = float(np.arctan2(p_amp_lead_avf, p_amp_lead_i) * 180 / np.pi)

        # Territory-Specific Markers (requires specific leads to be present)
        # ====================================================================
        # 1. SEPTAL WALL (LAD Territory - V1-V2)
        # ====================================================================
        v1_v2_lead_names = [lead for lead in ["V1", "V2"] if lead in lead_order]
        if len(v1_v2_lead_names) >= 2:
            features.update(MorphologicalExtractor._calculate_territory_features("V1_V2", v1_v2_lead_names, features))

        # ====================================================================
        # 2. ANTERIOR WALL (LAD Territory - V3-V4)
        # ====================================================================
        v3_v4_lead_names = [lead for lead in ["V3", "V4"] if lead in lead_order]
        if len(v3_v4_lead_names) >= 2:
            features.update(MorphologicalExtractor._calculate_territory_features("V3_V4", v3_v4_lead_names, features))

        # ====================================================================
        # 3. ANTEROSEPTAL WALL (LAD Territory - V1-V4)
        # ====================================================================
        v1_v4_lead_names = [lead for lead in ["V1", "V2", "V3", "V4"] if lead in lead_order]
        if len(v1_v4_lead_names) >= 4:
            features.update(MorphologicalExtractor._calculate_territory_features("V1_V4", v1_v4_lead_names, features))

        # ====================================================================
        # 4. INFERIOR WALL (RCA or LCx Territory - II, III, aVF)
        # ====================================================================
        inferior_lead_names = [lead for lead in ["II", "III", "aVF"] if lead in lead_order]
        if len(inferior_lead_names) >= 3:
            features.update(
                MorphologicalExtractor._calculate_territory_features("II_III_aVF", inferior_lead_names, features)
            )

        # Differentiate RCA vs LCx: ST elevation III > II suggests RCA
        if "II" in lead_order and "III" in lead_order:
            st_ii = features.get("morphological_st_level_II")
            st_iii = features.get("morphological_st_level_III")
            if st_ii is not None and st_iii is not None and st_ii > 0:
                features["morphological_III_II_st_level_ratio"] = float(st_iii / st_ii)
            else:
                features["morphological_III_II_st_level_ratio"] = np.nan

        # ====================================================================
        # 5. LATERAL WALL (LCx or Diagonal LAD Territory - I, aVL, V5, V6)
        # ====================================================================
        lateral_lead_names = [lead for lead in ["I", "aVL", "V5", "V6"] if lead in lead_order]
        if len(lateral_lead_names) >= 4:
            features.update(
                MorphologicalExtractor._calculate_territory_features("I_aVL_V5_V6", lateral_lead_names, features)
            )

        # Reciprocal ST depression in lateral leads (for inferior MI)
        i_avl_lead_names = [lead for lead in ["I", "aVL"] if lead in lead_order]
        if len(i_avl_lead_names) >= 2:
            i_avl_st_dep_values = [features.get(f"morphological_st_level_{lead}", np.nan) for lead in i_avl_lead_names]
            features["morphological_I_aVL_st_level"] = _safe_nanmean(i_avl_st_dep_values)

        # ====================================================================
        # 6. POSTERIOR WALL (RCA or LCx Territory - reciprocal in V1-V3)
        # ====================================================================
        v1_v3_lead_names_posterior = [lead for lead in ["V1", "V2", "V3"] if lead in lead_order]
        if len(v1_v3_lead_names_posterior) >= 3:
            # Tall R waves in V1-V3 (suggestive of posterior MI)
            v1_v3_r_amplitudes = []
            for lead in v1_v3_lead_names_posterior:
                r_amp = features.get(f"morphological_r_amplitude_mean_{lead}")
                if r_amp is not None:
                    v1_v3_r_amplitudes.append(r_amp)
            if v1_v3_r_amplitudes:
                features["morphological_V1_V3_r_amplitude"] = float(np.mean(v1_v3_r_amplitudes))
            else:
                features["morphological_V1_V3_r_amplitude"] = np.nan

        # ====================================================================
        # 7. RIGHT VENTRICULAR (RCA proximal Territory - V1, V4R)
        # ====================================================================
        # RV infarction: ST elevation in V1 with ST depression in V2
        if "V1" in lead_order and "V2" in lead_order:
            v1_st_elev = features.get("morphological_st_level_V1")
            v2_st_dep = features.get("morphological_st_level_V2")
            # Ratio for RV infarction pattern
            if v1_st_elev is not None and v2_st_dep is not None and v2_st_dep != 0:
                features["morphological_V1_V2_st_level_ratio"] = float(v1_st_elev / v2_st_dep)
            else:
                features["morphological_V1_V2_st_level_ratio"] = np.nan

        # ====================================================================
        # PHASE 1: EARLY MI MARKERS
        # ====================================================================
        # 1. T-wave inversion in lead aVL (early ACS indicator)
        if "aVL" in lead_order:
            avl_t_inv = features.get("morphological_t_inversion_depth_aVL")
            features["morphological_aVL_t_inversion"] = float(avl_t_inv) if avl_t_inv is not None else np.nan

        # 3. Loss of precordial T-wave balance
        # Disproportionate T-wave amplitude between precordial leads (early ischemia sign)
        precordial_leads = [lead for lead in ["V1", "V2", "V3", "V4", "V5", "V6"] if lead in lead_order]
        if len(precordial_leads) >= 3:
            t_amplitudes_precordial = []
            for lead in precordial_leads:
                t_amp = features.get(f"morphological_t_amplitude_mean_{lead}")
                if t_amp is not None:
                    t_amplitudes_precordial.append(abs(t_amp))

            if len(t_amplitudes_precordial) >= 3:
                t_amplitudes_array = np.array(t_amplitudes_precordial)
                # Calculate coefficient of variation (CV) as measure of imbalance
                mean_t_amp = np.mean(t_amplitudes_array)
                std_t_amp = np.std(t_amplitudes_array)
                if mean_t_amp > 0:
                    cv_t_balance = std_t_amp / mean_t_amp
                    features["morphological_precordial_t_balance_cv"] = float(cv_t_balance)
                    # High CV (>0.5) indicates significant imbalance
                    features["morphological_precordial_t_imbalance"] = 1.0 if cv_t_balance > 0.5 else 0.0

                # Also calculate max/min ratio (another measure of imbalance)
                if len(t_amplitudes_array) > 1 and np.min(t_amplitudes_array) > 0:
                    max_min_ratio = np.max(t_amplitudes_array) / np.min(t_amplitudes_array)
                    features["morphological_precordial_t_max_min_ratio"] = float(max_min_ratio)
                    # Ratio > 3 indicates significant imbalance
                    features["morphological_precordial_t_imbalance_ratio"] = 1.0 if max_min_ratio > 3.0 else 0.0

        # R WAVE PROGRESSION (across precordial leads V1-V6)
        # Get R amplitudes in V1-V6
        r_amplitudes = []
        precordial_leads = [lead for lead in ["V1", "V2", "V3", "V4", "V5", "V6"] if lead in lead_order]
        for lead_name in precordial_leads:
            r_amp = features.get(f"morphological_r_amplitude_mean_{lead_name}")
            if r_amp is not None:
                r_amplitudes.append(r_amp)

        # Calculate progression score (should increase from V1 to V4-V6)
        if len(r_amplitudes) >= 4:
            # Check if there's normal progression
            progression = np.diff(r_amplitudes[:4])
            progression_score = float(np.mean(progression > 0))  # Proportion of increases
            features["morphological_r_progression"] = progression_score
        else:
            features["morphological_r_progression"] = np.nan

        return features

    @staticmethod
    def _calculate_territory_features(
        territory_name: str,
        lead_names: list[str],
        features: dict[str, float],
        feature_types: list[str] | None = None,
    ) -> dict[str, float]:
        """Calculate aggregate features for a cardiac territory.

        Args:
            territory_name: Name of the territory (e.g., "V1_V2", "II_III_aVF")
            lead_names: List of lead names in the territory
            features: Dictionary of per-lead features
            feature_types: List of feature types to calculate. If None, calculates all.
                Available: "st_level", "j_amplitude", "t_r_ratio", "hyperacute_t", "t_inversion"

        Returns:
            Dictionary of territory features with keys like "morphological_{territory_name}_{feature_type}"
        """
        if feature_types is None:
            feature_types = ["st_level", "j_amplitude", "t_r_ratio", "hyperacute_t", "t_inversion"]

        territory_features: dict[str, float] = {}

        # ST level
        if "st_level" in feature_types:
            st_values = [features.get(f"morphological_st_level_{lead}", np.nan) for lead in lead_names]
            territory_features[f"morphological_{territory_name}_st_level"] = _safe_nanmean(st_values)

        # J-point amplitude
        if "j_amplitude" in feature_types:
            j_values = [features.get(f"morphological_j_amplitude_mean_{lead}", np.nan) for lead in lead_names]
            territory_features[f"morphological_{territory_name}_j_amplitude"] = _safe_nanmean(j_values)

        # T/R ratio (mean)
        if "t_r_ratio" in feature_types:
            t_r_ratios = [features.get(f"morphological_t_r_ratio_{lead}", np.nan) for lead in lead_names]
            territory_features[f"morphological_{territory_name}_t_r_ratio"] = _safe_nanmean(t_r_ratios)

        # Hyperacute T (any lead with |T/R| >= 1)
        if "hyperacute_t" in feature_types:
            hyperacute = any(
                abs(features.get(f"morphological_t_r_ratio_{lead}", np.nan)) >= 1.0
                for lead in lead_names
                if not np.isnan(features.get(f"morphological_t_r_ratio_{lead}", np.nan))
            )
            territory_features[f"morphological_{territory_name}_hyperacute_t"] = 1.0 if hyperacute else 0.0

        # T-wave inversion
        if "t_inversion" in feature_types:
            t_inv_values = [features.get(f"morphological_t_inversion_depth_{lead}", np.nan) for lead in lead_names]
            territory_features[f"morphological_{territory_name}_t_inversion"] = _safe_nanmean(t_inv_values)

        return territory_features

    @staticmethod
    def _calculate_beatwise_baseline(
        ch_data: np.ndarray,
        r_peak_indices: np.ndarray,
        p_onsets: np.ndarray[tuple[int], np.dtype[np.float64]],
        p_offsets: np.ndarray[tuple[int], np.dtype[np.float64]],
        t_offsets: np.ndarray[tuple[int], np.dtype[np.float64]],
        sfreq: float,
        tp_window_ms: float = 80.0,
        pr_window_ms: float = 80.0,
        tachycardia_threshold_bpm: float = 100.0,
    ) -> np.ndarray:
        """Calculate beat-wise isoelectric baseline from TP or PR segments.

        Returns:
            Array of baseline values, one per R-peak (NaN if not available)
        """
        n_beats = len(r_peak_indices)
        baseline_per_beat = np.full(n_beats, np.nan)

        if n_beats == 0:
            return baseline_per_beat

        is_tachycardic = False
        if n_beats > 1:
            rr_intervals = np.diff(r_peak_indices) / sfreq
            mean_hr = 60.0 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0.0
            is_tachycardic = mean_hr > tachycardia_threshold_bpm

        tp_window_samples = int(tp_window_ms * sfreq / 1000.0)
        pr_window_samples = int(pr_window_ms * sfreq / 1000.0)
        margin_before_r = int(50 * sfreq / 1000.0)
        window_before_r = int(250 * sfreq / 1000.0)
        window_size = int(80 * sfreq / 1000.0)

        for i, r_idx in enumerate(r_peak_indices):
            baseline_value = np.nan

            if is_tachycardic:
                baseline_value = MorphologicalExtractor._get_tp_baseline(
                    r_idx, ch_data, p_onsets, t_offsets, tp_window_samples
                )
            if np.isnan(baseline_value):
                baseline_value = MorphologicalExtractor._get_pr_baseline(
                    r_idx, ch_data, p_offsets, pr_window_samples, margin_before_r
                )
            if np.isnan(baseline_value):
                baseline_value = MorphologicalExtractor._get_r_baseline(r_idx, ch_data, window_before_r, window_size)

            baseline_per_beat[i] = baseline_value

        return baseline_per_beat

    @staticmethod
    def _get_r_baseline(r_idx: int, ch_data: np.ndarray, window_before_r: int, window_size: int) -> float:
        seg_start = max(0, r_idx - window_before_r - window_size)
        seg_end = max(0, r_idx - window_before_r)
        if seg_end > seg_start and seg_end <= len(ch_data):
            return float(np.median(ch_data[seg_start:seg_end]))
        return np.nan

    @staticmethod
    def _get_pr_baseline(
        r_idx: int, ch_data: np.ndarray, p_offsets_arr: np.ndarray, pr_window_samples: int, margin_before_r: int
    ) -> float:
        if len(p_offsets_arr) == 0:
            return np.nan

        p_offsets_before_r = p_offsets_arr[p_offsets_arr < r_idx]
        if len(p_offsets_before_r) == 0:
            return np.nan

        p_offset = int(p_offsets_before_r[-1])
        pr_seg_end = max(p_offset + pr_window_samples, r_idx - margin_before_r)
        pr_seg_start = pr_seg_end - pr_window_samples
        if pr_seg_start >= p_offset and pr_seg_end <= len(ch_data) and pr_seg_start >= 0:
            return float(np.median(ch_data[pr_seg_start:pr_seg_end]))
        return np.nan

    @staticmethod
    def _get_tp_baseline(
        r_idx: int, ch_data: np.ndarray, p_onsets_arr: np.ndarray, t_offsets_arr: np.ndarray, tp_window_samples: int
    ) -> float:
        if len(t_offsets_arr) == 0 or len(p_onsets_arr) == 0:
            return np.nan

        t_offsets_before_r = t_offsets_arr[t_offsets_arr < r_idx]
        if len(t_offsets_before_r) == 0:
            return np.nan

        t_offset = int(t_offsets_before_r[-1])
        p_onsets_after_t = p_onsets_arr[(p_onsets_arr > t_offset) & (p_onsets_arr < r_idx)]
        if len(p_onsets_after_t) == 0:
            return np.nan

        p_onset = int(p_onsets_after_t[0])
        tp_seg_duration = p_onset - t_offset
        if tp_seg_duration < tp_window_samples:
            return np.nan

        window_start = t_offset + (tp_seg_duration - tp_window_samples) // 2
        window_end = window_start + tp_window_samples
        if window_end > len(ch_data) or window_start < 0:
            return np.nan

        return float(np.median(ch_data[window_start:window_end]))

    @staticmethod
    def _get_beat_baseline(
        wave_idx: int,
        r_peak_indices: np.ndarray,
        baseline_per_beat: np.ndarray,
        fallback_baseline: float,
    ) -> float:
        """Get baseline for a specific wave by finding associated R-peak."""
        associated_r_idx = None
        for r_idx in r_peak_indices:
            if not np.isnan(r_idx):
                if wave_idx < r_idx:
                    break
                associated_r_idx = int(r_idx)

        if associated_r_idx is not None:
            r_peak_idx_in_array = np.where(r_peak_indices == associated_r_idx)[0]
            if len(r_peak_idx_in_array) > 0 and not np.isnan(baseline_per_beat[r_peak_idx_in_array[0]]):
                return float(baseline_per_beat[r_peak_idx_in_array[0]])

        return fallback_baseline

    @staticmethod
    def _process_single_channel(
        ch_data: np.ndarray,
        sfreq: float,
        r_peaks: np.ndarray | None = None,
        j_point_offset_ms: float = 0.0,
    ) -> dict[str, float]:
        """Static method for processing a single channel (multiprocessing compatible).

        Args:
            ch_data: Single channel ECG data with shape (n_timepoints,)
            sfreq: Sampling frequency in Hz
            r_peaks: Optional pre-detected R-peaks. If None, peaks are detected from ch_data.
            j_point_offset_ms: Additional offset in milliseconds to add to detected J-points.

        Returns:
            Dictionary of morphological features
        """
        features: dict[str, float] = {}
        if r_peaks is None:
            # Use detect_r_peaks() API for single channel detection
            results = detect_r_peaks(ch_data, sfreq)
            r_peaks, n_r_peaks, _ = results[0]
        else:
            n_r_peaks = len(r_peaks)

        if not n_r_peaks or r_peaks is None:
            logger.warning("No R-peaks detected. Skipping morphological features.")
            return {}

        # Use shared delineation function (returns Waves pydantic model)
        waves = ecg_delineate(ch_data, r_peaks, sfreq, j_point_offset_ms=j_point_offset_ms)

        # Extract wave arrays from Waves model
        p_peaks = waves.p_peaks
        q_peaks = waves.q_peaks
        s_peaks = waves.s_peaks
        t_peaks = waves.t_peaks

        p_onsets = waves.p_onsets
        p_offsets = waves.p_offsets
        r_onsets = waves.r_onsets
        t_onsets = waves.t_onsets
        t_offsets = waves.t_offsets

        j_points = waves.j_points

        n_p_peaks = len(p_peaks)
        n_q_peaks = len(q_peaks)
        n_s_peaks = len(s_peaks)
        n_t_peaks = len(t_peaks)
        n_r_onsets = len(r_onsets)
        n_p_onsets = len(p_onsets)
        n_p_offsets = len(p_offsets)
        n_t_onsets = len(t_onsets)
        n_t_offsets = len(t_offsets)
        n_j_points = len(j_points)

        n_samples = len(ch_data)

        wave_map: dict[WaveName, np.ndarray[tuple[int], np.dtype[np.float64]]] = {
            wave: waves.get_array(wave) for wave in WAVE_NAMES
        }

        baseline_per_beat = MorphologicalExtractor._calculate_beatwise_baseline(
            ch_data, r_peaks, wave_map["P_Onsets"], wave_map["P_Offsets"], wave_map["T_Offsets"], sfreq
        )

        if not np.all(np.isnan(baseline_per_beat)):
            global_baseline = float(np.nanmedian(baseline_per_beat))
        else:
            baseline_samples = int(30 * sfreq)
            global_baseline = float(np.nanmedian(ch_data[: min(baseline_samples, n_samples)]))
        features["baseline_median"] = global_baseline

        # Vectorized interval calculation for all pairs
        for interval_pair in WAVE_INTERVALS:
            wave1, wave2, feature_name, min_interval_ms, max_interval_ms = interval_pair

            peaks1 = wave_map[wave1]
            peaks2 = wave_map[wave2]

            features[f"{feature_name}_mean"] = np.nan
            features[f"{feature_name}_median"] = np.nan
            features[f"{feature_name}_std"] = np.nan

            if len(peaks1) == 0 or len(peaks2) == 0:
                continue

            stats = MorphologicalExtractor._calculate_interval_stats(
                peaks1, peaks2, sfreq, max_interval_ms, min_interval_ms
            )
            if stats:
                features[f"{feature_name}_mean"] = stats["mean"]
                features[f"{feature_name}_median"] = stats["median"]
                features[f"{feature_name}_std"] = stats["std"]

        # Flächen (Integrale unter den Kurven)
        features["p_area"] = np.nan
        if n_p_onsets and n_p_offsets:
            p_areas = []
            max_index = min(n_p_onsets, n_p_offsets)
            for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index], strict=True):
                if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                    continue
                p_areas.append(np.sum(np.abs(ch_data[p_on:p_off])))
            if p_areas:
                features["p_area"] = float(np.mean(p_areas))

        # T Area
        features["t_area"] = np.nan
        if n_t_onsets and n_t_offsets:
            t_areas = []
            max_index = min(n_t_onsets, n_t_offsets)
            for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index], strict=True):
                if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                    continue
                t_areas.append(np.sum(np.abs(ch_data[t_on:t_off])))
            if t_areas:
                features["t_area"] = float(np.mean(t_areas))

        # R Slope
        features["r_slope"] = np.nan
        if n_r_peaks and n_q_peaks and r_peaks is not None:
            r_slopes = []
            max_index = min(n_r_peaks, n_q_peaks)
            for r, q in zip(r_peaks[:max_index], q_peaks[:max_index], strict=True):
                if r < q or np.isnan(r) or np.isnan(q):
                    continue
                delta_y = ch_data[r] - ch_data[q]
                delta_x = (r - q) / sfreq
                if delta_x > 0:
                    r_slopes.append(delta_y / delta_x)
            if r_slopes:
                features["r_slope"] = float(np.mean(r_slopes))

        # T Slope
        features["t_slope"] = np.nan
        if n_t_onsets and n_t_offsets:
            t_slopes = []
            max_index = min(n_t_onsets, n_t_offsets)
            for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index], strict=True):
                if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                    continue
                delta_y = ch_data[t_on] - ch_data[t_off]
                delta_x = (t_on - t_off) / sfreq
                if delta_x > 0:
                    t_slopes.append(delta_y / delta_x)
            if t_slopes:
                features["t_slope"] = float(np.mean(t_slopes))

        # Amplituden
        features["p_amplitude_mean"] = np.nan
        features["p_amplitude_std"] = np.nan
        if n_p_peaks:
            p_peaks_array = wave_map["P_Peaks"]
            valid_mask = ~np.isnan(p_peaks_array)
            if np.any(valid_mask):
                valid_peaks = p_peaks_array[valid_mask].astype(int)
                p_amplitudes = ch_data[valid_peaks]
                features["p_amplitude_mean"] = float(np.mean(p_amplitudes))
                features["p_amplitude_std"] = float(np.std(p_amplitudes)) if len(p_amplitudes) > 1 else 0.0

        features["q_amplitude_mean"] = np.nan
        features["q_amplitude_std"] = np.nan
        if n_q_peaks:
            q_peaks_array = wave_map["Q_Peaks"]
            valid_mask = ~np.isnan(q_peaks_array)
            if np.any(valid_mask):
                valid_peaks = q_peaks_array[valid_mask].astype(int)
                q_amplitudes = ch_data[valid_peaks]
                features["q_amplitude_mean"] = float(np.mean(q_amplitudes))
                features["q_amplitude_std"] = float(np.std(q_amplitudes)) if len(q_amplitudes) > 1 else 0.0

        features["r_amplitude_mean"] = np.nan
        features["r_amplitude_std"] = np.nan
        if n_r_peaks:
            r_peaks_array = wave_map["R_Peaks"]
            valid_mask = ~np.isnan(r_peaks_array)
            if np.any(valid_mask):
                valid_peaks = r_peaks_array[valid_mask].astype(int)
                r_amplitudes = ch_data[valid_peaks]
                features["r_amplitude_mean"] = float(np.mean(r_amplitudes))
                features["r_amplitude_std"] = float(np.std(r_amplitudes)) if len(r_amplitudes) > 1 else 0.0

        features["rr_interval_mean"] = np.nan
        features["rr_interval_std"] = np.nan
        features["rr_interval_median"] = np.nan
        features["rr_interval_iqr"] = np.nan
        features["rr_interval_skewness"] = np.nan
        features["rr_interval_kurtosis"] = np.nan
        features["rr_interval_sd1"] = np.nan
        features["rr_interval_sd2"] = np.nan
        features["rr_interval_sd1_sd2_ratio"] = np.nan
        if n_r_peaks > 1 and r_peaks is not None:
            rr_intervals = np.diff(r_peaks) / sfreq
            rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
            if len(rr_intervals) > 1:
                mean_rr = float(np.mean(rr_intervals))
                std_rr = float(np.std(rr_intervals))
                features["rr_interval_mean"] = mean_rr
                features["rr_interval_std"] = std_rr
                features["rr_interval_median"] = float(np.median(rr_intervals))
                features["rr_interval_iqr"] = float(np.percentile(rr_intervals, 75) - np.percentile(rr_intervals, 25))
                cv = std_rr / (abs(mean_rr) + utils.EPS)
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
                sd1 = float(np.nanstd(diff_rr / np.sqrt(2)))
                # SD2: long-term variability
                sdrr = np.nanstd(rr_intervals)  # overall HRV
                interm = 2 * math.pow(sdrr, 2) - math.pow(sd1, 2)
                sd2 = float(np.sqrt(interm)) if interm > 0 else np.nan
                features["rr_interval_sd1"] = sd1
                features["rr_interval_sd2"] = sd2
                features["rr_interval_sd1_sd2_ratio"] = sd1 / (sd2 + utils.EPS) if not np.isnan(sd2) else np.nan

        features["s_amplitude_mean"] = np.nan
        features["s_amplitude_std"] = np.nan
        if n_s_peaks:
            s_peaks_array = wave_map["S_Peaks"]
            valid_mask = ~np.isnan(s_peaks_array)
            if np.any(valid_mask):
                valid_peaks = s_peaks_array[valid_mask].astype(int)
                s_amplitudes = ch_data[valid_peaks]
                features["s_amplitude_mean"] = float(np.mean(s_amplitudes))
                features["s_amplitude_std"] = float(np.std(s_amplitudes)) if len(s_amplitudes) > 1 else 0.0

        features["j_amplitude_mean"] = np.nan
        features["j_amplitude_std"] = np.nan
        if j_points is not None and len(j_points) > 0:
            j_points_array = wave_map["J_Points"]
            valid_mask = ~np.isnan(j_points_array)
            if np.any(valid_mask):
                valid_j_points = j_points_array[valid_mask].astype(int)
                # Ensure indices are within bounds
                valid_j_points = valid_j_points[valid_j_points < n_samples]
                if len(valid_j_points) > 0:
                    # Calculate j-point amplitude relative to baseline
                    j_amplitudes = []
                    for j_point_idx in valid_j_points:
                        beat_baseline = MorphologicalExtractor._get_beat_baseline(
                            j_point_idx, r_peaks, baseline_per_beat, global_baseline
                        )
                        j_amp = ch_data[j_point_idx] - beat_baseline
                        j_amplitudes.append(j_amp)
                    if j_amplitudes:
                        features["j_amplitude_mean"] = float(np.mean(j_amplitudes))
                        features["j_amplitude_std"] = float(np.std(j_amplitudes)) if len(j_amplitudes) > 1 else 0.0

        # QRS Fragmentation (count notches/direction changes in QRS complex)
        features["qrs_fragmentation"] = np.nan
        if n_q_peaks and n_s_peaks:
            fragmentations = []
            for q_idx, s_idx in zip(q_peaks, s_peaks, strict=True):
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

        # Initialize T wave morphological features
        features["t_amplitude_mean"] = np.nan
        features["t_amplitude_std"] = np.nan
        features["t_inversion_depth"] = np.nan
        if n_t_peaks:
            t_peaks_array = wave_map["T_Peaks"]
            valid_mask = ~np.isnan(t_peaks_array)
            if np.any(valid_mask):
                valid_peaks = t_peaks_array[valid_mask].astype(int)
                t_amplitudes = ch_data[valid_peaks]
                features["t_amplitude_mean"] = float(np.mean(t_amplitudes))
                features["t_amplitude_std"] = float(np.std(t_amplitudes)) if len(t_amplitudes) > 1 else 0.0

                # T-wave inversion depth (for negative T-waves)
                negative_mask = t_amplitudes < 0
                if np.any(negative_mask):
                    t_inversion_depths = np.abs(t_amplitudes[negative_mask])
                    features["t_inversion_depth"] = float(np.mean(t_inversion_depths))

        # T-wave symmetry (ratio of ascending to descending limb duration)
        features["t_symmetry"] = np.nan
        features["t_biphasic"] = np.nan
        features["t_preterminal_negative"] = np.nan
        features["t_peak_to_end_interval"] = np.nan
        if n_t_onsets > 0 and n_t_offsets > 0 and n_t_peaks:
            symmetry_ratios = []
            biphasic_flags = []
            preterminal_negative_flags = []
            t_peak_to_end_intervals = []

            for onset, peak, offset in zip(t_onsets, t_peaks, t_offsets, strict=True):
                if np.isnan(onset) or np.isnan(peak) or np.isnan(offset):
                    continue

                ascending_duration = peak - onset
                descending_duration = offset - peak
                if ascending_duration <= 0 or descending_duration <= 0:
                    continue

                ratio = ascending_duration / descending_duration
                ratio = max(0.1, min(2.0, ratio))
                symmetry_ratios.append(ratio)

                tpeak_tend = (offset - peak) / sfreq * 1000
                t_peak_to_end_intervals.append(tpeak_tend)

                t_start = max(int(onset), 0)
                t_end = min(int(offset), n_samples)
                if t_end <= t_start:
                    continue

                t_segment = ch_data[t_start:t_end]
                has_positive = np.any(t_segment > 0.0)
                has_negative = np.any(t_segment < 0.0)
                biphasic_flags.append(1.0 if (has_positive and has_negative) else 0.0)

                beat_baseline = MorphologicalExtractor._get_beat_baseline(
                    int(onset), r_peaks, baseline_per_beat, global_baseline
                )

                t_duration = t_end - t_start
                terminal_start = int(t_start + 0.6 * t_duration)
                terminal_end = t_end
                if terminal_end <= terminal_start or terminal_start >= n_samples:
                    continue

                terminal_segment = ch_data[terminal_start:terminal_end]
                terminal_below_baseline = terminal_segment < (beat_baseline - 0.05)
                preterminal_negative = np.sum(terminal_below_baseline) / len(terminal_segment) > 0.5
                preterminal_negative_flags.append(1.0 if preterminal_negative else 0.0)

            if symmetry_ratios:
                features["t_symmetry"] = float(np.mean(symmetry_ratios))
            if t_peak_to_end_intervals:
                features["t_peak_to_end_interval"] = float(np.mean(t_peak_to_end_intervals))
            if biphasic_flags:
                features["t_biphasic"] = float(np.mean(biphasic_flags))
            if preterminal_negative_flags:
                features["t_preterminal_negative"] = float(np.mean(preterminal_negative_flags))

        # ST Segment Features
        # Use R-offset (J-point) and T-onset when available, otherwise fall back to S-peak based method
        features["st_level"] = np.nan
        features["st_slope"] = np.nan
        features["st60_level"] = np.nan
        features["st80_level"] = np.nan
        features["st_area"] = np.nan
        if n_s_peaks and r_peaks is not None:
            st_elevations = []
            st_slopes = []
            st60_levels = []
            st80_levels = []
            st_areas = []

            # Determine if we can use J-point based method
            use_j_point = n_j_points > 0 and n_t_onsets > 0

            for i, s_peak in enumerate(s_peaks):
                if np.isnan(s_peak):
                    continue

                s_idx = int(s_peak)
                beat_baseline = MorphologicalExtractor._get_beat_baseline(
                    s_idx, r_peaks, baseline_per_beat, global_baseline
                )

                # Try to use J-point based method first
                st_start = None
                st_end = None
                j_point_idx = s_idx  # Default to S-peak

                if use_j_point and i < n_j_points and i < n_t_onsets and j_points is not None and t_onsets is not None:
                    j_point = j_points[i]
                    t_onset = t_onsets[i]

                    if not np.isnan(j_point) and not np.isnan(t_onset):
                        j_point_idx = int(j_point)
                        t_onset_idx = int(t_onset)

                        # ST segment from J-point to T-onset
                        if j_point_idx < t_onset_idx and j_point_idx < n_samples:
                            st_start = j_point_idx
                            st_end = min(n_samples, t_onset_idx)

                # Fallback to S-peak based method if J-point method not available
                if st_start is None or st_end is None:
                    # Use S-peak + 20ms to S-peak + 80ms as fallback
                    st_start = s_idx + int(0.02 * sfreq)
                    st_end = min(n_samples, s_idx + int(0.08 * sfreq))
                    j_point_idx = s_idx  # Use S-peak as approximation

                if st_end > st_start and st_start < n_samples:
                    st_segment = ch_data[st_start:st_end]

                    st_level = np.mean(st_segment) - beat_baseline
                    st_elevations.append(st_level)

                    if len(st_segment) > 1:
                        slope = (st_segment[-1] - st_segment[0]) / len(st_segment)
                        st_slopes.append(slope)

                    # ST measurements at J+60ms and J+80ms (or S+60ms/S+80ms as fallback)
                    st60_idx = min(n_samples, j_point_idx + int(0.06 * sfreq))
                    if st60_idx < n_samples:
                        st60_levels.append(ch_data[st60_idx] - beat_baseline)

                    st80_idx = min(n_samples, j_point_idx + int(0.08 * sfreq))
                    if st80_idx < n_samples:
                        st80_levels.append(ch_data[st80_idx] - beat_baseline)

                    st_segment_relative = st_segment - beat_baseline
                    st_areas.append(np.trapezoid(st_segment_relative))

            if st_elevations:
                features["st_level"] = float(np.mean(st_elevations))
            if st_slopes:
                features["st_slope"] = float(np.mean(st_slopes))
            if st60_levels:
                features["st60_level"] = float(np.mean(st60_levels))
            if st80_levels:
                features["st80_level"] = float(np.mean(st80_levels))
            if st_areas:
                features["st_area"] = float(np.mean(st_areas))

        # QTc (Corrected QT interval using Bazett's formula)
        # QTc = QT / √(RR) where QT is in ms and RR is in seconds
        features["qtc_bazett"] = np.nan
        features["qtc_fridericia"] = np.nan
        if "qt_interval_mean" in features and "rr_interval_mean" in features:
            qt_ms = features["qt_interval_mean"]
            rr_sec = features["rr_interval_mean"]
            if rr_sec > 0:
                features["qtc_bazett"] = qt_ms / np.sqrt(rr_sec)
                # Fridericia formula: QTc = QT / (RR^(1/3))
                features["qtc_fridericia"] = qt_ms / math.cbrt(rr_sec)
            else:
                features["qtc_bazett"] = qt_ms  # Fallback if RR is invalid
                features["qtc_fridericia"] = qt_ms  # Fallback if RR is invalid

        # Interval Ratios
        features["qt_rr_ratio"] = np.nan
        features["pr_rr_ratio"] = np.nan
        features["t_qt_ratio"] = np.nan
        features["t_r_ratio"] = np.nan
        features["q_to_r_ratio"] = np.nan

        rr_mean = features.get("rr_interval_mean")
        rr_ms = rr_mean * 1000 if rr_mean is not None and not np.isnan(rr_mean) else None

        # QT/RR ratio
        if "qt_interval_mean" in features and rr_ms is not None and rr_ms > 0:
            features["qt_rr_ratio"] = features["qt_interval_mean"] / rr_ms

        # PR/RR ratio (using pq_interval as PR interval)
        if "pq_interval_mean" in features and rr_ms is not None and rr_ms > 0:
            features["pr_rr_ratio"] = features["pq_interval_mean"] / rr_ms

        # T/QT ratio
        if "t_duration_mean" in features and "qt_interval_mean" in features:
            qt_ms = features["qt_interval_mean"]
            if qt_ms > 0:
                features["t_qt_ratio"] = features["t_duration_mean"] / qt_ms

        # T/R amplitude ratio
        if "t_amplitude_mean" in features and "r_amplitude_mean" in features:
            r_amp = features["r_amplitude_mean"]
            t_amp = features["t_amplitude_mean"]
            if r_amp is not None and abs(r_amp) > 0:
                features["t_r_ratio"] = float(t_amp / r_amp)

        # Q/R amplitude ratio
        if "q_amplitude_mean" in features and "r_amplitude_mean" in features:
            r_amp = features["r_amplitude_mean"]
            q_amp = features["q_amplitude_mean"]
            if r_amp is not None and abs(r_amp) > 0 and q_amp is not None:
                features["q_to_r_ratio"] = float(abs(q_amp) / abs(r_amp))

        return features

    @staticmethod
    def _calculate_interval_stats(
        peaks1: np.ndarray[tuple[int], np.dtype[np.float64]],
        peaks2: np.ndarray[tuple[int], np.dtype[np.float64]],
        sfreq: float,
        max_interval_ms: float | None = None,
        min_interval_ms: float | None = None,
    ) -> dict[str, float]:
        """Interval statistics calculation with intelligent peak pairing.

        Pairs peaks from peaks1 with peaks from peaks2 by finding the closest valid match.
        Handles edge cases where recordings start or end mid-wave, resulting in unpaired peaks.

        Args:
            peaks1: First wave peaks (e.g., Q peaks, P onsets)
            peaks2: Second wave peaks (e.g., S peaks, P offsets)
            sfreq: Sampling frequency in Hz
            max_interval_ms: Maximum allowed interval in milliseconds. If specified,
                only pairs where (peaks2 - peaks1) <= max_interval_ms will be considered.
                Defaults to None (no limit).
            min_interval_ms: Minimum allowed interval in milliseconds. If specified,
                only pairs where (peaks2 - peaks1) >= min_interval_ms will be considered.
                Defaults to None (no limit).

        Returns:
            Dictionary with mean, median, std in milliseconds, or empty dict if no valid intervals
        """
        intervals_ms = _pair_peaks(peaks1, peaks2, sfreq, max_interval_ms, min_interval_ms)

        if intervals_ms.size == 0:
            return {}

        return {
            "mean": float(np.mean(intervals_ms)),
            "median": float(np.median(intervals_ms)),
            "std": float(np.std(intervals_ms)),
        }


def _pair_peaks(
    peaks1: np.ndarray[tuple[int], np.dtype[np.float64]],
    peaks2: np.ndarray[tuple[int], np.dtype[np.float64]],
    sfreq: float,
    max_interval_ms: float | None = None,
    min_interval_ms: float | None = None,
) -> np.ndarray:
    """
    For each peak in peaks1, find the closest valid peak in peaks2.
    - peaks2 can be reused multiple times.
    - peaks must satisfy: peak2 > peak1 and (optionally) min_interval_ms <= interval <= max_interval_ms.
    - NaNs in either input are ignored.
    Returns:
        interval_ms (1D numpy array; rows with no valid candidate are dropped)
    """
    if peaks1.size == 0 or peaks2.size == 0:
        return np.array([])

    mask1 = ~np.isnan(peaks1)
    mask2 = ~np.isnan(peaks2)
    p1v = peaks1[mask1]
    p2v = peaks2[mask2]

    if p1v.size == 0 or p2v.size == 0:
        return np.array([])

    intervals_ms = (p2v[None, :] - p1v[:, None]) / sfreq * 1000.0
    cand_mask = intervals_ms > 0

    if min_interval_ms is not None:
        cand_mask &= intervals_ms >= min_interval_ms

    if max_interval_ms is not None:
        cand_mask &= intervals_ms <= max_interval_ms

    intervals_ms_valid = np.where(cand_mask, intervals_ms, np.inf)
    best_j = np.argmin(intervals_ms_valid, axis=1)
    best_intervals = intervals_ms_valid[np.arange(p1v.size), best_j]
    valid_rows = np.isfinite(best_intervals)
    return best_intervals[valid_rows]
