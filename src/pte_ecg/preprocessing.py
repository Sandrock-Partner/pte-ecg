"""Preprocessing module for ECG signal processing.

This module provides functionality for preprocessing ECG signals, including:
- Resampling
- Bandpass filtering
- Notch filtering
- Normalization

The module is designed to work with multi-channel ECG data and provides a
configurable preprocessing pipeline.
"""

import warnings
from typing import Literal

import numpy as np
import pydantic
from pydantic import Field
from scipy import signal
from scipy.signal import resample

from ._logging import logger


class ResampleArgs(pydantic.BaseModel):
    """Settings for resampling the ECG signal.

    Attributes:
        enabled: Whether to apply resampling.
        sfreq_new: New sampling frequency in Hz. If None, no resampling is performed.
    """

    enabled: bool = False
    sfreq_new: int | float | None = None


class BandpassArgs(pydantic.BaseModel):
    """Settings for bandpass filtering the ECG signal.

    Attributes:
        enabled: Whether to apply bandpass filtering.
        l_freq: Lower cutoff frequency in Hz. If None, no high-pass filtering is applied.
        h_freq: Higher cutoff frequency in Hz. If None, no low-pass filtering is applied.
    """

    enabled: bool = False
    l_freq: int | float | None = 0.5
    h_freq: int | float | None = None


class NotchArgs(pydantic.BaseModel):
    """Settings for notch filtering the ECG signal.

    Attributes:
        enabled: Whether to apply notch filtering.
        freq: Frequency to notch filter in Hz. If None, no notch filtering is applied.
    """

    enabled: bool = False
    freq: int | float | None = None


class NormalizeArgs(pydantic.BaseModel):
    """Settings for normalizing the ECG signal.

    Attributes:
        enabled: Whether to apply normalization.
        mode: Normalization method to use. One of:
            - 'mean': Subtract mean of each channel
            - 'ratio': Divide by mean of each channel
            - 'logratio': Log of ratio
            - 'percent': Scale to percentage of mean
            - 'zscore': Standard score (z-score) normalization
            - 'zlogratio': Z-score of log ratio
    """

    enabled: bool = False
    mode: Literal["mean", "ratio", "logratio", "percent", "zscore", "zlogratio"] = (
        "zscore"
    )


class PreprocessingSettings(pydantic.BaseModel):
    """Container for all preprocessing settings.

    Attributes:
        enabled: Whether to apply any preprocessing.
        resample: Settings for resampling.
        bandpass: Settings for bandpass filtering.
        notch: Settings for notch filtering.
        normalize: Settings for normalization.
    """

    enabled: bool = True
    resample: ResampleArgs = Field(default_factory=ResampleArgs)
    bandpass: BandpassArgs = Field(default_factory=BandpassArgs)
    notch: NotchArgs = Field(default_factory=NotchArgs)
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)


def preprocess(
    ecg: np.ndarray, sfreq: float, preprocessing: PreprocessingSettings
) -> tuple[np.ndarray, float]:
    """Apply preprocessing steps to ECG data.

    This function applies the following preprocessing steps in order:
    1. Resampling (if enabled)
    2. Bandpass filtering (if enabled)
    3. Notch filtering (if enabled)
    4. Normalization (if enabled)

    Args:
        ecg: Input ECG data with shape (n_samples, n_channels, n_timepoints).
        sfreq: Sampling frequency of the input data in Hz.
        preprocessing: Preprocessing settings.

    Returns:
            Tuple containing:
            - Processed ECG data with the same shape as input.
            - Updated sampling frequency (may change if resampling is applied).

    Raises:
        ValueError: If input data has invalid dimensions or preprocessing settings are invalid.
        RuntimeError: If preprocessing fails.
    """

    if sfreq <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {sfreq}")

    if not isinstance(ecg, np.ndarray) or ecg.ndim != 3:
        raise ValueError(
            "ECG data must be a 3D numpy array with shape (n_samples, n_channels, n_timepoints)"
        )

    if ecg.shape[-1] < ecg.shape[-2]:
        warnings.warn(
            "ECG data must be a 3D numpy array with shape (n_samples, n_channels, n_timepoints)."
            f"ECG data may have more channels than timepoints. Got shape: {ecg.shape}. Reshaping data."
        )
        ecg = ecg.transpose(0, 2, 1)

    if not preprocessing.enabled:
        logger.debug("Preprocessing is disabled, returning original data.")
        return ecg, sfreq

    ecg = _check_flats(ecg, drop_flat_recs=True)

    n_samples, n_channels, n_timepoints = ecg.shape
    logger.info(
        f"Starting preprocessing of {n_samples} samples with {n_channels} channels "
        f"and {n_timepoints} timepoints at {sfreq} Hz"
    )
    sfreq_new = sfreq
    if preprocessing.resample.enabled and preprocessing.resample.sfreq_new is not None:
        sfreq_new = preprocessing.resample.sfreq_new
        logger.info(f"Resampling from {sfreq} Hz to {sfreq_new} Hz")
        # Calculate new number of samples
        new_n_timepoints = int(n_timepoints * sfreq_new / sfreq)
        # Resample each sample and channel
        ecg_resampled = np.zeros((n_samples, n_channels, new_n_timepoints))
        for i in range(n_samples):
            for j in range(n_channels):
                ecg_resampled[i, j, :] = resample(ecg[i, j, :], new_n_timepoints)
        ecg = ecg_resampled
        n_timepoints = new_n_timepoints

    if preprocessing.bandpass.enabled:
        # Apply bandpass filter using scipy
        l_freq = preprocessing.bandpass.l_freq
        h_freq = preprocessing.bandpass.h_freq

        if l_freq is not None or h_freq is not None:
            nyquist = sfreq_new / 2

            # Design filter
            if l_freq is not None and h_freq is not None:
                # Bandpass filter
                low = l_freq / nyquist
                high = h_freq / nyquist
                b, a = signal.butter(4, [low, high], btype="band")
            elif l_freq is not None:
                # High-pass filter
                low = l_freq / nyquist
                b, a = signal.butter(4, low, btype="high")
            else:
                # Low-pass filter
                high = h_freq / nyquist
                b, a = signal.butter(4, high, btype="low")

            # Apply filter to each sample and channel
            for i in range(n_samples):
                for j in range(n_channels):
                    ecg[i, j, :] = signal.filtfilt(b, a, ecg[i, j, :])
    if preprocessing.notch.enabled and preprocessing.notch.freq is not None:
        # Apply notch filter using scipy
        freq = preprocessing.notch.freq
        nyquist = sfreq_new / 2

        # Design notch filter (band-stop filter)
        # Use a narrow band around the target frequency
        low = (freq - 1) / nyquist
        high = (freq + 1) / nyquist
        b, a = signal.butter(4, [low, high], btype="bandstop")

        # Apply filter to each sample and channel
        for i in range(n_samples):
            for j in range(n_channels):
                ecg[i, j, :] = signal.filtfilt(b, a, ecg[i, j, :])
    ecg = ecg.reshape(n_samples, -1)
    if preprocessing.normalize.enabled:
        # Apply normalization using custom implementation
        mode = preprocessing.normalize.mode

        if mode == "mean":
            # Subtract mean of each channel
            ecg = ecg - np.mean(ecg, axis=-1, keepdims=True)
        elif mode == "ratio":
            # Divide by mean of each channel
            mean_vals = np.mean(ecg, axis=-1, keepdims=True)
            mean_vals = np.where(mean_vals == 0, 1, mean_vals)  # Avoid division by zero
            ecg = ecg / mean_vals
        elif mode == "logratio":
            # Log of ratio
            mean_vals = np.mean(ecg, axis=-1, keepdims=True)
            mean_vals = np.where(
                mean_vals <= 0, 1e-10, mean_vals
            )  # Avoid log of zero/negative
            ecg = np.log(ecg / mean_vals)
        elif mode == "percent":
            # Scale to percentage of mean
            mean_vals = np.mean(ecg, axis=-1, keepdims=True)
            mean_vals = np.where(mean_vals == 0, 1, mean_vals)  # Avoid division by zero
            ecg = (ecg / mean_vals) * 100
        elif mode == "zscore":
            # Standard score (z-score) normalization
            mean_vals = np.mean(ecg, axis=-1, keepdims=True)
            std_vals = np.std(ecg, axis=-1, keepdims=True)
            std_vals = np.where(std_vals == 0, 1, std_vals)  # Avoid division by zero
            ecg = (ecg - mean_vals) / std_vals
        elif mode == "zlogratio":
            # Z-score of log ratio
            mean_vals = np.mean(ecg, axis=-1, keepdims=True)
            mean_vals = np.where(
                mean_vals <= 0, 1e-10, mean_vals
            )  # Avoid log of zero/negative
            log_ratio = np.log(ecg / mean_vals)
            mean_log = np.mean(log_ratio, axis=-1, keepdims=True)
            std_log = np.std(log_ratio, axis=-1, keepdims=True)
            std_log = np.where(std_log == 0, 1, std_log)  # Avoid division by zero
            ecg = (log_ratio - mean_log) / std_log
    ecg = ecg.reshape(n_samples, n_channels, n_timepoints)
    return ecg, sfreq_new


def _check_flats(ecg: np.ndarray, drop_flat_recs: bool) -> np.ndarray:
    are_flat_chs = np.all(np.isclose(ecg, ecg[..., 0:1]), axis=-1)
    n_flats = np.sum(are_flat_chs)
    if n_flats == ecg.shape[0] * ecg.shape[1]:
        raise ValueError(
            f"All channels of all recordings are flat lines ({n_flats}). Check your data"
        )
    if n_flats > 0:
        logger.warning(
            f"{n_flats} channels of {ecg.shape[0]} recordings are flat lines."
        )
    are_empty_recordings = np.all(are_flat_chs, axis=-1)
    n_empty_recordings = np.sum(are_empty_recordings)
    empty_recordings = np.where(are_empty_recordings)[0]
    if n_empty_recordings > 0 and drop_flat_recs:
        logger.warning(
            f"Discarding {n_empty_recordings} recordings with flat lines in all channels."
            f" Recording indices: {empty_recordings}."
        )
        ecg = ecg[~are_empty_recordings]
    return ecg
