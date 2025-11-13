"""Visualize ECG peak detection on 12-lead ECG with traditional ECG print layout.

This script loads ECG data from an H5 file and visualizes peak detection
results across all 12 channels in a layout resembling clinical ECG printouts.
"""

from __future__ import annotations

import multiprocessing
import pathlib
import warnings
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy.signal

from pte_ecg._logging import logger

# Peak detection methods (from morphological.py)
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
]

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

# Standard 12-lead ECG channel names
CHANNEL_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def main():
    """Main function to load and visualize ECG data."""
    # Default H5 file path - adjust to your data location
    h5_path = list(pathlib.Path(r"C:\AI_Data\approach3A\v4.1.6").glob("*.h5"))[0]
    print(f"Loading ECG data from: {h5_path}")

    ecg_data_all, sfreq = load_ecg_from_h5(h5_path)

    print(f"Loaded ECG data: shape={ecg_data_all.shape}, sfreq={sfreq} Hz")

    n_patients = 20
    output_dir = pathlib.Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Create a single pool to reuse across all patients
    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
        for patient_idx, ecg_data in enumerate(ecg_data_all[:n_patients]):
            print(f"Patient index: {patient_idx}")

            save_path = output_dir / f"ecg_patient_{patient_idx}_peaks.png"

            plot_12_lead_ecg(
                ecg_data=ecg_data,
                sfreq=sfreq,
                patient_idx=patient_idx,
                duration=10.0,  # Plot first 10 seconds
                save_path=save_path,
                block=False,
                pool=pool,
            )

    print(f"\nProcessed {n_patients} patients. Figures saved to: {output_dir}")


def detect_r_peaks(ch_data: np.ndarray, sfreq: float) -> tuple[np.ndarray | None, int, PeakMethod | None]:
    """Detect R-peaks using multiple methods with automatic fallback.

    Args:
        ch_data: Single channel ECG data with shape (n_timepoints,)
        sfreq: Sampling frequency in Hz

    Returns:
        Tuple of (r_peaks array, number of peaks, method used)
    """
    peaks_per_method: dict[PeakMethod, np.ndarray] = {}
    max_n_peaks = 0

    for method in _METHODS_FINDPEAKS:
        _, peaks_info = nk.ecg_peaks(
            ch_data,
            sampling_rate=np.rint(sfreq).astype(int) if method in ["zong", "emrich2023"] else sfreq,
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

    return None, 0, None


def delineate_waves(ch_data: np.ndarray, r_peaks: np.ndarray, sfreq: float, n_r_peaks: int) -> dict:
    """Delineate P, Q, S, T waves using neurokit2.

    Args:
        ch_data: Single channel ECG data
        r_peaks: R-peak indices
        sfreq: Sampling frequency in Hz
        n_r_peaks: Number of R-peaks detected

    Returns:
        Dictionary with wave indices for each wave type
    """
    methods = ["dwt", "prominence", "peak", "cwt"]
    waves_dict = {}

    for method in methods:
        if n_r_peaks < 2 and method in {"prominence", "cwt"}:
            logger.info(f"Not enough R-peaks ({n_r_peaks}) for {method} method.")
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", nk.misc.NeuroKitWarning)
                warnings.simplefilter("ignore", scipy.signal._peak_finding_utils.PeakPropertyWarning)
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
                continue
            raise
        except Exception as e:
            logger.warning(f"Delineation failed with method '{method}': {e}")
            continue

    return waves_dict


def load_ecg_from_h5(h5_path: pathlib.Path) -> tuple[np.ndarray, float]:
    """Load ECG data from H5 file.

    Args:
        h5_path: Path to H5 file

    Returns:
        Tuple of (ecg_data, sampling_frequency)
        ecg_data has shape (n_channels, n_timepoints)
    """
    with h5py.File(h5_path, "r") as root:
        sequences = np.array(root["ecg_data/sequences"], dtype=float)
        sfreq = root["ecg_data"].attrs["sampling_rate"]

    return sequences.transpose((0, 2, 1)), sfreq


def _process_channel_peaks(args: tuple) -> tuple:
    """Worker function to process a single channel for parallel execution.

    Args:
        args: Tuple of (ch_idx, ch_data, sfreq)

    Returns:
        Tuple of (ch_idx, r_peaks, n_r_peaks, method, waves_dict)
    """
    ch_idx, ch_data, sfreq = args

    # Detect R-peaks
    r_peaks, n_r_peaks, method = detect_r_peaks(ch_data, sfreq)

    # Delineate waves if R-peaks found
    waves_dict = {}
    if r_peaks is not None and len(r_peaks) > 0:
        waves_dict = delineate_waves(ch_data, r_peaks, sfreq, n_r_peaks)

    return (ch_idx, r_peaks, n_r_peaks, method, waves_dict)


def plot_12_lead_ecg(
    ecg_data: np.ndarray,
    sfreq: float,
    patient_idx: int = 0,
    duration: float = 10.0,
    save_path: pathlib.Path | None = None,
    block: bool = True,
    pool: multiprocessing.Pool | None = None,
):
    """Plot 12-lead ECG with detected peaks in traditional ECG print style.

    Args:
        ecg_data: ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        patient_idx: Patient index for title
        duration: Duration to plot in seconds (None = all)
        save_path: Optional path to save figure
        block: Whether to block and show the plot (True) or close it (False)
        pool: Optional multiprocessing pool to reuse (creates new one if None)
    """
    n_channels, n_timepoints = ecg_data.shape

    if n_channels != 12:
        print(f"Warning: Expected 12 channels, got {n_channels}")

    # Limit duration if requested
    if duration is not None:
        n_samples = int(duration * sfreq)
        n_samples = min(n_samples, n_timepoints)
        ecg_data = ecg_data[:, :n_samples]
        n_timepoints = n_samples

    # Create time array
    time = np.arange(n_timepoints) / sfreq

    # Traditional ECG layout: 3 rows x 4 columns
    # Row 1: I, aVR, V1, V4
    # Row 2: II, aVL, V2, V5
    # Row 3: III, aVF, V3, V6
    # I, aVR, V1, V4, II, aVL, V2, V5, III, aVF, V3, V6
    lead_order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    # Calculate global y-axis limits for consistent scaling
    global_min = np.min(ecg_data)
    global_max = np.max(ecg_data)
    y_margin = (global_max - global_min) * 0.1
    ylim = (global_min - y_margin, global_max + y_margin)

    # Process all 12 channels in parallel
    channel_args = [(ch_idx, ecg_data[ch_idx], sfreq) for ch_idx in range(n_channels)]

    if pool is not None:
        # Reuse existing pool
        results = pool.map(_process_channel_peaks, channel_args)
    else:
        # Create temporary pool if none provided
        with multiprocessing.Pool(processes=min(12, multiprocessing.cpu_count())) as temp_pool:
            results = temp_pool.map(_process_channel_peaks, channel_args)

    # Store results in a dictionary for easy access
    peak_results = {
        ch_idx: (r_peaks, n_r_peaks, method, waves_dict) for ch_idx, r_peaks, n_r_peaks, method, waves_dict in results
    }

    # Create figure with 12 subplots (3 rows x 4 columns for traditional layout)
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    fig.suptitle(
        f"12-Lead ECG with Peak Detection - Patient {patient_idx}",
        fontsize=16,
        fontweight="bold",
    )

    # Helper to safely extract valid peak indices
    def get_valid_peaks(peak_data):
        if peak_data is None or len(peak_data) == 0:
            return np.array([], dtype=int)
        peak_array = np.asarray(peak_data)
        if peak_array.ndim == 0:
            return np.array([], dtype=int)
        valid_mask = ~np.isnan(peak_array)
        return peak_array[valid_mask].astype(int)

    # Plot each channel in traditional ECG order
    for plot_idx, ch_idx in enumerate(lead_order):
        if ch_idx >= n_channels:
            continue

        row = plot_idx // 4
        col = plot_idx % 4
        ax = axes[row, col]
        ch_data = ecg_data[ch_idx]

        # Get pre-computed results
        r_peaks, n_r_peaks, method, waves_dict = peak_results[ch_idx]

        # Plot ECG signal
        ax.plot(time, ch_data, "k-", linewidth=0.8, label="ECG")

        # Plot R-peaks
        if r_peaks is not None and len(r_peaks) > 0:
            ax.plot(
                time[r_peaks],
                ch_data[r_peaks],
                "ro",
                markersize=8,
                alpha=0.4,
                label=f"R-peaks (n={n_r_peaks}, {method})",
            )

            # Plot P-peaks
            p_peaks_valid = get_valid_peaks(waves_dict.get("ECG_P_Peaks"))
            if len(p_peaks_valid) > 0:
                ax.plot(
                    time[p_peaks_valid],
                    ch_data[p_peaks_valid],
                    "go",
                    markersize=6,
                    alpha=0.4,
                    label=f"P-peaks (n={len(p_peaks_valid)})",
                )

            # Plot Q-peaks
            q_peaks_valid = get_valid_peaks(waves_dict.get("ECG_Q_Peaks"))
            if len(q_peaks_valid) > 0:
                ax.plot(
                    time[q_peaks_valid],
                    ch_data[q_peaks_valid],
                    "bo",
                    markersize=6,
                    alpha=0.4,
                    label=f"Q-peaks (n={len(q_peaks_valid)})",
                )

            # Plot S-peaks
            s_peaks_valid = get_valid_peaks(waves_dict.get("ECG_S_Peaks"))
            if len(s_peaks_valid) > 0:
                ax.plot(
                    time[s_peaks_valid],
                    ch_data[s_peaks_valid],
                    "co",
                    markersize=6,
                    alpha=0.4,
                    label=f"S-peaks (n={len(s_peaks_valid)})",
                )

            # Plot T-peaks
            t_peaks_valid = get_valid_peaks(waves_dict.get("ECG_T_Peaks"))
            if len(t_peaks_valid) > 0:
                ax.plot(
                    time[t_peaks_valid],
                    ch_data[t_peaks_valid],
                    "mo",
                    markersize=6,
                    alpha=0.4,
                    label=f"T-peaks (n={len(t_peaks_valid)})",
                )

        # Formatting with consistent scaling
        lead_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"Ch{ch_idx}"
        ax.set_title(lead_name, fontsize=12, fontweight="bold", pad=8)

        # Apply consistent y-axis scaling
        ax.set_ylim(ylim)

        # Only show axis labels on bottom row and left column
        if row == 2:  # Bottom row
            ax.set_xlabel("Time (s)", fontsize=10)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        if col == 0:  # Left column
            ax.set_ylabel("Amplitude (mV)", fontsize=10)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # Grid styling to resemble ECG paper
        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.8,
            alpha=0.4,
            color="#FF9999",
        )
        ax.grid(
            True,
            which="minor",
            linestyle="-",
            linewidth=0.3,
            alpha=0.2,
            color="#FFCCCC",
        )
        ax.minorticks_on()
        ax.set_axisbelow(True)
        ax.set_facecolor("#FFF8F8")

        # Legend only if peaks detected
        if r_peaks is not None and len(r_peaks) > 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if block:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
