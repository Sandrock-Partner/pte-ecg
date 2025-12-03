"""Visualize ECG peak detection on 12-lead ECG with traditional ECG print layout.

This script loads ECG data from an H5 file and visualizes peak detection
results across all 12 channels in a layout resembling clinical ECG printouts.
"""

from __future__ import annotations

import argparse
import cProfile
import multiprocessing
import pathlib
import pstats
from typing import Literal

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import neurokit2 as nk
import numpy as np
import pandas as pd

import pte_ecg
from pte_ecg._logging import logger

# Import PeakMethod type from pte_ecg
from pte_ecg.feature_extractors.morphological import PeakMethod

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
    "V6"
]


def main():
    """Main function to load and visualize ECG data."""
    # Default H5 file path - adjust to your data location
    h5_path = list(pathlib.Path(r"C:\AI_Data\approach3A\v4.1.9").glob("*.h5"))[0]
    n_recordings = 20
    print(f"Loading ECG data from: {h5_path}")

    data = load_ecg_from_h5(h5_path, n_recordings)
    ecg_data_all, sfreq, test_ids, m1ziffer = data

    print(f"Loaded ECG data: shape={ecg_data_all.shape}, sfreq={sfreq} Hz")
    print(f"Total samples in H5: {len(test_ids)}")

    # Create feature extraction settings
    settings = create_feature_settings()
    n_jobs = 8
    output_dir = pathlib.Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    n_patients = 0
    feature_names: set[str] = {}
    with multiprocessing.Pool(processes=min(n_jobs, multiprocessing.cpu_count())) as pool:
        for patient_idx, (ecg_data, test_id, m1z) in enumerate(
            zip(ecg_data_all, test_ids, m1ziffer, strict=True)
        ):
            print(f"Patient {patient_idx}: TestID={test_id}, M1Ziffer={m1z}")
            n_patients = patient_idx + 1

            save_path = output_dir / f"ecg_patient_{patient_idx}_peaks.png"
            features_df = pte_ecg.get_features(ecg_data[np.newaxis, :, :], sfreq=sfreq, settings=settings)
            new_feature_names = set(features_df.columns)
            if feature_names and not new_feature_names == feature_names:
                raise ValueError(f"Feature names changed. Difference: {new_feature_names ^ feature_names}")
            feature_names = new_feature_names
            st_levels = get_st_levels(features_df, 0)  # Index 0 since we have only one sample
            baseline_medians = get_baseline_medians(features_df, 0)  # Index 0 since we have only one sample

            plot_12_lead_ecg(
                ecg_data=ecg_data,
                sfreq=sfreq,
                patient_idx=patient_idx,
                test_id=test_id,
                features_df=features_df,
                sample_idx=0,  # First (and only) sample in the batch
                st_levels=st_levels,
                baseline_medians=baseline_medians,
                duration=10.0,  # Plot first 10 seconds
                save_path=save_path,
                block=False,
                pool=pool,
                j_point_offset_ms=settings.features.morphological["j_point_offset_ms"]
            )

    print(f"\nProcessed {n_patients} patients. Figures saved to: {output_dir}")


def delineate_waves(ch_data: np.ndarray, r_peaks: np.ndarray, sfreq: float, n_r_peaks: int, j_point_offset_ms: float = 0.0):
    """Delineate P, Q, S, T waves using the shared pte_ecg.ecg_delineate function.

    Args:
        ch_data: Single channel ECG data
        r_peaks: R-peak indices
        sfreq: Sampling frequency in Hz
        n_r_peaks: Number of R-peaks detected (unused, kept for compatibility)

    Returns:
        Waves object with wave indices for each wave type, or None if failed
    """
    if r_peaks is None or len(r_peaks) == 0:
        return None
    
    try:
        return pte_ecg.ecg_delineate(ch_data, r_peaks, sfreq, j_point_offset_ms=j_point_offset_ms)
    except pte_ecg.ECGDelineationError as e:
        logger.warning(f"ECG delineation failed: {e}")
        return None


def load_ecg_from_h5(h5_path: pathlib.Path, n_recordings: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Load ECG data from H5 file.

    Args:
        h5_path: Path to H5 file

    Returns:
        Tuple of (ecg_data, sampling_frequency, test_ids, m1ziffer, ecg_type)
        ecg_data has shape (n_samples, n_channels, n_timepoints)
        test_ids is an array of TestID strings for each sample
        m1ziffer is an array of M1Ziffer values for each sample
    """
    with h5py.File(h5_path, "r") as root:
        sequences = np.array(root["ecg_data/sequences"], dtype=float)
        sfreq = root["ecg_data"].attrs["sampling_rate"]
        
        # Load metadata
        metadata = root["ecg_data/metadata"]
        test_ids = metadata["TestID"][()]
        m1ziffer = metadata["M1Ziffer"][()]
        ecg_type = metadata["ecg_type"][()]

    
    # Filter for resting ECG type only
    resting_mask = ecg_type == b"resting"  # Note: might be bytes
    if not np.any(resting_mask):
        # Try string comparison if bytes comparison failed
        resting_mask = ecg_type == "resting"
    
    sequences = sequences.transpose((0, 2, 1))
    sequences = sequences[resting_mask][:n_recordings]
    test_ids = test_ids[resting_mask][:n_recordings]
    m1ziffer = m1ziffer[resting_mask][:n_recordings]

    return sequences, sfreq, test_ids, m1ziffer


def create_feature_settings() -> pte_ecg.Settings:
    """Create feature extraction settings from configuration dictionary.

    Returns:
        Settings object configured for morphological feature extraction
    """
    config = {
        "preprocessing": {
            "enabled": True,
            "resample": {
                "enabled": True,
                "sfreq_new": 80
            },
            "bandpass": {
                "enabled": True,
                "l_freq": 0.5,
                "h_freq": None
            },
            "notch": {
                "enabled": False,
                "freq": None
            },
            "normalize": {
                "enabled": False,
                "mode": "zscore"
            },
            "drop_mode": "any",
        },
        "features": {
            "morphological": {
                "enabled": True,
                "j_point_offset_ms": 20.0
            }
        },
        "lead_order": [
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
            "V6"
        ]
    }
    
    # Create Settings object from dictionary
    settings = pte_ecg.Settings(**config)
    return settings


def get_st_levels(features_df: pd.DataFrame, sample_idx: int) -> dict[str, float] | None:
    """Extract ST elevation values for a specific sample.

    Args:
        features_df: DataFrame with morphological features
        sample_idx: Sample index in the features DataFrame

    Returns:
        Dictionary mapping channel name to ST elevation value, or None if not found
    """
    if features_df is None or sample_idx >= len(features_df):
        return None
    
    row = features_df.iloc[sample_idx]
    
    # Extract ST elevation for each channel
    st_levels = {}
    for channel in CHANNEL_NAMES:
        col_name = f"morphological_st_level_{channel}"
        if col_name in row.index:
            st_levels[channel] = row[col_name] / 1000
    
    return st_levels


def get_baseline_medians(features_df: pd.DataFrame, sample_idx: int) -> dict[str, float] | None:
    """Extract baseline median values for a specific sample.

    Args:
        features_df: DataFrame with morphological features
        sample_idx: Sample index in the features DataFrame

    Returns:
        Dictionary mapping channel name to baseline median value, or None if not found
    """
    if features_df is None or sample_idx >= len(features_df):
        return None
    
    row = features_df.iloc[sample_idx]
    
    # Extract baseline median for each channel
    baseline_medians = {}
    for channel in CHANNEL_NAMES:
        col_name = f"morphological_baseline_median_{channel}"
        if col_name in row.index:
            baseline_medians[channel] = row[col_name]
    
    return baseline_medians


def extract_feature_summary(features_df: pd.DataFrame, sample_idx: int, sfreq: float) -> dict:
    """Extract summary statistics from features DataFrame.

    Args:
        features_df: DataFrame with morphological features
        sample_idx: Sample index in the features DataFrame
        sfreq: Sampling frequency in Hz

    Returns:
        Dictionary with average values and per-lead statistics
    """
    if features_df is None or sample_idx >= len(features_df):
        return {}
    
    row = features_df.iloc[sample_idx]
    summary = {
        "averages": {},
        "per_lead": {channel: {} for channel in CHANNEL_NAMES}
    }
    
    # Extract feature types to average
    feature_types = [
        ("p_duration_mean", "P Duration"),
        ("pq_interval_mean", "PQ Interval"),
        ("qrs_duration_mean", "QRS Duration"),
        ("qt_interval_mean", "QT Interval"),
        ("qtc_bazett", "QTc Bazett"),
        ("qtc_fridericia", "QTc Fridericia"),
        ("rr_interval_mean", "RR Interval Mean"),
        ("st_level", "ST Elevation"),
        ("baseline_median", "Baseline Median"),
    ]
    
    # Calculate averages across channels
    for feature_name, display_name in feature_types:
        values = []
        for channel in CHANNEL_NAMES:
            col_name = f"morphological_{feature_name}_{channel}"
            if col_name in row.index and not pd.isna(row[col_name]):
                value = row[col_name]
                if feature_name == "st_level":
                    value = value / 1000
                values.append(value)
                summary["per_lead"][channel][display_name] = value
            else:
                summary["per_lead"][channel][display_name] = np.nan
        
        if values:
            summary["averages"][display_name] = np.mean(values)
    
    # Convert RR intervals to heart rate (RR interval is already in seconds from feature extractor)
    if "RR Interval Mean" in summary["averages"]:
        rr_seconds = summary["averages"]["RR Interval Mean"]
        summary["averages"]["HR from RR Mean"] = 60.0 / rr_seconds if rr_seconds > 0 else np.nan
        
        # Also convert per-lead
        for channel in CHANNEL_NAMES:
            if "RR Interval Mean" in summary["per_lead"][channel]:
                rr_seconds = summary["per_lead"][channel]["RR Interval Mean"]
                summary["per_lead"][channel]["HR from RR Mean"] = 60.0 / rr_seconds if rr_seconds > 0 else np.nan
    
    return summary


def is_pathological(key: str, value: float) -> bool:
    """Check if a feature value is pathological.

    Args:
        key: Feature name
        value: Feature value

    Returns:
        True if the value is pathological
    """
    if pd.isna(value):
        return False
    
    # Define pathological thresholds
    thresholds = {
        "ST Elevation": lambda v: abs(v) > 0.1,  # >0.1 mV or <-0.1 mV
        "QTc Bazett": lambda v: v > 450 or v < 350,  # Prolonged or short QTc
        "QTc Fridericia": lambda v: v > 450 or v < 350,  # Prolonged or short QTc
        "QRS Duration": lambda v: v > 120,  # Wide QRS
        "PQ Interval": lambda v: v > 200 or v < 120,  # AV block or pre-excitation
        "HR from RR Mean": lambda v: v > 100 or v < 60,  # Tachycardia or bradycardia
    }
    
    if key in thresholds:
        return thresholds[key](value)
    return False


def format_statistics_text(summary: dict, channel: str | None = None) -> list[tuple[str, bool]]:
    """Format statistics as text for display with pathological highlighting.

    Args:
        summary: Dictionary with average values and per-lead data from extract_feature_summary
        channel: Channel name (e.g., "I", "II", "V1") or None for summary

    Returns:
        List of tuples (line_text, is_pathological) for each line
    """
    # Determine if we're formatting summary or per-lead data
    if channel is None:
        if not summary or "averages" not in summary:
            return [("No feature data available", False)]
        data = summary["averages"]
        title = "All-Channel Average"
    else:
        if not summary or "per_lead" not in summary or channel not in summary["per_lead"]:
            return [(f"Lead {channel}", False), ("="*22, False), ("No data available", False)]
        data = summary["per_lead"][channel]
        title = f"Lead {channel}"
    
    lines: list[tuple[str, bool]] = [(title, False), ("="*22, False)]
    
    # Heart Rate from RR intervals
    if "HR from RR Mean" in data:
        value = data["HR from RR Mean"]
        if pd.isna(value):
            value_str = f"{'N/A':>9}"
            pathological = False
        else:
            value_str = f"{value:>8.0f} bpm"
            pathological = is_pathological("HR from RR Mean", value)
        lines.append((f"HR (mean):{value_str}", pathological))
    
    # Intervals in ms
    intervals = [
        ("P Duration", "P Duration:"),
        ("PQ Interval", "PQ Interval:"),
        ("QRS Duration", "QRS Duration:"),
        ("QT Interval", "QT Interval:"),
        ("QTc Bazett", "QTc (Bazett):"),
        ("QTc Fridericia", "QTc (Frid.):"),
        ("ST Elevation", "ST Elevation:"),
        # ("Baseline Median", "Baseline Median:"),
    ]
    
    for key, label in intervals:
        if key in data:
            value = data[key]
            if key == "ST Elevation":
                unit = "mV"
            elif key == "Baseline Median":
                unit = "µV"
            else:
                unit = "ms"
            if pd.isna(value):
                lines.append((f"{label:<14}  N/A", False))
            else:
                value_str = f"{value:>5.2f}" if unit in ("mV", "µV") else f"{value:>5.0f}"
                pathological = is_pathological(key, value)
                lines.append((f"{label:<14}{value_str} {unit}", pathological))
    return lines


def _process_channel_peaks(ch_idx: int, ch_data: np.ndarray, sfreq: float, j_point_offset_ms: float = 0.0) -> tuple:
    """Worker function to process a single channel for parallel execution.

    Args:
        ch_idx: Channel index
        ch_data: Single channel ECG data
        sfreq: Sampling frequency
        j_point_offset_ms: Additional offset in milliseconds to add to detected J-points

    Returns:
        Tuple of (ch_idx, r_peaks, n_r_peaks, method, waves_obj)
        where waves_obj is a Waves object or None
    """
    # For aVR, invert signal for peak detection and delineation only
    lead_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"Ch{ch_idx}"
    ch_data_for_peaks = ch_data.copy() if lead_name == "aVR" else ch_data
    if lead_name == "aVR":
        ch_data_for_peaks, _ = nk.ecg_invert(ch_data_for_peaks, sampling_rate=sfreq, force=True)
    
    # Detect R-peaks per-channel (using inverted data for aVR)
    r_peaks, n_r_peaks, method = pte_ecg.detect_r_peaks(ch_data_for_peaks, sfreq)[0]

    # Delineate waves if R-peaks found (using inverted data for aVR)
    waves_obj = None
    if r_peaks is not None and len(r_peaks) > 0:
        waves_obj = delineate_waves(ch_data_for_peaks, r_peaks, sfreq, n_r_peaks, j_point_offset_ms)

    return (ch_idx, r_peaks, n_r_peaks, method, waves_obj)


def plot_12_lead_ecg(
    ecg_data: np.ndarray,
    sfreq: float,
    pool: multiprocessing.Pool,
    patient_idx: int = 0,
    test_id: str | None = None,
    features_df: pd.DataFrame | None = None,
    sample_idx: int = 0,
    st_levels: dict[str, float] | None = None,
    baseline_medians: dict[str, float] | None = None,
    duration: float = 10.0,
    j_point_offset_ms: float = 0.0,
    save_path: pathlib.Path | None = None,
    block: bool = True,
):
    """Plot 12-lead ECG with detected peaks in traditional ECG print style.

    Args:
        ecg_data: ECG data with shape (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        patient_idx: Patient index for title
        test_id: TestID for this recording
        features_df: DataFrame with morphological features
        sample_idx: Sample index in features DataFrame
        st_levels: Dictionary mapping channel name to ST elevation value
        baseline_medians: Dictionary mapping channel name to baseline median value
        duration: Duration to plot in seconds (None = all)
        save_path: Optional path to save figure
        j_point_offset_ms: Additional offset in milliseconds to add to detected J-points.
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

    # Convert ECG data from µV to mV for plotting
    ecg_data_mv = ecg_data / 1000.0

    # Traditional ECG layout: 3 rows x 4 columns
    # Row 1: I, aVR, V1, V4
    # Row 2: II, aVL, V2, V5
    # Row 3: III, aVF, V3, V6
    # I, aVR, V1, V4, II, aVL, V2, V5, III, aVF, V3, V6
    lead_order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    # Calculate global y-axis limits for consistent scaling (in mV)
    global_min = np.min(ecg_data_mv)
    global_max = np.max(ecg_data_mv)
    y_margin = (global_max - global_min) * 0.1
    ylim = (global_min - y_margin, global_max + y_margin)

    # Detect peaks per-channel
    channel_args = ((ch_idx, ecg_data[ch_idx], sfreq, j_point_offset_ms) for ch_idx in range(n_channels))

    results = pool.starmap(_process_channel_peaks, channel_args)
    
    # Store results in a dictionary for easy access
    peak_results = {
        ch_idx: (r_peaks, n_r_peaks, method, waves_obj) for ch_idx, r_peaks, n_r_peaks, method, waves_obj in results
    }

    # Extract feature summary if available
    summary = extract_feature_summary(features_df, sample_idx, sfreq) if features_df is not None else {}

    # Create figure with subfigures for better layout control
    # Top section for ECGs (3 rows x 4 cols), bottom section for 13 statistics boxes
    fig = plt.figure(figsize=(32, 14), constrained_layout=True)
    
    # Create subfigures: top for ECGs, bottom for statistics (reduced bottom height)
    subfigs = fig.subfigures(2, 1, height_ratios=[3.5, 0.8])
    
    title = f"12-Lead ECG with Peak Detection - Patient {patient_idx}"
    if test_id:
        title += f" (TestID: {test_id})"
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )
    
    # Create axes for ECG plots in top subfigure (3 rows x 4 columns)
    axes = subfigs[0].subplots(3, 4)
    axes = np.array(axes)

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
        ch_data = ecg_data_mv[ch_idx]  # Use mV data for plotting (original polarity)

        # Get pre-computed results
        r_peaks, n_r_peaks, method, waves_obj = peak_results[ch_idx]

        # Plot ECG signal
        ax.plot(time, ch_data, "k-", linewidth=0.8, label="_ECG")

        # Plot R-peaks
        if r_peaks is not None and len(r_peaks) > 0:
            ax.plot(
                time[r_peaks],
                ch_data[r_peaks],
                "ro",  # R-wave family color: red
                markersize=8,
                alpha=0.4,
                label=f"R-peaks (n={n_r_peaks}, {method})",
            )

            # P-wave family (green)
            p_peaks_valid = get_valid_peaks(waves_obj.p_peaks if waves_obj else None)
            if len(p_peaks_valid) > 0:
                ax.plot(
                    time[p_peaks_valid],
                    ch_data[p_peaks_valid],
                    "go",
                    markersize=6,
                    alpha=0.4,
                    label=f"P-peaks (n={len(p_peaks_valid)})",
                )

            p_onsets_valid = get_valid_peaks(waves_obj.p_onsets if waves_obj else None)
            if len(p_onsets_valid) > 0:
                ax.plot(
                    time[p_onsets_valid],
                    ch_data[p_onsets_valid],
                    "g^",  # Green upward triangle (P onset)
                    markersize=5,
                    alpha=0.6,
                    label=f"P-onsets (n={len(p_onsets_valid)})",
                )

            p_offsets_valid = get_valid_peaks(waves_obj.p_offsets if waves_obj else None)
            if len(p_offsets_valid) > 0:
                ax.plot(
                    time[p_offsets_valid],
                    ch_data[p_offsets_valid],
                    "gv",  # Green downward triangle (P offset)
                    markersize=5,
                    alpha=0.6,
                    label=f"P-offsets (n={len(p_offsets_valid)})",
                )

            # Q-wave family (blue)
            q_peaks_valid = get_valid_peaks(waves_obj.q_peaks if waves_obj else None)
            if len(q_peaks_valid) > 0:
                ax.plot(
                    time[q_peaks_valid],
                    ch_data[q_peaks_valid],
                    "bo",
                    markersize=6,
                    alpha=0.4,
                    label=f"Q-peaks (n={len(q_peaks_valid)})",
                )

            q_onsets_valid = get_valid_peaks(waves_obj.q_onsets if waves_obj else None)
            if len(q_onsets_valid) > 0:
                ax.plot(
                    time[q_onsets_valid],
                    ch_data[q_onsets_valid],
                    "b^",  # Blue upward triangle (Q onset)
                    markersize=5,
                    alpha=0.6,
                    label=f"Q-onsets (n={len(q_onsets_valid)})",
                )

            # q_offsets_valid = get_valid_peaks(waves_obj.q_offsets if waves_obj else None)
            # if len(q_offsets_valid) > 0:
            #     ax.plot(
            #         time[q_offsets_valid],
            #         ch_data[q_offsets_valid],
            #         "bv",  # Blue downward triangle (Q offset)
            #         markersize=5,
            #         alpha=0.6,
            #         label=f"Q-offsets (n={len(q_offsets_valid)})",
            #     )

            # S-wave family (cyan)
            s_peaks_valid = get_valid_peaks(waves_obj.s_peaks if waves_obj else None)
            if len(s_peaks_valid) > 0:
                ax.plot(
                    time[s_peaks_valid],
                    ch_data[s_peaks_valid],
                    "co",
                    markersize=6,
                    alpha=0.4,
                    label=f"S-peaks (n={len(s_peaks_valid)})",
                )

            # s_onsets_valid = get_valid_peaks(waves_obj.s_onsets if waves_obj else None)
            # if len(s_onsets_valid) > 0:
            #     ax.plot(
            #         time[s_onsets_valid],
            #         ch_data[s_onsets_valid],
            #         "c^",  # Cyan upward triangle (S onset)
            #         markersize=5,
            #         alpha=0.6,
            #         label=f"S-onsets (n={len(s_onsets_valid)})",
            #     )

            s_offsets_valid = get_valid_peaks(waves_obj.s_offsets if waves_obj else None)
            if len(s_offsets_valid) > 0:
                ax.plot(
                    time[s_offsets_valid],
                    ch_data[s_offsets_valid],
                    "cv",  # Cyan downward triangle (S offset)
                    markersize=5,
                    alpha=0.6,
                    label=f"S-offsets (n={len(s_offsets_valid)})",
                )

            j_points_valid = get_valid_peaks(waves_obj.j_points if waves_obj else None)
            if len(j_points_valid) > 0:
                ax.plot(
                    time[j_points_valid],
                    ch_data[j_points_valid],
                    "yo",
                    markersize=6,
                    alpha=0.7,
                    label=f"J-points (n={len(j_points_valid)})",
                )

            # R on/offsets (red)
            r_onsets_valid = get_valid_peaks(waves_obj.r_onsets if waves_obj else None)
            if len(r_onsets_valid) > 0:
                ax.plot(
                    time[r_onsets_valid],
                    ch_data[r_onsets_valid],
                    "r^",  # Red upward triangle (R-wave onset)
                    markersize=5,
                    alpha=0.6,
                    label=f"R-onsets (n={len(r_onsets_valid)})",
                )

            r_offsets_valid = get_valid_peaks(waves_obj.r_offsets if waves_obj else None)
            if len(r_offsets_valid) > 0:
                ax.plot(
                    time[r_offsets_valid],
                    ch_data[r_offsets_valid],
                    "rv",  # Red downward triangle (R-wave offset)
                    markersize=5,
                    alpha=0.5,
                    label=f"R-offsets (n={len(r_offsets_valid)})",
                )

            # T-wave family (magenta)
            t_onsets_valid = get_valid_peaks(waves_obj.t_onsets if waves_obj else None)
            if len(t_onsets_valid) > 0:
                ax.plot(
                    time[t_onsets_valid],
                    ch_data[t_onsets_valid],
                    "m^",  # Magenta upward triangle (T-wave onset)
                    markersize=5,
                    alpha=0.5,
                    label=f"T-onsets (n={len(t_onsets_valid)})",
                )

            t_offsets_valid = get_valid_peaks(waves_obj.t_offsets if waves_obj else None)
            if len(t_offsets_valid) > 0:
                ax.plot(
                    time[t_offsets_valid],
                    ch_data[t_offsets_valid],
                    "mv",  # Magenta downward triangle (T-wave offset)
                    markersize=5,
                    alpha=0.5,
                    label=f"T-offsets (n={len(t_offsets_valid)})",
                )

            t_peaks_valid = get_valid_peaks(waves_obj.t_peaks if waves_obj else None)
            if len(t_peaks_valid) > 0:
                ax.plot(
                    time[t_peaks_valid],
                    ch_data[t_peaks_valid],
                    "mo",
                    markersize=6,
                    alpha=0.4,
                    label="T-peaks",
                )

        lead_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"Ch{ch_idx}"
        if st_levels is not None and lead_name in st_levels:
            st_elev_relative = st_levels[lead_name]  # ST level relative to baseline (in mV)
            if np.isnan(st_elev_relative) or st_elev_relative == 0.0:
                continue
            # Get baseline median for this channel (in µV, convert to mV)
            baseline_mv = None
            if baseline_medians is not None and lead_name in baseline_medians:
                baseline_mv = baseline_medians[lead_name] / 1000.0  # Convert µV to mV
            
            if baseline_mv is not None and not np.isnan(baseline_mv):
                st_elev_absolute_mv = st_elev_relative + baseline_mv
            else:
                st_elev_absolute_mv = st_elev_relative
            
            ax.axhline(
                y=st_elev_absolute_mv,
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Mean ST level: {st_elev_absolute_mv:.2f} mV",
            )
        
        # Plot baseline median as horizontal line
        if baseline_medians is not None and lead_name in baseline_medians:
            baseline_mv = baseline_medians[lead_name] / 1000.0  # Convert µV to mV
            if np.isnan(baseline_mv):
                continue
            ax.axhline(
                y=baseline_mv,
                color="purple",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                label=f"Mean baseline: {baseline_mv:.2f} mV",
            )

        ax.set_title(lead_name, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(ylim)

        # ECG paper-style grid: 0.2 s thick vertical, 0.04 s thin vertical, 0.5 mV thick horizontal
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))

        # Show x-axis labels only at whole seconds
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x:.0f}" if np.isclose(x % 1.0, 0.0) else "")
        )
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

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
            linewidth=1.0,
            alpha=0.6,
            color="#FF9999",
        )
        ax.grid(
            True,
            which="minor",
            linestyle="-",
            linewidth=0.4,
            alpha=0.3,
            color="#FFCCCC",
        )
        ax.minorticks_on()
        ax.set_axisbelow(True)
        ax.set_facecolor("#FFF8F8")

        # Legend only if peaks detected
        if r_peaks is not None and len(r_peaks) > 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # Add 13 statistics boxes at the bottom (1 summary + 12 per-lead)
    if summary:
        # Create 13 axes in bottom subfigure (1 row x 13 columns) with minimal spacing
        stats_axes = subfigs[1].subplots(1, 13, gridspec_kw={'wspace': 0.1})
        
        # Helper function to render statistics with pathological highlighting
        def render_stats_text(ax, lines_with_flags: list[tuple[str, bool]], facecolor: str):
            """Render statistics text with pathological values in red."""
            ax.axis('off')
            
            # Build separate text blocks for normal and pathological lines
            # Replace pathological lines with spaces in the normal text, and vice versa
            normal_lines = []
            pathological_lines = []
            for line_text, is_pathological in lines_with_flags:
                if is_pathological:
                    # Keep pathological line, replace normal with spaces of same length
                    normal_lines.append(" " * len(line_text))
                    pathological_lines.append(line_text)
                else:
                    # Keep normal line, replace pathological with spaces
                    normal_lines.append(line_text)
                    pathological_lines.append(" " * len(line_text))
            
            normal_text = "\n".join(normal_lines)
            pathological_text = "\n".join(pathological_lines)
            
            # Render normal text (black) with background box
            ax.text(0.02, 0.98, normal_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   color='black',
                   bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.3, pad=0.25))
            
            # Render pathological text (red) on top - no bbox so it overlays cleanly
            ax.text(0.02, 0.98, pathological_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   color='red', fontweight='bold')
        
        # Summary box (first position)
        summary_lines = format_statistics_text(summary, channel=None)
        render_stats_text(stats_axes[0], summary_lines, 'wheat')
        
        # Per-lead boxes (one for each channel)
        for idx, channel in enumerate(CHANNEL_NAMES):
            lead_lines = format_statistics_text(summary, channel=channel)
            render_stats_text(stats_axes[idx + 1], lead_lines, 'lightblue')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if block:
        plt.show(block=True)
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ECG peak detection with optional profiling")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling and save results to profile_stats.prof",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="profile_stats.prof",
        help="Output file for profiling results (default: profile_stats.prof)",
    )
    parser.add_argument(
        "--profile-sort",
        type=str,
        default="cumulative",
        choices=["cumulative", "time", "calls", "name"],
        help="Sort order for profiling report (default: cumulative)",
    )
    parser.add_argument(
        "--profile-lines",
        type=int,
        default=50,
        help="Number of lines to show in profiling report (default: 50)",
    )
    
    args = parser.parse_args()
    
    if args.profile:
        print("Profiling enabled. This may slow down execution slightly.")
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            main()
        finally:
            profiler.disable()
            
            # Save profile to file
            profiler.dump_stats(args.profile_output)
            print(f"\nProfile saved to: {args.profile_output}")
            
            # Print summary
            stats = pstats.Stats(profiler)
            stats.sort_stats(args.profile_sort)
            print(f"\n{'='*80}")
            print(f"Top {args.profile_lines} functions by {args.profile_sort} time:")
            print(f"{'='*80}")
            stats.print_stats(args.profile_lines)
            
            # Also print callers/callees for top functions
            print(f"\n{'='*80}")
            print("Callers of top functions:")
            print(f"{'='*80}")
            stats.print_callers(args.profile_lines)
    else:
        main()
