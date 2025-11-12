"""Morphological ECG feature extractor.

This extractor performs comprehensive waveform analysis including peak detection,
interval calculations, ST segment analysis, and territory-specific markers.
"""

import multiprocessing
import warnings
from typing import Literal

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from tqdm import tqdm

from .._logging import logger
from . import utils
from .base import BaseFeatureExtractor

# Peak detection methods to try (in order of preference)
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

# Type alias for peak detection methods
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

# Interval pairs for comprehensive statistics (wave1, wave2, feature_name)
_INTERVAL_PAIRS = [
    ("P", "R", "pr_interval"),
    ("P", "Q", "pq_interval"),
    ("Q", "R", "qr_interval"),
    ("Q", "S", "qrs_duration"),
    ("R", "S", "rs_interval"),
    ("S", "T_onset", "st_duration"),
    ("Q", "T", "qt_interval"),
    ("R", "T_onset", "rt_duration"),
    ("R", "T", "rt_interval"),
    ("P", "T", "pt_interval"),
    ("P_onset", "P_offset", "p_duration"),
    ("T_onset", "T_offset", "t_duration"),
]


class ECGDelineationError(Exception):
    """Raised when all ECG delineation methods fail to detect peaks properly."""

    pass


# Module-level helper for multiprocessing
def _starmap_helper_morph(args: tuple) -> dict:
    """Helper to unpack args for _process_single_sample in multiprocessing."""
    return MorphologicalExtractor._process_single_sample_static(*args)


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
        - Intervals (mean, std, min, max): qrs_duration, qt_interval, pq_interval,
          pr_interval, qr_interval, rs_interval, p_duration, t_duration,
          st_duration, rt_duration, rt_interval, pt_interval
        - Amplitudes: p_amplitude, q_amplitude, r_amplitude, etc.
        - Areas: p_area, t_area
        - Slopes: r_slope, t_slope
        - ST segment: st_elevation, st_depression, j_point_elevation, st_slope
        - T-wave: t_wave_inversion_depth, t_symmetry
        - RR intervals: rr_interval_mean, rr_interval_std, etc.
        - Advanced: qrs_fragmentation, qtc_interval, qt_rr_ratio, etc.
        - Global (12-lead): qrs_axis, p_axis, territory markers

    Args:
        selected_features: List of features to extract (not yet implemented for filtering)
        n_jobs: Number of parallel jobs

    Examples:
        # Extract all morphological features
        extractor = MorphologicalExtractor()
        features = extractor.get_features(ecg_data, sfreq=1000)
    """

    name = "morphological"

    available_features = [
        # Interval means
        "qrs_duration",
        "qt_interval",
        "qtc_interval",
        "pq_interval",
        "pr_interval",
        "qr_interval",
        "rs_interval",
        "p_duration",
        "t_duration",
        "st_duration",
        "rt_duration",
        "rt_interval",
        "pt_interval",
        # Interval std deviations
        "qrs_duration_std",
        "qt_interval_std",
        "pq_interval_std",
        "pr_interval_std",
        "qr_interval_std",
        "rs_interval_std",
        "p_duration_std",
        "t_duration_std",
        "st_duration_std",
        "rt_duration_std",
        "rt_interval_std",
        "pt_interval_std",
        # Interval minimums
        "qrs_duration_min",
        "qt_interval_min",
        "pq_interval_min",
        "pr_interval_min",
        "qr_interval_min",
        "rs_interval_min",
        "p_duration_min",
        "t_duration_min",
        "st_duration_min",
        "rt_duration_min",
        "rt_interval_min",
        "pt_interval_min",
        # Interval maximums
        "qrs_duration_max",
        "qt_interval_max",
        "pq_interval_max",
        "pr_interval_max",
        "qr_interval_max",
        "rs_interval_max",
        "p_duration_max",
        "t_duration_max",
        "st_duration_max",
        "rt_duration_max",
        "rt_interval_max",
        "pt_interval_max",
        # Amplitudes
        "p_amplitude",
        "q_amplitude",
        "r_amplitude",
        "s_amplitude",
        "t_amplitude",
        # Areas and slopes
        "p_area",
        "t_area",
        "r_slope",
        "t_slope",
        # ST segment
        "st_elevation",
        "st_depression",
        "j_point_elevation",
        "st_slope",
        # T-wave analysis
        "t_wave_inversion_depth",
        "t_symmetry",
        # RR intervals
        "rr_interval_mean",
        "rr_interval_std",
        "rr_interval_median",
        "rr_interval_iqr",
        "rr_interval_skewness",
        "rr_interval_kurtosis",
        "sd1",
        "sd2",
        "sd1_sd2_ratio",
        # Advanced
        "qrs_fragmentation",
        "qt_rr_ratio",
        "pr_rr_ratio",
        "t_qt_ratio",
        # Multi-lead (global features)
        "qrs_axis",
        "p_axis",
        # Territory-specific (12-lead only)
        "V1_V3_ST_elevation",
        "V1_V4_T_inversion",
        "V1_Q_amplitude",
        "V1_Q_to_R_ratio",
        "II_III_aVF_ST_elevation",
        "II_III_aVF_T_inversion",
        "III_Q_amplitude",
        "III_Q_to_R_ratio",
        "I_aVL_V5_V6_ST_elevation",
        "I_aVL_V5_V6_T_inversion",
        "V5_Q_amplitude",
        "V5_Q_to_R_ratio",
        "V6_Q_amplitude",
        "V6_Q_to_R_ratio",
        "aVR_ST_elevation",
    ]

    def get_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract morphological features from ECG data.

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
        utils.assert_3_dims(ecg)

        start = utils.log_start("Morphological", ecg.shape[0])
        n_samples = ecg.shape[0]
        args_list = [(ecg_single, sfreq) for ecg_single in ecg]
        processes = utils.get_n_processes(self.n_jobs, n_samples)

        if processes == 1:
            results = list(
                tqdm(
                    (self._process_single_sample(sample_data, sfreq) for sample_data, sfreq in args_list),
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
        utils.log_end("Morphological", start, feature_df.shape)
        return feature_df

    def _process_single_sample(self, sample_data: np.ndarray, sfreq: float) -> dict[str, float]:
        """Extract morphological features from a single sample (all channels).

        Args:
            sample_data: Single sample ECG data with shape (n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            Dictionary with keys: morphological_{feature}_ch{N} (per-channel)
                               and morphological_{feature} (global features)
        """
        return self._process_single_sample_static(sample_data, sfreq)

    @staticmethod
    def _process_single_sample_static(sample_data: np.ndarray, sfreq: float) -> dict[str, float]:
        """Static version for multiprocessing compatibility.

        Args:
            sample_data: Single sample ECG data with shape (n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            Dictionary with keys: morphological_{feature}_ch{N} and morphological_{feature}
        """
        features: dict[str, float] = {}
        flat_chs = np.all(np.isclose(sample_data, sample_data[:, 0:1]), axis=1)
        if np.all(flat_chs):
            logger.warning("All channels are flat lines. Skipping morphological features.")
            return features
        for ch_num, (ch_data, is_flat) in enumerate(zip(sample_data, flat_chs)):
            if is_flat:
                logger.warning(f"Channel {ch_num} is a flat line. Skipping morphological features.")
                continue
            ch_feat = MorphologicalExtractor._process_single_channel_static(ch_data, sfreq)
            features.update((f"morphological_{key}_ch{ch_num}", value) for key, value in ch_feat.items())

        # Calculate electrical axes (requires combining data from multiple channels)
        # Assumes standard 12-lead ECG ordering: I, II, III, aVR, aVL, aVF, V1-V6
        # Lead I = ch0, aVF = ch5
        # QRS axis from R-wave amplitudes
        r_amp_lead_i = features.get("morphological_r_amplitude_ch0")
        r_amp_lead_avf = features.get("morphological_r_amplitude_ch5")
        if r_amp_lead_i is not None and r_amp_lead_avf is not None and (r_amp_lead_i != 0 or r_amp_lead_avf != 0):
            features["morphological_qrs_axis"] = float(np.arctan2(r_amp_lead_avf, r_amp_lead_i) * 180 / np.pi)

        # P axis from P-wave amplitudes
        p_amp_lead_i = features.get("morphological_p_amplitude_ch0")
        p_amp_lead_avf = features.get("morphological_p_amplitude_ch5")
        if p_amp_lead_i is not None and p_amp_lead_avf is not None and (p_amp_lead_i != 0 or p_amp_lead_avf != 0):
            features["morphological_p_axis"] = float(np.arctan2(p_amp_lead_avf, p_amp_lead_i) * 180 / np.pi)

        # Territory-Specific Markers (requires 12-lead ECG)
        # Standard 12-lead ordering: I, II, III, aVR, aVL, aVF, V1-V6
        if sample_data.shape[0] >= 12:
            # ANTERIOR WALL (LAD Territory - V1-V4)
            v1_v3_leads = [6, 7, 8]
            v1_v4_leads = [6, 7, 8, 9]
            v1_v3_st_elev = np.mean([features.get(f"morphological_st_elevation_ch{ch}", 0.0) for ch in v1_v3_leads])
            features["morphological_V1_V3_ST_elevation"] = float(v1_v3_st_elev)

            v1_v4_t_inv = np.mean(
                [features.get(f"morphological_t_wave_inversion_depth_ch{ch}", 0.0) for ch in v1_v4_leads]
            )
            features["morphological_V1_V4_T_inversion"] = float(v1_v4_t_inv)

            q_v1 = abs(features.get("morphological_q_amplitude_ch6", 0.0))
            r_v1 = features.get("morphological_r_amplitude_ch6", 1.0)
            features["morphological_V1_Q_amplitude"] = float(q_v1)
            features["morphological_V1_Q_to_R_ratio"] = float(q_v1 / r_v1) if r_v1 > 0 else 0.0

            # INFERIOR WALL (RCA Territory - II, III, aVF)
            inferior_leads = [1, 2, 5]
            inf_st_elev = np.mean([features.get(f"morphological_st_elevation_ch{ch}", 0.0) for ch in inferior_leads])
            features["morphological_II_III_aVF_ST_elevation"] = float(inf_st_elev)

            inf_t_inv = np.mean(
                [features.get(f"morphological_t_wave_inversion_depth_ch{ch}", 0.0) for ch in inferior_leads]
            )
            features["morphological_II_III_aVF_T_inversion"] = float(inf_t_inv)

            q_iii = abs(features.get("morphological_q_amplitude_ch2", 0.0))
            r_iii = features.get("morphological_r_amplitude_ch2", 1.0)
            features["morphological_III_Q_amplitude"] = float(q_iii)
            features["morphological_III_Q_to_R_ratio"] = float(q_iii / r_iii) if r_iii > 0 else 0.0

            # LATERAL WALL (LCX Territory - I, aVL, V5, V6)
            lateral_leads = [0, 4, 10, 11]
            lat_st_elev = np.mean([features.get(f"morphological_st_elevation_ch{ch}", 0.0) for ch in lateral_leads])
            features["morphological_I_aVL_V5_V6_ST_elevation"] = float(lat_st_elev)

            lat_t_inv = np.mean(
                [features.get(f"morphological_t_wave_inversion_depth_ch{ch}", 0.0) for ch in lateral_leads]
            )
            features["morphological_I_aVL_V5_V6_T_inversion"] = float(lat_t_inv)

            q_v5 = abs(features.get("morphological_q_amplitude_ch10", 0.0))
            r_v5 = features.get("morphological_r_amplitude_ch10", 1.0)
            q_v6 = abs(features.get("morphological_q_amplitude_ch11", 0.0))
            r_v6 = features.get("morphological_r_amplitude_ch11", 1.0)

            features["morphological_V5_Q_amplitude"] = float(q_v5)
            features["morphological_V5_Q_to_R_ratio"] = float(q_v5 / r_v5) if r_v5 > 0 else 0.0
            features["morphological_V6_Q_amplitude"] = float(q_v6)
            features["morphological_V6_Q_to_R_ratio"] = float(q_v6 / r_v6) if r_v6 > 0 else 0.0

            # GLOBAL PATTERNS (aVR)
            features["morphological_aVR_ST_elevation"] = float(features.get("morphological_st_elevation_ch3", 0.0))

        return features

    def _process_single_channel(self, ch_data: np.ndarray, sfreq: float) -> dict[str, float]:
        """Extract morphological features from a single channel.

        Args:
            ch_data: Single channel ECG data with shape (n_timepoints,)
            sfreq: Sampling frequency in Hz

        Returns:
            Dictionary of morphological features
        """
        return self._process_single_channel_static(ch_data, sfreq)

    @staticmethod
    def _process_single_channel_static(ch_data: np.ndarray, sfreq: float) -> dict[str, float]:
        """Static method for processing a single channel (multiprocessing compatible).

        Args:
            ch_data: Single channel ECG data with shape (n_timepoints,)
            sfreq: Sampling frequency in Hz

        Returns:
            Dictionary of morphological features
        """
        features: dict[str, float] = {}
        r_peaks, n_r_peaks, r_peak_method = MorphologicalExtractor._detect_r_peaks(ch_data, sfreq)
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
            logger.debug(f"Using high-frequency optimized methods for {sfreq} Hz: {methods}")

        for method in methods:
            if n_r_peaks < 2 and method in {"prominence", "cwt"}:
                logger.info(f"Not enough R-peaks ({n_r_peaks}) for {method} method.")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", nk.misc.NeuroKitWarning)  # type: ignore
                    warnings.simplefilter(
                        "ignore",
                        scipy.signal._peak_finding_utils.PeakPropertyWarning,  # type: ignore
                    )
                    _, waves_dict = nk.ecg_delineate(
                        ch_data,
                        rpeaks=r_peaks,
                        sampling_rate=sfreq,
                        method=method,
                    )
                logger.debug(f"ECG delineation successful with method: {method}")
                break
            except nk.misc.NeuroKitWarning as e:  # type: ignore
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

        # Build wave indices dictionary for vectorized interval calculation
        wave_indices = {
            "P": p_peaks if n_p_peaks else np.array([]),
            "Q": q_peaks if n_q_peaks else np.array([]),
            "R": r_peaks if (n_r_peaks and r_peaks is not None) else np.array([]),
            "S": s_peaks if n_s_peaks else np.array([]),
            "T": t_peaks if n_t_peaks else np.array([]),
            "P_onset": p_onsets if n_p_onsets else np.array([]),
            "P_offset": p_offsets if n_p_offsets else np.array([]),
            "T_onset": t_onsets if n_t_onsets else np.array([]),
            "T_offset": t_offsets if n_t_offsets else np.array([]),
        }

        # Vectorized interval calculation for all pairs
        for wave1, wave2, feature_name in _INTERVAL_PAIRS:
            peaks1 = wave_indices.get(wave1)
            peaks2 = wave_indices.get(wave2)

            if peaks1 is None or peaks2 is None:
                continue
            if len(peaks1) == 0 or len(peaks2) == 0:
                continue

            stats = MorphologicalExtractor._calculate_interval_stats(peaks1, peaks2, sfreq)

            if stats:
                features[feature_name] = stats["mean"]
                features[f"{feature_name}_std"] = stats["std"]
                features[f"{feature_name}_min"] = stats["min"]
                features[f"{feature_name}_max"] = stats["max"]

        # Flächen (Integrale unter den Kurven)
        if n_p_onsets and n_p_offsets:
            p_areas = []
            max_index = min(n_p_onsets, n_p_offsets)
            for p_on, p_off in zip(p_onsets[:max_index], p_offsets[:max_index]):
                if p_on >= p_off or np.isnan(p_on) or np.isnan(p_off):
                    continue
                p_areas.append(np.sum(np.abs(ch_data[p_on:p_off])))
            if p_areas:
                features["p_area"] = float(np.mean(p_areas))

        # T Area
        if n_t_onsets and n_t_offsets:
            t_areas = []
            max_index = min(n_t_onsets, n_t_offsets)
            for t_on, t_off in zip(t_onsets[:max_index], t_offsets[:max_index]):
                if t_on >= t_off or np.isnan(t_on) or np.isnan(t_off):
                    continue
                t_areas.append(np.sum(np.abs(ch_data[t_on:t_off])))
            if t_areas:
                features["t_area"] = float(np.mean(t_areas))

        # R Slope
        if n_r_peaks and n_q_peaks and r_peaks is not None:
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
                features["r_slope"] = float(np.mean(r_slopes))

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
                features["t_slope"] = float(np.mean(t_slopes))

        # Amplituden
        if n_p_peaks:
            p_amplitudes = [ch_data[p] for p in p_peaks if not np.isnan(p)]
            if p_amplitudes:
                features["p_amplitude"] = float(np.mean(p_amplitudes))

        if n_q_peaks:
            q_amplitudes = [ch_data[q] for q in q_peaks if not np.isnan(q)]
            if q_amplitudes:
                features["q_amplitude"] = float(np.mean(q_amplitudes))

        if n_r_peaks and r_peaks is not None:
            r_amplitudes = [ch_data[r] for r in r_peaks if not np.isnan(r)]
            if r_amplitudes:
                features["r_amplitude"] = float(np.mean(r_amplitudes))

        if n_r_peaks > 1 and r_peaks is not None:
            rr_intervals = np.diff(r_peaks) / sfreq
            rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
            mean_rr = float(np.mean(rr_intervals))
            std_rr = float(np.std(rr_intervals))
            features["rr_interval_mean"] = mean_rr
            features["rr_interval_std"] = std_rr
            if len(rr_intervals) > 1:
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
                interm = 2 * sdrr**2 - sd1**2
                sd2 = float(np.sqrt(interm)) if interm > 0 else np.nan
                features["sd1"] = sd1
                features["sd2"] = sd2
                features["sd1_sd2_ratio"] = sd1 / (sd2 + utils.EPS) if not np.isnan(sd2) else np.nan

        if n_s_peaks:
            s_amplitudes = [ch_data[s] for s in s_peaks if not np.isnan(s)]
            if s_amplitudes:
                features["s_amplitude"] = float(np.mean(s_amplitudes))

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
                features["t_amplitude"] = float(np.mean(t_amplitudes))

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
                features["j_point_elevation"] = features.get("st_elevation", 0.0) - features.get("st_depression", 0.0)
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

    @staticmethod
    def _detect_r_peaks(ch_data: np.ndarray, sfreq: float) -> tuple[np.ndarray | None, int, PeakMethod | None]:
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

    @staticmethod
    def _calculate_interval_stats(peaks1: np.ndarray, peaks2: np.ndarray, sfreq: float) -> dict[str, float]:
        """Vectorized interval statistics calculation.

        Args:
            peaks1: First wave peaks (e.g., Q peaks)
            peaks2: Second wave peaks (e.g., S peaks)
            sfreq: Sampling frequency in Hz

        Returns:
            Dictionary with mean, std, min, max in milliseconds, or empty dict if no valid intervals
        """
        # Ensure arrays are numpy arrays (might be lists from neurokit2)
        p1 = np.asarray(peaks1)
        p2 = np.asarray(peaks2)

        n = min(len(p1), len(p2))
        if n == 0:
            return {}

        p1 = p1[:n]
        p2 = p2[:n]

        valid_mask = (p1 < p2) & ~np.isnan(p1) & ~np.isnan(p2)

        if not np.any(valid_mask):
            return {}

        intervals_ms = (p2[valid_mask] - p1[valid_mask]) / sfreq * 1000

        return {
            "mean": float(np.mean(intervals_ms)),
            "std": float(np.std(intervals_ms)),
            "min": float(np.min(intervals_ms)),
            "max": float(np.max(intervals_ms)),
        }
