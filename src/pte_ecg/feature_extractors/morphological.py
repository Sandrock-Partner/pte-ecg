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
    """Helper function for multiprocessing morphological feature extraction."""
    sample_data, sfreq, lead_order = args
    return MorphologicalExtractor._process_single_sample_static(sample_data, sfreq, lead_order)


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
        - ST segment: st_elevation, st_depression, j_point_elevation, st_slope
        - T-wave: t_wave_inversion_depth, t_symmetry
        - RR intervals: rr_interval_mean, rr_interval_std, etc.
        - Advanced: qrs_fragmentation, q_wave_width, pathological_q, qtc_bazett, qtc_fridericia, qt_rr_ratio, etc.
        - Global (12-lead): qrs_axis, p_axis, territory markers
        - Early MI markers: aVL_T_inversion, terminal_qrs_distortion, precordial_t_wave_imbalance

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
        "st60_amplitude",
        "st80_amplitude",
        "st_area",
        "early_repolarization",
        # T-wave analysis
        "t_wave_inversion_depth",
        "t_symmetry",
        "biphasic_t",
        "tpeak_tend_interval",
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
        "pathological_q",
        "q_wave_width",
        "pathological_q_width",
        "qt_rr_ratio",
        "pr_rr_ratio",
        "t_qt_ratio",
        "qtc_bazett",
        "qtc_fridericia",
        # Multi-lead (global features)
        "qrs_axis",
        "p_axis",
        "r_wave_progression",
        # Territory-specific (12-lead only)
        # Septal (V1-V2)
        "V1_V2_ST_elevation",
        # Anterior (V3-V4)
        "V3_V4_ST_elevation",
        # Anteroseptal (V1-V4)
        "V1_V3_ST_elevation",
        "V1_V4_ST_elevation",
        "V1_V4_T_inversion",
        "V1_Q_amplitude",
        "V1_Q_to_R_ratio",
        # Inferior (II, III, aVF)
        "II_III_aVF_ST_elevation",
        "II_III_aVF_ST_depression",
        "II_III_aVF_T_inversion",
        "III_vs_II_ST_elevation_ratio",
        "III_Q_amplitude",
        "III_Q_to_R_ratio",
        # Lateral (I, aVL, V5, V6)
        "I_aVL_V5_V6_ST_elevation",
        "I_aVL_V5_V6_T_inversion",
        "I_aVL_ST_depression",
        "V5_Q_amplitude",
        "V5_Q_to_R_ratio",
        "V6_Q_amplitude",
        "V6_Q_to_R_ratio",
        # Posterior (reciprocal in V1-V3)
        "V1_V3_ST_depression",
        "V1_V3_R_wave_amplitude",
        # Right Ventricular (V1, V2)
        "V1_ST_elevation",
        "V2_ST_depression",
        "V1_V2_ST_elevation_depression_ratio",
        # Phase 1: Early MI Markers
        "aVL_T_inversion",
        "terminal_qrs_distortion",
        "V2_terminal_qrs_distortion",
        "V3_terminal_qrs_distortion",
        "precordial_t_wave_balance_cv",
        "precordial_t_wave_imbalance",
        "precordial_t_wave_max_min_ratio",
        "precordial_t_wave_imbalance_ratio",
        # Add T wave per region
        # Überhöhte T Wellen (hyperakute t-welle)
        # Präterminal negative T-Welle
        # Biphasische T-Welle (v.a. V2-V3, Wellens-Zeichen)
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
            Column names follow pattern: morphological_{feature_name}_{lead_name} for per-channel
            features, and morphological_{feature_name} for global features.

        Raises:
            ValueError: If ecg does not have 3 dimensions
        """
        utils.assert_3_dims(ecg)

        start = utils.log_start("Morphological", ecg.shape[0])
        n_samples = ecg.shape[0]
        args_list = [(ecg_single, sfreq, self.lead_order) for ecg_single in ecg]
        processes = utils.get_n_processes(self.n_jobs, n_samples)

        if processes == 1:
            results = list(
                tqdm(
                    (
                        self._process_single_sample(sample_data, sfreq, self.lead_order)
                        for sample_data, sfreq, self.lead_order in args_list
                    ),
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

    def _process_single_sample(self, sample_data: np.ndarray, sfreq: float, lead_order: list[str]) -> dict[str, float]:
        """Extract morphological features from a single sample (all channels).

        Args:
            sample_data: Single sample ECG data with shape (n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz
            lead_order: List of lead names in the same order as channels in sample_data

        Returns:
            Dictionary with keys: morphological_{feature}_{lead_name} (per-channel)
                               and morphological_{feature} (global features)
        """
        return self._process_single_sample_static(sample_data, sfreq, lead_order)

    @staticmethod
    def _process_single_sample_static(sample_data: np.ndarray, sfreq: float, lead_order: list[str]) -> dict[str, float]:
        """Static version for multiprocessing compatibility.

        Args:
            sample_data: Single sample ECG data with shape (n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz
            lead_order: List of lead names in the same order as channels in sample_data

        Returns:
            Dictionary with keys: morphological_{feature}_{lead_name} and morphological_{feature}
        """
        features: dict[str, float] = {}
        flat_chs = np.all(np.isclose(sample_data, sample_data[:, 0:1]), axis=1)
        if np.all(flat_chs):
            logger.warning("All channels are flat lines. Skipping morphological features.")
            return features
        for ch_num, (ch_data, is_flat) in enumerate(zip(sample_data, flat_chs)):
            if is_flat:
                lead_name = lead_order[ch_num] if ch_num < len(lead_order) else f"ch{ch_num}"
                logger.warning(f"Channel {ch_num} ({lead_name}) is a flat line. Skipping morphological features.")
                continue
            ch_feat = MorphologicalExtractor._process_single_channel_static(ch_data, sfreq)
            lead_name = lead_order[ch_num] if ch_num < len(lead_order) else f"ch{ch_num}"
            features.update((f"morphological_{key}_{lead_name}", value) for key, value in ch_feat.items())

        # Calculate electrical axes (requires combining data from multiple channels)
        # QRS axis from R-wave amplitudes (using Lead I and aVF)
        lead_i_idx = lead_order.index("I") if "I" in lead_order else None
        lead_avf_idx = lead_order.index("aVF") if "aVF" in lead_order else None

        if lead_i_idx is not None and lead_avf_idx is not None:
            lead_i_name = lead_order[lead_i_idx]
            lead_avf_name = lead_order[lead_avf_idx]
            r_amp_lead_i = features.get(f"morphological_r_amplitude_{lead_i_name}")
            r_amp_lead_avf = features.get(f"morphological_r_amplitude_{lead_avf_name}")
            if r_amp_lead_i is not None and r_amp_lead_avf is not None and (r_amp_lead_i != 0 or r_amp_lead_avf != 0):
                features["morphological_qrs_axis"] = float(np.arctan2(r_amp_lead_avf, r_amp_lead_i) * 180 / np.pi)

        # P axis from P-wave amplitudes (using Lead I and aVF)
        if lead_i_idx is not None and lead_avf_idx is not None:
            lead_i_name = lead_order[lead_i_idx]
            lead_avf_name = lead_order[lead_avf_idx]
            p_amp_lead_i = features.get(f"morphological_p_amplitude_{lead_i_name}")
            p_amp_lead_avf = features.get(f"morphological_p_amplitude_{lead_avf_name}")
            if p_amp_lead_i is not None and p_amp_lead_avf is not None and (p_amp_lead_i != 0 or p_amp_lead_avf != 0):
                features["morphological_p_axis"] = float(np.arctan2(p_amp_lead_avf, p_amp_lead_i) * 180 / np.pi)

        # Territory-Specific Markers (requires specific leads to be present)
        # ====================================================================
        # 1. SEPTAL WALL (LAD Territory - V1-V2)
        # ====================================================================
        v1_v2_lead_names = [lead for lead in ["V1", "V2"] if lead in lead_order]
        if len(v1_v2_lead_names) >= 2:
            v1_v2_st_elev = np.mean(
                [features.get(f"morphological_st_elevation_{lead}", 0.0) for lead in v1_v2_lead_names]
            )
            features["morphological_V1_V2_ST_elevation"] = float(v1_v2_st_elev)

        # ====================================================================
        # 2. ANTERIOR WALL (LAD Territory - V3-V4)
        # ====================================================================
        v3_v4_lead_names = [lead for lead in ["V3", "V4"] if lead in lead_order]
        if len(v3_v4_lead_names) >= 2:
            v3_v4_st_elev = np.mean(
                [features.get(f"morphological_st_elevation_{lead}", 0.0) for lead in v3_v4_lead_names]
            )
            features["morphological_V3_V4_ST_elevation"] = float(v3_v4_st_elev)

        # ====================================================================
        # 3. ANTEROSEPTAL WALL (LAD Territory - V1-V4)
        # ====================================================================
        v1_v3_lead_names = [lead for lead in ["V1", "V2", "V3"] if lead in lead_order]
        v1_v4_lead_names = [lead for lead in ["V1", "V2", "V3", "V4"] if lead in lead_order]

        if len(v1_v3_lead_names) >= 3:
            v1_v3_st_elev = np.mean(
                [features.get(f"morphological_st_elevation_{lead}", 0.0) for lead in v1_v3_lead_names]
            )
            features["morphological_V1_V3_ST_elevation"] = float(v1_v3_st_elev)

        if len(v1_v4_lead_names) >= 4:
            v1_v4_st_elev = np.mean(
                [features.get(f"morphological_st_elevation_{lead}", 0.0) for lead in v1_v4_lead_names]
            )
            features["morphological_V1_V4_ST_elevation"] = float(v1_v4_st_elev)

            v1_v4_t_inv = np.mean(
                [features.get(f"morphological_t_wave_inversion_depth_{lead}", 0.0) for lead in v1_v4_lead_names]
            )
            features["morphological_V1_V4_T_inversion"] = float(v1_v4_t_inv)

        if "V1" in lead_order:
            v1_name = "V1"
            q_v1 = abs(features.get(f"morphological_q_amplitude_{v1_name}", 0.0))
            r_v1 = features.get(f"morphological_r_amplitude_{v1_name}", 1.0)
            features["morphological_V1_Q_amplitude"] = float(q_v1)
            features["morphological_V1_Q_to_R_ratio"] = float(q_v1 / r_v1) if r_v1 > 0 else 0.0

        # ====================================================================
        # 4. INFERIOR WALL (RCA or LCx Territory - II, III, aVF)
        # ====================================================================
        inferior_lead_names = [lead for lead in ["II", "III", "aVF"] if lead in lead_order]
        if len(inferior_lead_names) >= 3:
            inf_st_elev = np.mean(
                [features.get(f"morphological_st_elevation_{lead}", 0.0) for lead in inferior_lead_names]
            )
            features["morphological_II_III_aVF_ST_elevation"] = float(inf_st_elev)

            inf_t_inv = np.mean(
                [features.get(f"morphological_t_wave_inversion_depth_{lead}", 0.0) for lead in inferior_lead_names]
            )
            features["morphological_II_III_aVF_T_inversion"] = float(inf_t_inv)

            # Differentiate RCA vs LCx: ST elevation III > II suggests RCA
            if "II" in lead_order and "III" in lead_order:
                st_ii = features.get("morphological_st_elevation_II", 0.0)
                st_iii = features.get("morphological_st_elevation_III", 0.0)
                if st_ii > 0:
                    features["morphological_III_vs_II_ST_elevation_ratio"] = float(st_iii / st_ii)
                else:
                    features["morphological_III_vs_II_ST_elevation_ratio"] = float(st_iii) if st_iii > 0 else 0.0

        if "III" in lead_order:
            iii_name = "III"
            q_iii = abs(features.get(f"morphological_q_amplitude_{iii_name}", 0.0))
            r_iii = features.get(f"morphological_r_amplitude_{iii_name}", 1.0)
            features["morphological_III_Q_amplitude"] = float(q_iii)
            features["morphological_III_Q_to_R_ratio"] = float(q_iii / r_iii) if r_iii > 0 else 0.0

        # ====================================================================
        # 5. LATERAL WALL (LCx or Diagonal LAD Territory - I, aVL, V5, V6)
        # ====================================================================
        lateral_lead_names = [lead for lead in ["I", "aVL", "V5", "V6"] if lead in lead_order]
        if len(lateral_lead_names) >= 4:
            lat_st_elev = np.mean(
                [features.get(f"morphological_st_elevation_{lead}", 0.0) for lead in lateral_lead_names]
            )
            features["morphological_I_aVL_V5_V6_ST_elevation"] = float(lat_st_elev)

            lat_t_inv = np.mean(
                [features.get(f"morphological_t_wave_inversion_depth_{lead}", 0.0) for lead in lateral_lead_names]
            )
            features["morphological_I_aVL_V5_V6_T_inversion"] = float(lat_t_inv)

        if "V5" in lead_order:
            v5_name = "V5"
            q_v5 = abs(features.get(f"morphological_q_amplitude_{v5_name}", 0.0))
            r_v5 = features.get(f"morphological_r_amplitude_{v5_name}", 1.0)
            features["morphological_V5_Q_amplitude"] = float(q_v5)
            features["morphological_V5_Q_to_R_ratio"] = float(q_v5 / r_v5) if r_v5 > 0 else 0.0

        if "V6" in lead_order:
            v6_name = "V6"
            q_v6 = abs(features.get(f"morphological_q_amplitude_{v6_name}", 0.0))
            r_v6 = features.get(f"morphological_r_amplitude_{v6_name}", 1.0)
            features["morphological_V6_Q_amplitude"] = float(q_v6)
            features["morphological_V6_Q_to_R_ratio"] = float(q_v6 / r_v6) if r_v6 > 0 else 0.0

        # ====================================================================
        # 6. POSTERIOR WALL (RCA or LCx Territory - reciprocal in V1-V3)
        # ====================================================================
        # Posterior MI shows ST depression in V1-V3 (reciprocal of posterior ST elevation)
        v1_v3_lead_names_posterior = [lead for lead in ["V1", "V2", "V3"] if lead in lead_order]
        if len(v1_v3_lead_names_posterior) >= 3:
            v1_v3_st_dep = np.mean(
                [features.get(f"morphological_st_depression_{lead}", 0.0) for lead in v1_v3_lead_names_posterior]
            )
            features["morphological_V1_V3_ST_depression"] = float(v1_v3_st_dep)

            # Tall R waves in V1-V3 (suggestive of posterior MI)
            v1_v3_r_amplitudes = []
            for lead in v1_v3_lead_names_posterior:
                r_amp = features.get(f"morphological_r_amplitude_{lead}")
                if r_amp is not None:
                    v1_v3_r_amplitudes.append(r_amp)
            if v1_v3_r_amplitudes:
                features["morphological_V1_V3_R_wave_amplitude"] = float(np.mean(v1_v3_r_amplitudes))

        # ====================================================================
        # 7. RIGHT VENTRICULAR (RCA proximal Territory - V1, V4R)
        # ====================================================================
        # RV infarction: ST elevation in V1 with ST depression in V2
        if "V1" in lead_order and "V2" in lead_order:
            v1_st_elev = features.get("morphological_st_elevation_V1", 0.0)
            v2_st_dep = features.get("morphological_st_depression_V2", 0.0)
            features["morphological_V1_ST_elevation"] = float(v1_st_elev)
            features["morphological_V2_ST_depression"] = float(v2_st_dep)
            # Ratio for RV infarction pattern
            if v2_st_dep > 0:
                features["morphological_V1_V2_ST_elevation_depression_ratio"] = float(v1_st_elev / v2_st_dep)
            else:
                features["morphological_V1_V2_ST_elevation_depression_ratio"] = (
                    float(v1_st_elev) if v1_st_elev > 0 else 0.0
                )

        # ====================================================================
        # RECIPROCAL CHANGES (ST depression in reciprocal leads)
        # ====================================================================
        # Reciprocal ST depression in inferior leads (for anterior/septal/anteroseptal MI)
        if len(inferior_lead_names) >= 3:
            inf_st_dep = np.mean(
                [features.get(f"morphological_st_depression_{lead}", 0.0) for lead in inferior_lead_names]
            )
            features["morphological_II_III_aVF_ST_depression"] = float(inf_st_dep)

        # Reciprocal ST depression in lateral leads (for inferior MI)
        i_avl_lead_names = [lead for lead in ["I", "aVL"] if lead in lead_order]
        if len(i_avl_lead_names) >= 2:
            i_avl_st_dep = np.mean(
                [features.get(f"morphological_st_depression_{lead}", 0.0) for lead in i_avl_lead_names]
            )
            features["morphological_I_aVL_ST_depression"] = float(i_avl_st_dep)

        # ====================================================================
        # PHASE 1: EARLY MI MARKERS
        # ====================================================================
        # 1. T-wave inversion in lead aVL (early ACS indicator)
        if "aVL" in lead_order:
            avl_t_inv = features.get("morphological_t_wave_inversion_depth_aVL", 0.0)
            features["morphological_aVL_T_inversion"] = float(avl_t_inv)

        # 2. Terminal QRS distortion (absence of S-wave in V2/V3)
        # Early sign of acute MI, especially LAD occlusion
        terminal_qrs_distortion_v2 = 0.0
        terminal_qrs_distortion_v3 = 0.0
        if "V2" in lead_order:
            # Check if S-wave is present (S amplitude should be negative and significant)
            s_amp_v2 = features.get("morphological_s_amplitude_V2")
            if s_amp_v2 is None or s_amp_v2 >= -0.05:  # No significant S-wave (threshold: -0.05mV)
                terminal_qrs_distortion_v2 = 1.0
        if "V3" in lead_order:
            s_amp_v3 = features.get("morphological_s_amplitude_V3")
            if s_amp_v3 is None or s_amp_v3 >= -0.05:  # No significant S-wave
                terminal_qrs_distortion_v3 = 1.0

        if "V2" in lead_order or "V3" in lead_order:
            # Terminal QRS distortion if either V2 or V3 lacks S-wave
            features["morphological_terminal_qrs_distortion"] = float(
                max(terminal_qrs_distortion_v2, terminal_qrs_distortion_v3)
            )
            if "V2" in lead_order:
                features["morphological_V2_terminal_qrs_distortion"] = float(terminal_qrs_distortion_v2)
            if "V3" in lead_order:
                features["morphological_V3_terminal_qrs_distortion"] = float(terminal_qrs_distortion_v3)

        # 3. Loss of precordial T-wave balance
        # Disproportionate T-wave amplitude between precordial leads (early ischemia sign)
        precordial_leads = [lead for lead in ["V1", "V2", "V3", "V4", "V5", "V6"] if lead in lead_order]
        if len(precordial_leads) >= 3:
            t_amplitudes_precordial = []
            for lead in precordial_leads:
                t_amp = features.get(f"morphological_t_amplitude_{lead}")
                if t_amp is not None:
                    t_amplitudes_precordial.append(abs(t_amp))

            if len(t_amplitudes_precordial) >= 3:
                t_amplitudes_array = np.array(t_amplitudes_precordial)
                # Calculate coefficient of variation (CV) as measure of imbalance
                mean_t_amp = np.mean(t_amplitudes_array)
                std_t_amp = np.std(t_amplitudes_array)
                if mean_t_amp > 0:
                    cv_t_balance = std_t_amp / mean_t_amp
                    features["morphological_precordial_t_wave_balance_cv"] = float(cv_t_balance)
                    # High CV (>0.5) indicates significant imbalance
                    features["morphological_precordial_t_wave_imbalance"] = 1.0 if cv_t_balance > 0.5 else 0.0

                # Also calculate max/min ratio (another measure of imbalance)
                if len(t_amplitudes_array) > 1 and np.min(t_amplitudes_array) > 0:
                    max_min_ratio = np.max(t_amplitudes_array) / np.min(t_amplitudes_array)
                    features["morphological_precordial_t_wave_max_min_ratio"] = float(max_min_ratio)
                    # Ratio > 3 indicates significant imbalance
                    features["morphological_precordial_t_wave_imbalance_ratio"] = 1.0 if max_min_ratio > 3.0 else 0.0

        # ====================================================================
        # R WAVE PROGRESSION (across precordial leads V1-V6)
        # ====================================================================
        # Get R amplitudes in V1-V6 (leads 6-11)
        r_amplitudes = []
        for lead_idx in range(6, 12):
            r_amp = features.get(f"morphological_r_amplitude_ch{lead_idx}")
            if r_amp is not None:
                r_amplitudes.append(r_amp)

        # Calculate progression score (should increase from V1 to V4-V5)
        if len(r_amplitudes) >= 4:
            # Check if there's normal progression
            progression = np.diff(r_amplitudes[:4])
            progression_score = float(np.mean(progression > 0))  # Proportion of increases
            features["morphological_r_wave_progression"] = progression_score
        else:
            features["morphological_r_wave_progression"] = np.nan

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
                features[f"{feature_name}_mean"] = stats["mean"]
                features[f"{feature_name}_median"] = stats["median"]
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

            # Q wave width (duration from Q onset to Q peak)
            # Pathological if > 40ms
            features["pathological_q_width"] = 0.0  # Initialize
            q_onsets = waves_dict.get("ECG_Q_Onsets")
            if q_onsets is not None and len(q_onsets) > 0:
                q_widths = []
                max_index = min(len(q_peaks), len(q_onsets))
                for q_peak, q_onset in zip(q_peaks[:max_index], q_onsets[:max_index]):
                    if np.isnan(q_peak) or np.isnan(q_onset) or q_peak <= q_onset:
                        continue
                    q_width_ms = (q_peak - q_onset) / sfreq * 1000
                    if q_width_ms > 0:
                        q_widths.append(q_width_ms)
                if q_widths:
                    features["q_wave_width"] = float(np.mean(q_widths))
                    # Pathological Q wave width (>40ms)
                    features["pathological_q_width"] = 1.0 if np.mean(q_widths) > 40.0 else 0.0

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

        # Pathological Q wave detection
        # Pathological if QRS duration > 40ms AND Q amplitude > 25% of R amplitude
        features["pathological_q"] = 0.0
        if "qrs_duration_mean" in features and "q_amplitude" in features and "r_amplitude" in features:
            qrs_dur = features["qrs_duration_mean"]
            q_amp = abs(features["q_amplitude"])
            r_amp = abs(features["r_amplitude"])
            if qrs_dur > 40 and r_amp > 0 and q_amp > 0.25 * r_amp:
                features["pathological_q"] = 1.0

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
        features["biphasic_t"] = 0.0
        features["tpeak_tend_interval"] = np.nan
        t_onsets = waves_dict.get("ECG_T_Onsets")
        t_offsets = waves_dict.get("ECG_T_Offsets")
        if t_onsets is not None and t_offsets is not None and n_t_peaks:
            symmetry_ratios = []
            biphasic_flags = []
            tpeak_tend_intervals = []
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

                # Tpeak-Tend interval (repolarization dispersion)
                tpeak_tend = (offset - peak) / sfreq * 1000  # Convert to milliseconds
                tpeak_tend_intervals.append(tpeak_tend)

                # Biphasic T wave detection
                # Extract T segment for this beat
                t_start = max(int(onset), 0)
                t_end = min(int(offset), len(ch_data))
                if t_end > t_start:
                    t_segment = ch_data[t_start:t_end]
                    has_positive = np.any(t_segment > 0.1)
                    has_negative = np.any(t_segment < -0.1)
                    biphasic_flags.append(1.0 if (has_positive and has_negative) else 0.0)

            if symmetry_ratios:
                features["t_symmetry"] = float(np.mean(symmetry_ratios))
            if tpeak_tend_intervals:
                features["tpeak_tend_interval"] = float(np.mean(tpeak_tend_intervals))
            if biphasic_flags:
                features["biphasic_t"] = float(np.mean(biphasic_flags))

        # ST Segment Features
        # Calculate global baseline (isoelectric line) from first 200ms of signal
        baseline_samples = int(0.2 * sfreq)  # 200ms
        global_baseline = np.mean(ch_data[: min(baseline_samples, len(ch_data))])

        if n_s_peaks and n_r_peaks:
            st_elevations = []
            st_depressions = []
            st_slopes = []
            st60_amplitudes = []
            st80_amplitudes = []
            st_areas = []

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

                    # ST amplitude at J+60ms
                    st60_idx = min(len(ch_data), s_idx + int(0.06 * sfreq))
                    if st60_idx < len(ch_data):
                        st60_amplitudes.append(ch_data[st60_idx] - global_baseline)

                    # ST amplitude at J+80ms
                    st80_idx = min(len(ch_data), s_idx + int(0.08 * sfreq))
                    if st80_idx < len(ch_data):
                        st80_amplitudes.append(ch_data[st80_idx] - global_baseline)

                    # ST segment area under curve
                    st_segment_relative = st_segment - global_baseline
                    st_areas.append(np.trapezoid(st_segment_relative))

            # Store averaged ST segment features
            if st_elevations:
                features["st_elevation"] = float(np.mean(st_elevations))
            if st_depressions:
                features["st_depression"] = float(np.mean(st_depressions))
            if st_elevations and st_depressions:
                features["j_point_elevation"] = features.get("st_elevation", 0.0) - features.get("st_depression", 0.0)
            if st_slopes:
                features["st_slope"] = float(np.mean(st_slopes))
            if st60_amplitudes:
                features["st60_amplitude"] = float(np.mean(st60_amplitudes))
            if st80_amplitudes:
                features["st80_amplitude"] = float(np.mean(st80_amplitudes))
            if st_areas:
                features["st_area"] = float(np.mean(st_areas))

            # Early repolarization pattern (J-point elevation > 0.1mV)
            if "j_point_elevation" in features:
                features["early_repolarization"] = 1.0 if features["j_point_elevation"] > 0.1 else 0.0
            else:
                features["early_repolarization"] = 0.0
        else:
            # Initialize early_repolarization even when no ST segments found
            features["early_repolarization"] = 0.0

        # QTc (Corrected QT interval using Bazett's formula)
        # QTc = QT / √(RR) where QT is in ms and RR is in seconds
        if "qt_interval_mean" in features and "rr_interval_mean" in features:
            qt_ms = features["qt_interval_mean"]
            rr_sec = features["rr_interval_mean"]
            if rr_sec > 0:
                features["qtc_bazett"] = qt_ms / np.sqrt(rr_sec)
                # Fridericia formula: QTc = QT / (RR^(1/3))
                features["qtc_fridericia"] = qt_ms / ((rr_sec) ** (1 / 3))
            else:
                features["qtc_bazett"] = qt_ms  # Fallback if RR is invalid
                features["qtc_fridericia"] = qt_ms  # Fallback if RR is invalid

        # Interval Ratios
        # Convert RR interval from seconds to milliseconds for ratio calculations
        rr_ms = features.get("rr_interval_mean", 0.0) * 1000

        # QT/RR ratio
        if "qt_interval_mean" in features and rr_ms > 0:
            features["qt_rr_ratio"] = features["qt_interval_mean"] / rr_ms

        # PR/RR ratio (using pq_interval as PR interval)
        if "pq_interval_mean" in features and rr_ms > 0:
            features["pr_rr_ratio"] = features["pq_interval_mean"] / rr_ms

        # T/QT ratio
        if "t_duration_mean" in features and "qt_interval_mean" in features:
            qt_ms = features["qt_interval_mean"]
            if qt_ms > 0:
                features["t_qt_ratio"] = features["t_duration_mean"] / qt_ms

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
            Dictionary with mean, median, std, min, max in milliseconds, or empty dict if no valid intervals
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
            "median": float(np.median(intervals_ms)),
            "std": float(np.std(intervals_ms)),
            "min": float(np.min(intervals_ms)),
            "max": float(np.max(intervals_ms)),
        }
