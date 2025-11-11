"""
ECG Feature Extractor
Pure technical feature extraction from ECG signals without medical interpretation

Extracts comprehensive set of ECG features:
- Wave characteristics (P, QRS, T amplitudes, durations)
- Interval features (PR, QT, QTc, RR)
- ST segment features (elevation, depression, slope)
- HRV features (time-domain and frequency-domain)
- Electrical axes
"""

from pathlib import Path

import h5py
import numpy as np
from scipy import signal as scipy_signal


class ECGFeatureExtractor:
    """
    Pure ECG Feature Extraction
    Extracts measurable technical features from ECG signals
    No medical interpretation - that happens in medical_knowledge_enrichment layer
    """

    def __init__(self, sampling_rate: int = 500):
        """
        Initialize ECG Feature Extractor

        Args:
            sampling_rate: ECG sampling rate in Hz (default: 500)
        """
        self.sampling_rate = sampling_rate
        self.lead_names = [
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

    def extract_features(self, ecg_signal: np.ndarray) -> dict[str, float]:
        """
        Extract all technical features from ECG signal using R-peak based waveform detection

        Args:
            ecg_signal: ECG signal array (samples, 12 leads)

        Returns:
            Dictionary with all extracted features
        """
        print("[ECGFeatureExtractor] Extracting features...")

        features = {}

        # Step 1: Detect R-peaks from Lead II (best for R-peak detection)
        lead_ii = ecg_signal[:, 1] if ecg_signal.shape[1] > 1 else ecg_signal[:, 0]
        r_peaks = self._detect_r_peaks(lead_ii)
        rr_intervals = np.diff(r_peaks) if len(r_peaks) > 1 else np.array([])

        # Step 2: Calculate heart rate from RR intervals (Problem #5 FIX)
        if len(rr_intervals) > 0:
            rr_mean_sec = np.mean(rr_intervals) / self.sampling_rate
            features["heart_rate"] = (
                float(60 / rr_mean_sec) if rr_mean_sec > 0 else 70.0
            )
            features["RR_mean"] = float(rr_mean_sec * 1000)  # in ms
        else:
            features["heart_rate"] = 70.0  # Fallback
            features["RR_mean"] = 857.0  # 70 bpm

        # Step 3: Per-lead features with R-peak based waveform detection
        for lead_idx, lead_name in enumerate(self.lead_names):
            if lead_idx >= ecg_signal.shape[1]:
                continue

            lead_data_raw = ecg_signal[:, lead_idx]

            # Apply baseline wander correction
            lead_data = self._remove_baseline_wander(lead_data_raw)

            # Calculate global baseline for this lead (ONCE, not per beat)
            # Use first 200ms after filtering as isoelectric baseline
            baseline_samples = int(
                0.2 * self.sampling_rate
            )  # 200ms at 500 Hz = 100 samples
            global_baseline = np.mean(
                lead_data[: min(baseline_samples, len(lead_data))]
            )

            # Basic signal statistics (Problem #12 - can be vectorized later)
            features[f"{lead_name.lower()}_mean"] = float(np.mean(lead_data))
            features[f"{lead_name.lower()}_std"] = float(np.std(lead_data))
            features[f"{lead_name.lower()}_max"] = float(np.max(lead_data))
            features[f"{lead_name.lower()}_min"] = float(np.min(lead_data))
            features[f"{lead_name.lower()}_ptp"] = float(np.ptp(lead_data))

            # Initialize waveform-based features (will be averaged across beats)
            p_amplitudes, p_durations = [], []
            qrs_durations, q_amplitudes, r_amplitudes, s_amplitudes = [], [], [], []
            st_elevations, st_depressions, st_slopes = [], [], []
            t_amplitudes, t_durations, t_inversion_depths = [], [], []
            qt_intervals, pr_intervals = [], []
            qrs_fragmentations = []

            # Process each heartbeat (R-peak based waveform detection - Problems #2, #3 FIX)
            for i, r_peak in enumerate(r_peaks):
                # Get RR interval for this beat (for adaptive windows)
                rr_interval = rr_intervals[i] if i < len(rr_intervals) else None

                # Detect QRS boundaries
                qrs_onset, qrs_offset = self._detect_qrs_boundaries(lead_data, r_peak)

                # QRS duration (Problem #1 FIX - no more hardcoded 90ms)
                qrs_dur_ms = (qrs_offset - qrs_onset) / self.sampling_rate * 1000
                qrs_durations.append(qrs_dur_ms)

                # Q, R, S amplitudes (measured relative to R-peak within QRS)
                qrs_region = lead_data[qrs_onset : qrs_offset + 1]

                # Find R-peak within QRS complex
                r_peak_local = np.argmax(np.abs(qrs_region))
                r_amp = float(np.abs(qrs_region[r_peak_local]))

                # Q amplitude: minimum BEFORE R-peak (always take measured value)
                q_region = (
                    qrs_region[:r_peak_local] if r_peak_local > 0 else np.array([0.0])
                )
                q_amp = abs(float(np.min(q_region))) if len(q_region) > 0 else 0.0

                # S amplitude: minimum AFTER R-peak (always take measured value)
                s_region = (
                    qrs_region[r_peak_local + 1 :]
                    if r_peak_local < len(qrs_region) - 1
                    else np.array([0.0])
                )
                s_amp = abs(float(np.min(s_region))) if len(s_region) > 0 else 0.0

                q_amplitudes.append(q_amp)
                r_amplitudes.append(r_amp)
                s_amplitudes.append(s_amp)

                # QRS fragmentation
                qrs_fragmentations.append(self._count_notches(qrs_region))

                # Detect P-wave (Problem #6 FIX)
                p_wave = self._detect_p_wave(lead_data, r_peak, qrs_onset, rr_interval)
                if p_wave[0] is not None:
                    p_onset, p_peak, p_end = p_wave
                    p_amplitudes.append(float(lead_data[p_peak]))
                    p_dur_ms = (p_end - p_onset) / self.sampling_rate * 1000
                    p_durations.append(p_dur_ms)

                    # PR interval (Problem #1 FIX - measured, not hardcoded)
                    pr_int_ms = (qrs_onset - p_onset) / self.sampling_rate * 1000
                    pr_intervals.append(pr_int_ms)

                # Detect T-wave (Problem #7 FIX)
                next_r_peak = r_peaks[i + 1] if i + 1 < len(r_peaks) else None
                t_wave = self._detect_t_wave(lead_data, qrs_offset, next_r_peak)
                if t_wave[0] is not None:
                    t_onset, t_peak, t_end = t_wave
                    t_amp = float(lead_data[t_peak])
                    t_amplitudes.append(t_amp)

                    t_dur_ms = (t_end - t_onset) / self.sampling_rate * 1000
                    t_durations.append(t_dur_ms)

                    # T-wave inversion depth
                    if t_amp < 0:
                        t_inversion_depths.append(abs(t_amp))
                    else:
                        t_inversion_depths.append(0.0)

                    # QT interval (Problem #1 FIX - measured, not hardcoded 400ms)
                    qt_int_ms = (t_end - qrs_onset) / self.sampling_rate * 1000
                    qt_intervals.append(qt_int_ms)

                # ST segment (Problem #2 FIX - relative to QRS offset, not "middle of signal")
                st_start = qrs_offset + int(0.02 * self.sampling_rate)  # J+20ms
                st_end = min(
                    len(lead_data), qrs_offset + int(0.08 * self.sampling_rate)
                )  # J+80ms

                if st_end > st_start and st_start < len(lead_data):
                    st_segment = lead_data[st_start:st_end]
                    # Use global baseline (not per-beat baseline) for consistent ST measurement
                    st_level = np.mean(st_segment) - global_baseline

                    st_elevations.append(max(0, st_level))
                    st_depressions.append(max(0, -st_level))

                    # ST slope
                    if len(st_segment) > 1:
                        slope = (st_segment[-1] - st_segment[0]) / len(st_segment)
                        st_slopes.append(slope)

            # Store averaged per-lead features
            features[f"P_amplitude_{lead_name}"] = (
                float(np.mean(p_amplitudes)) if p_amplitudes else 0.0
            )
            features[f"P_duration_{lead_name}"] = (
                float(np.mean(p_durations)) if p_durations else 80.0
            )

            features[f"Q_amplitude_{lead_name}"] = (
                float(np.mean(q_amplitudes)) if q_amplitudes else 0.0
            )
            features[f"R_amplitude_{lead_name}"] = (
                float(np.mean(r_amplitudes)) if r_amplitudes else 0.0
            )
            features[f"S_amplitude_{lead_name}"] = (
                float(np.mean(s_amplitudes)) if s_amplitudes else 0.0
            )
            features[f"QRS_duration_{lead_name}"] = (
                float(np.mean(qrs_durations)) if qrs_durations else 90.0
            )
            features[f"QRS_fragmentation_{lead_name}"] = (
                float(np.mean(qrs_fragmentations)) if qrs_fragmentations else 0.0
            )

            features[f"ST_elevation_{lead_name}"] = (
                float(np.mean(st_elevations)) if st_elevations else 0.0
            )
            features[f"ST_depression_{lead_name}"] = (
                float(np.mean(st_depressions)) if st_depressions else 0.0
            )
            features[f"J_point_elevation_{lead_name}"] = (
                features[f"ST_elevation_{lead_name}"]
                - features[f"ST_depression_{lead_name}"]
            )
            features[f"ST_slope_{lead_name}"] = (
                float(np.mean(st_slopes)) if st_slopes else 0.0
            )

            features[f"T_amplitude_{lead_name}"] = (
                float(np.mean(t_amplitudes)) if t_amplitudes else 0.0
            )
            features[f"T_duration_{lead_name}"] = (
                float(np.mean(t_durations)) if t_durations else 160.0
            )
            features[f"T_wave_inversion_depth_{lead_name}"] = (
                float(np.mean(t_inversion_depths)) if t_inversion_depths else 0.0
            )

            # Backward compatibility (deprecated, will be removed)
            features[f"qrs_amplitude_{lead_name.lower()}"] = features[
                f"R_amplitude_{lead_name}"
            ]
            features[f"st_level_{lead_name.lower()}"] = features[
                f"J_point_elevation_{lead_name}"
            ]

        # Global interval features (Problem #1 FIX)
        features["PR_interval"] = (
            float(np.mean(pr_intervals)) if pr_intervals else 160.0
        )
        features["QT_interval"] = (
            float(np.mean(qt_intervals)) if qt_intervals else 400.0
        )

        # QTc (Bazett's formula) with real QT and RR
        rr_sec = features["RR_mean"] / 1000
        features["QTc_interval"] = (
            float(features["QT_interval"] / np.sqrt(rr_sec))
            if rr_sec > 0
            else features["QT_interval"]
        )

        # Interval ratios
        features["QT_RR_ratio"] = (
            float(features["QT_interval"] / features["RR_mean"])
            if features["RR_mean"] > 0
            else 0.4
        )
        features["PR_RR_ratio"] = (
            float(features["PR_interval"] / features["RR_mean"])
            if features["RR_mean"] > 0
            else 0.16
        )
        features["T_QT_ratio"] = (
            float(
                np.mean([features[f"T_duration_{lead}"] for lead in ["II", "V5"]])
                / features["QT_interval"]
            )
            if features["QT_interval"] > 0
            else 0.6
        )

        # HRV features (time-domain)
        if len(rr_intervals) > 5:
            rr_ms = rr_intervals / self.sampling_rate * 1000
            features["SDNN"] = float(np.std(rr_ms))

            successive_diffs = np.diff(rr_ms)
            features["RMSSD"] = float(np.sqrt(np.mean(successive_diffs**2)))
            features["pNN50"] = float(
                np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100
            )

            # HRV triangular index
            if len(rr_ms) > 20:
                hist, _ = np.histogram(rr_ms, bins=50)
                if np.max(hist) > 0:
                    features["HRV_triangular_index"] = float(len(rr_ms) / np.max(hist))

            # HRV frequency-domain (Problem #4 FIX - Welch's PSD instead of pseudocode)
            if len(rr_ms) > 20:
                hrv_psd_features = self._calculate_hrv_psd(rr_ms)
                features.update(hrv_psd_features)
            else:
                features["VLF_power"] = 0.0
                features["LF_power"] = 0.0
                features["HF_power"] = 0.0
                features["LF_HF_ratio"] = 0.0

        # Electrical axes
        if all(f"R_amplitude_{lead}" in features for lead in ["I", "aVF"]):
            features["QRS_axis"] = self._calculate_axis(
                features["R_amplitude_I"], features["R_amplitude_aVF"]
            )
            features["P_axis"] = self._calculate_axis(
                features["P_amplitude_I"], features["P_amplitude_aVF"]
            )

        # Territory-specific markers (will be fixed in next task - Problem #9)
        territory_features = self._calculate_territory_markers(features)
        features.update(territory_features)

        # T-wave symmetry (Problem #8 FIX - R-peak based, not "last 20%")
        t_symmetry = self._calculate_t_symmetry_rpeak_based(ecg_signal, r_peaks)
        features["T_symmetry"] = t_symmetry

        print(f"[ECGFeatureExtractor] Extracted {len(features)} features")
        return features

    def _detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks using adaptive threshold"""
        # Bandpass filter (5-15 Hz for QRS enhancement)
        sos = scipy_signal.butter(
            4, [5, 15], btype="bandpass", fs=self.sampling_rate, output="sos"
        )
        filtered = scipy_signal.sosfilt(sos, signal)

        # Square and integrate
        squared = filtered**2
        window_size = int(0.12 * self.sampling_rate)
        integrated = np.convolve(
            squared, np.ones(window_size) / window_size, mode="same"
        )

        # Adaptive threshold
        threshold = 0.5 * np.max(integrated)
        peaks = []
        min_distance = int(0.2 * self.sampling_rate)

        for i in range(1, len(integrated) - 1):
            if (
                integrated[i] > threshold
                and integrated[i] > integrated[i - 1]
                and integrated[i] > integrated[i + 1]
            ):
                if len(peaks) == 0 or (i - peaks[-1]) > min_distance:
                    search_window = int(0.05 * self.sampling_rate)
                    start = max(0, i - search_window)
                    end = min(len(signal), i + search_window)
                    local_peak = start + np.argmax(np.abs(signal[start:end]))
                    peaks.append(local_peak)

        return np.array(peaks)

    def _calculate_axis(self, lead_i: float, lead_avf: float) -> float:
        """Calculate electrical axis in degrees"""
        if lead_i == 0 and lead_avf == 0:
            return 0.0
        return float(np.arctan2(lead_avf, lead_i) * 180 / np.pi)

    def _count_notches(self, signal: np.ndarray, threshold: float = 0.1) -> int:
        """Count notches/fragmentations in a signal"""
        if len(signal) < 3:
            return 0

        diff = np.diff(signal)
        sign_changes = np.diff(np.sign(diff))
        notches = np.sum(np.abs(sign_changes) > threshold)
        return int(notches)

    def _remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove baseline wander using highpass filter

        Baseline wander (< 0.5 Hz) can interfere with ST-segment measurements.
        This removes low-frequency drift while preserving clinical features.

        Args:
            signal: Raw ECG signal

        Returns:
            Filtered signal with baseline wander removed
        """
        if len(signal) < 10:
            return signal

        # Highpass filter at 0.5 Hz to remove baseline wander
        # Preserves ST segment (0.05-5 Hz) and all other clinical features
        try:
            sos = scipy_signal.butter(
                2, 0.5, btype="highpass", fs=self.sampling_rate, output="sos"
            )
            filtered = scipy_signal.sosfilt(sos, signal)
            return filtered
        except Exception:
            # If filtering fails, return original signal
            return signal

    def _detect_qrs_boundaries(
        self, signal: np.ndarray, r_peak: int
    ) -> tuple[int, int]:
        """
        Detect QRS onset and offset using gradient-based method

        Searches backwards from R-peak for QRS onset (where signal starts rising rapidly)
        Searches forwards from R-peak for QRS offset (where signal returns to baseline)

        Args:
            signal: ECG lead signal
            r_peak: Index of R-peak

        Returns:
            Tuple of (qrs_onset_index, qrs_offset_index)
        """
        # Default search windows (typical QRS is 60-120ms = 30-60 samples at 500 Hz)
        search_back = min(r_peak, int(0.08 * self.sampling_rate))  # 80ms before R
        search_forward = min(
            len(signal) - r_peak, int(0.08 * self.sampling_rate)
        )  # 80ms after R

        # Calculate gradient (slope)
        gradient = np.gradient(signal)

        # QRS Onset: Search backwards for where gradient exceeds threshold
        onset_threshold = 0.1 * np.max(
            np.abs(gradient[max(0, r_peak - search_back) : r_peak])
        )
        qrs_onset = r_peak - search_back

        for i in range(r_peak - 1, max(0, r_peak - search_back), -1):
            if np.abs(gradient[i]) < onset_threshold:
                qrs_onset = i
                break

        # QRS Offset: Search forwards for where gradient returns near zero
        offset_threshold = 0.1 * np.max(
            np.abs(gradient[r_peak : min(len(signal), r_peak + search_forward)])
        )
        qrs_offset = r_peak + search_forward

        for i in range(r_peak + 1, min(len(signal), r_peak + search_forward)):
            if np.abs(gradient[i]) < offset_threshold:
                qrs_offset = i
                break

        # Fallback: if detection failed, use typical durations
        if qrs_offset <= qrs_onset:
            qrs_onset = max(0, r_peak - int(0.04 * self.sampling_rate))  # 40ms before R
            qrs_offset = min(
                len(signal) - 1, r_peak + int(0.06 * self.sampling_rate)
            )  # 60ms after R

        return qrs_onset, qrs_offset

    def _detect_p_wave(
        self,
        signal: np.ndarray,
        r_peak: int,
        qrs_onset: int,
        rr_interval: float | None = None,
    ) -> tuple[int | None, int | None, int | None]:
        """
        Detect P-wave before QRS complex

        P-wave occurs during atrial depolarization, typically 120-200ms before QRS onset.
        Search window is adaptive to heart rate (60% of RR interval if available).

        Args:
            signal: ECG lead signal
            r_peak: Index of R-peak
            qrs_onset: Index of QRS onset
            rr_interval: RR interval in samples (optional, for adaptive window)

        Returns:
            Tuple of (p_onset_index, p_peak_index, p_end_index) or (None, None, None) if not found
        """
        # Determine search window (adaptive to heart rate)
        if rr_interval is not None and rr_interval > 0:
            # Adaptive: search in 60% of RR interval before QRS onset
            search_window = int(0.6 * rr_interval)
        else:
            # Default: 200ms before QRS onset
            search_window = int(0.2 * self.sampling_rate)

        search_start = max(0, qrs_onset - search_window)
        search_end = qrs_onset - int(
            0.04 * self.sampling_rate
        )  # At least 40ms before QRS

        if search_end <= search_start:
            return None, None, None

        # Extract P-wave search region
        p_region = signal[search_start:search_end]

        if len(p_region) < 10:
            return None, None, None

        # Find P-wave peak (max absolute value in search region)
        p_peak_local = np.argmax(np.abs(p_region))
        p_peak = search_start + p_peak_local

        # Find P-wave onset (where signal starts rising before peak)
        gradient = np.gradient(p_region)
        p_onset_local = 0
        threshold = (
            0.2 * np.max(np.abs(gradient[: p_peak_local + 1]))
            if p_peak_local > 0
            else 0
        )

        for i in range(p_peak_local - 1, -1, -1):
            if np.abs(gradient[i]) < threshold:
                p_onset_local = i
                break

        # Find P-wave end (where signal flattens after peak)
        p_end_local = len(p_region) - 1
        for i in range(p_peak_local + 1, len(p_region)):
            if np.abs(gradient[i]) < threshold:
                p_end_local = i
                break

        p_onset = search_start + p_onset_local
        p_end = search_start + p_end_local

        # Sanity check: P-wave duration should be 60-120ms
        p_duration_ms = (p_end - p_onset) / self.sampling_rate * 1000
        if p_duration_ms < 40 or p_duration_ms > 200:
            return None, None, None

        return p_onset, p_peak, p_end

    def _detect_t_wave(
        self, signal: np.ndarray, qrs_offset: int, next_r_peak: int | None = None
    ) -> tuple[int | None, int | None, int | None]:
        """
        Detect T-wave after QRS complex

        T-wave occurs during ventricular repolarization, typically 200-400ms after QRS offset.
        Search ends at next R-peak or 500ms after QRS offset.

        Args:
            signal: ECG lead signal
            qrs_offset: Index of QRS offset (end of QRS)
            next_r_peak: Index of next R-peak (optional, limits search window)

        Returns:
            Tuple of (t_onset_index, t_peak_index, t_end_index) or (None, None, None) if not found
        """
        # Determine search window
        if next_r_peak is not None and next_r_peak > qrs_offset:
            # Search until next R-peak
            search_end = next_r_peak - int(
                0.05 * self.sampling_rate
            )  # Stop 50ms before next R
        else:
            # Default: 500ms after QRS offset
            search_end = min(len(signal), qrs_offset + int(0.5 * self.sampling_rate))

        search_start = qrs_offset + int(
            0.05 * self.sampling_rate
        )  # T starts ~50ms after QRS

        if search_end <= search_start or search_start >= len(signal):
            return None, None, None

        # Extract T-wave search region
        t_region = signal[search_start:search_end]

        if len(t_region) < 10:
            return None, None, None

        # T-onset is approximately at QRS offset (ST segment transition)
        t_onset = qrs_offset

        # Find T-wave peak (max absolute value in search region)
        t_peak_local = np.argmax(np.abs(t_region))
        t_peak = search_start + t_peak_local

        # Find T-wave end using tangent method
        # T-end is where signal returns to baseline (small slope)
        gradient = np.gradient(t_region)
        baseline_threshold = 0.1 * np.max(np.abs(gradient))

        t_end_local = len(t_region) - 1
        for i in range(t_peak_local + 1, len(t_region)):
            if np.abs(gradient[i]) < baseline_threshold:
                t_end_local = i
                break

        t_end = search_start + t_end_local

        # Sanity check: T-wave duration should be 100-300ms
        t_duration_ms = (t_end - t_onset) / self.sampling_rate * 1000
        if t_duration_ms < 80 or t_duration_ms > 400:
            return None, None, None

        return t_onset, t_peak, t_end

    def _calculate_hrv_psd(self, rr_intervals: np.ndarray) -> dict[str, float]:
        """
        Calculate HRV frequency-domain features using Welch's Power Spectral Density

        Computes power in VLF, LF, and HF bands according to HRV standards:
        - VLF (Very Low Frequency): 0.003-0.04 Hz
        - LF (Low Frequency): 0.04-0.15 Hz - sympathetic + parasympathetic
        - HF (High Frequency): 0.15-0.4 Hz - parasympathetic (respiratory)

        Args:
            rr_intervals: Array of RR intervals in milliseconds

        Returns:
            Dictionary with VLF_power, LF_power, HF_power, LF_HF_ratio
        """
        hrv_features = {}

        if len(rr_intervals) < 20:
            # Not enough data for PSD
            hrv_features["VLF_power"] = 0.0
            hrv_features["LF_power"] = 0.0
            hrv_features["HF_power"] = 0.0
            hrv_features["LF_HF_ratio"] = 0.0
            return hrv_features

        # Resample RR intervals to evenly spaced time series (required for PSD)
        # Use 4 Hz resampling rate (standard for HRV analysis)
        fs_resample = 4.0  # Hz
        # Time array starts at 0, then cumulative sum of RR intervals
        time_original = (
            np.concatenate([[0], np.cumsum(rr_intervals)]) / 1000
        )  # Convert to seconds
        time_resampled = np.arange(0, time_original[-1], 1 / fs_resample)

        # Interpolate RR intervals to uniform time grid
        # Prepend first RR value for interpolation at t=0
        rr_with_initial = np.concatenate([[rr_intervals[0]], rr_intervals])
        rr_resampled = np.interp(time_resampled, time_original, rr_with_initial)

        # Detrend to remove baseline drift
        rr_detrended = scipy_signal.detrend(rr_resampled)

        # Calculate Power Spectral Density using Welch's method
        try:
            frequencies, psd = scipy_signal.welch(
                rr_detrended,
                fs=fs_resample,
                nperseg=min(256, len(rr_detrended)),
                scaling="density",
            )

            # Calculate power in each frequency band
            # VLF: 0.003-0.04 Hz
            vlf_mask = (frequencies >= 0.003) & (frequencies < 0.04)
            vlf_power = (
                np.trapz(psd[vlf_mask], frequencies[vlf_mask])
                if np.any(vlf_mask)
                else 0.0
            )

            # LF: 0.04-0.15 Hz
            lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
            lf_power = (
                np.trapz(psd[lf_mask], frequencies[lf_mask]) if np.any(lf_mask) else 0.0
            )

            # HF: 0.15-0.4 Hz
            hf_mask = (frequencies >= 0.15) & (frequencies < 0.4)
            hf_power = (
                np.trapz(psd[hf_mask], frequencies[hf_mask]) if np.any(hf_mask) else 0.0
            )

            # LF/HF ratio
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

            hrv_features["VLF_power"] = float(vlf_power)
            hrv_features["LF_power"] = float(lf_power)
            hrv_features["HF_power"] = float(hf_power)
            hrv_features["LF_HF_ratio"] = float(lf_hf_ratio)

        except Exception:
            # If PSD calculation fails, return zeros
            hrv_features["VLF_power"] = 0.0
            hrv_features["LF_power"] = 0.0
            hrv_features["HF_power"] = 0.0
            hrv_features["LF_HF_ratio"] = 0.0

        return hrv_features

    def _calculate_territory_markers(
        self, features: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate territory-specific markers (ONLY MEASUREMENTS, NO INTERPRETATION)

        Territories:
        - Anterior Wall (LAD): V1-V4
        - Inferior Wall (RCA): II, III, aVF
        - Lateral Wall (LCX): I, aVL, V5, V6
        - Global Patterns: aVR

        Returns:
            Dictionary with territory-averaged raw measurements (no scores or binary flags)
        """
        territory_features = {}

        # === ANTERIOR WALL (LAD Territory - V1-V4) ===
        anterior_leads = ["V1", "V2", "V3", "V4"]

        # Average ST elevation in V1-V3 (measurement only)
        v1_v3_st_elev = np.mean(
            [features.get(f"ST_elevation_{lead}", 0.0) for lead in ["V1", "V2", "V3"]]
        )
        territory_features["V1_V3_ST_elevation"] = float(v1_v3_st_elev)

        # Average T-wave inversion depth in V1-V4 (measurement only)
        v1_v4_t_inv = np.mean(
            [
                features.get(f"T_wave_inversion_depth_{lead}", 0.0)
                for lead in anterior_leads
            ]
        )
        territory_features["V1_V4_T_inversion"] = float(v1_v4_t_inv)

        # Q-wave measurements for V1 (raw values, no binary flag)
        q_v1 = abs(features.get("Q_amplitude_V1", 0.0))
        r_v1 = features.get("R_amplitude_V1", 1.0)
        territory_features["V1_Q_amplitude"] = float(q_v1)
        territory_features["V1_Q_to_R_ratio"] = float(q_v1 / r_v1) if r_v1 > 0 else 0.0

        # === INFERIOR WALL (RCA Territory - II, III, aVF) ===
        inferior_leads = ["II", "III", "aVF"]

        # Average ST elevation in inferior leads (measurement only)
        inf_st_elev = np.mean(
            [features.get(f"ST_elevation_{lead}", 0.0) for lead in inferior_leads]
        )
        territory_features["II_III_aVF_ST_elevation"] = float(inf_st_elev)

        # Average T-wave inversion in inferior leads (measurement only)
        inf_t_inv = np.mean(
            [
                features.get(f"T_wave_inversion_depth_{lead}", 0.0)
                for lead in inferior_leads
            ]
        )
        territory_features["II_III_aVF_T_inversion"] = float(inf_t_inv)

        # Q-wave measurements for III (raw values, no binary flag)
        q_iii = abs(features.get("Q_amplitude_III", 0.0))
        r_iii = features.get("R_amplitude_III", 1.0)
        territory_features["III_Q_amplitude"] = float(q_iii)
        territory_features["III_Q_to_R_ratio"] = (
            float(q_iii / r_iii) if r_iii > 0 else 0.0
        )

        # === LATERAL WALL (LCX Territory - I, aVL, V5, V6) ===
        lateral_leads = ["I", "aVL", "V5", "V6"]

        # Average ST elevation in lateral leads (measurement only)
        lat_st_elev = np.mean(
            [features.get(f"ST_elevation_{lead}", 0.0) for lead in lateral_leads]
        )
        territory_features["I_aVL_V5_V6_ST_elevation"] = float(lat_st_elev)

        # Average T-wave inversion in lateral leads (measurement only)
        lat_t_inv = np.mean(
            [
                features.get(f"T_wave_inversion_depth_{lead}", 0.0)
                for lead in lateral_leads
            ]
        )
        territory_features["I_aVL_V5_V6_T_inversion"] = float(lat_t_inv)

        # Q-wave measurements for V5 and V6 (raw values, no binary flags)
        q_v5 = abs(features.get("Q_amplitude_V5", 0.0))
        r_v5 = features.get("R_amplitude_V5", 1.0)
        q_v6 = abs(features.get("Q_amplitude_V6", 0.0))
        r_v6 = features.get("R_amplitude_V6", 1.0)

        territory_features["V5_Q_amplitude"] = float(q_v5)
        territory_features["V5_Q_to_R_ratio"] = float(q_v5 / r_v5) if r_v5 > 0 else 0.0
        territory_features["V6_Q_amplitude"] = float(q_v6)
        territory_features["V6_Q_to_R_ratio"] = float(q_v6 / r_v6) if r_v6 > 0 else 0.0

        # === GLOBAL PATTERNS ===
        # aVR ST elevation (already measured per-lead, just copy for convenience)
        territory_features["aVR_ST_elevation"] = float(
            features.get("ST_elevation_aVR", 0.0)
        )

        return territory_features

    def _calculate_t_symmetry_global(self, ecg_signal: np.ndarray) -> float:
        """
        Calculate global T-wave symmetry across all leads (DEPRECATED - uses last 20%)

        DEPRECATED: This function uses "last 20% of signal" which is incorrect
        for variable RR intervals. Use _calculate_t_symmetry_rpeak_based() instead.

        Args:
            ecg_signal: ECG signal array (samples, 12 leads)

        Returns:
            T-wave symmetry ratio (0.0 - 2.0, with 1.0 being perfectly symmetric)
        """
        symmetry_values = []

        for lead_idx in range(min(12, ecg_signal.shape[1])):
            lead_data = ecg_signal[:, lead_idx]

            # T-wave is in last 20% of signal (WRONG for variable RR intervals)
            t_wave_region = lead_data[-int(0.2 * self.sampling_rate) :]

            if len(t_wave_region) < 10:
                continue

            # Find T-wave peak (max absolute value)
            t_peak_idx = np.argmax(np.abs(t_wave_region))

            # Ascending limb: start to peak
            ascending_duration = t_peak_idx

            # Descending limb: peak to end
            descending_duration = len(t_wave_region) - t_peak_idx

            # Calculate ratio (avoid division by zero)
            if descending_duration > 0:
                ratio = ascending_duration / descending_duration
                # Clamp to reasonable range
                ratio = max(0.1, min(2.0, ratio))
                symmetry_values.append(ratio)

        # Return average symmetry across all leads
        if symmetry_values:
            return float(np.mean(symmetry_values))
        else:
            return 1.0  # Default: perfectly symmetric

    def _calculate_t_symmetry_rpeak_based(
        self, ecg_signal: np.ndarray, r_peaks: np.ndarray
    ) -> float:
        """
        Calculate global T-wave symmetry using R-peak based T-wave detection (Problem #8 FIX)

        T-wave symmetry = ratio of ascending to descending limb duration
        - Symmetric T-wave: ratio ~ 1.0
        - Asymmetric T-wave: ratio << 1.0 or >> 1.0 (pathological)

        Args:
            ecg_signal: ECG signal array (samples, 12 leads)
            r_peaks: Array of R-peak indices

        Returns:
            T-wave symmetry ratio (0.0 - 2.0, with 1.0 being perfectly symmetric)
        """
        if len(r_peaks) < 2:
            return 1.0  # Not enough data

        symmetry_values = []

        # Analyze T-waves across all leads
        for lead_idx in range(min(12, ecg_signal.shape[1])):
            lead_data_raw = ecg_signal[:, lead_idx]
            lead_data = self._remove_baseline_wander(lead_data_raw)

            # Process each heartbeat
            for i in range(len(r_peaks) - 1):
                r_peak = r_peaks[i]
                next_r_peak = r_peaks[i + 1]

                # Detect QRS offset
                _, qrs_offset = self._detect_qrs_boundaries(lead_data, r_peak)

                # Detect T-wave
                t_wave = self._detect_t_wave(lead_data, qrs_offset, next_r_peak)
                if t_wave[0] is None:
                    continue

                t_onset, t_peak, t_end = t_wave

                # Calculate T-wave limb durations
                ascending_duration = t_peak - t_onset
                descending_duration = t_end - t_peak

                # Calculate symmetry ratio
                if descending_duration > 0 and ascending_duration > 0:
                    ratio = ascending_duration / descending_duration
                    # Clamp to reasonable range
                    ratio = max(0.1, min(2.0, ratio))
                    symmetry_values.append(ratio)

        # Return average symmetry across all leads and beats
        if symmetry_values:
            return float(np.mean(symmetry_values))
        else:
            return 1.0  # Default: perfectly symmetric


def main() -> None:
    fpath = Path(r"C:\Users\R.Koehler\Data\v2.1.0\cad_int_rest_approach3A_test1.h5")
    ecg_signals, sampling_rate = load_ecg(fpath)
    extractor = ECGFeatureExtractor(sampling_rate=sampling_rate)
    features = extractor.extract_features(ecg_signals[10])
    ...


def load_ecg(fpath: Path) -> tuple[np.ndarray, int]:
    with h5py.File(fpath, "r") as root:
        sequences = np.array(root["ecg_data/sequences"], dtype=float)
        sfreq = root["ecg_data"].attrs["sampling_rate"]
    return sequences, sfreq


if __name__ == "__main__":
    main()
