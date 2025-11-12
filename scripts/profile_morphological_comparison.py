#!/usr/bin/env python3
"""
Profiling script for morphological feature extraction comparing different sampling frequencies.

This script profiles the _morph_single_patient function at different sampling frequencies
to identify how sampling rate affects performance and accuracy.
"""

import time
import warnings
from typing import Any

import neurokit2 as nk
import numpy as np

# Import the function to profile
from pte_ecg.features import _get_r_peaks, _morph_single_patient


def load_real_ecg_data(n_channels: int = 12, duration: float = 10.0, sfreq: float = 500.0) -> np.ndarray:
    """Load real ECG data from NeuroKit2 and replicate it across channels."""
    # Load a real ECG signal from NeuroKit2
    ecg_signal = nk.data("ecg_3000hz")

    # Resample to desired frequency if needed
    original_sfreq = 3000  # Hz
    if sfreq != original_sfreq:
        from scipy import signal

        n_original = len(ecg_signal)
        n_resampled = int(n_original * sfreq / original_sfreq)
        ecg_signal = signal.resample(ecg_signal, n_resampled)

    # Extract desired duration
    n_samples = int(duration * sfreq)
    if len(ecg_signal) > n_samples:
        ecg_signal = ecg_signal[:n_samples]
    elif len(ecg_signal) < n_samples:
        # Repeat the signal if it's too short
        repeats = int(np.ceil(n_samples / len(ecg_signal)))
        ecg_signal = np.tile(ecg_signal, repeats)[:n_samples]

    # Create multi-channel data by adding slight variations
    np.random.seed(42)  # For reproducible results
    ecg_data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Add slight amplitude and phase variations for each channel
        amplitude_variation = 0.8 + 0.4 * np.random.randn()
        noise_level = 0.05 * np.std(ecg_signal)
        noise = noise_level * np.random.randn(n_samples)

        # Add small phase shift to simulate different lead positions
        phase_shift = np.random.randint(-10, 11)  # ±10 samples
        if phase_shift > 0:
            shifted_signal = np.concatenate([ecg_signal[phase_shift:], ecg_signal[:phase_shift]])
        elif phase_shift < 0:
            shifted_signal = np.concatenate([ecg_signal[phase_shift:], ecg_signal[:phase_shift]])
        else:
            shifted_signal = ecg_signal

        ecg_data[ch] = amplitude_variation * shifted_signal + noise

    return ecg_data


def profile_with_line_profiler(func, *args, **kwargs) -> tuple[Any, float]:
    """Profile a function with simple timing."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    return result, execution_time


def profile_ecg_delineation_methods(sample_data: np.ndarray, sfreq: float) -> dict[str, dict[str, Any]]:
    """Profile each ECG delineation method separately."""
    # Use first channel for method comparison
    ch_data = sample_data[0]

    # Get R-peaks first
    r_peaks, n_r_peaks, r_peak_method = _get_r_peaks(ch_data, sfreq)

    if n_r_peaks == 0:
        return {"error": {"message": "No R-peaks detected"}}

    methods = ["dwt", "prominence", "peak", "cwt"]
    method_results = {}

    for method in methods:
        method_info = {"success": False, "time": 0.0, "error": None, "features_detected": 0}

        # Skip methods that require more R-peaks
        if n_r_peaks < 2 and method in {"prominence", "cwt"}:
            method_info["error"] = "Not enough R-peaks for this method"
            method_results[method] = method_info
            continue

        # Time the delineation
        start_time = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", nk.misc.NeuroKitWarning)
                _, waves_dict = nk.ecg_delineate(
                    ch_data,
                    rpeaks=r_peaks,
                    sampling_rate=sfreq,
                    method=method,
                )

            delineation_time = time.perf_counter() - start_time
            method_info["time"] = delineation_time
            method_info["success"] = True

            # Count detected features
            feature_count = 0
            for key, values in waves_dict.items():
                if values is not None and len(values) > 0:
                    feature_count += len([v for v in values if not np.isnan(v)])
            method_info["features_detected"] = feature_count

        except nk.misc.NeuroKitWarning as e:
            delineation_time = time.perf_counter() - start_time
            method_info["time"] = delineation_time
            method_info["error"] = str(e)
        except Exception as e:
            delineation_time = time.perf_counter() - start_time
            method_info["time"] = delineation_time
            method_info["error"] = f"Unexpected error: {str(e)}"

        method_results[method] = method_info

    return method_results


def detailed_timing_profile(sample_data: np.ndarray, sfreq: float) -> dict[str, Any]:
    """Manually profile different parts of the morphological feature extraction."""
    timing_results = {}

    # Time the overall function
    start_time = time.perf_counter()
    result = _morph_single_patient(0, sample_data, sfreq)
    total_time = time.perf_counter() - start_time
    timing_results["total_time"] = total_time

    # Now profile individual components for a single channel
    ch_data = sample_data[0]  # Use first channel

    # Time R-peak detection
    start_time = time.perf_counter()
    r_peaks, n_r_peaks, r_peak_method = _get_r_peaks(ch_data, sfreq)
    r_peak_time = time.perf_counter() - start_time
    timing_results["r_peak_detection"] = r_peak_time
    timing_results["n_r_peaks"] = n_r_peaks

    return timing_results


def profile_at_frequency(sfreq: float, n_channels: int = 12, duration: float = 10.0) -> dict[str, Any]:
    """Profile morphological feature extraction at a specific sampling frequency."""
    print(f"\n=== Profiling at {sfreq} Hz ===")
    print("Loading real ECG data from NeuroKit2...")

    sample_data = load_real_ecg_data(n_channels, duration, sfreq)
    print(f"Loaded ECG data: {sample_data.shape} (channels x samples)")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Duration: {duration:.1f} seconds\n")

    results = {}

    # 1. Simple timing profile
    print("1. Simple Timing Profile:")
    print("-" * 40)
    result, execution_time = profile_with_line_profiler(_morph_single_patient, 0, sample_data, sfreq)
    print(f"Total execution time: {execution_time:.4f} seconds")
    print(f"Features extracted: {len(result)}")
    results["total_time"] = execution_time
    results["features_extracted"] = len(result)
    print()

    # 2. ECG Delineation Method Comparison
    print("2. ECG Delineation Method Comparison:")
    print("-" * 50)
    method_results = profile_ecg_delineation_methods(sample_data, sfreq)

    print(f"{'Method':<12} {'Success':<8} {'Time (s)':<10} {'Features':<10} {'Error':<30}")
    print("-" * 70)

    successful_methods = []
    for method, info in method_results.items():
        success_str = "Y" if info["success"] else "N"
        time_str = f"{info['time']:.4f}" if info["time"] is not None and info["time"] > 0 else "N/A"
        features_str = str(info["features_detected"]) if info["success"] else "N/A"
        error_msg = info.get("error", "") or ""
        error_str = error_msg[:28] + "..." if error_msg and len(error_msg) > 28 else error_msg

        print(f"{method:<12} {success_str:<8} {time_str:<10} {features_str:<10} {error_str:<30}")

        if info["success"]:
            successful_methods.append((method, info["time"]))

    print()
    if successful_methods:
        successful_methods.sort(key=lambda x: x[1])  # Sort by time
        print("Ranking by speed (fastest first):")
        for i, (method, time_taken) in enumerate(successful_methods, 1):
            print(f"{i}. {method}: {time_taken:.4f}s")
        results["fastest_method"] = successful_methods[0][0]
        results["fastest_time"] = successful_methods[0][1]
    print()

    # 3. Detailed timing breakdown
    print("3. Detailed Timing Breakdown:")
    print("-" * 40)
    timing_results = detailed_timing_profile(sample_data, sfreq)
    for component, value in timing_results.items():
        if component == "total_time":
            print(f"{component:25s}: {value:.4f}s (100.0%)")
        elif component == "n_r_peaks":
            print(f"{component:25s}: {value}")
            results["n_r_peaks"] = value
        else:
            percentage = (value / timing_results["total_time"]) * 100 if "total_time" in timing_results else 0
            print(f"{component:25s}: {value:.4f}s ({percentage:5.1f}%)")
    print()

    results["method_results"] = method_results
    results["timing_breakdown"] = timing_results

    return results


def main():
    """Main profiling function comparing different sampling frequencies."""
    print("=== Morphological Feature Extraction Profiling ===\n")
    print("Comparing performance at different sampling frequencies...\n")

    n_channels = 12
    duration = 10.0  # seconds
    frequencies = [500.0, 80.0]

    all_results = {}

    for sfreq in frequencies:
        all_results[sfreq] = profile_at_frequency(sfreq, n_channels, duration)

    # Comparison summary
    print("\n" + "=" * 60)
    print("SAMPLING FREQUENCY COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'500 Hz':<15} {'80 Hz':<15} {'Ratio (500/80)':<15}")
    print("-" * 75)

    # Total execution time comparison
    time_500 = all_results[500.0]["total_time"]
    time_80 = all_results[80.0]["total_time"]
    ratio_time = time_500 / time_80 if time_80 > 0 else float("inf")
    print(f"{'Total Time (s)':<30} {time_500:<15.4f} {time_80:<15.4f} {ratio_time:<15.2f}")

    # Features extracted comparison
    feat_500 = all_results[500.0]["features_extracted"]
    feat_80 = all_results[80.0]["features_extracted"]
    ratio_feat = feat_500 / feat_80 if feat_80 > 0 else float("inf")
    print(f"{'Features Extracted':<30} {feat_500:<15} {feat_80:<15} {ratio_feat:<15.2f}")

    # R-peaks detected comparison
    rpeaks_500 = all_results[500.0]["n_r_peaks"]
    rpeaks_80 = all_results[80.0]["n_r_peaks"]
    ratio_rpeaks = rpeaks_500 / rpeaks_80 if rpeaks_80 > 0 else float("inf")
    print(f"{'R-peaks Detected':<30} {rpeaks_500:<15} {rpeaks_80:<15} {ratio_rpeaks:<15.2f}")

    # Fastest method comparison
    if "fastest_method" in all_results[500.0] and "fastest_method" in all_results[80.0]:
        fastest_500 = all_results[500.0]["fastest_method"]
        fastest_80 = all_results[80.0]["fastest_method"]
        fastest_time_500 = all_results[500.0]["fastest_time"]
        fastest_time_80 = all_results[80.0]["fastest_time"]

        print(f"\n{'Fastest Method':<30} {fastest_500:<15} {fastest_80:<15}")
        print(
            f"{'Fastest Method Time (s)':<30} {fastest_time_500:<15.4f} {fastest_time_80:<15.4f} {fastest_time_500 / fastest_time_80:<15.2f}"
        )

    print("\n" + "=" * 60)
    print("METHOD-BY-METHOD COMPARISON")
    print("=" * 60)

    methods = ["dwt", "prominence", "peak", "cwt"]
    print(
        f"\n{'Method':<12} {'500Hz Time':<12} {'80Hz Time':<12} {'500Hz Feat':<12} {'80Hz Feat':<12} {'Time Ratio':<12}"
    )
    print("-" * 72)

    for method in methods:
        info_500 = all_results[500.0]["method_results"].get(method, {})
        info_80 = all_results[80.0]["method_results"].get(method, {})

        time_500 = info_500.get("time", 0) if info_500.get("success", False) else 0
        time_80 = info_80.get("time", 0) if info_80.get("success", False) else 0
        feat_500 = info_500.get("features_detected", 0) if info_500.get("success", False) else 0
        feat_80 = info_80.get("features_detected", 0) if info_80.get("success", False) else 0

        time_ratio = time_500 / time_80 if time_80 > 0 else float("inf")

        time_500_str = f"{time_500:.4f}" if time_500 > 0 else "FAIL"
        time_80_str = f"{time_80:.4f}" if time_80 > 0 else "FAIL"
        time_ratio_str = f"{time_ratio:.2f}" if time_ratio != float("inf") else "N/A"

        print(f"{method:<12} {time_500_str:<12} {time_80_str:<12} {feat_500:<12} {feat_80:<12} {time_ratio_str:<12}")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    # Performance insights
    if time_500 < time_80:
        print(f"• 500 Hz is {time_80 / time_500:.1f}x FASTER than 80 Hz")
    else:
        print(f"• 80 Hz is {time_500 / time_80:.1f}x FASTER than 500 Hz")

    # Feature detection insights
    if feat_500 > feat_80:
        print(f"• 500 Hz extracts {feat_500 - feat_80} MORE features than 80 Hz")
    elif feat_80 > feat_500:
        print(f"• 80 Hz extracts {feat_80 - feat_500} MORE features than 500 Hz")
    else:
        print("• Both frequencies extract the same number of features")

    # R-peak detection insights
    if rpeaks_500 > rpeaks_80:
        print(f"• 500 Hz detects {rpeaks_500 - rpeaks_80} MORE R-peaks than 80 Hz")
    elif rpeaks_80 > rpeaks_500:
        print(f"• 80 Hz detects {rpeaks_80 - rpeaks_500} MORE R-peaks than 500 Hz")
    else:
        print("• Both frequencies detect the same number of R-peaks")

    print("\n=== Profiling Complete ===")


if __name__ == "__main__":
    main()
