#!/usr/bin/env python3
"""
Profiling script for morphological feature extraction.

This script profiles the _morph_single_patient function to identify performance bottlenecks
in the morphological feature extraction pipeline.
"""

import cProfile
import pstats
import time
import warnings
from io import StringIO
from pathlib import Path
from typing import Any

import neurokit2 as nk
import numpy as np

# Import the function to profile
from src.pte_ecg.features import _get_r_peaks, _morph_single_patient


def load_real_ecg_data(
    n_channels: int = 12, duration: float = 10.0, sfreq: float = 500.0
) -> np.ndarray:
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
            shifted_signal = np.concatenate(
                [ecg_signal[phase_shift:], ecg_signal[:phase_shift]]
            )
        elif phase_shift < 0:
            shifted_signal = np.concatenate(
                [ecg_signal[phase_shift:], ecg_signal[:phase_shift]]
            )
        else:
            shifted_signal = ecg_signal

        ecg_data[ch] = amplitude_variation * shifted_signal + noise

    return ecg_data


def profile_function_with_cprofile(func, *args, **kwargs) -> tuple[Any, pstats.Stats]:
    """Profile a function using cProfile and return results and stats."""
    profiler = cProfile.Profile()

    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Create stats object
    stats = pstats.Stats(profiler)

    return result, stats


def profile_with_line_profiler(func, *args, **kwargs) -> tuple[Any, float]:
    """Profile a function with simple timing."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    return result, execution_time


def profile_ecg_delineation_methods(
    sample_data: np.ndarray, sfreq: float
) -> dict[str, dict[str, float]]:
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
        method_info = {
            "success": False,
            "time": 0.0,
            "error": None,
            "features_detected": 0,
        }

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


def detailed_timing_profile(sample_data: np.ndarray, sfreq: float) -> dict[str, float]:
    """Manually profile different parts of the morphological feature extraction."""
    from src.pte_ecg.features import _get_r_peaks

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

    if n_r_peaks > 0:
        # Time ECG delineation for each method
        methods = ["dwt", "prominence", "peak", "cwt"]
        for method in methods:
            if n_r_peaks < 2 and method in {"prominence", "cwt"}:
                continue

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
                timing_results[f"ecg_delineate_{method}"] = delineation_time
                break  # Only time the first successful method
            except nk.misc.NeuroKitWarning:
                delineation_time = time.perf_counter() - start_time
                timing_results[f"ecg_delineate_{method}_failed"] = delineation_time
                continue

    return timing_results


def profile_at_frequency(sfreq: float, n_channels: int = 12, duration: float = 10.0) -> dict:
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
    results['total_time'] = execution_time
    results['features_extracted'] = len(result)
    print()

    # 2. ECG Delineation Method Comparison
    print("2. ECG Delineation Method Comparison:")
    print("-" * 50)
    method_results = profile_ecg_delineation_methods(sample_data, sfreq)

    print(
        f"{'Method':<12} {'Success':<8} {'Time (s)':<10} {'Features':<10} {'Error':<30}"
    )
    print("-" * 70)

    successful_methods = []
    for method, info in method_results.items():
        success_str = "Y" if info["success"] else "N"
        time_str = f"{info['time']:.4f}" if info['time'] is not None and info['time'] > 0 else "N/A"
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
        else:
            percentage = (
                (value / timing_results["total_time"]) * 100
                if "total_time" in timing_results
                else 0
            )
            print(f"{component:25s}: {value:.4f}s ({percentage:5.1f}%)")
    print()

    # 4. cProfile analysis
    print("4. cProfile Analysis:")
    print("-" * 40)
    result, stats = profile_function_with_cprofile(
        _morph_single_patient, 0, sample_data, sfreq
    )

    # Sort by cumulative time and show top functions
    stats.sort_stats("cumulative")

    # Capture the stats output
    s = StringIO()
    stats.print_stats(20)  # Top 20 functions
    profile_output = s.getvalue()
    print(profile_output)

    # 5. Memory usage analysis (if available)
    try:
        import os

        import psutil

        print("5. Memory Usage Analysis:")
        print("-" * 40)

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run the function
        result = _morph_single_patient(0, sample_data, sfreq)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after:  {memory_after:.2f} MB")
        print(f"Memory used:   {memory_used:.2f} MB")
        print()

    except ImportError:
        print("5. Memory Usage Analysis: psutil not available")
        print()

    # 6. Scaling analysis
    print("6. Scaling Analysis:")
    print("-" * 40)
    channel_counts = [1, 4, 8, 12, 16]

    for n_ch in channel_counts:
        if n_ch <= n_channels:
            test_data = sample_data[:n_ch]
            _, exec_time = profile_with_line_profiler(
                _morph_single_patient, 0, test_data, sfreq
            )
            time_per_channel = exec_time / n_ch
            print(
                f"{n_ch:2d} channels: {exec_time:.4f}s total, {time_per_channel:.4f}s per channel"
            )

    print("\n=== Profiling Complete ===")

    # Save detailed profile to file
    output_file = Path("morphological_profile_results.txt")
    with open(output_file, "w") as f:
        f.write("=== Morphological Feature Extraction Profiling Results ===\n\n")
        f.write(f"Test data: {sample_data.shape} (channels x samples)\n")
        f.write(f"Sampling frequency: {sfreq} Hz\n")
        f.write(f"Duration: {duration:.1f} seconds\n\n")

        f.write("ECG Delineation Method Comparison:\n")
        f.write("-" * 50 + "\n")
        f.write(
            f"{'Method':<12} {'Success':<8} {'Time (s)':<10} {'Features':<10} {'Error':<30}\n"
        )
        f.write("-" * 70 + "\n")

        for method, info in method_results.items():
            success_str = "Y" if info["success"] else "N"
            time_str = f"{info['time']:.4f}" if info["time"] > 0 else "N/A"
            features_str = str(info["features_detected"]) if info["success"] else "N/A"
            error_msg = info.get("error", "")
            error_str = (
                error_msg[:28] + "..."
                if error_msg and len(error_msg) > 28
                else error_msg
            )
            f.write(
                f"{method:<12} {success_str:<8} {time_str:<10} {features_str:<10} {error_str:<30}\n"
            )

        f.write("\nDetailed Timing Breakdown:\n")
        f.write("-" * 40 + "\n")
        for component, value in timing_results.items():
            if component == "total_time":
                f.write(f"{component:25s}: {value:.4f}s (100.0%)\n")
            elif component == "n_r_peaks":
                f.write(f"{component:25s}: {value}\n")
            else:
                percentage = (
                    (value / timing_results["total_time"]) * 100
                    if "total_time" in timing_results
                    else 0
                )
                f.write(f"{component:25s}: {value:.4f}s ({percentage:5.1f}%)\n")

        f.write("\ncProfile Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(profile_output)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
    main()
