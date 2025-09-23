#!/usr/bin/env python3
"""Test script to verify preprocessing functionality after MNE removal."""

import numpy as np
from src.pte_ecg.preprocessing import preprocess, PreprocessingSettings, BandpassArgs, NotchArgs, NormalizeArgs, ResampleArgs


def test_preprocessing_without_mne():
    """Test that preprocessing works correctly without MNE dependency."""
    
    # Create synthetic ECG data
    np.random.seed(42)
    n_samples, n_channels, n_timepoints = 2, 3, 1000
    sfreq = 250.0  # Hz
    
    # Generate synthetic ECG-like signal
    t = np.linspace(0, n_timepoints / sfreq, n_timepoints)
    ecg_data = np.zeros((n_samples, n_channels, n_timepoints))
    
    for i in range(n_samples):
        for j in range(n_channels):
            # Create a signal with some frequency components
            signal = (
                np.sin(2 * np.pi * 1.2 * t) +  # Heart rate component
                0.5 * np.sin(2 * np.pi * 0.3 * t) +  # Breathing component
                0.1 * np.random.randn(n_timepoints)  # Noise
            )
            ecg_data[i, j, :] = signal
    
    print(f"Original ECG shape: {ecg_data.shape}")
    print(f"Original sampling frequency: {sfreq} Hz")
    
    # Test 1: Basic preprocessing (no operations)
    print("\n=== Test 1: No preprocessing ===")
    settings = PreprocessingSettings(enabled=False)
    processed_ecg, new_sfreq = preprocess(ecg_data, sfreq, settings)
    print(f"Processed ECG shape: {processed_ecg.shape}")
    print(f"New sampling frequency: {new_sfreq} Hz")
    assert processed_ecg.shape == ecg_data.shape
    assert new_sfreq == sfreq
    print("[PASS] No preprocessing test passed")
    
    # Test 2: Resampling
    print("\n=== Test 2: Resampling ===")
    settings = PreprocessingSettings(
        resample=ResampleArgs(enabled=True, sfreq_new=125.0)
    )
    processed_ecg, new_sfreq = preprocess(ecg_data, sfreq, settings)
    expected_timepoints = int(n_timepoints * 125.0 / sfreq)
    print(f"Processed ECG shape: {processed_ecg.shape}")
    print(f"New sampling frequency: {new_sfreq} Hz")
    assert processed_ecg.shape == (n_samples, n_channels, expected_timepoints)
    assert new_sfreq == 125.0
    print("[PASS] Resampling test passed")
    
    # Test 3: Bandpass filtering
    print("\n=== Test 3: Bandpass filtering ===")
    settings = PreprocessingSettings(
        bandpass=BandpassArgs(enabled=True, l_freq=0.5, h_freq=40.0)
    )
    processed_ecg, new_sfreq = preprocess(ecg_data, sfreq, settings)
    print(f"Processed ECG shape: {processed_ecg.shape}")
    print(f"New sampling frequency: {new_sfreq} Hz")
    assert processed_ecg.shape == ecg_data.shape
    assert new_sfreq == sfreq
    print("[PASS] Bandpass filtering test passed")
    
    # Test 4: Notch filtering
    print("\n=== Test 4: Notch filtering ===")
    settings = PreprocessingSettings(
        notch=NotchArgs(enabled=True, freq=50.0)
    )
    processed_ecg, new_sfreq = preprocess(ecg_data, sfreq, settings)
    print(f"Processed ECG shape: {processed_ecg.shape}")
    print(f"New sampling frequency: {new_sfreq} Hz")
    assert processed_ecg.shape == ecg_data.shape
    assert new_sfreq == sfreq
    print("[PASS] Notch filtering test passed")
    
    # Test 5: Z-score normalization
    print("\n=== Test 5: Z-score normalization ===")
    settings = PreprocessingSettings(
        normalize=NormalizeArgs(enabled=True, mode="zscore")
    )
    processed_ecg, new_sfreq = preprocess(ecg_data, sfreq, settings)
    print(f"Processed ECG shape: {processed_ecg.shape}")
    print(f"New sampling frequency: {new_sfreq} Hz")
    
    # Check that normalization worked (mean should be close to 0, std close to 1)
    flattened = processed_ecg.reshape(n_samples, -1)
    means = np.mean(flattened, axis=1)
    stds = np.std(flattened, axis=1)
    print(f"Means after normalization: {means}")
    print(f"Stds after normalization: {stds}")
    
    assert processed_ecg.shape == ecg_data.shape
    assert new_sfreq == sfreq
    assert np.allclose(means, 0, atol=1e-10)
    assert np.allclose(stds, 1, atol=1e-10)
    print("[PASS] Z-score normalization test passed")
    
    # Test 6: Combined preprocessing
    print("\n=== Test 6: Combined preprocessing ===")
    settings = PreprocessingSettings(
        resample=ResampleArgs(enabled=True, sfreq_new=200.0),
        bandpass=BandpassArgs(enabled=True, l_freq=1.0, h_freq=30.0),
        notch=NotchArgs(enabled=True, freq=50.0),
        normalize=NormalizeArgs(enabled=True, mode="zscore")
    )
    processed_ecg, new_sfreq = preprocess(ecg_data, sfreq, settings)
    expected_timepoints = int(n_timepoints * 200.0 / sfreq)
    print(f"Processed ECG shape: {processed_ecg.shape}")
    print(f"New sampling frequency: {new_sfreq} Hz")
    
    assert processed_ecg.shape == (n_samples, n_channels, expected_timepoints)
    assert new_sfreq == 200.0
    
    # Check normalization
    flattened = processed_ecg.reshape(n_samples, -1)
    means = np.mean(flattened, axis=1)
    stds = np.std(flattened, axis=1)
    assert np.allclose(means, 0, atol=1e-10)
    assert np.allclose(stds, 1, atol=1e-10)
    print("[PASS] Combined preprocessing test passed")
    
    print("\n[SUCCESS] All preprocessing tests passed! MNE removal was successful.")


if __name__ == "__main__":
    test_preprocessing_without_mne()
