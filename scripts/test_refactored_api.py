"""Test script for refactored feature extraction API."""

import numpy as np

import pte_ecg

# Create synthetic ECG data
# Shape: (n_samples, n_channels, n_timepoints)
n_samples = 5
n_channels = 12  # Standard 12-lead ECG
n_timepoints = 10000  # 10 seconds at 1000 Hz
sfreq = 1000

ecg_data = np.random.randn(n_samples, n_channels, n_timepoints)

print("=" * 80)
print("Testing Refactored PTE-ECG API")
print("=" * 80)
print(f"ECG data shape: {ecg_data.shape}")
print(f"Sampling frequency: {sfreq} Hz")
print()

# Test 1: Default settings
print("Test 1: Extract features with default settings")
print("-" * 80)
try:
    features = pte_ecg.get_features(ecg_data, sfreq)
    print(f"[OK] Success! Extracted {features.shape[1]} features from {features.shape[0]} samples")
    print(f"  Feature columns: {list(features.columns[:5])}...")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
print()

# Test 2: Custom Settings object
print("Test 2: Extract features with custom Settings object")
print("-" * 80)
try:
    settings = pte_ecg.Settings()
    settings.preprocessing.enabled = False
    settings.features.fft.features = ["sum_freq", "mean_freq", "dominant_frequency"]

    features = pte_ecg.get_features(ecg_data, sfreq, settings=settings)
    print(f"[OK] Success! Extracted {features.shape[1]} features from {features.shape[0]} samples")
    print(f"  Selected features: {list(features.columns)[:6]}...")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
print()

# Test 3: ExtractorRegistry
print("Test 3: Check ExtractorRegistry")
print("-" * 80)
try:
    from pte_ecg.feature_extractors import ExtractorRegistry

    registry = ExtractorRegistry.get_instance()
    extractors = registry.list_extractors()
    print(f"[OK] Registered extractors: {extractors}")

    if "fft" in extractors:
        fft_class = registry.get("fft")
        print(f"  FFT extractor: {fft_class}")
        print(f"  Available features: {fft_class.available_features[:5]}...")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
print()

# Test 4: Direct extractor usage
print("Test 4: Use FFTExtractor directly")
print("-" * 80)
try:
    from pte_ecg.feature_extractors.fft import FFTExtractor

    extractor = FFTExtractor(selected_features=["dominant_frequency", "spectral_entropy"])
    features = extractor.get_features(ecg_data, sfreq)
    print(f"[OK] Success! Extracted {features.shape[1]} features from {features.shape[0]} samples")
    print(f"  Features: {list(features.columns)}")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
print()

print("=" * 80)
print("Testing complete!")
print("=" * 80)
