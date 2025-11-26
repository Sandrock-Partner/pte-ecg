"""Comprehensive test script for all feature extractors."""

import numpy as np

import pte_ecg

# Create synthetic ECG data
n_samples = 2
n_channels = 12  # Standard 12-lead ECG
n_timepoints = 5000  # 5 seconds at 1000 Hz
sfreq = 1000

np.random.seed(42)
ecg_data = np.random.randn(n_samples, n_channels, n_timepoints)

print("=" * 80)
print("Testing All Feature Extractors")
print("=" * 80)
print(f"ECG data shape: {ecg_data.shape}")
print(f"Sampling frequency: {sfreq} Hz")
print()

# Test each extractor individually
extractors_to_test = [
    ("FFT", "fft", True),
    ("Statistical", "statistical", True),
    ("Welch", "welch", True),
    ("Morphological", "morphological", True),
    ("Nonlinear", "nonlinear", False),  # Optional dependency
    ("WaveShape", "waveshape", False),  # Optional dependency
]

results = {}

for name, extractor_name, is_required in extractors_to_test:
    print(f"Test: {name} Extractor")
    print("-" * 80)

    try:
        # Create settings with only this extractor enabled
        settings = pte_ecg.Settings()

        # Disable all extractors first
        settings.features.fft = {"enabled": False}
        settings.features.statistical = {"enabled": False}
        settings.features.welch = {"enabled": False}
        settings.features.morphological = {"enabled": False}
        settings.features.nonlinear = {"enabled": False}
        settings.features.waveshape = {"enabled": False}

        # Enable only the one we're testing
        setattr(settings.features, extractor_name, {"enabled": True})

        # Disable preprocessing for faster testing
        settings.preprocessing.enabled = False

        features = pte_ecg.get_features(ecg_data, sfreq, settings=settings)

        print(f"[OK] Extracted {features.shape[1]} features from {features.shape[0]} samples")
        print(f"  Sample columns: {list(features.columns[:3])}...")
        results[name] = "SUCCESS"

    except ImportError as e:
        if not is_required:
            print(f"[SKIP] Optional dependency not installed: {e}")
            results[name] = "SKIPPED"
        else:
            print(f"[FAIL] Required dependency missing: {e}")
            results[name] = "FAILED"
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        results[name] = "FAILED"

    print()

# Test with multiple extractors enabled
print("Test: Multiple Extractors Simultaneously")
print("-" * 80)
try:
    settings = pte_ecg.Settings()
    # Enable core extractors (skip optional ones)
    settings.features.fft = {"enabled": True}
    settings.features.statistical = {"enabled": True}
    settings.features.welch = {"enabled": True}
    settings.features.morphological = {"enabled": True}
    settings.preprocessing.enabled = False

    features = pte_ecg.get_features(ecg_data, sfreq, settings=settings)
    print(f"[OK] Extracted {features.shape[1]} total features from {features.shape[0]} samples")
    results["Multiple"] = "SUCCESS"
except Exception as e:
    print(f"[FAIL] Error: {e}")
    results["Multiple"] = "FAILED"

print()

# Test Registry
print("Test: ExtractorRegistry Discovery")
print("-" * 80)
try:
    from pte_ecg.feature_extractors import ExtractorRegistry

    registry = ExtractorRegistry.get_instance()
    extractors = registry.list_extractors()
    print(f"[OK] Discovered {len(extractors)} extractors: {extractors}")
    results["Registry"] = "SUCCESS"
except Exception as e:
    print(f"[FAIL] Error: {e}")
    results["Registry"] = "FAILED"

print()

# Summary
print("=" * 80)
print("Test Summary")
print("=" * 80)
for name, status in results.items():
    status_icon = {
        "SUCCESS": "[OK]",
        "SKIPPED": "[SKIP]",
        "FAILED": "[FAIL]",
    }.get(status, "[?]")
    print(f"{status_icon} {name}: {status}")

print()
successful = sum(1 for s in results.values() if s == "SUCCESS")
skipped = sum(1 for s in results.values() if s == "SKIPPED")
failed = sum(1 for s in results.values() if s == "FAILED")
print(f"Total: {successful} passed, {skipped} skipped, {failed} failed")
print("=" * 80)
