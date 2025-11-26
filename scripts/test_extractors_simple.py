"""Simple test script without multiprocessing (Windows-compatible)."""

import numpy as np

import pte_ecg

# Create synthetic ECG data
n_samples = 2
n_channels = 12
n_timepoints = 5000
sfreq = 1000

np.random.seed(42)
ecg_data = np.random.randn(n_samples, n_channels, n_timepoints)

print("=" * 80)
print("Testing All Feature Extractors (n_jobs=1, no multiprocessing)")
print("=" * 80)
print(f"ECG data shape: {ecg_data.shape}")
print(f"Sampling frequency: {sfreq} Hz")
print()

# Test each extractor with n_jobs=1 to disable multiprocessing
extractors_to_test = [
    ("FFT", "fft"),
    ("Statistical", "statistical"),
    ("Welch", "welch"),
    ("Morphological", "morphological"),
]

results = {}

for name, extractor_name in extractors_to_test:
    print(f"Test: {name} Extractor")
    print("-" * 80)

    try:
        settings = pte_ecg.Settings()

        # Disable all extractors
        for ex in ["fft", "statistical", "welch", "morphological", "nonlinear", "waveshape"]:
            setattr(settings.features, ex, {"enabled": False})

        # Enable only the one we're testing (with n_jobs=1 to disable multiprocessing)
        setattr(settings.features, extractor_name, {"enabled": True, "n_jobs": 1})

        settings.preprocessing.enabled = False

        features = pte_ecg.get_features(ecg_data, sfreq, settings=settings)

        print(f"[OK] Extracted {features.shape[1]} features from {features.shape[0]} samples")
        print(f"  Sample columns: {list(features.columns[:3])}...")
        results[name] = "SUCCESS"

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        results[name] = "FAILED"

    print()

# Test all together
print("Test: All Core Extractors Together (n_jobs=1)")
print("-" * 80)
try:
    settings = pte_ecg.Settings()
    settings.features.fft = {"enabled": True}
    settings.features.statistical = {"enabled": True}
    settings.features.welch = {"enabled": True}
    settings.features.morphological = {"enabled": True, "n_jobs": 1}  # Disable multiprocessing
    settings.preprocessing.enabled = False

    features = pte_ecg.get_features(ecg_data, sfreq, settings=settings)
    print(f"[OK] Extracted {features.shape[1]} total features from {features.shape[0]} samples")
    results["All Core"] = "SUCCESS"
except Exception as e:
    print(f"[FAIL] Error: {e}")
    results["All Core"] = "FAILED"

print()

# Test Registry
print("Test: ExtractorRegistry")
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
    status_icon = "[OK]" if status == "SUCCESS" else "[FAIL]"
    print(f"{status_icon} {name}: {status}")

print()
successful = sum(1 for s in results.values() if s == "SUCCESS")
failed = sum(1 for s in results.values() if s == "FAILED")
print(f"Total: {successful} passed, {failed} failed")
print("=" * 80)
