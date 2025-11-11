# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PTE-ECG** (Python Tools for Electrocardiography) is a modular ECG feature extraction library providing:
- Configurable preprocessing pipelines (resampling, filtering, normalization)
- Comprehensive feature extraction (FFT, morphological, nonlinear, statistical, Welch, bispectrum)
- Efficient multi-subject, multi-channel processing
- Clinical-grade morphological features

**Current Version**: 0.3.1 (Pre-Alpha)
**Python**: 3.11+ required
**License**: MIT

## Development Commands

### Environment Setup
```bash
# Clone and setup
git clone https://github.com/richardkoehler/pte-ecg.git
cd pte-ecg
uv sync                    # Install all dependencies including optional groups
```

### Code Quality (Run after major edits)
```bash
uvx ruff check --fix       # Lint with auto-fixes
uvx ty check               # Type checking
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_pipelines.py

# Run specific test function
uv run pytest tests/test_pipelines.py::test_get_features_default
```

### Running Scripts
```bash
uv run <script.py>         # Execute Python scripts with proper environment
```

### Package Management
```bash
uv add <package>           # Add new dependency
uv sync                    # Sync after pulling changes
```

## Architecture

### Core Pipeline Flow

```
ECG Data (numpy array: n_channels × n_samples)
    ↓
[Preprocessing] (preprocessing.py)
    - Resample → Bandpass Filter → Notch Filter → Normalize
    ↓
[Feature Extraction] (features.py)
    - FFT Features
    - Morphological Features (peak detection, waveform analysis)
    - Nonlinear Features (optional: nolds)
    - Statistical Features
    - Welch Features
    - Bispectrum Features (optional: pybispectra)
    ↓
pandas DataFrame (n_samples × n_features)
```

### Module Structure

**`pipelines.py`** - Orchestration layer
- `get_features()`: Main entry point
- Handles Settings configuration (Pydantic models)
- Coordinates preprocessing → feature extraction workflow

**`preprocessing.py`** - Signal conditioning
- Uses scipy.signal for filtering
- Supports MNE-style preprocessing if available
- All preprocessing is optional and configurable

**`features.py`** - Feature extraction engine (MAIN FILE)
- **Per-channel processing**: Most features computed independently per ECG lead
- **Morphological features**: Uses neurokit2's `ecg_delineate()` for waveform detection
- **Peak detection**: Multiple methods (prominence, dwt, cwt, etc.) with automatic fallback
- **Parallel processing**: Uses multiprocessing for multi-sample datasets
- **Territory-specific features**: Aggregates leads for regional cardiac markers

**`_logging.py`** - Centralized logging
- Logger instance: `from ._logging import logger`

**`ecg_feature_extractor.py`** - Legacy/reference implementation
- Keep for comparison but new features go in `features.py`

### Settings Architecture

Configuration uses Pydantic models with nested structure:
```python
settings = pte_ecg.Settings()
settings.preprocessing.bandpass.enabled = True
settings.preprocessing.bandpass.l_freq = 0.5
settings.features.morphological.enabled = True
```

Use `"default"` string for default settings.

## Critical Implementation Details

### ECG Data Format
- **Input shape**: `(n_channels, n_samples)` or `(n_samples, n_channels, n_samples)`
- **Standard 12-lead ordering**: I, II, III, aVR, aVL, aVF, V1-V6
- Lead indices: I=ch0, II=ch1, III=ch2, aVR=ch3, aVL=ch4, aVF=ch5, V1-6=ch6-11

### Morphological Features Implementation

**Key principle**: Reuse existing infrastructure from neurokit2, don't duplicate peak detection.

**Peak Detection** (`_get_r_peaks()`):
- Tries multiple methods: prominence (fastest), dwt, peak, cwt, etc.
- Returns first successful method
- R-peaks are foundation for all other waveform detection

**Waveform Delineation** (`nk.ecg_delineate()`):
- Provides P, Q, S, T peaks/onsets/offsets
- Methods: `waves_dict.get("ECG_T_Peaks")`, `waves_dict.get("ECG_Q_Peaks")`, etc.
- Returns np.ndarray with NaN for undetected peaks

**Feature Naming Convention**:
- Per-channel: `morphological_{feature}_ch{n}` (e.g., `morphological_st_elevation_ch0`)
- Global (multi-lead): `morphological_{feature}` (e.g., `morphological_qrs_axis`)

**Baseline Handling**:
- Baseline wander removal happens BEFORE calling `features.py`
- Global baseline for ST segments: first 200ms of signal
- ST segment measured from J-point+20ms to J-point+80ms

### Type Safety Requirements

**All numpy scalar returns must be converted to Python types**:
```python
# Required for dict[str, float] compatibility
features["feature_name"] = float(np.mean(values))  # ✅ Good
features["feature_name"] = np.mean(values)          # ❌ Type error
```

This is critical because the function signature is `dict[str, float]` not `dict[str, np.floating]`.

## Coding Standards

### Comments
Use sparingly - code should be self-documenting. Only comment:
- Non-obvious "why" explanations
- Clinical/medical rationale
- Complex algorithms

### Exception Handling
- Use try-except sparingly
- Keep try blocks minimal (single risky operation)
- Catch specific exceptions only
- Avoid bare `except:` clauses

```python
# Good
try:
    result = risky_operation()
except SpecificError:
    handle_error()

# Bad - too broad
try:
    step1()
    step2()
    step3()
except Exception:
    pass
```

### Type Annotations
- All functions must have type hints
- Use modern syntax: `list[int]`, `dict[str, float]` (not `List`, `Dict`)
- Pydantic for configuration validation

### Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_underscore_prefix`

## Current Development Status

### ✅ Completed: Feature Consolidation
All morphological features successfully ported from `ecg_feature_extractor.py` to `features.py`:

1. ST segment features (elevation, depression, J-point, slope)
2. T-wave inversion depth
3. QTc interval (Bazett's formula)
4. Interval ratios (QT/RR, PR/RR, T/QT)
5. Electrical axes (QRS, P)
6. T-wave symmetry
7. QRS fragmentation
8. Territory-specific markers (15 features: anterior/inferior/lateral walls, aVR)

**Location in code**: `features.py` lines 1081-1100 (QRS fragmentation), 1093-1111 (T-symmetry), 794-883 (territory markers)

### Next Development Phases

**Phase 1: Testing & Validation**
- Unit tests for each morphological feature
- Integration tests with synthetic ECG data
- Validation on PhysioNet datasets (MIT-BIH, PTB-XL)

**Phase 2: Performance Optimization**
- Profile morphological feature extraction (current: ~500ms per 10s ECG)
- Optimize parallel processing
- Target: <200ms per 10s ECG

**Phase 3: Documentation & Cleanup**
- Add clinical interpretation guide
- Document expected ranges for each feature
- Decide fate of `ecg_feature_extractor.py` (archive/delete/keep as reference)

## Dependencies

### Core
- numpy >= 2.0.0
- pandas >= 2.3.0
- scipy >= 1.15.3
- neurokit2 >= 0.2.11
- pydantic >= 2.11.7

### Optional (install with `uv sync --group <group>`)
- **nonlinear**: nolds >= 0.6.2
- **bispectrum**: pybispectra >= 1.2.1, numba >= 0.61.2
- **dev**: mypy, pre-commit, tox
- **test**: pytest >= 6.0, pytest-cov >= 4.2.0

Default: All groups installed with `uv sync` (configured in pyproject.toml)

## Known Limitations

1. **12-Lead ECG Assumption**: Electrical axes and territory markers require standard 12-lead ordering
2. **Peak Detection Dependency**: Relies on neurokit2's `ecg_delineate()` which may fail on noisy/pathological ECGs
3. **Type System**: Pre-existing type errors throughout codebase (numpy floats vs Python floats) - new code must use `float()` conversions

## Usage Example

```python
import numpy as np
import pte_ecg

# ECG data: (n_channels, n_samples)
ecg_data = np.random.randn(12, 10000)  # 12 leads, 10 seconds at 1000 Hz
sfreq = 1000

# Configure features
settings = pte_ecg.Settings()
settings.preprocessing.bandpass.enabled = True
settings.preprocessing.bandpass.l_freq = 0.5
settings.preprocessing.bandpass.h_freq = 40
settings.features.morphological.enabled = True
settings.features.fft.enabled = True

# Extract features
features_df = pte_ecg.get_features(ecg=ecg_data, sfreq=sfreq, settings=settings)
```

## References

- NeuroKit2: https://neuropsychology.github.io/NeuroKit/
- PhysioNet PTB-XL: https://physionet.org/content/ptb-xl/
- AHA/ACCF/HRS ECG Standardization Guidelines
