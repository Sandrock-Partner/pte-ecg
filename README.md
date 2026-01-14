![ECG Trace](docs/ecg_trace.svg)

# PTE-ECG

![License](https://img.shields.io/github/license/Sandrock-Partner/pte-ecg)
<!-- ![PyPI version](https://img.shields.io/pypi/v/pte-ecg?color=blue) -->
<!-- ![Build Status](https://img.shields.io/github/actions/workflow/status/richardkoehler/pte-ecg/python-package.yml?branch=main) -->

**Python Tools for Electrocardiography (PTE) - ECG**

A Python package for extracting features from ECG signals with a modern plugin-based architecture.

This package provides an extensible and pluggable interface to extract features from raw ECG data using
an extractor registry and configuration-driven pipeline. The current version (`1.0.0-alpha.1`) ships with a
plugin-based architecture that allows easy customization and extension of feature extractors.

**Requirements**: Python >=3.12

## Table of Contents
- [✨ Highlights](#highlights)
- [🚀 Installation](#installation)
- [💻 Development Setup](#development-setup)
- [🩺 Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Custom Configuration](#custom-configuration)
  - [Creating Custom Extractors](#creating-custom-extractors)
  - [Discovering Available Extractors](#discovering-available-extractors)
- [📄 License](#license)
- [🤝 Contributing](#contributing)


## Highlights

- 🔌 **Plugin-Based Architecture** - Easily extensible with custom feature extractors
- 🔩 **Configurable Pipeline** - Enable/disable extractors and configure parameters
- ⚡️ **Efficient Processing** - Multi-subject, multi-channel data support
- 🛠️ **Preprocessing Methods** (enabled by default)
  - Resampling
  - Bandpass filtering
  - Notch filtering
  - Normalization (multiple modes: mean, ratio, logratio, percent, zscore, zlogratio)
- 📊 **Feature Extraction Methods** (5 built-in extractors)
  - **FFT** - Frequency domain features (21 features/channel)
  - **Statistical** - Basic statistical measures (13 features/channel)
  - **Welch** - Power spectral density (19 features/channel)
  - **Morphological** - Waveform analysis, intervals, axes (50+ features/channel)
  - **Nonlinear** - Entropy, fractal dimension, DFA (30+ features/channel) [Optional]


## 🚀 Installation

### Basic Installation

```bash
pip install git+https://github.com/richardkoehler/pte-ecg.git
```

### Optional Dependencies

The package supports an optional nonlinear feature group that can be installed as needed:

```bash
# For nonlinear features (requires nolds)
pip install git+https://github.com/richardkoehler/pte-ecg.git[nonlinear]

# Install all optional dependencies
pip install git+https://github.com/richardkoehler/pte-ecg.git[all]
```

### Using uv

```bash
uv add git+https://github.com/richardkoehler/pte-ecg
```

## Development setup

```bash
# Clone the repository
git clone https://github.com/richardkoehler/pte-ecg.git
cd pte-ecg

# Install with pip
pip install -e .

# Or install with uv
uv sync
```

## Usage

### Basic Usage

Here's a basic example of how to use the package to extract features from ECG data.
The main API expects a 3D numpy array with shape \\((n\\_ecgs, n\\_channels, n\\_timepoints)\\):

```python
import numpy as np
import pte_ecg

# Generate some synthetic ECG data (replace with your actual data)
# Shape: (n_ecgs, n_channels, n_timepoints)
sfreq = 1000  # Sampling frequency in Hz
n_ecgs = 5
ecg_data = np.random.randn(n_ecgs, 12, 10_000)  # 5 ECGs, 12 leads, 10 seconds at 1000 Hz

# Use default settings (morphological extractor enabled, standard 12‑lead order)
features = pte_ecg.get_features(ecg=ecg_data, sfreq=sfreq)

print(f"Extracted {len(features.columns)} features for {len(features)} ECGs")
print(features.head())
```

### Custom Configuration

```python
import pte_ecg

# Create custom settings
settings = pte_ecg.Settings()

# Configure preprocessing
settings.preprocessing.bandpass.enabled = True
settings.preprocessing.bandpass.l_freq = 0.5
settings.preprocessing.bandpass.h_freq = 40

settings.preprocessing.notch.enabled = True
settings.preprocessing.notch.freq = 50  # Remove 50 Hz powerline noise

settings.preprocessing.normalize.enabled = True
settings.preprocessing.normalize.mode = "zscore"  # Options: "mean", "ratio", "logratio", "percent", "zscore", "zlogratio"

# Enable/disable specific extractors (morphological is enabled by default)
settings.features.fft = {"enabled": True}
settings.features.statistical = {"enabled": True}
settings.features.welch = {"enabled": True}
# settings.features.nonlinear = {"enabled": True}  # Enable optional nonlinear extractor

# Extract features with custom settings
features = pte_ecg.get_features(ecg=ecg_data, sfreq=sfreq, settings=settings)

# You can also load settings from a config file (JSON or TOML)
# features = pte_ecg.get_features(ecg=ecg_data, sfreq=sfreq, settings="config.json")

# Or customize the lead order (default is standard 12-lead)
# settings = pte_ecg.Settings(lead_order=["I", "II", "III", "aVR", "aVL", "aVF"])  # Limb leads only
```

### Creating Custom Extractors

The plugin architecture allows you to create custom feature extractors that integrate
with the central `FeatureExtractor` via dependency injection:

```python
from pte_ecg.feature_extractors.base import BaseFeatureExtractor
from pte_ecg.core import FeatureExtractor
import numpy as np
import pandas as pd


class MyCustomExtractor(BaseFeatureExtractor):
    """Custom feature extractor example."""

    # Entry‑point name used in configuration / registry
    name = "my_custom"
    available_features = ["custom_feature_1", "custom_feature_2"]

    def __init__(self, parent: FeatureExtractor):
        # parent gives access to sfreq, lead_order, settings, etc.
        self.parent = parent

    def get_features(self, ecg: np.ndarray) -> pd.DataFrame:
        """Extract custom features from ECG data.

        Args:
            ecg: ECG data with shape (n_ecgs, n_channels, n_timepoints)

        Returns:
            DataFrame with shape (n_ecgs, n_features)
        """
        n_ecgs, n_channels, _ = ecg.shape
        features_list: list[dict[str, float]] = []

        for sample_idx in range(n_ecgs):
            sample_features: dict[str, float] = {}
            for ch_idx in range(n_channels):
                channel_data = ecg[sample_idx, ch_idx, :]

                # Your custom feature extraction logic here
                sample_features[f"custom_feature_1_ch{ch_idx}"] = float(np.mean(channel_data))
                sample_features[f"custom_feature_2_ch{ch_idx}"] = float(np.std(channel_data))

            features_list.append(sample_features)

        return pd.DataFrame(features_list)


# Register your extractor in pyproject.toml:
# [project.entry-points."pte_ecg.extractors"]
# my_custom = "your_package.module:MyCustomExtractor"
```

### Discovering Available Extractors

```python
import pte_ecg

# Get the registry instance (singleton pattern)
from pte_ecg.feature_extractors.registry import ExtractorRegistry

registry = ExtractorRegistry.get_instance()

# List all available extractors
extractors = registry.list_extractors()
print(f"Available extractors: {extractors}")

# Get a specific extractor class
FFTExtractor = registry.get("fft")

# Note: Extractors are typically instantiated internally by FeatureExtractor.
# To use directly, you need to provide a parent FeatureExtractor instance:
# from pte_ecg.core import FeatureExtractor
# parent = FeatureExtractor(sfreq=1000)
# fft_extractor = FFTExtractor(parent=parent)
```

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit an issue if you find a bug or have a feature request. Pull requests are also welcome.
