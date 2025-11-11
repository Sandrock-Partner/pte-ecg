"""PTE-ECG: A Python package for ECG signal processing and feature extraction.

This package provides tools for preprocessing ECG signals and extracting various types of features
including statistical, morphological, and nonlinear features. It is designed to work with
multi-channel ECG data and supports parallel processing for efficient computation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ._logging import logger, set_log_file, set_log_level
from .config import ConfigLoader, ExtractorConfig, FeaturesConfig, Settings
from .core import FeatureExtractor
from .preprocessing import (
    BandpassArgs,
    NormalizeArgs,
    NotchArgs,
    PreprocessingSettings,
    ResampleArgs,
    preprocess,
)

__version__ = "0.4.0"
__all__ = [
    "__version__",
    "logger",
    "set_log_level",
    "set_log_file",
    "get_features",
    "preprocess",
    "Settings",
    "PreprocessingSettings",
    "ResampleArgs",
    "BandpassArgs",
    "NotchArgs",
    "NormalizeArgs",
    "FeaturesConfig",
    "ExtractorConfig",
    "ConfigLoader",
    "FeatureExtractor",
]


def __dir__():
    return __all__


def get_features(
    ecg: np.ndarray,
    sfreq: float,
    settings: Settings | str | Path | None = None,
) -> pd.DataFrame:
    """Extract features from ECG data.

    This is the main high-level API for feature extraction. It handles
    configuration loading and orchestrates the complete feature extraction pipeline.

    Args:
        ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        settings: Configuration for feature extraction. Can be:
            - Settings object: Use directly
            - str or Path: Load from JSON/TOML config file
            - None: Use default settings

    Returns:
        DataFrame with shape (n_samples, n_features) containing all extracted features.
        Column names follow pattern: {extractor_name}_{feature_name}_ch{N}

    Raises:
        ValueError: If input data has invalid shape or settings are invalid
        FileNotFoundError: If settings is a path that doesn't exist
        ValidationError: If config file doesn't match schema

    Examples:
        # Use default settings
        features = pte_ecg.get_features(ecg_data, sfreq=1000)

        # Use Settings object
        settings = pte_ecg.Settings()
        settings.features.morphological.features = ["st_elevation", "qtc_interval"]
        settings.features.fft.enabled = False
        features = pte_ecg.get_features(ecg_data, sfreq=1000, settings=settings)

        # Load from config file
        features = pte_ecg.get_features(ecg_data, sfreq=1000, settings="config.json")
    """
    # Handle settings parameter
    if settings is None or settings == "default":
        # Use default settings
        settings_obj = Settings()
    elif isinstance(settings, (str, Path)):
        # Load from config file
        settings_obj = ConfigLoader.from_file(settings)
    else:
        # Assume it's already a Settings object
        settings_obj = settings

    # Create extractor and run feature extraction
    extractor = FeatureExtractor(settings_obj)
    return extractor.extract_features(ecg, sfreq)
