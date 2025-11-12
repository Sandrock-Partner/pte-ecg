"""PTE-ECG: A Python package for ECG signal processing and feature extraction.

This package provides tools for preprocessing ECG signals and extracting various types of features
including statistical, morphological, and nonlinear features. It is designed to work with
multi-channel ECG data and supports parallel processing for efficient computation.
"""

from ._logging import logger, set_log_file, set_log_level
from .config import ConfigLoader, ExtractorConfig, FeaturesConfig, Settings
from .core import FeatureExtractor
from .feature_extractors.registry import ExtractorRegistry
from .preprocessing import (
    BandpassArgs,
    NormalizeArgs,
    NotchArgs,
    PreprocessingSettings,
    ResampleArgs,
    preprocess,
)

__version__ = "1.0.0-alpha.1"
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
    "ExtractorRegistry",
]


def __dir__():
    return __all__
