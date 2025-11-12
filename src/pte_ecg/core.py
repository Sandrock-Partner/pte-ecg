"""Main feature extraction orchestrator."""

from pathlib import Path

import numpy as np
import pandas as pd

from ._logging import logger
from .config import ConfigLoader, ExtractorConfig, Settings
from .feature_extractors.base import FeatureExtractorProtocol
from .preprocessing import preprocess


class FeatureExtractor:
    """Main orchestrator for ECG feature extraction.

    This class coordinates preprocessing and feature extraction using
    configured extractors. It handles the complete pipeline from raw
    ECG data to extracted features.

    Args:
        settings: Complete settings for preprocessing and feature extraction

    Examples:
        # Use default settings
        extractor = FeatureExtractor()
        features = extractor.extract_features(ecg_data, sfreq=1000)

        # Use custom settings
        settings = Settings()
        settings.features.morphological.features = ["st_elevation", "qtc_interval"]
        settings.features.fft.enabled = False
        extractor = FeatureExtractor(settings)
        features = extractor.extract_features(ecg_data, sfreq=1000)
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize the feature extractor.

        Args:
            settings: Settings object. If None, uses default settings.
        """
        self.settings = settings or Settings()
        self._extractors = self._initialize_extractors()

    def _initialize_extractors(self) -> dict[str, FeatureExtractorProtocol]:
        """Initialize enabled feature extractors.

        Uses the ExtractorRegistry to discover and instantiate all registered extractors.
        Supports both hardcoded and plugin-registered extractors.

        Returns:
            Dictionary mapping extractor names to initialized extractor instances
        """
        from .feature_extractors import ExtractorRegistry

        extractors = {}
        registry = ExtractorRegistry.get_instance()

        registered_extractors = registry.list_extractors()
        if not registered_extractors:
            raise ValueError(
                "ExtractorRegistry is empty. "
                "Define entry points in pyproject.toml to use the registry."
            )

        # Iterate through all registry-discovered extractors
        for extractor_name in registered_extractors:
            # Get config for this extractor, with fallback to default config
            extractor_config = getattr(self.settings.features, extractor_name, ExtractorConfig())
            
            # Skip if extractor is disabled
            if not extractor_config.enabled:
                logger.debug(f"Extractor '{extractor_name}' is disabled, skipping")
                continue

            # Get extractor class from registry and initialize
            extractor_class = registry.get(extractor_name)
            selected_features = None if extractor_config.features == "all" else extractor_config.features
            extractors[extractor_name] = extractor_class(
                selected_features=selected_features, n_jobs=extractor_config.n_jobs
            )
            logger.info(
                f"Initialized {extractor_name} extractor with "
                f"{len(extractors[extractor_name].selected_features)} features"
            )

        if not extractors:
            raise ValueError("No feature extractors enabled. Enable at least one extractor in settings.")

        logger.info(f"Initialized {len(extractors)} feature extractor(s): {list(extractors.keys())}")
        return extractors

    def extract_features(
        self,
        ecg: np.ndarray,
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract features from ECG data.

        This is the main entry point for feature extraction. It:
        1. Applies preprocessing (if enabled)
        2. Runs each enabled feature extractor
        3. Concatenates results into a single DataFrame

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_samples, n_features) containing all extracted features.
            Column names follow pattern: {extractor_name}_{feature_name}_ch{N}

        Raises:
            ValueError: If input data has invalid shape or no extractors are enabled

        Examples:
            extractor = FeatureExtractor()
            ecg_data = np.random.randn(10, 12, 10000)  # 10 samples, 12 leads, 10s at 1kHz
            features = extractor.extract_features(ecg_data, sfreq=1000)
        """
        # Validate input shape
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_samples, n_channels, n_timepoints), got shape {ecg.shape}"
            )

        n_samples, n_channels, n_timepoints = ecg.shape
        logger.info(
            f"Extracting features from {n_samples} samples with {n_channels} channels "
            f"and {n_timepoints} timepoints at {sfreq} Hz"
        )

        # Apply preprocessing
        if self.settings.preprocessing.enabled:
            logger.info("Applying preprocessing pipeline")
            ecg, sfreq = preprocess(ecg, sfreq, self.settings.preprocessing)
            logger.info(f"Preprocessing complete. New sampling frequency: {sfreq} Hz")
        else:
            logger.info("Preprocessing disabled, using raw ECG data")

        # Extract features from each extractor
        feature_dfs = []
        for extractor_name, extractor in self._extractors.items():
            logger.info(f"Extracting {extractor_name} features...")
            features = extractor.get_features(ecg, sfreq)
            feature_dfs.append(features)
            logger.info(f"Extracted {features.shape[1]} {extractor_name} features from {features.shape[0]} samples")

        # Concatenate all features horizontally
        if len(feature_dfs) == 1:
            result = feature_dfs[0]
        else:
            result = pd.concat(feature_dfs, axis=1)

        logger.info(f"Feature extraction complete. Total features: {result.shape[1]} from {result.shape[0]} samples")

        return result


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
    elif isinstance(settings, Settings):
        # Use provided Settings object
        settings_obj = settings
    else:
        raise TypeError(f"settings must be a Settings object, str, Path, or None, got {type(settings).__name__}")

    # Create extractor and run feature extraction
    extractor = FeatureExtractor(settings_obj)
    return extractor.extract_features(ecg, sfreq)
