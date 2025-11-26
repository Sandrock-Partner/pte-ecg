"""Main feature extraction orchestrator."""

from pathlib import Path

import pandas as pd

from ._logging import logger
from .config import ConfigLoader, Settings
from .feature_extractors.base import FeatureExtractorProtocol
from .preprocessing import preprocess
from .types import ECGData


class FeatureExtractor:
    """Main orchestrator for ECG feature extraction.

    This class coordinates preprocessing and feature extraction using
    configured extractors. It handles the complete pipeline from raw
    ECG data to extracted features.

    Individual extractors receive this instance via dependency injection,
    allowing them to access sfreq, lead_order, and settings as needed.

    Args:
        sfreq: Sampling frequency in Hz
        settings: Complete settings for preprocessing and feature extraction

    Examples:
        # Use default settings (only morphological enabled)
        extractor = FeatureExtractor(sfreq=1000)
        features = extractor.extract_features(ecg_data)

        # Enable additional extractors
        settings = Settings()
        settings.features.statistical = {"enabled": True}
        extractor = FeatureExtractor(sfreq=1000, settings=settings)
        features = extractor.extract_features(ecg_data)
    """

    def __init__(self, sfreq: float, settings: Settings | None = None):
        """Initialize the feature extractor.

        Args:
            sfreq: Sampling frequency in Hz
            settings: Settings object. If None, uses default settings.
        """
        self.sfreq = sfreq
        self.settings = settings or Settings()
        self._extractors = self._initialize_extractors()

    @property
    def lead_order(self) -> list[str]:
        """Return the lead order from settings."""
        return self.settings.lead_order

    def _initialize_extractors(self) -> dict[str, FeatureExtractorProtocol]:
        """Initialize enabled feature extractors using dependency injection.

        Each extractor receives this FeatureExtractor instance, allowing them
        to access sfreq, lead_order, and other settings as needed.

        Returns:
            Dictionary mapping extractor names to initialized extractor instances
        """
        from .feature_extractors import ExtractorRegistry

        extractors = {}
        registry = ExtractorRegistry.get_instance()

        registered_extractors = registry.list_extractors()
        if not registered_extractors:
            raise ValueError("ExtractorRegistry is empty. Define entry points in pyproject.toml to use the registry.")

        for extractor_name in registered_extractors:
            extractor_config = self.settings.features.get_extractor_config(extractor_name)

            if not extractor_config.get("enabled", False):
                logger.debug(f"Extractor '{extractor_name}' is disabled, skipping")
                continue

            # Pass all config kwargs except 'enabled' to the extractor
            config_kwargs = {k: v for k, v in extractor_config.items() if k != "enabled"}

            # Get extractor class from registry and initialize with dependency injection + config
            extractor_class = registry.get(extractor_name)
            extractors[extractor_name] = extractor_class(self, **config_kwargs)
            logger.info(f"Initialized {extractor_name} extractor")

        if not extractors:
            raise ValueError("No feature extractors enabled. Enable at least one extractor in settings.")

        logger.info(f"Initialized {len(extractors)} feature extractor(s): {list(extractors.keys())}")
        return extractors

    def extract_features(
        self,
        ecg: ECGData,  # Shape: (n_ecgs, n_channels, n_timepoints)
    ) -> pd.DataFrame:
        """Extract features from ECG data.

        This is the main entry point for feature extraction. It:
        1. Applies preprocessing (if enabled)
        2. Runs each enabled feature extractor
        3. Concatenates results into a single DataFrame

        Args:
            ecg: ECG data with shape (n_ecgs, n_channels, n_timepoints)

        Returns:
            DataFrame with shape (n_ecgs, n_features) containing all extracted features.
            Column names follow pattern: {extractor_name}_{feature_name}_{lead_name}

        Raises:
            ValueError: If input data has invalid shape or no extractors are enabled

        Examples:
            extractor = FeatureExtractor(sfreq=1000)
            ecg_data = np.random.randn(10, 12, 10000)  # 10 ECGs, 12 leads, 10s at 1kHz
            features = extractor.extract_features(ecg_data)
        """
        # Validate input shape
        if ecg.ndim != 3:
            raise ValueError(
                f"ECG data must have 3 dimensions (n_ecgs, n_channels, n_timepoints), got shape {ecg.shape}"
            )

        n_ecgs, n_channels, n_times = ecg.shape

        # Validate that number of channels matches the lead_order
        expected_n_channels = len(self.lead_order)
        if n_channels != expected_n_channels:
            raise ValueError(
                f"ECG data has {n_channels} channels, but lead_order specifies {expected_n_channels} leads. "
                f"Expected lead order: {self.lead_order}. "
                f"Please ensure your ECG data has the same number of channels as specified in lead_order."
            )

        logger.info(
            f"Extracting features from {n_ecgs} ECGs with {n_channels} channels "
            f"(leads: {self.lead_order}) and {n_times} timepoints at {self.sfreq} Hz"
        )

        # Apply preprocessing (may update self.sfreq)
        if self.settings.preprocessing.enabled:
            logger.info("Applying preprocessing pipeline")
            ecg, new_sfreq = preprocess(ecg, self.sfreq, self.settings.preprocessing)
            if new_sfreq != self.sfreq:
                logger.info(f"Sampling frequency changed: {self.sfreq} Hz -> {new_sfreq} Hz")
                self.sfreq = new_sfreq
            logger.info("Preprocessing complete")
        else:
            logger.info("Preprocessing disabled, using raw ECG data")

        # Extract features from each extractor
        feature_dfs = []
        for extractor_name, extractor in self._extractors.items():
            logger.info(f"Extracting {extractor_name} features...")
            features = extractor.get_features(ecg)
            feature_dfs.append(features)
            logger.info(f"Extracted {features.shape[1]} {extractor_name} features from {features.shape[0]} samples")

        # Concatenate all features horizontally
        if len(feature_dfs) == 1:
            result = feature_dfs[0]
        else:
            result = pd.concat(feature_dfs, axis=1)

        logger.info(f"Feature extraction complete. Total features: {result.shape[1]} from {result.shape[0]} ECGs")

        return result


def get_features(
    ecg: ECGData,  # Shape: (n_ecgs, n_channels, n_timepoints)
    sfreq: float,
    settings: Settings | str | Path | None = None,
) -> pd.DataFrame:
    """Extract features from ECG data.

    This is the main high-level API for feature extraction. It handles
    configuration loading and orchestrates the complete feature extraction pipeline.

    Args:
        ecg: ECG data with shape (n_ecgs, n_channels, n_timepoints).
            The number of channels must match the number of leads specified in
            settings.lead_order. By default, expects 12 leads in standard order.
        sfreq: Sampling frequency in Hz
        settings: Configuration for feature extraction. Can be:
            - Settings object: Use directly
            - str or Path: Load from JSON/TOML config file
            - None: Use default settings (12-lead standard order)

    Returns:
        DataFrame with shape (n_ecgs, n_features) containing all extracted features.
        Column names follow pattern: {extractor_name}_{feature_name}_{lead_name}

    Raises:
        ValueError: If input data has invalid shape, number of channels doesn't match
            lead_order, or settings are invalid
        FileNotFoundError: If settings is a path that doesn't exist
        ValidationError: If config file doesn't match schema

    Examples:
        # Use default settings (morphological enabled, 12-lead standard order)
        features = pte_ecg.get_features(ecg_data, sfreq=1000)

        # Enable additional extractors
        settings = pte_ecg.Settings()
        settings.features.statistical = {"enabled": True}
        features = pte_ecg.get_features(ecg_data, sfreq=1000, settings=settings)

        # Custom lead order (limb leads only)
        settings = pte_ecg.Settings(lead_order=["I", "II", "III", "aVR", "aVL", "aVF"])
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
    extractor = FeatureExtractor(sfreq, settings_obj)
    return extractor.extract_features(ecg)
