"""Base classes and protocols for feature extractors."""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..config.models import Settings
from ..types import ECGData


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Protocol that all feature extractors must implement.

    This protocol defines the interface that all feature extractors must follow,
    enabling a plugin-based architecture where custom extractors can be added
    without modifying the core library.

    Attributes:
        name: Unique identifier for the extractor (e.g., "morphological", "fft")
        available_features: List of all features this extractor can compute
    """

    name: str
    available_features: list[str]

    def get_features(
        self,
        ecg: ECGData,  # Shape: (n_ecgs, n_channels, n_timepoints)
        sfreq: float,
    ) -> pd.DataFrame:
        """Extract features from ECG data.

        Args:
            ecg: ECG data with shape (n_ecgs, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz

        Returns:
            DataFrame with shape (n_ecgs, n_features) containing extracted features.
            Column names follow the pattern: {extractor_name}_{feature_name}_{lead_name}
        """
        ...


class BaseFeatureExtractor:
    """Base class providing common functionality for feature extractors.

    This class provides validation and utility methods that all feature extractors
    can inherit. Subclasses must define:
    - name: str attribute
    - available_features: list[str] attribute
    - get_features() method implementation

    Args:
        settings: Complete settings object containing configuration for the extractor.
            Extractors can access:
            - settings.lead_order: List of lead names
            - settings.features.{extractor_name}.features: Selected features (or "all")
            - settings.features.{extractor_name}.n_jobs: Number of parallel jobs
            - Any other settings they need

    Raises:
        ValueError: If selected_features contains features not in available_features
    """

    name: str = ""
    available_features: list[str] = []

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lead_order = settings.lead_order
        
        # Get extractor-specific config
        extractor_config = getattr(settings.features, self.name, None)
        if extractor_config is None:
            # Fallback for extractors not in settings
            self.n_jobs = -1
            self.selected_features = self.available_features.copy()
        else:
            self.n_jobs = extractor_config.n_jobs
            if extractor_config.features == "all":
                self.selected_features = self.available_features.copy()
            else:
                self.selected_features = extractor_config.features
        
        self._validate_features()

    def _validate_features(self) -> None:
        """Validate that selected_features are all available in this extractor.

        Raises:
            ValueError: If any selected feature is not in available_features
        """
        invalid = set(self.selected_features) - set(self.available_features)
        if invalid:
            raise ValueError(
                f"Invalid features for {self.name} extractor: {sorted(invalid)}. "
                f"Available features: {sorted(self.available_features)}"
            )

    def _should_extract_feature(self, feature_name: str) -> bool:
        """Check if a feature should be extracted based on selected_features.

        Args:
            feature_name: Name of the feature to check

        Returns:
            True if feature should be extracted, False otherwise
        """
        return feature_name in self.selected_features
