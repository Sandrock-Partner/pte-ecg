"""Base classes and protocols for feature extractors."""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


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
        ecg: np.ndarray,
        sfreq: float,
        selected_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract features from ECG data.

        Args:
            ecg: ECG data with shape (n_samples, n_channels, n_timepoints)
            sfreq: Sampling frequency in Hz
            selected_features: List of features to extract. If None, extract all
                available features.

        Returns:
            DataFrame with shape (n_samples, n_features) containing extracted features.
            Column names follow the pattern: {extractor_name}_{feature_name}_ch{N}
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
        selected_features: List of features to extract. If None, all available
            features will be extracted. Must be a subset of available_features.
        n_jobs: Number of parallel jobs for extractors that support parallelization.
            -1 means use all CPUs, positive values specify exact count.

    Raises:
        ValueError: If selected_features contains features not in available_features
    """

    name: str = ""
    available_features: list[str] = []

    def __init__(
        self, selected_features: list[str] | None = None, n_jobs: int = -1
    ) -> None:
        self.n_jobs = n_jobs
        self.selected_features = selected_features or self.available_features.copy()
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
