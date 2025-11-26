"""Base classes and protocols for feature extractors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

from ..types import ECGData

if TYPE_CHECKING:
    from ..core import FeatureExtractor


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Protocol that all feature extractors must implement.

    This protocol defines the interface that all feature extractors must follow,
    enabling a plugin-based architecture where custom extractors can be added
    without modifying the core library.

    Attributes:
        name: Unique identifier for the extractor (e.g., "morphological", "fft")
        available_features: List of all features this extractor can compute
        parent: Reference to the parent FeatureExtractor for accessing sfreq, lead_order, etc.
    """

    name: str
    available_features: list[str]

    def get_features(
        self,
        ecg: ECGData,  # Shape: (n_ecgs, n_channels, n_timepoints)
    ) -> pd.DataFrame:
        """Extract features from ECG data.

        Args:
            ecg: ECG data with shape (n_ecgs, n_channels, n_timepoints)

        Returns:
            DataFrame with shape (n_ecgs, n_features) containing extracted features.
            Column names follow the pattern: {extractor_name}_{feature_name}_{lead_name}
        """
        ...


class BaseFeatureExtractor:
    """Base class providing common functionality for feature extractors.

    Feature extractors receive the parent FeatureExtractor via dependency injection,
    allowing them to access:
    - parent.sfreq: Sampling frequency in Hz
    - parent.lead_order: List of lead names
    - parent.settings: Full settings object

    Subclasses must define:
    - name: str attribute
    - available_features: list[str] attribute
    - __init__(self, parent: FeatureExtractor) method
    - get_features(self, ecg) method implementation
    """

    name: str = ""
    available_features: list[str] = []
    parent: FeatureExtractor

    @property
    def sfreq(self) -> float:
        """Return sampling frequency from parent."""
        return self.parent.sfreq

    @property
    def lead_order(self) -> list[str]:
        """Return lead order from parent."""
        return self.parent.lead_order
