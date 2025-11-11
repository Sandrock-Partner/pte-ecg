"""Pydantic models for configuration."""

from typing import Literal

import pydantic
from pydantic import BaseModel, Field

from ..preprocessing import PreprocessingSettings


class ExtractorConfig(BaseModel):
    """Configuration for a single feature extractor.

    Args:
        enabled: Whether this extractor should be used
        features: List of specific features to extract, or "all" for all features
        n_jobs: Number of parallel jobs (-1 = all CPUs, >0 = specific count)

    Examples:
        # Extract all features
        config = ExtractorConfig(enabled=True, features="all")

        # Extract specific features only
        config = ExtractorConfig(
            enabled=True,
            features=["st_elevation", "qtc_interval"],
            n_jobs=4
        )
    """

    enabled: bool = True
    features: list[str] | Literal["all"] = "all"
    n_jobs: int = -1


class FeaturesConfig(BaseModel):
    """Top-level configuration for all feature extractors.

    Each attribute corresponds to a feature extractor and uses ExtractorConfig
    for configuration. New extractors can be added as attributes with
    ExtractorConfig type.

    Attributes:
        fft: FFT-based frequency domain features
        morphological: Waveform shape and interval measurements
        statistical: Basic statistical summaries
        welch: Power spectral density features
        nonlinear: Complexity and entropy features (disabled by default)
        waveshape: Bispectrum-based features (disabled by default)
    """

    fft: ExtractorConfig = Field(default_factory=ExtractorConfig)
    morphological: ExtractorConfig = Field(default_factory=ExtractorConfig)
    statistical: ExtractorConfig = Field(default_factory=ExtractorConfig)
    welch: ExtractorConfig = Field(default_factory=ExtractorConfig)
    nonlinear: ExtractorConfig = Field(
        default_factory=lambda: ExtractorConfig(enabled=False)
    )
    waveshape: ExtractorConfig = Field(
        default_factory=lambda: ExtractorConfig(enabled=False)
    )

    @pydantic.model_validator(mode="after")
    def check_any_features(self) -> "FeaturesConfig":
        """Validate that at least one feature extractor is enabled.

        Raises:
            ValueError: If all feature extractors are disabled
        """
        enabled_extractors = [
            name
            for name, config in self.model_dump().items()
            if isinstance(config, dict) and config.get("enabled", False)
        ]

        if not enabled_extractors:
            raise ValueError(
                "At least one feature extractor must be enabled. "
                "Set enabled=True for at least one extractor in features config."
            )

        return self


class Settings(BaseModel):
    """Complete settings for ECG feature extraction.

    This is the top-level configuration object that combines preprocessing
    and feature extraction settings.

    Args:
        preprocessing: Preprocessing pipeline configuration
        features: Feature extraction configuration

    Examples:
        # Default settings (all features enabled)
        settings = Settings()

        # Custom settings
        settings = Settings()
        settings.features.morphological.features = ["st_elevation", "qtc_interval"]
        settings.features.fft.enabled = False
        settings.preprocessing.bandpass.enabled = True
        settings.preprocessing.bandpass.l_freq = 0.5
    """

    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
