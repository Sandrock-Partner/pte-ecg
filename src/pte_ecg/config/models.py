"""Pydantic models for configuration."""

from typing import Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field

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
    for configuration. Supports both hardcoded extractors (for backward compatibility)
    and dynamically discovered extractors via plugins.

    Attributes:
        fft: FFT-based frequency domain features
        morphological: Waveform shape and interval measurements
        statistical: Basic statistical summaries
        welch: Power spectral density features
        nonlinear: Complexity and entropy features (disabled by default)
        waveshape: Bispectrum-based features (disabled by default)

    Dynamic extractors registered via plugins are automatically supported and
    can be accessed via attribute access. They will use default ExtractorConfig
    values if not explicitly configured.
    """

    model_config = ConfigDict(extra="allow")

    fft: ExtractorConfig = Field(default_factory=ExtractorConfig)
    morphological: ExtractorConfig = Field(default_factory=ExtractorConfig)
    statistical: ExtractorConfig = Field(default_factory=ExtractorConfig)
    welch: ExtractorConfig = Field(default_factory=ExtractorConfig)
    nonlinear: ExtractorConfig = Field(default_factory=lambda: ExtractorConfig(enabled=False))
    waveshape: ExtractorConfig = Field(default_factory=lambda: ExtractorConfig(enabled=False))

    def __getattr__(self, name: str) -> ExtractorConfig:
        """Return default ExtractorConfig for dynamic extractor names.

        This allows plugin-registered extractors to be accessed via attribute
        access even if they're not explicitly defined in the config.

        Args:
            name: Name of the extractor

        Returns:
            Default ExtractorConfig instance
        """
        # Only handle extractor configs, not other attributes
        # Check if this is a known Pydantic field first
        if name in self.model_fields:
            return super().__getattribute__(name)
        
        # For dynamic extractors, return a default config
        # Store it in __dict__ so it persists
        if name not in self.__dict__:
            self.__dict__[name] = ExtractorConfig()
        return self.__dict__[name]

    @pydantic.model_validator(mode="after")
    def check_any_features(self) -> "FeaturesConfig":
        """Validate that at least one feature extractor is enabled.

        Checks both hardcoded and dynamically added extractor configs.

        Raises:
            ValueError: If all feature extractors are disabled
        """
        enabled_extractors = []
        checked_names = set()
        
        # Check hardcoded fields
        for name, field in self.model_fields.items():
            config = getattr(self, name, None)
            if isinstance(config, ExtractorConfig) and config.enabled:
                enabled_extractors.append(name)
            checked_names.add(name)
        
        # Check dynamic fields from model_dump (includes extra='allow' fields)
        dumped = self.model_dump()
        for name, value in dumped.items():
            if name not in checked_names:
                # This is a dynamic field
                if isinstance(value, dict) and value.get("enabled", False):
                    enabled_extractors.append(name)
                elif isinstance(value, ExtractorConfig) and value.enabled:
                    enabled_extractors.append(name)
                checked_names.add(name)
        
        # Also check __dict__ for dynamically added configs (fallback)
        for name, value in self.__dict__.items():
            if name not in checked_names and isinstance(value, ExtractorConfig):
                if value.enabled:
                    enabled_extractors.append(name)

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
