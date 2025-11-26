"""Pydantic models for configuration."""

from typing import Any

import pydantic
from pydantic import BaseModel, Field

from ..constants import ALLOWED_LEAD_NAMES, STANDARD_LEAD_NAMES
from ..preprocessing import PreprocessingSettings


class FeaturesConfig(BaseModel):
    """Top-level configuration for all feature extractors.

    Each attribute is a dictionary with extractor-specific settings.
    The only required key is 'enabled' (bool). All other keys are
    passed to the extractor as keyword arguments.

    Attributes:
        morphological: Waveform shape and interval measurements (enabled by default)
        fft: FFT-based frequency domain features (disabled by default)
        statistical: Basic statistical summaries (disabled by default)
        welch: Power spectral density features (disabled by default)
        nonlinear: Complexity and entropy features (disabled by default)
        waveshape: Bispectrum-based features (disabled by default)

    Examples:
        # Enable morphological (default) and statistical extractors
        config = FeaturesConfig(
            statistical={"enabled": True}
        )

        # Enable fft with custom settings
        config = FeaturesConfig(
            fft={"enabled": True, "n_jobs": 4}
        )
    """

    morphological: dict[str, Any] = Field(default_factory=lambda: {"enabled": True})
    fft: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    statistical: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    welch: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    nonlinear: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    waveshape: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})

    def get_extractor_config(self, name: str) -> dict[str, Any]:
        """Get configuration for an extractor by name.

        Args:
            name: Name of the extractor

        Returns:
            Dictionary with extractor config. Returns {"enabled": False}
            for unknown extractors.
        """
        return getattr(self, name, {"enabled": False})

    @pydantic.model_validator(mode="after")
    def check_any_features(self) -> "FeaturesConfig":
        """Validate that at least one feature extractor is enabled.

        Raises:
            ValueError: If all feature extractors are disabled
        """
        enabled_extractors = []

        for name in self.model_fields:
            config = getattr(self, name, {})
            if isinstance(config, dict) and config.get("enabled", False):
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
        lead_order: List of lead names specifying the order of leads in the ECG data.
            Each lead name must be from the allowed set: I, II, III, aVR, aVL, aVF, V1-V6.
            The number of leads determines the expected number of channels in the ECG data.
            Default is the standard 12-lead order: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    Examples:
        # Default settings (morphological enabled, standard 12-lead order)
        settings = Settings()

        # Enable additional extractors
        settings = Settings()
        settings.features.statistical = {"enabled": True}

        # Custom lead order (e.g., only limb leads)
        settings = Settings(lead_order=["I", "II", "III", "aVR", "aVL", "aVF"])
    """

    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    lead_order: list[str] = Field(
        default_factory=lambda: list(STANDARD_LEAD_NAMES),
        description="Order of leads in the ECG data. Each name must be from the allowed set.",
    )

    @pydantic.field_validator("lead_order")
    @classmethod
    def validate_lead_order(cls, v: list[str]) -> list[str]:
        """Validate that all lead names are from the allowed set.

        Args:
            v: List of lead names

        Returns:
            Validated list of lead names

        Raises:
            ValueError: If any lead name is not in the allowed set
        """
        if not v:
            raise ValueError("lead_order cannot be empty. Specify at least one lead name.")

        invalid_leads = set(v) - ALLOWED_LEAD_NAMES
        if invalid_leads:
            raise ValueError(
                f"Invalid lead names found: {sorted(invalid_leads)}. "
                f"Allowed lead names are: {sorted(ALLOWED_LEAD_NAMES)}"
            )

        return v
