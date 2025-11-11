"""Configuration file loaders for JSON and TOML formats."""

import json
import tomllib  # Python 3.11+ built-in
from pathlib import Path

from .models import Settings


class ConfigLoader:
    """Utility class for loading Settings from configuration files.

    Supports JSON and TOML formats. YAML is not supported to avoid
    external dependencies.

    Examples:
        # Load from JSON
        settings = ConfigLoader.from_json("config.json")

        # Load from TOML
        settings = ConfigLoader.from_toml("config.toml")

        # Auto-detect format
        settings = ConfigLoader.from_file("config.json")
        settings = ConfigLoader.from_file("config.toml")
    """

    @staticmethod
    def from_json(path: str | Path) -> Settings:
        """Load settings from a JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            Settings object validated against Pydantic models

        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If file is not valid JSON
            pydantic.ValidationError: If config doesn't match schema

        Examples:
            settings = ConfigLoader.from_json("config.json")
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return Settings(**data)

    @staticmethod
    def from_toml(path: str | Path) -> Settings:
        """Load settings from a TOML file.

        Args:
            path: Path to TOML configuration file

        Returns:
            Settings object validated against Pydantic models

        Raises:
            FileNotFoundError: If file does not exist
            tomllib.TOMLDecodeError: If file is not valid TOML
            pydantic.ValidationError: If config doesn't match schema

        Examples:
            settings = ConfigLoader.from_toml("config.toml")
        """
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return Settings(**data)

    @staticmethod
    def from_file(path: str | Path) -> Settings:
        """Load settings from a file, auto-detecting format by extension.

        Supports .json and .toml file extensions.

        Args:
            path: Path to configuration file (.json or .toml)

        Returns:
            Settings object validated against Pydantic models

        Raises:
            ValueError: If file extension is not .json or .toml
            FileNotFoundError: If file does not exist
            Exception: Various parsing/validation errors depending on format

        Examples:
            settings = ConfigLoader.from_file("config.json")
            settings = ConfigLoader.from_file("config.toml")
        """
        path = Path(path)

        if path.suffix == ".json":
            return ConfigLoader.from_json(path)
        elif path.suffix == ".toml":
            return ConfigLoader.from_toml(path)
        else:
            raise ValueError(
                f"Unsupported config file format: {path.suffix}. "
                "Only .json and .toml are supported."
            )
