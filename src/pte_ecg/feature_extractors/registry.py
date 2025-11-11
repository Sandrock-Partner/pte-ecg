"""Plugin registry for feature extractors."""

from importlib.metadata import entry_points

from .._logging import logger
from .base import FeatureExtractorProtocol


class ExtractorRegistry:
    """Registry for feature extractor plugins.

    This class manages a registry of available feature extractors. Extractors
    can be registered manually or discovered automatically via entry points.

    The registry uses the 'pte_ecg.extractors' entry point group for plugin
    discovery. Plugins should define their entry point in pyproject.toml:

        [project.entry-points."pte_ecg.extractors"]
        fft = "pte_ecg.feature_extractors.fft:FFTExtractor"
        morphological = "pte_ecg.feature_extractors.morphological:MorphologicalExtractor"

    Examples:
        # Get singleton instance
        registry = ExtractorRegistry.get_instance()

        # List available extractors
        names = registry.list_extractors()

        # Get an extractor class
        FFTExtractor = registry.get("fft")

        # Register a custom extractor
        registry.register("custom", CustomExtractor)
    """

    _instance: "ExtractorRegistry | None" = None
    _extractors: dict[str, type[FeatureExtractorProtocol]]

    def __init__(self):
        """Initialize the registry and discover plugins."""
        self._extractors = {}
        self._discover_plugins()

    @classmethod
    def get_instance(cls) -> "ExtractorRegistry":
        """Get the singleton registry instance.

        Returns:
            The singleton ExtractorRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _discover_plugins(self) -> None:
        """Discover and register extractors via entry points."""
        try:
            # Python 3.10+ API
            eps = entry_points(group="pte_ecg.extractors")
        except TypeError:
            # Python 3.9 fallback
            eps = entry_points().get("pte_ecg.extractors", [])

        for ep in eps:
            try:
                extractor_class = ep.load()
                self._extractors[ep.name] = extractor_class
                logger.debug(f"Discovered extractor plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load extractor plugin '{ep.name}': {e}")

        if self._extractors:
            logger.info(
                f"Discovered {len(self._extractors)} extractor plugin(s): "
                f"{list(self._extractors.keys())}"
            )
        else:
            logger.warning(
                "No extractor plugins discovered via entry points. "
                "Make sure entry points are defined in pyproject.toml."
            )

    def register(
        self, name: str, extractor_class: type[FeatureExtractorProtocol]
    ) -> None:
        """Manually register an extractor.

        Args:
            name: Unique name for the extractor
            extractor_class: Extractor class implementing FeatureExtractorProtocol

        Raises:
            ValueError: If name is already registered or extractor doesn't implement protocol
        """
        if name in self._extractors:
            raise ValueError(
                f"Extractor '{name}' is already registered. "
                "Use a different name or unregister the existing one first."
            )

        # Verify it implements the protocol (basic check)
        if not hasattr(extractor_class, "name") or not hasattr(
            extractor_class, "get_features"
        ):
            raise ValueError(
                f"Extractor class {extractor_class} does not implement FeatureExtractorProtocol. "
                "It must have 'name' attribute and 'get_features' method."
            )

        self._extractors[name] = extractor_class
        logger.info(f"Registered extractor: {name}")

    def unregister(self, name: str) -> None:
        """Unregister an extractor.

        Args:
            name: Name of the extractor to unregister

        Raises:
            KeyError: If extractor is not registered
        """
        if name not in self._extractors:
            raise KeyError(f"Extractor '{name}' is not registered")

        del self._extractors[name]
        logger.info(f"Unregistered extractor: {name}")

    def get(self, name: str) -> type[FeatureExtractorProtocol]:
        """Get an extractor class by name.

        Args:
            name: Name of the extractor

        Returns:
            Extractor class

        Raises:
            KeyError: If extractor is not registered
        """
        if name not in self._extractors:
            raise KeyError(
                f"Extractor '{name}' not found. Available extractors: "
                f"{list(self._extractors.keys())}"
            )
        return self._extractors[name]

    def list_extractors(self) -> list[str]:
        """List all registered extractor names.

        Returns:
            List of extractor names
        """
        return list(self._extractors.keys())

    def has_extractor(self, name: str) -> bool:
        """Check if an extractor is registered.

        Args:
            name: Name of the extractor

        Returns:
            True if extractor is registered, False otherwise
        """
        return name in self._extractors
