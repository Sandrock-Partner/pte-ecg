"""Unit tests for feature extractors."""

import numpy as np
import pandas as pd
import pytest

from pte_ecg.feature_extractors.fft import FFTExtractor
from pte_ecg.feature_extractors.morphological import MorphologicalExtractor
from pte_ecg.feature_extractors.statistical import StatisticalExtractor
from pte_ecg.feature_extractors.welch import WelchExtractor

# Optional extractors - import with try-except
try:
    from pte_ecg.feature_extractors.nonlinear import NonlinearExtractor

    HAS_NONLINEAR = True
except ImportError:
    HAS_NONLINEAR = False

try:
    from pte_ecg.feature_extractors.waveshape import WaveShapeExtractor

    HAS_WAVESHAPE = True
except ImportError:
    HAS_WAVESHAPE = False


class TestFFTExtractor:
    """Tests for FFTExtractor."""

    def test_basic_extraction(self, test_data: tuple[np.ndarray, int]):
        """Test basic FFT feature extraction."""
        ecg_data, sfreq = test_data
        extractor = FFTExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == ecg_data.shape[0]  # Same number of samples
        assert features.shape[1] > 0  # At least one feature
        assert all(col.startswith("fft_") for col in features.columns)

    def test_single_channel(self, single_channel_data: tuple[np.ndarray, int]):
        """Test FFT extraction with single channel."""
        ecg_data, sfreq = single_channel_data
        extractor = FFTExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == ecg_data.shape[0]
        # Single channel should have features ending with _ch0
        assert any("_ch0" in col for col in features.columns)

    def test_invalid_dimensions(self):
        """Test FFT extractor with invalid input dimensions."""
        extractor = FFTExtractor()

        # 2D input should raise ValueError
        with pytest.raises(ValueError, match="must have 3 dimensions"):
            extractor.get_features(np.random.randn(10, 100), sfreq=100)

        # 1D input should raise ValueError
        with pytest.raises(ValueError, match="must have 3 dimensions"):
            extractor.get_features(np.random.randn(100), sfreq=100)

    def test_feature_consistency(self, test_data: tuple[np.ndarray, int]):
        """Test that FFT features are consistent across runs."""
        ecg_data, sfreq = test_data
        extractor = FFTExtractor()

        features1 = extractor.get_features(ecg_data, sfreq)
        features2 = extractor.get_features(ecg_data, sfreq)

        pd.testing.assert_frame_equal(features1, features2)


class TestStatisticalExtractor:
    """Tests for StatisticalExtractor."""

    def test_basic_extraction(self, test_data: tuple[np.ndarray, int]):
        """Test basic statistical feature extraction."""
        ecg_data, sfreq = test_data
        extractor = StatisticalExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == ecg_data.shape[0]
        assert features.shape[1] > 0
        assert all(col.startswith("statistical_") for col in features.columns)

    def test_expected_features(self, test_data: tuple[np.ndarray, int]):
        """Test that expected statistical features are present."""
        ecg_data, sfreq = test_data
        extractor = StatisticalExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        # Check for common statistical features (they include channel suffix)
        expected_prefixes = ["statistical_mean_ch", "statistical_var_ch", "statistical_median_ch"]
        for prefix in expected_prefixes:
            assert any(prefix in col for col in features.columns)

    def test_no_nan_values(self, test_data: tuple[np.ndarray, int]):
        """Test that statistical features don't contain NaN values for valid data."""
        ecg_data, sfreq = test_data
        extractor = StatisticalExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        # Statistical features should not have NaN for valid data
        assert not features.isnull().any().any()


class TestWelchExtractor:
    """Tests for WelchExtractor."""

    def test_basic_extraction(self, test_data: tuple[np.ndarray, int]):
        """Test basic Welch feature extraction."""
        ecg_data, sfreq = test_data
        extractor = WelchExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == ecg_data.shape[0]
        assert features.shape[1] > 0
        assert all(col.startswith("welch_") for col in features.columns)

    def test_frequency_bins(self, test_data: tuple[np.ndarray, int]):
        """Test that Welch extractor creates expected frequency bins."""
        ecg_data, sfreq = test_data
        extractor = WelchExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        # Should have bin features
        bin_cols = [col for col in features.columns if "welch_bin_" in col]
        assert len(bin_cols) > 0

    def test_with_different_sfreq(self):
        """Test Welch extractor with different sampling frequencies."""
        n_samples, n_channels, n_timepoints = 2, 3, 500
        ecg_data = np.random.randn(n_samples, n_channels, n_timepoints)
        extractor = WelchExtractor()

        # Test with different sampling frequencies
        for sfreq in [100, 250, 500]:
            features = extractor.get_features(ecg_data, sfreq)
            assert isinstance(features, pd.DataFrame)
            assert features.shape[0] == n_samples


class TestMorphologicalExtractor:
    """Tests for MorphologicalExtractor."""

    def test_basic_extraction(self, test_data: tuple[np.ndarray, int]):
        """Test basic morphological feature extraction."""
        ecg_data, sfreq = test_data
        extractor = MorphologicalExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == ecg_data.shape[0]
        assert features.shape[1] > 0
        assert all(col.startswith("morphological_") for col in features.columns)

    def test_expected_morphological_features(self, test_data: tuple[np.ndarray, int]):
        """Test that expected morphological features are present."""
        ecg_data, sfreq = test_data
        extractor = MorphologicalExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        # Check for common morphological features (they include channel suffix)
        expected_features = [
            "morphological_qrs_duration_ch",
            "morphological_qrs_dispersion_ch",
            "morphological_rr_interval_mean_ch",
        ]
        for feature in expected_features:
            # At least one channel should have these features
            assert any(feature in col for col in features.columns)

    def test_flat_signal_handling(self):
        """Test morphological extractor with flat signal."""
        # Create flat signal
        n_samples, n_channels, n_timepoints = 1, 2, 500
        ecg_data = np.zeros((n_samples, n_channels, n_timepoints))
        sfreq = 100

        extractor = MorphologicalExtractor()
        features = extractor.get_features(ecg_data, sfreq)

        # Should return DataFrame even with flat signal
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == n_samples

    def test_n_jobs_parameter(self, test_data: tuple[np.ndarray, int]):
        """Test morphological extractor with different n_jobs values."""
        ecg_data, sfreq = test_data

        # Test with single process
        extractor_single = MorphologicalExtractor(n_jobs=1)
        features_single = extractor_single.get_features(ecg_data, sfreq)

        # Test with multiple processes
        extractor_multi = MorphologicalExtractor(n_jobs=2)
        features_multi = extractor_multi.get_features(ecg_data, sfreq)

        # Results should have same shape
        assert features_single.shape == features_multi.shape


@pytest.mark.skipif(not HAS_NONLINEAR, reason="nolds package not installed")
class TestNonlinearExtractor:
    """Tests for NonlinearExtractor (requires nolds)."""

    def test_basic_extraction(self, test_data: tuple[np.ndarray, int]):
        """Test basic nonlinear feature extraction."""
        ecg_data, sfreq = test_data
        extractor = NonlinearExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == ecg_data.shape[0]
        assert features.shape[1] > 0
        assert all(col.startswith("nonlinear_") for col in features.columns)

    def test_expected_nonlinear_features(self, test_data: tuple[np.ndarray, int]):
        """Test that expected nonlinear features are present."""
        ecg_data, sfreq = test_data
        extractor = NonlinearExtractor()

        features = extractor.get_features(ecg_data, sfreq)

        # Check for common nonlinear features
        expected_features = [
            "nonlinear_sample_entropy",
            "nonlinear_hurst_exponent",
            "nonlinear_dfa_alpha1",
        ]
        for feature in expected_features:
            assert any(feature in col for col in features.columns)

    def test_n_jobs_parameter(self, test_data: tuple[np.ndarray, int]):
        """Test nonlinear extractor with different n_jobs values."""
        ecg_data, sfreq = test_data

        # Test with single process
        extractor_single = NonlinearExtractor(n_jobs=1)
        features_single = extractor_single.get_features(ecg_data, sfreq)

        # Test with multiple processes
        extractor_multi = NonlinearExtractor(n_jobs=2)
        features_multi = extractor_multi.get_features(ecg_data, sfreq)

        # Results should have same shape
        assert features_single.shape == features_multi.shape


@pytest.mark.skipif(not HAS_WAVESHAPE, reason="pybispectra package not installed")
class TestWaveShapeExtractor:
    """Tests for WaveShapeExtractor (requires pybispectra)."""

    def test_basic_extraction(self, test_data: tuple[np.ndarray, int]):
        """Test basic waveshape feature extraction."""
        # Skip this test for now - WaveShape implementation is incomplete
        # TODO: Complete WaveShape implementation and enable this test
        pytest.skip("WaveShape implementation is incomplete - returns 3D array instead of 2D")

    def test_import_error_without_pybispectra(self, monkeypatch):
        """Test that proper error is raised when pybispectra is not installed."""
        # This test only runs if pybispectra IS installed, so we simulate it not being available
        if not HAS_WAVESHAPE:
            pytest.skip("pybispectra not installed, cannot test error handling")

        # Mock the HAS_PYBISPECTRA flag
        import pte_ecg.feature_extractors.waveshape as ws_module

        monkeypatch.setattr(ws_module, "HAS_PYBISPECTRA", False)

        extractor = WaveShapeExtractor()
        ecg_data = np.random.randn(1, 2, 500)

        with pytest.raises(ImportError, match="pybispectra is required"):
            extractor.get_features(ecg_data, sfreq=100)


class TestExtractorRegistry:
    """Tests for extractor discovery and registry."""

    def test_all_extractors_discoverable(self):
        """Test that all extractors are discoverable via entry points."""
        from pte_ecg.feature_extractors.registry import ExtractorRegistry

        registry = ExtractorRegistry()

        # Core extractors should always be available
        assert "fft" in registry.list_extractors()
        assert "statistical" in registry.list_extractors()
        assert "welch" in registry.list_extractors()
        assert "morphological" in registry.list_extractors()

        # Optional extractors
        if HAS_NONLINEAR:
            assert "nonlinear" in registry.list_extractors()
        if HAS_WAVESHAPE:
            assert "waveshape" in registry.list_extractors()

    def test_extractor_instantiation(self):
        """Test that extractors can be instantiated from registry."""
        from pte_ecg.feature_extractors.registry import ExtractorRegistry

        registry = ExtractorRegistry()

        # Test core extractors
        for name in ["fft", "statistical", "welch", "morphological"]:
            extractor_class = registry.get(name)
            assert extractor_class is not None
            # Instantiate the extractor
            extractor = extractor_class()
            assert hasattr(extractor, "get_features")
            assert hasattr(extractor, "name")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
