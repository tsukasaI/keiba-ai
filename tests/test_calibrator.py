"""Tests for calibrator module."""

import numpy as np
import pytest

from src.models.calibrator import (
    BinningCalibration,
    CalibrationAnalyzer,
    CalibrationBin,
    ExactaCalibrator,
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
)


class TestPlattScaling:
    """Test Platt scaling calibration."""

    def test_fit_and_calibrate(self):
        """Test fitting and calibrating with Platt scaling."""
        np.random.seed(42)
        probs = np.random.rand(100)
        labels = (np.random.rand(100) < probs).astype(int)

        calibrator = PlattScaling()
        calibrator.fit(probs, labels)

        calibrated = calibrator.calibrate(probs)
        assert len(calibrated) == len(probs)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_calibrate_without_fit_raises(self):
        """Test that calibrate without fit raises error."""
        calibrator = PlattScaling()
        with pytest.raises(ValueError, match="not fitted"):
            calibrator.calibrate(np.array([0.5]))


class TestIsotonicCalibration:
    """Test isotonic regression calibration."""

    def test_fit_and_calibrate(self):
        """Test fitting and calibrating."""
        np.random.seed(42)
        probs = np.random.rand(100)
        labels = (np.random.rand(100) < probs).astype(int)

        calibrator = IsotonicCalibration()
        calibrator.fit(probs, labels)

        calibrated = calibrator.calibrate(probs)
        assert len(calibrated) == len(probs)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_monotonic_output(self):
        """Test that output is monotonically increasing."""
        np.random.seed(42)
        probs = np.random.rand(100)
        labels = (np.random.rand(100) < probs).astype(int)

        calibrator = IsotonicCalibration()
        calibrator.fit(probs, labels)

        # Test on sorted input
        sorted_probs = np.sort(probs)
        calibrated = calibrator.calibrate(sorted_probs)

        # Should be monotonically increasing (or equal)
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1] + 1e-10


class TestTemperatureScaling:
    """Test temperature scaling calibration."""

    def test_fit_and_calibrate(self):
        """Test fitting and calibrating."""
        np.random.seed(42)
        probs = np.random.rand(100)
        labels = (np.random.rand(100) < probs).astype(int)

        calibrator = TemperatureScaling()
        calibrator.fit(probs, labels)

        calibrated = calibrator.calibrate(probs)
        assert len(calibrated) == len(probs)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_temperature_adjustment(self):
        """Test that temperature adjusts probabilities."""
        calibrator = TemperatureScaling()
        calibrator.temperature = 2.0  # Higher temp = lower confidence
        calibrator.fitted = True

        probs = np.array([0.9, 0.1])
        calibrated = calibrator.calibrate(probs)

        # High temp should push toward 0.5
        assert calibrated[0] < 0.9
        assert calibrated[1] > 0.1


class TestBinningCalibration:
    """Test histogram binning calibration."""

    def test_fit_and_calibrate(self):
        """Test fitting and calibrating."""
        np.random.seed(42)
        probs = np.random.rand(100)
        labels = (np.random.rand(100) < probs).astype(int)

        calibrator = BinningCalibration(n_bins=10)
        calibrator.fit(probs, labels)

        calibrated = calibrator.calibrate(probs)
        assert len(calibrated) == len(probs)
        assert all(0 <= p <= 1 for p in calibrated)


class TestCalibrationAnalyzer:
    """Test calibration analysis."""

    def test_analyze_creates_bins(self):
        """Test that analyze creates calibration bins."""
        np.random.seed(42)
        probs = np.random.rand(100)
        labels = (np.random.rand(100) < probs).astype(int)

        analyzer = CalibrationAnalyzer(n_bins=5)
        bins = analyzer.analyze(probs, labels)

        # May have fewer bins if some are empty
        assert len(bins) <= 5
        assert all(isinstance(b, CalibrationBin) for b in bins)

    def test_brier_score(self):
        """Test Brier score calculation."""
        probs = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])

        analyzer = CalibrationAnalyzer()
        brier = analyzer.brier_score(probs, labels)

        # Should be low for well-calibrated predictions
        assert 0 <= brier <= 1
        assert brier < 0.1

    def test_brier_score_perfect(self):
        """Test Brier score with perfect predictions."""
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        labels = np.array([1, 0, 1, 0])

        analyzer = CalibrationAnalyzer()
        brier = analyzer.brier_score(probs, labels)

        assert brier == 0.0

    def test_expected_calibration_error(self):
        """Test ECE calculation."""
        bins = [
            CalibrationBin(
                bin_start=0.0,
                bin_end=0.5,
                bin_center=0.25,
                num_samples=50,
                avg_predicted_prob=0.25,
                actual_hit_rate=0.25,  # Perfect calibration
                num_hits=12,
                confidence_error=0.0,
            ),
            CalibrationBin(
                bin_start=0.5,
                bin_end=1.0,
                bin_center=0.75,
                num_samples=50,
                avg_predicted_prob=0.75,
                actual_hit_rate=0.75,  # Perfect calibration
                num_hits=37,
                confidence_error=0.0,
            ),
        ]

        analyzer = CalibrationAnalyzer()
        ece = analyzer.expected_calibration_error(bins)

        assert ece == 0.0

    def test_detect_issues_overconfident(self):
        """Test detecting overconfidence."""
        bins = [
            CalibrationBin(
                bin_start=0.0,
                bin_end=0.5,
                bin_center=0.25,
                num_samples=100,
                avg_predicted_prob=0.4,
                actual_hit_rate=0.2,  # Overconfident
                num_hits=20,
                confidence_error=0.2,
            ),
        ]

        analyzer = CalibrationAnalyzer()
        issues = analyzer.detect_issues(bins)

        assert "overall" in issues
        assert "Overconfident" in issues["overall"]


class TestExactaCalibrator:
    """Test exacta probability calibrator."""

    def test_fit_and_calibrate(self):
        """Test fitting and calibrating exacta probabilities."""
        np.random.seed(42)
        predictions = [
            {"predicted_prob": np.random.rand(), "won": np.random.rand() < 0.1}
            for _ in range(100)
        ]

        calibrator = ExactaCalibrator(method="platt")
        calibrator.fit_from_backtest(predictions)

        exacta_probs = {
            ("A", "B"): 0.05,
            ("A", "C"): 0.03,
            ("B", "A"): 0.04,
        }

        calibrated = calibrator.calibrate_dict(exacta_probs)

        assert len(calibrated) == len(exacta_probs)
        assert all(0 <= p <= 1 for p in calibrated.values())

    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            ExactaCalibrator(method="invalid")

    def test_available_methods(self):
        """Test that all methods are available."""
        for method in ["platt", "isotonic", "temperature", "binning"]:
            calibrator = ExactaCalibrator(method=method)
            assert calibrator.method == method
