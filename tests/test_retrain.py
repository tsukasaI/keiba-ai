"""Tests for retrain script functions."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from retrain import (
    fit_calibration,
    step_banner,
    BASELINE_ROI,
    MODEL_PKL_PATH,
    ONNX_PATH,
    CALIBRATION_PATH,
)


class TestStepBanner:
    """Test step_banner utility function."""

    def test_step_banner_output(self, capsys):
        """Test that step banner prints correctly."""
        step_banner(1, "TEST STEP")
        captured = capsys.readouterr()

        assert "=" * 60 in captured.out
        assert "STEP 1: TEST STEP" in captured.out

    def test_step_banner_different_numbers(self, capsys):
        """Test step banner with different step numbers."""
        step_banner(5, "ANOTHER STEP")
        captured = capsys.readouterr()

        assert "STEP 5: ANOTHER STEP" in captured.out


class TestFitCalibration:
    """Test fit_calibration function."""

    def test_empty_predictions_returns_default(self):
        """Test that empty predictions returns default calibration."""
        result = fit_calibration([])

        assert result["type"] == "temperature"
        assert result["temperature"] == 1.0

    def test_temperature_method(self):
        """Test temperature scaling calibration."""
        np.random.seed(42)
        predictions = [
            {"predicted_prob": np.random.rand() * 0.1, "won": np.random.rand() < 0.05}
            for _ in range(100)
        ]

        result = fit_calibration(predictions, method="temperature")

        assert result["type"] == "temperature"
        assert "temperature" in result
        assert 0.1 <= result["temperature"] <= 10.0

    def test_binning_method(self):
        """Test binning calibration."""
        np.random.seed(42)
        predictions = [
            {"predicted_prob": np.random.rand() * 0.1, "won": np.random.rand() < 0.05}
            for _ in range(100)
        ]

        result = fit_calibration(predictions, method="binning")

        assert result["type"] == "binning"
        assert "n_bins" in result
        assert result["n_bins"] == 15
        assert "bin_edges" in result
        assert "bin_values" in result
        assert len(result["bin_edges"]) == result["n_bins"] + 1
        assert len(result["bin_values"]) == result["n_bins"]

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        predictions = [{"predicted_prob": 0.5, "won": True}]

        with pytest.raises(ValueError, match="Unknown calibration method"):
            fit_calibration(predictions, method="invalid")

    def test_calibration_values_valid(self):
        """Test that calibration values are valid."""
        np.random.seed(42)
        predictions = [
            {"predicted_prob": p, "won": np.random.rand() < p}
            for p in np.linspace(0.01, 0.1, 100)
        ]

        result = fit_calibration(predictions, method="temperature")

        assert isinstance(result["temperature"], float)
        assert not np.isnan(result["temperature"])
        assert not np.isinf(result["temperature"])

    def test_binning_values_valid(self):
        """Test that binning values are all valid floats."""
        np.random.seed(42)
        predictions = [
            {"predicted_prob": np.random.rand() * 0.1, "won": np.random.rand() < 0.05}
            for _ in range(100)
        ]

        result = fit_calibration(predictions, method="binning")

        for edge in result["bin_edges"]:
            assert isinstance(edge, float)
            assert not np.isnan(edge)

        for value in result["bin_values"]:
            assert isinstance(value, float)
            assert 0 <= value <= 1


class TestConstants:
    """Test module constants."""

    def test_baseline_roi(self):
        """Test baseline ROI is reasonable."""
        assert BASELINE_ROI > 0
        assert BASELINE_ROI < 100

    def test_paths_are_paths(self):
        """Test that paths are Path objects."""
        assert isinstance(MODEL_PKL_PATH, Path)
        assert isinstance(ONNX_PATH, Path)
        assert isinstance(CALIBRATION_PATH, Path)

    def test_model_path_ends_with_pkl(self):
        """Test model path has correct extension."""
        assert MODEL_PKL_PATH.suffix == ".pkl"

    def test_onnx_path_ends_with_onnx(self):
        """Test ONNX path has correct extension."""
        assert ONNX_PATH.suffix == ".onnx"

    def test_calibration_path_ends_with_json(self):
        """Test calibration path has correct extension."""
        assert CALIBRATION_PATH.suffix == ".json"


class TestCalibrationExportFormat:
    """Test that calibration export format is compatible with Rust API."""

    def test_temperature_format_compatible(self):
        """Test temperature calibration format matches Rust API expectations."""
        predictions = [
            {"predicted_prob": 0.05, "won": False},
            {"predicted_prob": 0.08, "won": True},
            {"predicted_prob": 0.03, "won": False},
        ]

        result = fit_calibration(predictions, method="temperature")

        # Must have 'type' and 'temperature' keys
        assert "type" in result
        assert result["type"] == "temperature"
        assert "temperature" in result

        # Temperature must be a float (not int)
        assert isinstance(result["temperature"], float)

        # Must be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == result

    def test_binning_format_compatible(self):
        """Test binning calibration format matches Rust API expectations."""
        np.random.seed(42)
        predictions = [
            {"predicted_prob": np.random.rand() * 0.1, "won": np.random.rand() < 0.05}
            for _ in range(100)
        ]

        result = fit_calibration(predictions, method="binning")

        # Must have required keys
        assert result["type"] == "binning"
        assert "n_bins" in result
        assert "bin_edges" in result
        assert "bin_values" in result

        # Types must match Rust expectations
        assert isinstance(result["n_bins"], int)
        assert all(isinstance(x, float) for x in result["bin_edges"])
        assert all(isinstance(x, float) for x in result["bin_values"])

        # Must be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == result


class TestCalibrationWithRealisticData:
    """Test calibration with realistic exacta prediction data."""

    def test_low_hit_rate_data(self):
        """Test with realistic low hit rate (exacta ~3-5%)."""
        np.random.seed(42)
        n_predictions = 1000
        hit_rate = 0.04  # 4% hit rate

        predictions = []
        for _ in range(n_predictions):
            # Predictions are typically in 0.01-0.10 range
            pred_prob = np.random.uniform(0.01, 0.10)
            won = np.random.rand() < hit_rate
            predictions.append({"predicted_prob": pred_prob, "won": won})

        result = fit_calibration(predictions, method="temperature")

        # Should successfully fit
        assert result["type"] == "temperature"
        assert 0.1 <= result["temperature"] <= 10.0

    def test_extreme_probabilities(self):
        """Test with some extreme probability values."""
        predictions = [
            {"predicted_prob": 0.001, "won": False},
            {"predicted_prob": 0.999, "won": True},
            {"predicted_prob": 0.5, "won": False},
            {"predicted_prob": 0.5, "won": True},
        ]

        result = fit_calibration(predictions, method="temperature")

        # Should handle extreme values without error
        assert result["type"] == "temperature"
        assert not np.isnan(result["temperature"])
        assert not np.isinf(result["temperature"])

    def test_all_same_outcome(self):
        """Test with all predictions having same outcome."""
        # All losses
        predictions_loss = [
            {"predicted_prob": 0.05, "won": False}
            for _ in range(100)
        ]

        result = fit_calibration(predictions_loss, method="temperature")
        assert result["type"] == "temperature"

        # All wins (unrealistic but should handle)
        predictions_win = [
            {"predicted_prob": 0.05, "won": True}
            for _ in range(100)
        ]

        result = fit_calibration(predictions_win, method="temperature")
        assert result["type"] == "temperature"
