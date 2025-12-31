"""Tests for the ModelEvaluator class."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.models.evaluator import ModelEvaluator
from src.models.config import RACE_ID_COL, HORSE_NAME_COL, TARGET_COL


class TestModelEvaluatorInit:
    """Test ModelEvaluator initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        evaluator = ModelEvaluator()

        assert evaluator.exacta_calc is not None


class TestModelEvaluatorCalculateBasicMetrics:
    """Test calculate_basic_metrics method."""

    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator()

    def test_perfect_predictions(self, evaluator):
        """Perfect predictions should give 100% accuracy."""
        y_true = np.array([0, 1, 2])

        # Perfect predictions
        y_pred_proba = np.zeros((3, 18))
        y_pred_proba[0, 0] = 1.0
        y_pred_proba[1, 1] = 1.0
        y_pred_proba[2, 2] = 1.0

        result = evaluator.calculate_basic_metrics(y_true, y_pred_proba)

        assert result["top1_accuracy"] == 1.0
        assert result["top3_accuracy"] == 1.0
        assert result["avg_correct_prob"] == pytest.approx(1.0, abs=0.01)

    def test_wrong_predictions(self, evaluator):
        """Wrong predictions should give 0% top-1 accuracy."""
        y_true = np.array([0, 1, 2])

        # All wrong predictions
        y_pred_proba = np.zeros((3, 18))
        y_pred_proba[0, 5] = 1.0  # Predicts 5, actual 0
        y_pred_proba[1, 6] = 1.0  # Predicts 6, actual 1
        y_pred_proba[2, 7] = 1.0  # Predicts 7, actual 2

        result = evaluator.calculate_basic_metrics(y_true, y_pred_proba)

        assert result["top1_accuracy"] == 0.0
        assert result["top3_accuracy"] == 0.0

    def test_top3_accuracy(self, evaluator):
        """Top-3 accuracy should count predictions in top 3."""
        y_true = np.array([0, 1])

        y_pred_proba = np.zeros((2, 18))
        # First sample: actual 0 is in top 3 but not top 1
        y_pred_proba[0, 5] = 0.4  # Rank 1
        y_pred_proba[0, 6] = 0.3  # Rank 2
        y_pred_proba[0, 0] = 0.2  # Rank 3 (actual)
        y_pred_proba[0, 7] = 0.1  # Rank 4

        # Second sample: correct top-1 prediction
        y_pred_proba[1, 1] = 1.0

        result = evaluator.calculate_basic_metrics(y_true, y_pred_proba)

        assert result["top1_accuracy"] == 0.5  # Only second is top-1 correct
        assert result["top3_accuracy"] == 1.0  # Both are in top-3

    def test_log_loss_calculation(self, evaluator):
        """Log loss should be calculated correctly."""
        y_true = np.array([0, 1])

        y_pred_proba = np.zeros((2, 18))
        y_pred_proba[0, 0] = 0.5
        y_pred_proba[1, 1] = 0.5

        result = evaluator.calculate_basic_metrics(y_true, y_pred_proba)

        # Log loss = -mean(log(0.5)) = -log(0.5) ≈ 0.693
        assert result["log_loss"] == pytest.approx(0.693, abs=0.01)

    def test_returns_all_metrics(self, evaluator):
        """Should return all expected metrics."""
        y_true = np.array([0, 1, 2])
        y_pred_proba = np.ones((3, 18)) / 18

        result = evaluator.calculate_basic_metrics(y_true, y_pred_proba)

        assert "top1_accuracy" in result
        assert "top3_accuracy" in result
        assert "log_loss" in result
        assert "avg_correct_prob" in result


class TestModelEvaluatorCalculateExactaAccuracy:
    """Test calculate_exacta_accuracy method."""

    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator()

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        return model

    @pytest.fixture
    def sample_test_df(self):
        """Create sample test DataFrame."""
        return pd.DataFrame({
            RACE_ID_COL: [1001, 1001, 1001, 1002, 1002, 1002],
            HORSE_NAME_COL: ["A", "B", "C", "D", "E", "F"],
            TARGET_COL: [1, 2, 3, 1, 2, 3],  # A wins race 1, D wins race 2
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

    def test_returns_dict(self, evaluator, mock_model, sample_test_df):
        """Should return dictionary with metrics."""
        # Setup mock to predict correctly
        def mock_predict(X, horse_names):
            return {name: np.array([0.5 if i == 0 else 0.1] * 18)
                    for i, name in enumerate(horse_names)}
        mock_model.predict_race = mock_predict

        result = evaluator.calculate_exacta_accuracy(
            sample_test_df, mock_model, ["feature_1"]
        )

        assert isinstance(result, dict)
        assert "total_races" in result
        assert "exacta_accuracy" in result
        assert "win_accuracy" in result
        assert "place_accuracy" in result

    def test_counts_total_races(self, evaluator, mock_model, sample_test_df):
        """Should count total races correctly."""
        def mock_predict(X, horse_names):
            return {name: np.ones(18) / 18 for name in horse_names}
        mock_model.predict_race = mock_predict

        result = evaluator.calculate_exacta_accuracy(
            sample_test_df, mock_model, ["feature_1"]
        )

        assert result["total_races"] == 2

    def test_skips_small_races(self, evaluator, mock_model):
        """Races with fewer than 2 horses should be skipped."""
        df = pd.DataFrame({
            RACE_ID_COL: [1001],  # Only 1 horse
            HORSE_NAME_COL: ["A"],
            TARGET_COL: [1],
            "feature_1": [1.0],
        })

        def mock_predict(X, horse_names):
            return {name: np.ones(18) / 18 for name in horse_names}
        mock_model.predict_race = mock_predict

        result = evaluator.calculate_exacta_accuracy(df, mock_model, ["feature_1"])

        assert result["total_races"] == 0


class TestModelEvaluatorFullEvaluation:
    """Test full_evaluation method."""

    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator()

    @pytest.fixture
    def mock_model(self):
        """Create mock model that returns uniform probabilities."""
        model = Mock()
        model.predict_proba = Mock(return_value=np.ones((6, 18)) / 18)

        def mock_predict_race(X, horse_names):
            return {name: np.ones(18) / 18 for name in horse_names}
        model.predict_race = mock_predict_race

        return model

    @pytest.fixture
    def sample_test_df(self):
        """Create sample test DataFrame."""
        return pd.DataFrame({
            RACE_ID_COL: [1001, 1001, 1001, 1002, 1002, 1002],
            HORSE_NAME_COL: ["A", "B", "C", "D", "E", "F"],
            TARGET_COL: [1, 2, 3, 1, 2, 3],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

    def test_returns_combined_metrics(self, evaluator, mock_model, sample_test_df):
        """Should return combined basic and exacta metrics."""
        result = evaluator.full_evaluation(
            sample_test_df, mock_model, ["feature_1"]
        )

        # Basic metrics
        assert "top1_accuracy" in result
        assert "top3_accuracy" in result
        assert "log_loss" in result

        # Exacta metrics
        assert "total_races" in result
        assert "exacta_accuracy" in result


class TestModelEvaluatorPrintReport:
    """Test print_evaluation_report method."""

    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator()

    def test_prints_report(self, evaluator, capsys):
        """Should print formatted report."""
        results = {
            "top1_accuracy": 0.10,
            "top3_accuracy": 0.30,
            "log_loss": 2.5,
            "avg_correct_prob": 0.08,
            "total_races": 100,
            "exacta_accuracy": 0.02,
            "win_accuracy": 0.08,
            "place_accuracy": 0.15,
        }

        evaluator.print_evaluation_report(results)

        captured = capsys.readouterr()
        assert "MODEL EVALUATION REPORT" in captured.out
        assert "Top-1 Accuracy" in captured.out
        assert "Top-3 Accuracy" in captured.out
        assert "Exacta Accuracy" in captured.out

    def test_handles_missing_values(self, evaluator, capsys):
        """Should handle missing values gracefully."""
        results = {}  # Empty results

        evaluator.print_evaluation_report(results)

        captured = capsys.readouterr()
        assert "MODEL EVALUATION REPORT" in captured.out


class TestModelEvaluatorEdgeCases:
    """Test edge cases for ModelEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return ModelEvaluator()

    def test_empty_predictions(self, evaluator):
        """Handle empty prediction arrays."""
        y_true = np.array([])
        y_pred_proba = np.zeros((0, 18))

        # Should not raise error
        result = evaluator.calculate_basic_metrics(y_true, y_pred_proba)

        # Metrics should be NaN or 0
        assert result["top1_accuracy"] == 0 or np.isnan(result["top1_accuracy"])

    def test_uniform_predictions(self, evaluator):
        """Handle uniform probability predictions."""
        y_true = np.array([0, 1, 2])
        y_pred_proba = np.ones((3, 18)) / 18  # Uniform distribution

        result = evaluator.calculate_basic_metrics(y_true, y_pred_proba)

        # Uniform predictions have low accuracy
        assert 0 <= result["top1_accuracy"] <= 1
        assert 0 <= result["top3_accuracy"] <= 1
        assert result["avg_correct_prob"] == pytest.approx(1/18, abs=0.01)
