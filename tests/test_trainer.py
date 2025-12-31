"""Tests for the ModelTrainer class."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.models.trainer import ModelTrainer
from src.models.config import DATE_COL, TRAINING_CONFIG


class TestModelTrainerInit:
    """Test ModelTrainer initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        trainer = ModelTrainer()

        assert trainer.n_splits == TRAINING_CONFIG["n_splits"]
        assert trainer.models == []
        assert trainer.cv_results == []

    def test_custom_n_splits(self):
        """Test initialization with custom n_splits."""
        trainer = ModelTrainer(n_splits=3)

        assert trainer.n_splits == 3


class TestModelTrainerPrepareTarget:
    """Test _prepare_target method."""

    @pytest.fixture
    def trainer(self):
        return ModelTrainer()

    def test_convert_positions_to_zero_indexed(self, trainer):
        """Positions should be converted to 0-indexed classes."""
        target = pd.Series([1, 2, 3, 10, 18])
        result = trainer._prepare_target(target)

        expected = pd.Series([0, 1, 2, 9, 17])
        pd.testing.assert_series_equal(result, expected)

    def test_clip_positions_out_of_range(self, trainer):
        """Positions outside 1-18 should be clipped."""
        target = pd.Series([0, 1, 18, 19, 25])
        result = trainer._prepare_target(target)

        # 0 -> 1 (clipped) -> 0; 19 -> 18 (clipped) -> 17
        expected = pd.Series([0, 0, 17, 17, 17])
        pd.testing.assert_series_equal(result, expected)

    def test_handle_string_values(self, trainer):
        """String values should be converted correctly."""
        target = pd.Series(["1", "5", "10"])
        result = trainer._prepare_target(target)

        expected = pd.Series([0, 4, 9])
        pd.testing.assert_series_equal(result, expected)


class TestModelTrainerEvaluateFold:
    """Test _evaluate_fold method."""

    @pytest.fixture
    def trainer(self):
        return ModelTrainer()

    @pytest.fixture
    def mock_model(self):
        """Create mock model with controlled predictions."""
        model = Mock()
        return model

    def test_evaluate_fold_returns_dict(self, trainer, mock_model):
        """_evaluate_fold should return a dictionary with metrics."""
        # Create predictions where first sample predicts class 0 correctly
        probs = np.array([
            [0.8, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        mock_model.predict_proba.return_value = probs

        X = pd.DataFrame({"feature": [1, 2]})
        y = pd.Series([0, 1])

        result = trainer._evaluate_fold(mock_model, X, y)

        assert "top1_accuracy" in result
        assert "top3_accuracy" in result
        assert "log_loss" in result

    def test_evaluate_fold_perfect_predictions(self, trainer, mock_model):
        """Perfect predictions should give 100% accuracy."""
        # Perfect predictions
        probs = np.zeros((3, 18))
        probs[0, 0] = 1.0  # Predicts class 0
        probs[1, 1] = 1.0  # Predicts class 1
        probs[2, 2] = 1.0  # Predicts class 2
        mock_model.predict_proba.return_value = probs

        X = pd.DataFrame({"feature": [1, 2, 3]})
        y = pd.Series([0, 1, 2])

        result = trainer._evaluate_fold(mock_model, X, y)

        assert result["top1_accuracy"] == 1.0
        assert result["top3_accuracy"] == 1.0

    def test_evaluate_fold_wrong_predictions(self, trainer, mock_model):
        """Wrong predictions should give 0% top-1 accuracy."""
        # All wrong predictions
        probs = np.zeros((3, 18))
        probs[0, 5] = 1.0  # Predicts class 5, actual 0
        probs[1, 6] = 1.0  # Predicts class 6, actual 1
        probs[2, 7] = 1.0  # Predicts class 7, actual 2
        mock_model.predict_proba.return_value = probs

        X = pd.DataFrame({"feature": [1, 2, 3]})
        y = pd.Series([0, 1, 2])

        result = trainer._evaluate_fold(mock_model, X, y)

        assert result["top1_accuracy"] == 0.0
        assert result["top3_accuracy"] == 0.0

    def test_evaluate_fold_top3_accuracy(self, trainer, mock_model):
        """Top-3 accuracy includes predictions in top 3 positions."""
        # Correct class is in top 3 but not top 1
        probs = np.zeros((2, 18))
        probs[0, 1] = 0.4  # Rank 1
        probs[0, 2] = 0.3  # Rank 2
        probs[0, 0] = 0.2  # Rank 3 (actual class)
        probs[0, 3] = 0.1  # Rank 4

        probs[1, 1] = 1.0  # Correct prediction
        mock_model.predict_proba.return_value = probs

        X = pd.DataFrame({"feature": [1, 2]})
        y = pd.Series([0, 1])  # First actual is 0, second is 1

        result = trainer._evaluate_fold(mock_model, X, y)

        assert result["top1_accuracy"] == 0.5  # Only second is top-1 correct
        assert result["top3_accuracy"] == 1.0  # Both are in top-3


class TestModelTrainerTrainWithCV:
    """Test train_with_cv method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with dates for CV testing."""
        np.random.seed(42)

        # Create 300 samples over 10 months
        base_date = datetime(2020, 1, 1)
        dates = []
        for i in range(300):
            days_offset = (i // 30) * 30 + (i % 30)  # Spread over ~10 months
            dates.append(base_date + timedelta(days=days_offset))

        df = pd.DataFrame({
            DATE_COL: dates,
            "feature_1": np.random.randn(300),
            "feature_2": np.random.randn(300),
            "着順": np.random.randint(1, 19, 300),
        })

        return df

    def test_train_with_cv_returns_models_and_results(self, sample_data):
        """train_with_cv should return list of models and results."""
        trainer = ModelTrainer(n_splits=2)

        with patch.object(trainer, '_evaluate_fold') as mock_eval:
            mock_eval.return_value = {
                "top1_accuracy": 0.1,
                "top3_accuracy": 0.3,
                "log_loss": 2.5,
            }

            models, results = trainer.train_with_cv(
                sample_data,
                feature_cols=["feature_1", "feature_2"],
            )

        assert len(models) == 2
        assert len(results) == 2

    def test_train_with_cv_results_contain_fold_info(self, sample_data):
        """Each result should contain fold number and sample counts."""
        trainer = ModelTrainer(n_splits=2)

        with patch.object(trainer, '_evaluate_fold') as mock_eval:
            # Return a new dict each time to avoid mutation issues
            mock_eval.side_effect = lambda *args: {
                "top1_accuracy": 0.1,
                "top3_accuracy": 0.3,
                "log_loss": 2.5,
            }

            _, results = trainer.train_with_cv(
                sample_data,
                feature_cols=["feature_1", "feature_2"],
            )

        # Results should have fold info
        for result in results:
            assert "fold" in result
            assert "train_samples" in result
            assert "val_samples" in result
            assert result["train_samples"] > 0
            assert result["val_samples"] > 0

        # Fold numbers should be sequential
        fold_nums = sorted([r["fold"] for r in results])
        assert fold_nums == list(range(1, len(results) + 1))

    def test_train_with_cv_expanding_window(self, sample_data):
        """Training data should expand or stay same with each fold."""
        trainer = ModelTrainer(n_splits=3)

        with patch.object(trainer, '_evaluate_fold') as mock_eval:
            # Return a new dict each time to avoid mutation issues
            mock_eval.side_effect = lambda *args: {
                "top1_accuracy": 0.1,
                "top3_accuracy": 0.3,
                "log_loss": 2.5,
            }

            _, results = trainer.train_with_cv(
                sample_data,
                feature_cols=["feature_1", "feature_2"],
            )

        # Training samples should be non-decreasing (expanding window)
        train_samples = [r["train_samples"] for r in results]
        for i in range(len(train_samples) - 1):
            assert train_samples[i] <= train_samples[i + 1]

        # Total samples across folds should be reasonable
        assert all(s > 0 for s in train_samples)

    def test_train_with_cv_stores_models(self, sample_data):
        """Models should be stored in trainer.models."""
        trainer = ModelTrainer(n_splits=2)

        with patch.object(trainer, '_evaluate_fold') as mock_eval:
            # Return a new dict each time to avoid mutation issues
            mock_eval.side_effect = lambda *args: {
                "top1_accuracy": 0.1,
                "top3_accuracy": 0.3,
                "log_loss": 2.5,
            }

            trainer.train_with_cv(
                sample_data,
                feature_cols=["feature_1", "feature_2"],
            )

        assert len(trainer.models) == 2
        assert len(trainer.cv_results) == 2


class TestModelTrainerTrainFinalModel:
    """Test train_final_model method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data spanning 12 months."""
        np.random.seed(42)

        base_date = datetime(2020, 1, 1)
        dates = []
        for i in range(360):  # ~12 months of data
            days_offset = i
            dates.append(base_date + timedelta(days=days_offset))

        df = pd.DataFrame({
            DATE_COL: dates,
            "feature_1": np.random.randn(360),
            "feature_2": np.random.randn(360),
            "着順": np.random.randint(1, 19, 360),
        })

        return df

    def test_train_final_model_returns_model_and_test_data(self, sample_data):
        """train_final_model should return model and test DataFrame."""
        trainer = ModelTrainer()

        model, test_df = trainer.train_final_model(
            sample_data,
            feature_cols=["feature_1", "feature_2"],
            test_months=3,
        )

        assert model is not None
        assert model.model is not None  # Model should be trained
        assert isinstance(test_df, pd.DataFrame)
        assert len(test_df) > 0

    def test_train_final_model_test_data_is_recent(self, sample_data):
        """Test data should be from the most recent period."""
        trainer = ModelTrainer()

        model, test_df = trainer.train_final_model(
            sample_data,
            feature_cols=["feature_1", "feature_2"],
            test_months=3,
        )

        # Test data should contain the latest dates
        max_date = pd.to_datetime(sample_data[DATE_COL]).max()
        test_max = pd.to_datetime(test_df[DATE_COL]).max()

        assert test_max == max_date

    def test_train_final_model_no_data_leakage(self, sample_data):
        """Training should not include test period data."""
        trainer = ModelTrainer()

        model, test_df = trainer.train_final_model(
            sample_data,
            feature_cols=["feature_1", "feature_2"],
            test_months=3,
        )

        # The model's feature names should be set
        assert model.feature_names == ["feature_1", "feature_2"]

        # Test data should be separate from training
        test_min_date = pd.to_datetime(test_df[DATE_COL]).min()
        max_train_date = pd.to_datetime(sample_data[DATE_COL]).max() - pd.DateOffset(months=3)

        assert test_min_date >= max_train_date


class TestModelTrainerSummarizeCVResults:
    """Test summarize_cv_results method."""

    @pytest.fixture
    def trainer(self):
        return ModelTrainer()

    def test_summarize_empty_results(self, trainer):
        """Should return empty DataFrame when no results."""
        result = trainer.summarize_cv_results()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_summarize_with_results(self, trainer, capsys):
        """Should print summary and return DataFrame."""
        trainer.cv_results = [
            {
                "fold": 1,
                "top1_accuracy": 0.10,
                "top3_accuracy": 0.30,
                "log_loss": 2.5,
                "train_samples": 1000,
                "val_samples": 200,
            },
            {
                "fold": 2,
                "top1_accuracy": 0.12,
                "top3_accuracy": 0.32,
                "log_loss": 2.4,
                "train_samples": 1200,
                "val_samples": 200,
            },
        ]

        result = trainer.summarize_cv_results()

        assert len(result) == 2
        assert "top1_accuracy" in result.columns
        assert "top3_accuracy" in result.columns
        assert "log_loss" in result.columns

        # Check printed output
        captured = capsys.readouterr()
        assert "CROSS-VALIDATION SUMMARY" in captured.out
        assert "Mean Top-1 Accuracy" in captured.out


class TestModelTrainerIntegration:
    """Integration tests for ModelTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data for integration test."""
        np.random.seed(42)

        base_date = datetime(2020, 1, 1)
        dates = []
        for i in range(200):
            days_offset = i * 2  # Every 2 days over ~13 months
            dates.append(base_date + timedelta(days=days_offset))

        df = pd.DataFrame({
            DATE_COL: dates,
            "feature_1": np.random.randn(200),
            "feature_2": np.random.randn(200),
            "着順": np.random.randint(1, 10, 200),  # Fewer classes for faster training
        })

        return df

    def test_full_cv_workflow(self, sample_data):
        """Test complete CV training workflow."""
        trainer = ModelTrainer(n_splits=2)

        models, results = trainer.train_with_cv(
            sample_data,
            feature_cols=["feature_1", "feature_2"],
        )

        # Verify models are trained
        assert len(models) == 2
        for model in models:
            assert model.model is not None

        # Verify results are valid
        assert len(results) == 2
        for result in results:
            assert 0 <= result["top1_accuracy"] <= 1
            assert 0 <= result["top3_accuracy"] <= 1
            assert result["log_loss"] > 0

        # Verify summary works
        summary_df = trainer.summarize_cv_results()
        assert len(summary_df) == 2

    def test_final_model_can_predict(self, sample_data):
        """Final model should be able to make predictions."""
        trainer = ModelTrainer()

        model, test_df = trainer.train_final_model(
            sample_data,
            feature_cols=["feature_1", "feature_2"],
            test_months=2,
        )

        # Make predictions on test data
        X_test = test_df[["feature_1", "feature_2"]]
        probs = model.predict_proba(X_test)

        assert probs.shape[0] == len(test_df)
        assert probs.shape[1] == 18
