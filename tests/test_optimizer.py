"""Tests for the hyperparameter optimizer module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.models.optimizer import ModelOptimizer


class TestModelOptimizerInit:
    """Tests for ModelOptimizer initialization."""

    def test_default_initialization(self):
        """Test default optimizer initialization."""
        optimizer = ModelOptimizer()
        assert optimizer.model_type == "lightgbm"
        assert optimizer.n_trials == 100
        assert optimizer.study_name == "keiba_optimization"
        assert optimizer.storage is None

    def test_custom_initialization(self):
        """Test custom optimizer initialization."""
        optimizer = ModelOptimizer(
            model_type="xgboost",
            n_trials=50,
            study_name="custom_study",
            storage="sqlite:///test.db",
        )
        assert optimizer.model_type == "xgboost"
        assert optimizer.n_trials == 50
        assert optimizer.study_name == "custom_study"
        assert optimizer.storage == "sqlite:///test.db"


class TestModelOptimizerSuggestParams:
    """Tests for parameter suggestion methods."""

    def test_suggest_lgbm_params(self):
        """Test LightGBM parameter suggestion."""
        optimizer = ModelOptimizer()

        # Create a mock trial
        mock_trial = Mock()
        mock_trial.suggest_int = Mock(side_effect=[31, 5, 20, 7])
        mock_trial.suggest_float = Mock(side_effect=[0.05, 0.8, 0.8, 0.01, 0.01])

        params = optimizer._suggest_lgbm_params(mock_trial)

        assert "objective" in params
        assert params["objective"] == "multiclass"
        assert params["num_class"] == 18
        assert "num_leaves" in params
        assert "learning_rate" in params


class TestModelOptimizerEvaluate:
    """Tests for model evaluation."""

    @pytest.fixture
    def mock_data(self):
        """Create mock training data."""
        n_samples = 100
        n_features = 36  # Updated feature count

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 18, n_samples)

        return X, y

    def test_evaluate_model_returns_float(self, mock_data):
        """Test that evaluate_model returns a float."""
        optimizer = ModelOptimizer()

        X, y = mock_data

        # Mock data loader
        optimizer.data_loader = Mock()
        optimizer.data_loader.get_features_and_target = Mock(
            return_value=(X, y)
        )
        optimizer._train_df = pd.DataFrame()
        optimizer._val_df = pd.DataFrame()

        params = {
            "objective": "multiclass",
            "num_class": 18,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
            "n_jobs": 1,
            "seed": 42,
        }

        # This should not raise
        result = optimizer._evaluate_model(params)

        assert isinstance(result, float)
        assert result > 0  # Log loss should be positive


class TestModelOptimizerOptimize:
    """Tests for the optimization process."""

    def test_optimize_runs_without_data(self):
        """Test that optimize raises error without data."""
        optimizer = ModelOptimizer(n_trials=1)

        with patch.object(optimizer, '_load_data', return_value=False):
            with pytest.raises(RuntimeError, match="Failed to load data"):
                optimizer.optimize()

    def test_get_best_params_before_optimize(self):
        """Test that get_best_params raises before optimization."""
        optimizer = ModelOptimizer()

        with pytest.raises(RuntimeError, match="Run optimize"):
            optimizer.get_best_params()


class TestModelOptimizerIntegration:
    """Integration tests for ModelOptimizer."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        n_samples = 200
        np.random.seed(42)

        df = pd.DataFrame({
            'レースID': [f'race_{i // 10}' for i in range(n_samples)],
            'レース日付': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            '着順': np.random.randint(1, 19, n_samples),
            'horse_age_num': np.random.randint(2, 8, n_samples),
            'horse_sex_encoded': np.random.randint(1, 4, n_samples),
            'post_position_num': np.random.randint(1, 19, n_samples),
            '斤量': np.random.uniform(50, 60, n_samples),
            '馬体重': np.random.uniform(400, 550, n_samples),
            'jockey_win_rate': np.random.uniform(0, 0.3, n_samples),
            'jockey_place_rate': np.random.uniform(0, 0.5, n_samples),
            'trainer_win_rate': np.random.uniform(0, 0.3, n_samples),
            'jockey_races': np.random.randint(0, 1000, n_samples),
            'trainer_races': np.random.randint(0, 500, n_samples),
            'distance_num': np.random.choice([1200, 1600, 2000, 2400], n_samples),
            'is_turf': np.random.randint(0, 2, n_samples),
            'is_dirt': np.random.randint(0, 2, n_samples),
            'track_condition_num': np.random.randint(0, 4, n_samples),
            'avg_position_last_3': np.random.uniform(1, 18, n_samples),
            'avg_position_last_5': np.random.uniform(1, 18, n_samples),
            'win_rate_last_3': np.random.uniform(0, 1, n_samples),
            'win_rate_last_5': np.random.uniform(0, 1, n_samples),
            'place_rate_last_3': np.random.uniform(0, 1, n_samples),
            'place_rate_last_5': np.random.uniform(0, 1, n_samples),
            'last_position': np.random.uniform(1, 18, n_samples),
            'career_races': np.random.randint(0, 50, n_samples),
            'odds_log': np.random.uniform(0, 5, n_samples),
            # Running style features
            'early_position': np.random.uniform(1, 18, n_samples),
            'late_position': np.random.uniform(1, 18, n_samples),
            'position_change': np.random.uniform(-10, 10, n_samples),
            # Aptitude features
            'aptitude_sprint': np.random.uniform(0, 1, n_samples),
            'aptitude_mile': np.random.uniform(0, 1, n_samples),
            'aptitude_intermediate': np.random.uniform(0, 1, n_samples),
            'aptitude_long': np.random.uniform(0, 1, n_samples),
            'aptitude_turf': np.random.uniform(0, 1, n_samples),
            'aptitude_dirt': np.random.uniform(0, 1, n_samples),
            'aptitude_course': np.random.uniform(0, 1, n_samples),
            # Pace features
            'last_3f_avg': np.random.uniform(33, 38, n_samples),
            'last_3f_best': np.random.uniform(32, 36, n_samples),
            'last_3f_last': np.random.uniform(33, 38, n_samples),
            # Race classification features
            'weight_change_kg': np.random.uniform(-10, 10, n_samples),
            'is_graded_race': np.random.randint(0, 2, n_samples),
            'grade_level': np.random.randint(1, 7, n_samples),
        })

        return df

    def test_short_optimization(self, sample_df):
        """Test a short optimization run."""
        optimizer = ModelOptimizer(n_trials=2)

        # Mock data loading
        with patch.object(optimizer, '_load_data') as mock_load:
            mock_load.return_value = True

            # Split data
            n = len(sample_df)
            optimizer._train_df = sample_df.iloc[:int(n*0.6)]
            optimizer._val_df = sample_df.iloc[int(n*0.6):int(n*0.8)]
            optimizer._test_df = sample_df.iloc[int(n*0.8):]

            # Mock data loader
            optimizer.data_loader = Mock()

            from src.models.config import FEATURES
            optimizer.data_loader.get_features_and_target = Mock(
                side_effect=lambda df, features: (
                    df[features].values,
                    df['着順'].values - 1  # Convert to 0-indexed
                )
            )

            # Run optimization
            best_params = optimizer.optimize(n_trials=2)

            assert isinstance(best_params, dict)
            assert optimizer.study is not None
            assert len(optimizer.study.trials) == 2
