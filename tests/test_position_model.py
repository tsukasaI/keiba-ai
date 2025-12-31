"""Tests for the PositionProbabilityModel class."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.models.position_model import PositionProbabilityModel
from src.models.config import LGBM_PARAMS


class TestPositionProbabilityModelInit:
    """Test PositionProbabilityModel initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = PositionProbabilityModel()

        assert model.params is not None
        assert model.model is None
        assert model.feature_names is None
        assert model.num_classes == 18

    def test_custom_params_initialization(self):
        """Test initialization with custom parameters."""
        custom_params = {
            "objective": "multiclass",
            "num_class": 10,
            "learning_rate": 0.1,
        }

        model = PositionProbabilityModel(params=custom_params)

        assert model.params == custom_params
        assert model.num_classes == 10

    def test_default_params_not_mutated(self):
        """Ensure default params are not mutated."""
        original_params = LGBM_PARAMS.copy()
        model = PositionProbabilityModel()
        model.params["learning_rate"] = 999

        assert LGBM_PARAMS["learning_rate"] == original_params["learning_rate"]


class TestPositionProbabilityModelTrain:
    """Test train method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        })

        # Labels are 0-indexed positions (0-17)
        y = pd.Series(np.random.randint(0, 18, n_samples))

        return X, y

    @pytest.fixture
    def fast_params(self):
        """Fast training parameters for tests."""
        return {
            "objective": "multiclass",
            "num_class": 18,
            "num_leaves": 4,
            "learning_rate": 0.3,
            "verbose": -1,
            "n_jobs": 1,
        }

    def test_train_basic(self, sample_data, fast_params):
        """Test basic training."""
        X, y = sample_data

        model = PositionProbabilityModel(params=fast_params)
        result = model.train(X, y)

        assert result is model  # Returns self
        assert model.model is not None
        assert model.feature_names == list(X.columns)

    def test_train_with_validation(self, sample_data, fast_params):
        """Test training with validation data."""
        X, y = sample_data

        # Split into train/val
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        model = PositionProbabilityModel(params=fast_params)
        model.train(X_train, y_train, X_val, y_val)

        assert model.model is not None
        assert model.model.best_iteration > 0

    def test_train_stores_feature_names(self, sample_data, fast_params):
        """Test that feature names are stored after training."""
        X, y = sample_data

        model = PositionProbabilityModel(params=fast_params)
        model.train(X, y)

        assert model.feature_names == ["feature_1", "feature_2", "feature_3"]


class TestPositionProbabilityModelPredict:
    """Test prediction methods."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for prediction tests."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 18, n_samples))

        params = {
            "objective": "multiclass",
            "num_class": 18,
            "num_leaves": 4,
            "learning_rate": 0.3,
            "verbose": -1,
            "n_jobs": 1,
        }

        model = PositionProbabilityModel(params=params)
        model.train(X, y)

        return model

    def test_predict_proba_not_trained_raises(self):
        """predict_proba should raise error if model not trained."""
        model = PositionProbabilityModel()
        X = pd.DataFrame({"feature_1": [1, 2, 3]})

        with pytest.raises(ValueError, match="Model not trained"):
            model.predict_proba(X)

    def test_predict_proba_shape(self, trained_model):
        """predict_proba should return correct shape."""
        X = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
            "feature_3": np.random.randn(5),
        })

        probs = trained_model.predict_proba(X)

        assert probs.shape == (5, 18)

    def test_predict_proba_sums_to_one(self, trained_model):
        """Each row of predict_proba should sum to ~1."""
        X = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
            "feature_3": np.random.randn(5),
        })

        probs = trained_model.predict_proba(X)

        row_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5), decimal=5)

    def test_predict_proba_values_in_range(self, trained_model):
        """All probability values should be in [0, 1]."""
        X = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
            "feature_3": np.random.randn(5),
        })

        probs = trained_model.predict_proba(X)

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


class TestPositionProbabilityModelPredictRace:
    """Test predict_race method."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for prediction tests."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 18, n_samples))

        params = {
            "objective": "multiclass",
            "num_class": 18,
            "num_leaves": 4,
            "learning_rate": 0.3,
            "verbose": -1,
            "n_jobs": 1,
        }

        model = PositionProbabilityModel(params=params)
        model.train(X, y)

        return model

    def test_predict_race_returns_dict(self, trained_model):
        """predict_race should return a dictionary."""
        X_race = pd.DataFrame({
            "feature_1": np.random.randn(6),
            "feature_2": np.random.randn(6),
            "feature_3": np.random.randn(6),
        })

        result = trained_model.predict_race(X_race)

        assert isinstance(result, dict)
        assert len(result) == 6

    def test_predict_race_with_horse_names(self, trained_model):
        """predict_race should use provided horse names as keys."""
        X_race = pd.DataFrame({
            "feature_1": np.random.randn(3),
            "feature_2": np.random.randn(3),
            "feature_3": np.random.randn(3),
        })
        horse_names = ["アーモンドアイ", "コントレイル", "デアリングタクト"]

        result = trained_model.predict_race(X_race, horse_names=horse_names)

        assert set(result.keys()) == set(horse_names)

    def test_predict_race_default_horse_names(self, trained_model):
        """predict_race should generate default names if not provided."""
        X_race = pd.DataFrame({
            "feature_1": np.random.randn(4),
            "feature_2": np.random.randn(4),
            "feature_3": np.random.randn(4),
        })

        result = trained_model.predict_race(X_race)

        assert set(result.keys()) == {"horse_0", "horse_1", "horse_2", "horse_3"}

    def test_predict_race_array_shape(self, trained_model):
        """Each horse's probability array should have 18 elements."""
        X_race = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
            "feature_3": np.random.randn(5),
        })

        result = trained_model.predict_race(X_race)

        for probs in result.values():
            assert len(probs) == 18

    def test_predict_race_normalized(self, trained_model):
        """With normalize=True, position probs should sum to 1."""
        X_race = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
            "feature_3": np.random.randn(5),
        })

        result = trained_model.predict_race(X_race, normalize=True)

        # Stack all probabilities
        all_probs = np.array(list(result.values()))

        # For the first 5 positions (num_horses), column sums should be ~1
        col_sums = all_probs[:, :5].sum(axis=0)
        np.testing.assert_array_almost_equal(col_sums, np.ones(5), decimal=2)

    def test_predict_race_not_normalized(self, trained_model):
        """With normalize=False, raw probabilities are returned."""
        X_race = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
            "feature_3": np.random.randn(5),
        })

        result = trained_model.predict_race(X_race, normalize=False)

        # All 18 positions should have values
        for probs in result.values():
            # Row should sum to 1 (softmax output)
            assert abs(sum(probs) - 1.0) < 0.01

    def test_predict_race_extra_positions_zero(self, trained_model):
        """Positions beyond num_horses should be zero when normalized."""
        X_race = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
            "feature_3": np.random.randn(5),
        })

        result = trained_model.predict_race(X_race, normalize=True)

        # Positions 6-18 (indices 5-17) should be zero
        for probs in result.values():
            assert np.allclose(probs[5:], 0)


class TestPositionProbabilityModelFeatureImportance:
    """Test get_feature_importance method."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "feature_a": np.random.randn(n_samples),
            "feature_b": np.random.randn(n_samples),
            "feature_c": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 18, n_samples))

        params = {
            "objective": "multiclass",
            "num_class": 18,
            "num_leaves": 4,
            "learning_rate": 0.3,
            "verbose": -1,
            "n_jobs": 1,
        }

        model = PositionProbabilityModel(params=params)
        model.train(X, y)

        return model

    def test_feature_importance_not_trained_raises(self):
        """get_feature_importance should raise error if model not trained."""
        model = PositionProbabilityModel()

        with pytest.raises(ValueError, match="Model not trained"):
            model.get_feature_importance()

    def test_feature_importance_returns_dataframe(self, trained_model):
        """get_feature_importance should return a DataFrame."""
        result = trained_model.get_feature_importance()

        assert isinstance(result, pd.DataFrame)

    def test_feature_importance_columns(self, trained_model):
        """DataFrame should have 'feature' and 'importance' columns."""
        result = trained_model.get_feature_importance()

        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_feature_importance_contains_all_features(self, trained_model):
        """All features should be in the importance DataFrame."""
        result = trained_model.get_feature_importance()

        assert set(result["feature"]) == {"feature_a", "feature_b", "feature_c"}

    def test_feature_importance_sorted_descending(self, trained_model):
        """Features should be sorted by importance descending."""
        result = trained_model.get_feature_importance()

        importance_values = result["importance"].values
        assert all(importance_values[i] >= importance_values[i+1]
                   for i in range(len(importance_values)-1))


class TestPositionProbabilityModelSaveLoad:
    """Test save and load methods."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 18, n_samples))

        params = {
            "objective": "multiclass",
            "num_class": 18,
            "num_leaves": 4,
            "learning_rate": 0.3,
            "verbose": -1,
            "n_jobs": 1,
        }

        model = PositionProbabilityModel(params=params)
        model.train(X, y)

        return model

    def test_save_creates_file(self, trained_model):
        """save should create a file at the specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            trained_model.save(path)

            assert path.exists()

    def test_save_creates_parent_directories(self, trained_model):
        """save should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "model.pkl"

            trained_model.save(path)

            assert path.exists()

    def test_load_restores_model(self, trained_model):
        """load should restore a functional model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            trained_model.save(path)

            loaded_model = PositionProbabilityModel.load(path)

            assert loaded_model.model is not None
            assert loaded_model.feature_names == trained_model.feature_names

    def test_load_preserves_predictions(self, trained_model):
        """Loaded model should produce same predictions as original."""
        X_test = pd.DataFrame({
            "feature_1": np.random.randn(5),
            "feature_2": np.random.randn(5),
        })

        original_preds = trained_model.predict_proba(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            trained_model.save(path)

            loaded_model = PositionProbabilityModel.load(path)
            loaded_preds = loaded_model.predict_proba(X_test)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_load_preserves_params(self, trained_model):
        """Loaded model should have same params as original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            trained_model.save(path)

            loaded_model = PositionProbabilityModel.load(path)

        assert loaded_model.params == trained_model.params


class TestPositionProbabilityModelEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def fast_params(self):
        """Fast training parameters for tests."""
        return {
            "objective": "multiclass",
            "num_class": 18,
            "num_leaves": 4,
            "learning_rate": 0.3,
            "verbose": -1,
            "n_jobs": 1,
        }

    def test_single_horse_race(self, fast_params):
        """Handle race with single horse."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 18, n_samples))

        model = PositionProbabilityModel(params=fast_params)
        model.train(X, y)

        # Single horse race
        X_race = pd.DataFrame({"feature_1": [0.5]})
        result = model.predict_race(X_race, normalize=True)

        assert len(result) == 1
        # Single horse should have 100% probability for position 1
        probs = list(result.values())[0]
        assert probs[0] == pytest.approx(1.0, abs=0.01)

    def test_maximum_horses(self, fast_params):
        """Handle race with maximum 18 horses."""
        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randint(0, 18, n_samples))

        model = PositionProbabilityModel(params=fast_params)
        model.train(X, y)

        # Full field of 18 horses
        X_race = pd.DataFrame({"feature_1": np.random.randn(18)})
        result = model.predict_race(X_race, normalize=True)

        assert len(result) == 18

        # All 18 positions should have non-zero probability
        all_probs = np.array(list(result.values()))
        assert np.all(all_probs > 0)
