"""Tests for XGBoost, CatBoost, and Ensemble models."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.models.xgboost_model import XGBoostPositionModel
from src.models.catboost_model import CatBoostPositionModel
from src.models.ensemble import EnsembleModel


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 36

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 18, n_samples)

    # Split into train/val
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    return X_train, y_train, X_val, y_val


class TestXGBoostPositionModel:
    """Tests for XGBoostPositionModel."""

    def test_train_and_predict(self, sample_data):
        """Test basic training and prediction."""
        X_train, y_train, X_val, y_val = sample_data

        model = XGBoostPositionModel()
        model.train(X_train, y_train, X_val, y_val, num_boost_round=10)

        proba = model.predict_proba(X_val)

        assert proba.shape == (len(X_val), 18)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_race(self, sample_data):
        """Test race-level prediction."""
        X_train, y_train, X_val, y_val = sample_data

        model = XGBoostPositionModel()
        model.train(X_train, y_train, num_boost_round=10)

        # Simulate a race with 8 horses
        X_race = X_val[:8]
        result = model.predict_race(X_race, normalize=True)

        assert len(result) == 8
        for name, proba in result.items():
            assert len(proba) == 18

    def test_save_load(self, sample_data):
        """Test model persistence."""
        X_train, y_train, X_val, y_val = sample_data

        model = XGBoostPositionModel()
        model.train(X_train, y_train, num_boost_round=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xgb_model.pkl"
            model.save(path)

            loaded = XGBoostPositionModel.load(path)

            # Predictions should be the same
            orig_pred = model.predict_proba(X_val)
            loaded_pred = loaded.predict_proba(X_val)

            np.testing.assert_array_almost_equal(orig_pred, loaded_pred)

    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X_train, y_train, _, _ = sample_data
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        model = XGBoostPositionModel()
        model.train(X_train, y_train, feature_names=feature_names, num_boost_round=10)

        importance = model.feature_importance()

        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) > 0


class TestCatBoostPositionModel:
    """Tests for CatBoostPositionModel."""

    def test_train_and_predict(self, sample_data):
        """Test basic training and prediction."""
        X_train, y_train, X_val, y_val = sample_data

        model = CatBoostPositionModel()
        model.train(X_train, y_train, X_val, y_val, num_boost_round=10)

        proba = model.predict_proba(X_val)

        assert proba.shape == (len(X_val), 18)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_race(self, sample_data):
        """Test race-level prediction."""
        X_train, y_train, X_val, y_val = sample_data

        model = CatBoostPositionModel()
        model.train(X_train, y_train, num_boost_round=10)

        # Simulate a race with 8 horses
        X_race = X_val[:8]
        result = model.predict_race(X_race, normalize=True)

        assert len(result) == 8
        for name, proba in result.items():
            assert len(proba) == 18

    def test_save_load(self, sample_data):
        """Test model persistence."""
        X_train, y_train, X_val, y_val = sample_data

        model = CatBoostPositionModel()
        model.train(X_train, y_train, num_boost_round=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catboost_model.pkl"
            model.save(path)

            loaded = CatBoostPositionModel.load(path)

            # Predictions should be the same
            orig_pred = model.predict_proba(X_val)
            loaded_pred = loaded.predict_proba(X_val)

            np.testing.assert_array_almost_equal(orig_pred, loaded_pred)


class TestEnsembleModel:
    """Tests for EnsembleModel."""

    def test_add_model(self, sample_data):
        """Test adding models to ensemble."""
        X_train, y_train, _, _ = sample_data

        ensemble = EnsembleModel()

        # Train and add XGBoost
        xgb_model = XGBoostPositionModel()
        xgb_model.train(X_train, y_train, num_boost_round=5)
        ensemble.add_model(xgb_model, weight=1.0)

        # Train and add CatBoost
        cb_model = CatBoostPositionModel()
        cb_model.train(X_train, y_train, num_boost_round=5)
        ensemble.add_model(cb_model, weight=1.5)

        assert len(ensemble.models) == 2
        assert len(ensemble.weights) == 2
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6

    def test_predict_proba(self, sample_data):
        """Test ensemble prediction."""
        X_train, y_train, X_val, y_val = sample_data

        ensemble = EnsembleModel()

        # Add XGBoost
        xgb_model = XGBoostPositionModel()
        xgb_model.train(X_train, y_train, num_boost_round=5)
        ensemble.add_model(xgb_model)

        # Add CatBoost
        cb_model = CatBoostPositionModel()
        cb_model.train(X_train, y_train, num_boost_round=5)
        ensemble.add_model(cb_model)

        proba = ensemble.predict_proba(X_val)

        assert proba.shape == (len(X_val), 18)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_train_all_models(self, sample_data):
        """Test training all models together."""
        X_train, y_train, X_val, y_val = sample_data

        ensemble = EnsembleModel()
        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            model_configs=[
                {"type": "xgboost", "params": {"max_depth": 3}},
                {"type": "catboost", "params": {"depth": 3}},
            ],
        )

        assert len(ensemble.models) == 2
        assert len(ensemble.weights) == 2

        proba = ensemble.predict_proba(X_val)
        assert proba.shape == (len(X_val), 18)

    def test_save_load(self, sample_data):
        """Test ensemble persistence."""
        X_train, y_train, X_val, y_val = sample_data

        ensemble = EnsembleModel()

        xgb_model = XGBoostPositionModel()
        xgb_model.train(X_train, y_train, num_boost_round=5)
        ensemble.add_model(xgb_model)

        cb_model = CatBoostPositionModel()
        cb_model.train(X_train, y_train, num_boost_round=5)
        ensemble.add_model(cb_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ensemble.pkl"
            ensemble.save(path)

            loaded = EnsembleModel.load(path)

            assert len(loaded.models) == 2
            assert loaded.weights == ensemble.weights

            # Predictions should be the same
            orig_pred = ensemble.predict_proba(X_val)
            loaded_pred = loaded.predict_proba(X_val)

            np.testing.assert_array_almost_equal(orig_pred, loaded_pred)

    def test_get_model_info(self, sample_data):
        """Test model info extraction."""
        X_train, y_train, _, _ = sample_data

        ensemble = EnsembleModel()

        xgb_model = XGBoostPositionModel()
        xgb_model.train(X_train, y_train, num_boost_round=5)
        ensemble.add_model(xgb_model)

        info = ensemble.get_model_info()

        assert info["n_models"] == 1
        assert "XGBoostPositionModel" in info["model_types"]
        assert info["strategy"] == "weighted_average"


class TestStackingEnsemble:
    """Tests for EnsembleModel with stacking strategy."""

    def test_stacking_train(self, sample_data):
        """Test stacking ensemble training."""
        X_train, y_train, X_val, y_val = sample_data

        ensemble = EnsembleModel(strategy="stacking", n_folds=3)
        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            model_configs=[
                {"type": "xgboost", "params": {"max_depth": 3}},
                {"type": "catboost", "params": {"depth": 3}},
            ],
        )

        assert len(ensemble.models) == 2
        assert ensemble.meta_learners is not None
        assert len(ensemble.meta_learners) == 18  # One per class

    def test_stacking_predict(self, sample_data):
        """Test stacking ensemble prediction."""
        X_train, y_train, X_val, y_val = sample_data

        ensemble = EnsembleModel(strategy="stacking", n_folds=3)
        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            model_configs=[
                {"type": "xgboost", "params": {"max_depth": 3}},
                {"type": "catboost", "params": {"depth": 3}},
            ],
        )

        proba = ensemble.predict_proba(X_val)

        assert proba.shape == (len(X_val), 18)
        # Probabilities should be non-negative
        assert np.all(proba >= 0)
        # Rows should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_stacking_save_load(self, sample_data):
        """Test stacking ensemble persistence."""
        X_train, y_train, X_val, y_val = sample_data

        ensemble = EnsembleModel(strategy="stacking", n_folds=3)
        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            model_configs=[
                {"type": "xgboost", "params": {"max_depth": 3}},
                {"type": "catboost", "params": {"depth": 3}},
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stacking_ensemble.pkl"
            ensemble.save(path)

            loaded = EnsembleModel.load(path)

            assert loaded.strategy == "stacking"
            assert loaded.meta_learners is not None
            assert len(loaded.meta_learners) == 18

            # Predictions should be the same
            orig_pred = ensemble.predict_proba(X_val)
            loaded_pred = loaded.predict_proba(X_val)

            np.testing.assert_array_almost_equal(orig_pred, loaded_pred)

    def test_stacking_model_info(self, sample_data):
        """Test stacking model info."""
        X_train, y_train, X_val, y_val = sample_data

        ensemble = EnsembleModel(strategy="stacking", n_folds=3)
        ensemble.train(
            X_train, y_train,
            X_val, y_val,
            model_configs=[
                {"type": "xgboost", "params": {"max_depth": 3}},
            ],
        )

        info = ensemble.get_model_info()

        assert info["strategy"] == "stacking"
        assert info["n_folds"] == 3
        assert info["has_meta_learners"] is True
        assert "meta_learner_alphas" in info
        assert len(info["meta_learner_alphas"]) == 18

    def test_stacking_vs_averaging(self, sample_data):
        """Compare stacking with weighted averaging predictions."""
        X_train, y_train, X_val, y_val = sample_data

        model_configs = [
            {"type": "xgboost", "params": {"max_depth": 3}},
            {"type": "catboost", "params": {"depth": 3}},
        ]

        # Train stacking ensemble
        stacking = EnsembleModel(strategy="stacking", n_folds=3)
        stacking.train(X_train, y_train, X_val, y_val, model_configs=model_configs)

        # Train weighted average ensemble
        averaging = EnsembleModel(strategy="weighted_average")
        averaging.train(X_train, y_train, X_val, y_val, model_configs=model_configs)

        # Both should produce valid probabilities
        stacking_pred = stacking.predict_proba(X_val)
        averaging_pred = averaging.predict_proba(X_val)

        assert stacking_pred.shape == averaging_pred.shape
        assert np.allclose(stacking_pred.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(averaging_pred.sum(axis=1), 1.0, atol=1e-5)

        # Predictions should be different (stacking learns weights)
        # This is a weak check; they could theoretically be the same
        # but in practice stacking should produce different results
        assert not np.allclose(stacking_pred, averaging_pred)

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            EnsembleModel(strategy="invalid_strategy")
