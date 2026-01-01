"""
Keiba AI Prediction System - XGBoost Model

XGBoost implementation for horse racing position prediction.
Same interface as PositionProbabilityModel for easy swapping.
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 18,
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.01,
    "reg_lambda": 0.01,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}


class XGBoostPositionModel:
    """XGBoost model for position probability prediction.

    Predicts probability distribution over finishing positions (1-18).
    Uses multi-class classification with softmax output.
    """

    def __init__(self, params: Optional[dict] = None):
        """Initialize the model.

        Args:
            params: XGBoost parameters (merged with defaults)
        """
        self.params = XGB_PARAMS.copy()
        if params:
            self.params.update(params)

        self.model: Optional[xgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.n_classes = self.params.get("num_class", 18)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels (0-indexed positions)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names
            num_boost_round: Maximum boosting rounds
            early_stopping_rounds: Early stopping patience
        """
        self.feature_names = feature_names

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            evals.append((dval, "val"))

        # Train
        callbacks = []
        if early_stopping_rounds > 0 and X_val is not None:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    metric_name="mlogloss",
                    data_name="val",
                )
            )

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False,
        )

        # best_iteration is only available with early stopping
        try:
            best_iter = self.model.best_iteration
        except AttributeError:
            best_iter = num_boost_round

        logger.info(f"XGBoost training complete, best iteration: {best_iter}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict position probabilities.

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Probability array (n_samples, 18)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        proba = self.model.predict(dmatrix)

        # Ensure shape is (n_samples, 18)
        if len(proba.shape) == 1:
            proba = proba.reshape(-1, self.n_classes)

        return proba

    def predict_race(
        self,
        X: np.ndarray,
        normalize: bool = True,
        horse_names: Optional[List[str]] = None,
    ) -> dict:
        """Predict probabilities for a single race.

        Args:
            X: Feature array for horses in race (n_horses, n_features)
            normalize: Apply Sinkhorn normalization
            horse_names: Horse names for output keys

        Returns:
            Dictionary mapping horse names to position probabilities
        """
        n_horses = len(X)

        if horse_names is None:
            horse_names = [f"Horse_{i+1}" for i in range(n_horses)]

        proba = self.predict_proba(X)

        # Only use positions up to n_horses
        proba = proba[:, :n_horses]

        if normalize:
            proba = self._sinkhorn_normalize(proba)

        # Pad to 18 positions if needed
        if proba.shape[1] < 18:
            padding = np.zeros((n_horses, 18 - proba.shape[1]))
            proba = np.hstack([proba, padding])

        return {
            name: proba[i]
            for i, name in enumerate(horse_names)
        }

    def _sinkhorn_normalize(
        self,
        proba: np.ndarray,
        n_iters: int = 10,
    ) -> np.ndarray:
        """Sinkhorn-Knopp normalization for doubly stochastic matrix.

        Args:
            proba: Raw probability matrix (n_horses, n_positions)
            n_iters: Number of iterations

        Returns:
            Normalized probability matrix
        """
        proba = proba.copy()
        eps = 1e-10

        for _ in range(n_iters):
            # Row normalization (each horse sums to 1)
            row_sums = proba.sum(axis=1, keepdims=True)
            proba = proba / (row_sums + eps)

            # Column normalization (each position sums to 1)
            col_sums = proba.sum(axis=0, keepdims=True)
            proba = proba / (col_sums + eps)

        # Final row normalization
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / (row_sums + eps)

        return proba

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.get_score(importance_type="gain")

        df = pd.DataFrame({
            "feature": list(importance.keys()),
            "importance": list(importance.values()),
        })

        return df.sort_values("importance", ascending=False)

    def save(self, path: Path) -> None:
        """Save model to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "n_classes": self.n_classes,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "XGBoostPositionModel":
        """Load model from file.

        Args:
            path: Model file path

        Returns:
            Loaded model instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.n_classes = data.get("n_classes", 18)

        return instance
