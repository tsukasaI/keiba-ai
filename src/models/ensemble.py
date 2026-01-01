"""
Keiba AI Prediction System - Ensemble Model

Combines multiple models for improved prediction accuracy.
Supports weighted averaging and stacking strategies.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
import pickle

from src.models.position_model import PositionProbabilityModel
from src.models.xgboost_model import XGBoostPositionModel
from src.models.catboost_model import CatBoostPositionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Type alias for base models
BaseModel = Union[PositionProbabilityModel, XGBoostPositionModel, CatBoostPositionModel]


class EnsembleModel:
    """Ensemble of multiple models for position prediction.

    Supports multiple strategies:
    - simple_average: Equal weight for all models
    - weighted_average: Weights based on validation performance
    """

    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        weights: Optional[List[float]] = None,
        strategy: str = "weighted_average",
    ):
        """Initialize ensemble.

        Args:
            models: List of trained base models
            weights: Weights for each model (normalized automatically)
            strategy: Ensemble strategy ("simple_average" or "weighted_average")
        """
        self.models = models or []
        self.weights = weights
        self.strategy = strategy
        self.feature_names: Optional[List[str]] = None
        self.n_classes = 18

        if self.weights is not None:
            self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        if self.weights is not None:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble.

        Args:
            model: Trained model to add
            weight: Weight for this model
        """
        self.models.append(model)

        if self.weights is None:
            self.weights = [1.0]
        else:
            self.weights.append(weight)

        self._normalize_weights()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        model_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Train all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            model_configs: Configuration for each model type
        """
        self.feature_names = feature_names

        if model_configs is None:
            # Default: use all three model types
            model_configs = [
                {"type": "lightgbm", "params": {}},
                {"type": "xgboost", "params": {}},
                {"type": "catboost", "params": {}},
            ]

        self.models = []
        val_scores = []

        for config in model_configs:
            model_type = config.get("type", "lightgbm")
            params = config.get("params", {})

            logger.info(f"Training {model_type} model...")

            if model_type == "lightgbm":
                model = PositionProbabilityModel(params=params)
            elif model_type == "xgboost":
                model = XGBoostPositionModel(params=params)
            elif model_type == "catboost":
                model = CatBoostPositionModel(params=params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.train(
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names,
            )

            self.models.append(model)

            # Calculate validation score for weighting
            if X_val is not None and y_val is not None:
                proba = model.predict_proba(X_val)
                eps = 1e-15
                proba = np.clip(proba, eps, 1 - eps)
                log_loss = -np.mean(
                    np.log(proba[np.arange(len(y_val)), y_val.astype(int)])
                )
                val_scores.append(1.0 / (log_loss + eps))  # Higher score = better
                logger.info(f"  {model_type} validation log loss: {log_loss:.4f}")

        # Set weights based on validation performance
        if self.strategy == "weighted_average" and val_scores:
            self.weights = val_scores
            self._normalize_weights()
            logger.info(f"Model weights: {self.weights}")
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict position probabilities.

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Weighted average probability array (n_samples, 18)
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)
            predictions.append(pred)

        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # (n_models, n_samples, 18)

        # Weighted average
        weights = np.array(self.weights).reshape(-1, 1, 1)
        ensemble_pred = np.sum(predictions * weights, axis=0)

        return ensemble_pred

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
        """Sinkhorn-Knopp normalization for doubly stochastic matrix."""
        proba = proba.copy()
        eps = 1e-10

        for _ in range(n_iters):
            row_sums = proba.sum(axis=1, keepdims=True)
            proba = proba / (row_sums + eps)
            col_sums = proba.sum(axis=0, keepdims=True)
            proba = proba / (col_sums + eps)

        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / (row_sums + eps)

        return proba

    def feature_importance(self) -> pd.DataFrame:
        """Get averaged feature importance across all models.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        all_importance = {}

        for i, model in enumerate(self.models):
            try:
                importance_df = model.feature_importance()
                weight = self.weights[i] if self.weights else 1.0 / len(self.models)

                for _, row in importance_df.iterrows():
                    feature = row["feature"]
                    imp = row["importance"] * weight
                    all_importance[feature] = all_importance.get(feature, 0) + imp
            except Exception:
                continue

        df = pd.DataFrame({
            "feature": list(all_importance.keys()),
            "importance": list(all_importance.values()),
        })

        return df.sort_values("importance", ascending=False)

    def save(self, path: Path) -> None:
        """Save ensemble to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "models": self.models,
            "weights": self.weights,
            "strategy": self.strategy,
            "feature_names": self.feature_names,
            "n_classes": self.n_classes,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        """Load ensemble from file.

        Args:
            path: Model file path

        Returns:
            Loaded ensemble instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            models=data["models"],
            weights=data["weights"],
            strategy=data["strategy"],
        )
        instance.feature_names = data["feature_names"]
        instance.n_classes = data.get("n_classes", 18)

        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about ensemble composition.

        Returns:
            Dictionary with ensemble info
        """
        return {
            "n_models": len(self.models),
            "model_types": [type(m).__name__ for m in self.models],
            "weights": self.weights,
            "strategy": self.strategy,
        }
