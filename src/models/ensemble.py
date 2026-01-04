"""
Keiba AI Prediction System - Ensemble Model

Combines multiple models for improved prediction accuracy.
Supports weighted averaging and stacking strategies.

Strategies:
- simple_average: Equal weight for all models
- weighted_average: Weights based on validation log loss
- stacking: Train meta-learner on cross-validated base predictions
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

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
    - stacking: Train meta-learner on cross-validated base model predictions
    """

    VALID_STRATEGIES = ["simple_average", "weighted_average", "stacking"]

    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        weights: Optional[List[float]] = None,
        strategy: str = "weighted_average",
        n_folds: int = 5,
    ):
        """Initialize ensemble.

        Args:
            models: List of trained base models
            weights: Weights for each model (normalized automatically)
            strategy: Ensemble strategy ("simple_average", "weighted_average", or "stacking")
            n_folds: Number of CV folds for stacking (default: 5)
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy: {strategy}. Valid: {self.VALID_STRATEGIES}")

        self.models = models or []
        self.weights = weights
        self.strategy = strategy
        self.n_folds = n_folds
        self.feature_names: Optional[List[str]] = None
        self.n_classes = 18

        # Stacking-specific attributes
        self.meta_learners: Optional[List[RidgeCV]] = None  # One per position class

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

        if self.strategy == "stacking":
            self._train_stacking(X_train, y_train, X_val, y_val, feature_names, model_configs)
        else:
            self._train_averaging(X_train, y_train, X_val, y_val, feature_names, model_configs)

    def _train_averaging(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        model_configs: List[Dict[str, Any]],
    ) -> None:
        """Train models for simple/weighted averaging strategies."""
        self.models = []
        val_scores = []

        for config in model_configs:
            model_type = config.get("type", "lightgbm")
            params = config.get("params", {})

            logger.info(f"Training {model_type} model...")

            model = self._create_base_model(model_type, params)
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

    def _train_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        model_configs: List[Dict[str, Any]],
    ) -> None:
        """Train models using stacking with cross-validation.

        Stacking process:
        1. Generate out-of-fold predictions using K-fold CV for each base model
        2. Stack predictions to create meta-features
        3. Train a meta-learner (RidgeCV) on stacked predictions
        4. Retrain base models on full training data
        """
        n_samples = len(X_train)
        n_models = len(model_configs)

        logger.info(f"Training stacking ensemble with {self.n_folds}-fold CV")
        logger.info(f"Base models: {[c.get('type', 'lightgbm') for c in model_configs]}")

        # Initialize out-of-fold predictions: (n_samples, n_models, n_classes)
        oof_predictions = np.zeros((n_samples, n_models, self.n_classes))

        # K-fold cross-validation for out-of-fold predictions
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            for model_idx, config in enumerate(model_configs):
                model_type = config.get("type", "lightgbm")
                params = config.get("params", {})

                # Train on fold
                model = self._create_base_model(model_type, params)
                model.train(
                    X_fold_train, y_fold_train,
                    X_fold_val, y_fold_val,
                    feature_names=feature_names,
                )

                # Predict on held-out fold
                fold_preds = model.predict_proba(X_fold_val)
                oof_predictions[val_idx, model_idx, :] = fold_preds

        # Prepare meta-features: flatten model predictions for each sample
        # Shape: (n_samples, n_models * n_classes)
        meta_features = oof_predictions.reshape(n_samples, -1)

        # Convert labels to one-hot for regression-based stacking
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y_train.astype(int)] = 1.0

        # Train meta-learners (one per class for better calibration)
        logger.info("Training meta-learners...")
        self.meta_learners = []

        for class_idx in range(self.n_classes):
            meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            meta_learner.fit(meta_features, y_onehot[:, class_idx])
            self.meta_learners.append(meta_learner)
            logger.debug(f"  Class {class_idx}: alpha = {meta_learner.alpha_:.4f}")

        logger.info(f"Meta-learners trained (alpha range: "
                   f"{min(m.alpha_ for m in self.meta_learners):.2f} - "
                   f"{max(m.alpha_ for m in self.meta_learners):.2f})")

        # Retrain base models on full training data
        logger.info("Retraining base models on full data...")
        self.models = []

        for config in model_configs:
            model_type = config.get("type", "lightgbm")
            params = config.get("params", {})

            logger.info(f"  Training {model_type}...")
            model = self._create_base_model(model_type, params)
            model.train(
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names,
            )
            self.models.append(model)

        # Evaluate on validation set if available
        if X_val is not None and y_val is not None:
            val_preds = self.predict_proba(X_val)
            eps = 1e-15
            val_preds = np.clip(val_preds, eps, 1 - eps)
            log_loss = -np.mean(
                np.log(val_preds[np.arange(len(y_val)), y_val.astype(int)])
            )
            logger.info(f"Stacking ensemble validation log loss: {log_loss:.4f}")

    def _create_base_model(self, model_type: str, params: Dict[str, Any]) -> BaseModel:
        """Create a base model instance."""
        if model_type == "lightgbm":
            return PositionProbabilityModel(params=params)
        elif model_type == "xgboost":
            return XGBoostPositionModel(params=params)
        elif model_type == "catboost":
            return CatBoostPositionModel(params=params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict position probabilities.

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Probability array (n_samples, 18)
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        if self.strategy == "stacking":
            return self._predict_stacking(X)
        else:
            return self._predict_averaging(X)

    def _predict_averaging(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted averaging strategy."""
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

    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking strategy with meta-learners."""
        if self.meta_learners is None:
            raise ValueError("Meta-learners not trained. Call train() first.")

        n_samples = len(X)
        n_models = len(self.models)

        # Get base model predictions
        base_predictions = np.zeros((n_samples, n_models, self.n_classes))
        for model_idx, model in enumerate(self.models):
            base_predictions[:, model_idx, :] = model.predict_proba(X)

        # Flatten to meta-features: (n_samples, n_models * n_classes)
        meta_features = base_predictions.reshape(n_samples, -1)

        # Get meta-learner predictions for each class
        stacked_pred = np.zeros((n_samples, self.n_classes))
        for class_idx, meta_learner in enumerate(self.meta_learners):
            stacked_pred[:, class_idx] = meta_learner.predict(meta_features)

        # Ensure non-negative and normalize to probabilities
        stacked_pred = np.maximum(stacked_pred, 0)
        row_sums = stacked_pred.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        stacked_pred = stacked_pred / row_sums

        return stacked_pred

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
            "n_folds": self.n_folds,
            "meta_learners": self.meta_learners,
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
            n_folds=data.get("n_folds", 5),
        )
        instance.feature_names = data["feature_names"]
        instance.n_classes = data.get("n_classes", 18)
        instance.meta_learners = data.get("meta_learners")

        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about ensemble composition.

        Returns:
            Dictionary with ensemble info
        """
        info = {
            "n_models": len(self.models),
            "model_types": [type(m).__name__ for m in self.models],
            "weights": self.weights,
            "strategy": self.strategy,
        }

        if self.strategy == "stacking":
            info["n_folds"] = self.n_folds
            info["has_meta_learners"] = self.meta_learners is not None
            if self.meta_learners:
                info["meta_learner_alphas"] = [m.alpha_ for m in self.meta_learners]

        return info
