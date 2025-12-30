"""
競馬AI予測システム - 着順予測モデル

LightGBM multiclass model for predicting finishing position probabilities.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import LGBM_PARAMS, TRAINING_CONFIG, RACE_ID_COL, HORSE_NAME_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PositionProbabilityModel:
    """
    Predict probability distribution over finishing positions.

    For each horse in a race, outputs:
    [P(1st), P(2nd), P(3rd), ..., P(18th)]
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or LGBM_PARAMS.copy()
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.num_classes = self.params.get("num_class", 18)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "PositionProbabilityModel":
        """
        Train the position prediction model.

        Args:
            X_train: Training features
            y_train: Training labels (0-indexed positions)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Train
        logger.info("Starting model training...")
        logger.info(f"Training samples: {len(X_train):,}")
        if X_val is not None:
            logger.info(f"Validation samples: {len(X_val):,}")

        callbacks = [
            lgb.log_evaluation(period=100),
        ]

        if X_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=TRAINING_CONFIG["early_stopping_rounds"],
                    verbose=True,
                )
            )

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=TRAINING_CONFIG["num_boost_round"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict raw position probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_samples, 18) with position probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        probs = self.model.predict(X, num_iteration=self.model.best_iteration)

        return probs

    def predict_race(
        self,
        X_race: pd.DataFrame,
        horse_names: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predict position probabilities for all horses in a race.

        Args:
            X_race: Features for horses in one race
            horse_names: Optional list of horse names/IDs
            normalize: Whether to normalize probabilities within race

        Returns:
            Dict mapping horse name/index to probability array
        """
        if horse_names is None:
            horse_names = [f"horse_{i}" for i in range(len(X_race))]

        # Get raw predictions
        raw_probs = self.predict_proba(X_race)
        num_horses = len(X_race)

        if normalize:
            # Normalize so probabilities sum to 1 for each position
            # Only consider positions up to num_horses
            probs = raw_probs[:, :num_horses].copy()

            # Iterative normalization (Sinkhorn-Knopp)
            for _ in range(10):
                # Normalize rows (each horse's probs sum to 1)
                row_sums = probs.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1)
                probs = probs / row_sums

                # Normalize columns (each position's probs sum to 1)
                col_sums = probs.sum(axis=0, keepdims=True)
                col_sums = np.where(col_sums > 0, col_sums, 1)
                probs = probs / col_sums

            # Pad back to 18 positions with zeros
            full_probs = np.zeros((num_horses, 18))
            full_probs[:, :num_horses] = probs
        else:
            full_probs = raw_probs

        return {name: full_probs[i] for i, name in enumerate(horse_names)}

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importance(importance_type="gain")

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "params": self.params,
            }, f)

        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Path) -> "PositionProbabilityModel":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(params=data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]

        logger.info(f"Model loaded from: {path}")

        return instance


def main():
    """Test model training."""
    from .data_loader import RaceDataLoader

    # Load data
    loader = RaceDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data()

    # Train model
    model = PositionProbabilityModel()
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n" + "=" * 60)
    print("MODEL TRAINING RESULTS")
    print("=" * 60)

    # Feature importance
    print("\nTop 10 Feature Importance:")
    importance = model.get_feature_importance()
    print(importance.head(10).to_string(index=False))

    # Predictions on test set
    probs = model.predict_proba(X_test)

    # Top-1 accuracy
    preds = probs.argmax(axis=1)
    accuracy = (preds == y_test.values).mean()
    print(f"\nTop-1 Accuracy (Test): {accuracy:.2%}")

    # Top-3 accuracy
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_accuracy = np.array([y in top3 for y, top3 in zip(y_test.values, top3_preds)]).mean()
    print(f"Top-3 Accuracy (Test): {top3_accuracy:.2%}")

    # Save model
    model_path = Path(__file__).parent.parent.parent / "data" / "models" / "position_model.pkl"
    model.save(model_path)


if __name__ == "__main__":
    main()
