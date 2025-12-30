"""
競馬AI予測システム - モデル訓練

Training pipeline with TimeSeriesSplit for proper validation.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from .config import TRAINING_CONFIG, DATE_COL, RACE_ID_COL
from .data_loader import RaceDataLoader
from .position_model import PositionProbabilityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrate model training with proper time-series validation.

    CRITICAL: Never use future data to predict past races.
    """

    def __init__(self, n_splits: int = TRAINING_CONFIG["n_splits"]):
        self.n_splits = n_splits
        self.models: List[PositionProbabilityModel] = []
        self.cv_results: List[Dict] = []

    def train_with_cv(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "着順",
        date_col: str = DATE_COL,
    ) -> Tuple[List[PositionProbabilityModel], List[Dict]]:
        """
        Train using TimeSeriesSplit.

        Uses expanding window: each fold trains on all data before validation period.
        """
        # Parse and sort by date
        df = df.copy()
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"])
        df = df.sort_values("_date")

        # Get unique dates for splitting
        unique_dates = sorted(df["_date"].unique())
        n_dates = len(unique_dates)

        logger.info(f"Total unique dates: {n_dates}")
        logger.info(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")

        # Calculate fold boundaries
        # Each fold uses expanding window for training
        fold_size = n_dates // (self.n_splits + 1)

        self.models = []
        self.cv_results = []

        for fold in range(self.n_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold + 1}/{self.n_splits}")
            logger.info("=" * 60)

            # Training: all data before validation period
            train_end_idx = (fold + 1) * fold_size
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + fold_size, n_dates)

            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            train_mask = df["_date"].isin(train_dates)
            val_mask = df["_date"].isin(val_dates)

            X_train = df.loc[train_mask, feature_cols]
            y_train = self._prepare_target(df.loc[train_mask, target_col])
            X_val = df.loc[val_mask, feature_cols]
            y_val = self._prepare_target(df.loc[val_mask, target_col])

            logger.info(f"Train period: {train_dates[0].date()} to {train_dates[-1].date()}")
            logger.info(f"Valid period: {val_dates[0].date()} to {val_dates[-1].date()}")
            logger.info(f"Train samples: {len(X_train):,}")
            logger.info(f"Valid samples: {len(X_val):,}")

            # Train model
            model = PositionProbabilityModel()
            model.train(X_train, y_train, X_val, y_val)
            self.models.append(model)

            # Evaluate
            metrics = self._evaluate_fold(model, X_val, y_val)
            metrics["fold"] = fold + 1
            metrics["train_samples"] = len(X_train)
            metrics["val_samples"] = len(X_val)
            self.cv_results.append(metrics)

            logger.info(f"Fold {fold + 1} Results:")
            logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2%}")
            logger.info(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.2%}")
            logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")

        return self.models, self.cv_results

    def _prepare_target(self, target: pd.Series) -> pd.Series:
        """Convert position to 0-indexed class."""
        target = pd.to_numeric(target, errors="coerce")
        target = target.clip(lower=1, upper=18)
        return (target - 1).astype(int)

    def _evaluate_fold(
        self, model: PositionProbabilityModel, X: pd.DataFrame, y: pd.Series
    ) -> Dict:
        """Calculate evaluation metrics for a fold."""
        probs = model.predict_proba(X)

        # Top-1 accuracy
        preds = probs.argmax(axis=1)
        top1_accuracy = (preds == y.values).mean()

        # Top-3 accuracy
        top3_preds = np.argsort(probs, axis=1)[:, -3:]
        top3_accuracy = np.array([
            y_true in top3 for y_true, top3 in zip(y.values, top3_preds)
        ]).mean()

        # Log loss (with clipping to avoid log(0))
        eps = 1e-10
        probs_clipped = np.clip(probs, eps, 1 - eps)
        log_loss = -np.mean([
            np.log(probs_clipped[i, y_true])
            for i, y_true in enumerate(y.values)
        ])

        return {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "log_loss": log_loss,
        }

    def train_final_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "着順",
        date_col: str = DATE_COL,
        test_months: int = TRAINING_CONFIG["test_size_months"],
    ) -> Tuple[PositionProbabilityModel, pd.DataFrame]:
        """
        Train final model on all data except test period.

        Returns:
            model: Trained model
            test_df: Test data for evaluation
        """
        df = df.copy()
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"])
        df = df.sort_values("_date")

        # Split: last N months for test
        max_date = df["_date"].max()
        test_start = max_date - pd.DateOffset(months=test_months)

        train_df = df[df["_date"] < test_start].copy()
        test_df = df[df["_date"] >= test_start].copy()

        logger.info(f"\n{'='*60}")
        logger.info("FINAL MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Train period: {train_df['_date'].min().date()} to {train_df['_date'].max().date()}")
        logger.info(f"Test period: {test_df['_date'].min().date()} to {test_df['_date'].max().date()}")
        logger.info(f"Train samples: {len(train_df):,}")
        logger.info(f"Test samples: {len(test_df):,}")

        X_train = train_df[feature_cols]
        y_train = self._prepare_target(train_df[target_col])

        # Use last 20% of training data for early stopping
        split_idx = int(len(X_train) * 0.8)
        X_train_fit = X_train.iloc[:split_idx]
        y_train_fit = y_train.iloc[:split_idx]
        X_val = X_train.iloc[split_idx:]
        y_val = y_train.iloc[split_idx:]

        model = PositionProbabilityModel()
        model.train(X_train_fit, y_train_fit, X_val, y_val)

        return model, test_df

    def summarize_cv_results(self) -> pd.DataFrame:
        """Summarize cross-validation results."""
        if not self.cv_results:
            return pd.DataFrame()

        df = pd.DataFrame(self.cv_results)

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False))
        print("-" * 60)
        print(f"Mean Top-1 Accuracy: {df['top1_accuracy'].mean():.2%} (+/- {df['top1_accuracy'].std():.2%})")
        print(f"Mean Top-3 Accuracy: {df['top3_accuracy'].mean():.2%} (+/- {df['top3_accuracy'].std():.2%})")
        print(f"Mean Log Loss: {df['log_loss'].mean():.4f} (+/- {df['log_loss'].std():.4f})")

        return df


def main():
    """Train model with cross-validation."""
    from .data_loader import RaceDataLoader

    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    # Train with CV
    trainer = ModelTrainer(n_splits=5)
    models, cv_results = trainer.train_with_cv(
        df,
        feature_cols=loader.feature_cols,
    )

    # Summarize results
    trainer.summarize_cv_results()

    # Train final model
    final_model, test_df = trainer.train_final_model(
        df,
        feature_cols=loader.feature_cols,
    )

    # Save final model
    model_path = Path(__file__).parent.parent.parent / "data" / "models" / "final_model.pkl"
    final_model.save(model_path)

    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    importance = final_model.get_feature_importance()
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
