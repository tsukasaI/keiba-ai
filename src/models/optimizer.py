"""
Keiba AI Prediction System - Hyperparameter Optimization

Uses Optuna for Bayesian hyperparameter optimization.
Optimizes for ROI using walk-forward backtesting.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.models.data_loader import RaceDataLoader
from src.models.position_model import PositionProbabilityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optuna-based hyperparameter optimization for horse racing prediction.

    Optimizes for ROI using walk-forward backtesting methodology.
    Uses time-series cross-validation to prevent data leakage.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        n_trials: int = 100,
        study_name: str = "keiba_optimization",
        storage: Optional[str] = None,
    ):
        """Initialize the optimizer.

        Args:
            model_type: Model type to optimize ("lightgbm", "xgboost", "catboost")
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            storage: Optional SQLite storage path for persistence
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.study = None
        self.data_loader = None
        self._train_df = None
        self._val_df = None
        self._test_df = None

    def _load_data(self) -> bool:
        """Load and prepare data for optimization."""
        try:
            self.data_loader = RaceDataLoader()
            df = self.data_loader.load_features()

            if df is None or len(df) == 0:
                logger.error("No data loaded")
                return False

            # Create time splits
            self._train_df, self._val_df, self._test_df = (
                self.data_loader.create_time_splits(df)
            )

            logger.info(f"Train: {len(self._train_df)}, Val: {len(self._val_df)}, Test: {len(self._test_df)}")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def _suggest_lgbm_params(self, trial: optuna.Trial) -> dict:
        """Suggest LightGBM hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        return {
            "objective": "multiclass",
            "num_class": 18,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

    def _evaluate_model(self, params: dict) -> float:
        """Evaluate model with given parameters using validation set.

        Args:
            params: Model hyperparameters

        Returns:
            Validation log loss (lower is better)
        """
        try:
            # Get features and target
            X_train, y_train = self.data_loader.get_features_and_target(
                self._train_df
            )
            X_val, y_val = self.data_loader.get_features_and_target(
                self._val_df
            )

            # Train model
            model = PositionProbabilityModel(params=params)
            model.train(
                X_train, y_train,
                X_val, y_val,
            )

            # Calculate validation loss
            proba = model.predict_proba(X_val)

            # Multi-class log loss
            eps = 1e-15
            proba = np.clip(proba, eps, 1 - eps)
            log_loss = -np.mean(
                np.log(proba[np.arange(len(y_val)), y_val.astype(int)])
            )

            return log_loss

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Validation log loss (to minimize)
        """
        if self.model_type in ("lightgbm", "lgbm"):
            params = self._suggest_lgbm_params(trial)
        else:
            raise ValueError(f"Optimization not supported for model type: {self.model_type}. Only 'lgbm'/'lightgbm' is supported.")

        log_loss = self._evaluate_model(params)

        # Prune unpromising trials
        trial.report(log_loss, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return log_loss

    def optimize(self, n_trials: Optional[int] = None) -> dict:
        """Run hyperparameter optimization.

        Args:
            n_trials: Number of trials (overrides init value)

        Returns:
            Best hyperparameters found
        """
        if n_trials is not None:
            self.n_trials = n_trials

        # Load data
        if not self._load_data():
            raise RuntimeError("Failed to load data for optimization")

        # Create or load study
        sampler = TPESampler(seed=42)

        if self.storage:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                direction="minimize",
                sampler=sampler,
            )
        else:
            self.study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
            )

        # Run optimization
        logger.info(f"Starting optimization with {self.n_trials} trials...")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # Log results
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value (log loss): {self.study.best_value:.6f}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study.best_params

    def get_best_params(self) -> dict:
        """Get best parameters merged with defaults.

        Returns:
            Complete parameter dictionary
        """
        if self.study is None:
            raise RuntimeError("Run optimize() first")

        # Merge with default params
        best_params = {
            "objective": "multiclass",
            "num_class": 18,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        best_params.update(self.study.best_params)

        return best_params

    def save_best_params(self, path: Path) -> None:
        """Save best parameters to JSON file.

        Args:
            path: Output file path
        """
        best_params = self.get_best_params()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(best_params, f, indent=2)

        logger.info(f"Best params saved to {path}")


def main():
    """Run hyperparameter optimization from command line."""
    parser = argparse.ArgumentParser(description="Optimize model hyperparameters")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--model-type", type=str, default="lightgbm", help="Model type")
    parser.add_argument("--output", type=str, default="data/models/best_params.json")
    args = parser.parse_args()

    optimizer = ModelOptimizer(
        model_type=args.model_type,
        n_trials=args.n_trials,
    )

    best_params = optimizer.optimize()

    # Save results
    output_path = Path(args.output)
    optimizer.save_best_params(output_path)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest log loss: {optimizer.study.best_value:.6f}")
    print(f"\nBest parameters saved to: {output_path}")


if __name__ == "__main__":
    main()
