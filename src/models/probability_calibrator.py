"""
競馬AI予測システム - 確率キャリブレーション

Probability calibration methods for improving prediction reliability.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import FEATURES, RACE_ID_COL, TARGET_COL, DATE_COL
from .position_model import PositionProbabilityModel
from .exacta_calculator import ExactaCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PlattScaling:
    """
    Platt Scaling: Fit logistic regression on model outputs.

    Maps raw probabilities to calibrated probabilities using:
    P_calibrated = 1 / (1 + exp(A * P_raw + B))
    """

    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattScaling":
        """
        Fit Platt scaling on validation data.

        Args:
            probs: Raw predicted probabilities (1D array)
            labels: Binary labels (1 if event occurred, 0 otherwise)
        """
        # Reshape for sklearn
        X = probs.reshape(-1, 1)
        y = labels.astype(int)

        self.model.fit(X, y)
        self.fitted = True

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        X = probs.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class IsotonicCalibration:
    """
    Isotonic Regression: Non-parametric calibration.

    Fits a monotonically increasing function to map
    raw probabilities to calibrated probabilities.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibration":
        """
        Fit isotonic regression on validation data.

        Args:
            probs: Raw predicted probabilities
            labels: Binary labels
        """
        self.model.fit(probs, labels)
        self.fitted = True

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probabilities."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self.model.predict(probs)


class TemperatureScaling:
    """
    Temperature Scaling: Scale logits by a learned temperature.

    Simple and effective for neural network outputs.
    P_calibrated = softmax(logits / T)
    """

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> "TemperatureScaling":
        """
        Fit temperature using gradient descent on NLL.

        Args:
            probs: Raw predicted probabilities
            labels: Binary labels
            lr: Learning rate
            max_iter: Maximum iterations
        """
        # Convert to logits
        eps = 1e-10
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        temperature = 1.0

        for _ in range(max_iter):
            # Scaled probabilities
            scaled_logits = logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)

            # Negative log likelihood
            nll = -np.mean(
                labels * np.log(scaled_probs) +
                (1 - labels) * np.log(1 - scaled_probs)
            )

            # Gradient
            grad = np.mean(
                (scaled_probs - labels) * logits / (temperature ** 2)
            )

            # Update
            temperature -= lr * grad
            temperature = max(0.1, min(10.0, temperature))  # Clip

        self.temperature = temperature
        self.fitted = True

        logger.info(f"Temperature scaling fitted: T = {self.temperature:.3f}")

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw probabilities."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        eps = 1e-10
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))


class BinningCalibration:
    """
    Histogram Binning: Simple binning-based calibration.

    Maps predictions to average actual rate within each bin.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_values = None
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "BinningCalibration":
        """Fit histogram binning."""
        # Create bin edges
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        self.bin_values = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (probs >= self.bin_edges[i]) & (probs < self.bin_edges[i + 1])
            if mask.sum() > 0:
                self.bin_values[i] = labels[mask].mean()
            else:
                # Use bin center if no samples
                self.bin_values[i] = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2

        self.fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply binning calibration."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        result = np.zeros_like(probs)
        for i in range(self.n_bins):
            mask = (probs >= self.bin_edges[i]) & (probs < self.bin_edges[i + 1])
            result[mask] = self.bin_values[i]

        # Handle edge case
        result[probs >= self.bin_edges[-1]] = self.bin_values[-1]

        return result


class ExactaCalibrator:
    """
    Calibrator specifically for exacta predictions.

    Calibrates the probability of hitting an exacta bet,
    accounting for the specific structure of exacta probabilities.
    """

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: Calibration method - "platt", "isotonic", "temperature", "binning"
        """
        self.method = method

        if method == "platt":
            self.calibrator = PlattScaling()
        elif method == "isotonic":
            self.calibrator = IsotonicCalibration()
        elif method == "temperature":
            self.calibrator = TemperatureScaling()
        elif method == "binning":
            self.calibrator = BinningCalibration(n_bins=15)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.fitted = False

    def fit_from_backtest(
        self,
        predictions: List[Dict],
    ) -> "ExactaCalibrator":
        """
        Fit calibrator from backtest results.

        Args:
            predictions: List of dicts with 'predicted_prob' and 'won' keys
        """
        probs = np.array([p["predicted_prob"] for p in predictions])
        labels = np.array([1 if p["won"] else 0 for p in predictions])

        self.calibrator.fit(probs, labels)
        self.fitted = True

        logger.info(f"Exacta calibrator fitted using {self.method}")
        logger.info(f"  Training samples: {len(probs)}")
        logger.info(f"  Hit rate: {labels.mean():.2%}")

        return self

    def fit_from_data(
        self,
        df: pd.DataFrame,
        model: PositionProbabilityModel,
        feature_cols: List[str],
        odds_lookup: Dict,
    ) -> "ExactaCalibrator":
        """
        Fit calibrator by running predictions on validation data.

        Args:
            df: Validation DataFrame
            model: Trained position model
            feature_cols: Feature column names
            odds_lookup: Dict with actual race results
        """
        exacta_calc = ExactaCalculator()
        predictions = []

        race_groups = df.groupby(RACE_ID_COL)

        for race_id, race_df in race_groups:
            if race_id not in odds_lookup:
                continue

            odds_data = odds_lookup[race_id]
            if "exacta" not in odds_data:
                continue

            actual_1st, actual_2nd, _ = odds_data["exacta"]

            if len(race_df) < 2:
                continue

            # Get predictions
            X_race = race_df[feature_cols]
            horse_numbers = race_df["馬番"].tolist()

            position_probs = model.predict_race(
                X_race,
                horse_names=[str(h) for h in horse_numbers],
            )

            exacta_probs = exacta_calc.calculate_exacta_probs(position_probs)

            # Record all predictions with outcomes
            for (h1, h2), prob in exacta_probs.items():
                won = (int(h1) == actual_1st) and (int(h2) == actual_2nd)
                predictions.append({
                    "predicted_prob": prob,
                    "won": won,
                })

        return self.fit_from_backtest(predictions)

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to exacta probabilities."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted.")

        return self.calibrator.calibrate(probs)

    def calibrate_dict(
        self, exacta_probs: Dict[Tuple[str, str], float]
    ) -> Dict[Tuple[str, str], float]:
        """Apply calibration to exacta probability dictionary."""
        if not self.fitted:
            raise ValueError("Calibrator not fitted.")

        combinations = list(exacta_probs.keys())
        probs = np.array([exacta_probs[k] for k in combinations])

        calibrated = self.calibrator.calibrate(probs)

        return {k: float(calibrated[i]) for i, k in enumerate(combinations)}

    def save(self, path: Path) -> None:
        """Save calibrator to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "method": self.method,
                "calibrator": self.calibrator,
                "fitted": self.fitted,
            }, f)

        logger.info(f"Calibrator saved to: {path}")

    @classmethod
    def load(cls, path: Path) -> "ExactaCalibrator":
        """Load calibrator from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(method=data["method"])
        instance.calibrator = data["calibrator"]
        instance.fitted = data["fitted"]

        logger.info(f"Calibrator loaded from: {path}")

        return instance


def compare_calibration_methods(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, Dict]:
    """
    Compare different calibration methods.

    Returns dict with metrics for each method.
    """
    methods = {
        "uncalibrated": None,
        "platt": PlattScaling(),
        "isotonic": IsotonicCalibration(),
        "temperature": TemperatureScaling(),
        "binning": BinningCalibration(n_bins=15),
    }

    results = {}

    for name, calibrator in methods.items():
        if calibrator is not None:
            calibrator.fit(train_probs, train_labels)
            calibrated_probs = calibrator.calibrate(test_probs)
        else:
            calibrated_probs = test_probs

        # Calculate metrics
        eps = 1e-10
        probs_clipped = np.clip(calibrated_probs, eps, 1 - eps)

        # Brier score
        brier = np.mean((calibrated_probs - test_labels) ** 2)

        # Log loss
        log_loss = -np.mean(
            test_labels * np.log(probs_clipped) +
            (1 - test_labels) * np.log(1 - probs_clipped)
        )

        # ECE (Expected Calibration Error)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            mask = (calibrated_probs >= bin_edges[i]) & (calibrated_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = test_labels[mask].mean()
                bin_conf = calibrated_probs[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
        ece /= len(test_probs)

        # Calibration error (predicted - actual)
        cal_error = calibrated_probs.mean() - test_labels.mean()

        results[name] = {
            "brier_score": brier,
            "log_loss": log_loss,
            "ece": ece,
            "calibration_error": cal_error,
            "avg_predicted": calibrated_probs.mean(),
            "avg_actual": test_labels.mean(),
        }

    return results


def print_calibration_comparison(results: Dict[str, Dict]) -> None:
    """Print calibration comparison table."""
    print("\n" + "=" * 70)
    print("CALIBRATION METHOD COMPARISON")
    print("=" * 70)
    print(f"{'Method':<15} {'Brier':>10} {'LogLoss':>10} {'ECE':>10} {'CalError':>12}")
    print("-" * 70)

    for method, metrics in results.items():
        print(
            f"{method:<15} "
            f"{metrics['brier_score']:>10.4f} "
            f"{metrics['log_loss']:>10.4f} "
            f"{metrics['ece']:>10.4f} "
            f"{metrics['calibration_error']:>+11.2%}"
        )

    print("-" * 70)
    print(f"{'Method':<15} {'AvgPred':>10} {'AvgActual':>10}")
    print("-" * 70)

    for method, metrics in results.items():
        print(
            f"{method:<15} "
            f"{metrics['avg_predicted']:>10.2%} "
            f"{metrics['avg_actual']:>10.2%}"
        )

    print("=" * 70)


def main():
    """Test calibration methods."""
    from .data_loader import RaceDataLoader
    from .backtester import Backtester
    from .odds_loader import OddsLoader

    # Load data
    logger.info("Loading data...")
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    # Load odds
    odds_loader = OddsLoader()
    odds_loader.load_odds()
    odds_lookup = odds_loader.create_race_odds_lookup()

    # Split into calibration train/test
    df = df.copy()
    df["_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=["_date"]).sort_values("_date")

    # Use first part for training model, middle for calibration, end for test
    split1 = df["_date"].quantile(0.5)
    split2 = df["_date"].quantile(0.75)

    train_df = df[df["_date"] < split1]
    cal_df = df[(df["_date"] >= split1) & (df["_date"] < split2)]
    test_df = df[df["_date"] >= split2]

    logger.info(f"Train: {len(train_df):,}, Cal: {len(cal_df):,}, Test: {len(test_df):,}")

    # Train model
    feature_cols = FEATURES.copy()
    X_train = train_df[feature_cols]
    y_train = pd.to_numeric(train_df[TARGET_COL], errors="coerce")
    y_train = (y_train.clip(lower=1, upper=18) - 1).astype(int)

    model = PositionProbabilityModel()
    model.train(X_train, y_train)

    # Collect calibration data
    logger.info("Collecting calibration data...")
    exacta_calc = ExactaCalculator()

    cal_predictions = []
    test_predictions = []

    for split_df, pred_list in [(cal_df, cal_predictions), (test_df, test_predictions)]:
        race_groups = split_df.groupby(RACE_ID_COL)

        for race_id, race_df in race_groups:
            if race_id not in odds_lookup:
                continue

            odds_data = odds_lookup[race_id]
            if "exacta" not in odds_data:
                continue

            actual_1st, actual_2nd, actual_odds = odds_data["exacta"]

            if len(race_df) < 2:
                continue

            X_race = race_df[feature_cols]
            horse_numbers = race_df["馬番"].tolist()

            position_probs = model.predict_race(
                X_race,
                horse_names=[str(h) for h in horse_numbers],
            )

            exacta_probs = exacta_calc.calculate_exacta_probs(position_probs)

            # Top 3 predictions
            top_exactas = exacta_calc.get_top_exactas(exacta_probs, n=3)

            for (h1, h2), prob in top_exactas:
                if prob < 0.03:
                    continue

                won = (int(h1) == actual_1st) and (int(h2) == actual_2nd)
                pred_list.append({
                    "predicted_prob": prob,
                    "won": won,
                    "actual_odds": actual_odds if won else 0,
                })

    logger.info(f"Calibration predictions: {len(cal_predictions):,}")
    logger.info(f"Test predictions: {len(test_predictions):,}")

    # Compare methods
    cal_probs = np.array([p["predicted_prob"] for p in cal_predictions])
    cal_labels = np.array([1 if p["won"] else 0 for p in cal_predictions])
    test_probs = np.array([p["predicted_prob"] for p in test_predictions])
    test_labels = np.array([1 if p["won"] else 0 for p in test_predictions])

    results = compare_calibration_methods(cal_probs, cal_labels, test_probs, test_labels)
    print_calibration_comparison(results)

    # Best method
    best_method = min(results.keys(), key=lambda k: results[k]["ece"])
    print(f"\nBest method by ECE: {best_method}")


if __name__ == "__main__":
    main()
