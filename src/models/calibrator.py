"""
競馬AI予測システム - 確率キャリブレーション

Unified calibration module:
- Calibration methods: Platt, Isotonic, Temperature, Binning
- Calibration analysis: metrics, diagnostics, reporting
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Calibration Methods
# =============================================================================

class PlattScaling:
    """
    Platt Scaling: Fit logistic regression on model outputs.
    Maps raw probabilities to calibrated probabilities.
    """

    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "PlattScaling":
        X = probs.reshape(-1, 1)
        y = labels.astype(int)
        self.model.fit(X, y)
        self.fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        X = probs.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class IsotonicCalibration:
    """
    Isotonic Regression: Non-parametric calibration.
    Fits a monotonically increasing function.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibration":
        self.model.fit(probs, labels)
        self.fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        return self.model.predict(probs)


class TemperatureScaling:
    """
    Temperature Scaling: Scale logits by a learned temperature.
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
        eps = 1e-10
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        temperature = 1.0

        for _ in range(max_iter):
            scaled_logits = logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)

            grad = np.mean(
                (scaled_probs - labels) * logits / (temperature ** 2)
            )
            temperature -= lr * grad
            temperature = max(0.1, min(10.0, temperature))

        self.temperature = temperature
        self.fitted = True
        logger.info(f"Temperature scaling fitted: T = {self.temperature:.3f}")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        eps = 1e-10
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))


class BinningCalibration:
    """Histogram Binning: Maps predictions to average actual rate within bins."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_values = None
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "BinningCalibration":
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        self.bin_values = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (probs >= self.bin_edges[i]) & (probs < self.bin_edges[i + 1])
            if mask.sum() > 0:
                self.bin_values[i] = labels[mask].mean()
            else:
                self.bin_values[i] = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2

        self.fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        result = np.zeros_like(probs)
        for i in range(self.n_bins):
            mask = (probs >= self.bin_edges[i]) & (probs < self.bin_edges[i + 1])
            result[mask] = self.bin_values[i]
        result[probs >= self.bin_edges[-1]] = self.bin_values[-1]
        return result


# =============================================================================
# Exacta Calibrator (wrapper for exacta betting)
# =============================================================================

class ExactaCalibrator:
    """Calibrator specifically for exacta predictions."""

    METHODS = {
        "platt": PlattScaling,
        "isotonic": IsotonicCalibration,
        "temperature": TemperatureScaling,
        "binning": lambda: BinningCalibration(n_bins=15),
    }

    def __init__(self, method: str = "isotonic"):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Valid: {list(self.METHODS.keys())}")

        self.method = method
        factory = self.METHODS[method]
        self.calibrator = factory() if callable(factory) else factory
        self.fitted = False

    def fit_from_backtest(self, predictions: List[Dict]) -> "ExactaCalibrator":
        """Fit calibrator from backtest predictions."""
        probs = np.array([p["predicted_prob"] for p in predictions])
        labels = np.array([1 if p["won"] else 0 for p in predictions])

        self.calibrator.fit(probs, labels)
        self.fitted = True

        logger.info(f"Exacta calibrator fitted using {self.method}")
        logger.info(f"  Training samples: {len(probs)}, Hit rate: {labels.mean():.2%}")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"method": self.method, "calibrator": self.calibrator, "fitted": self.fitted}, f)
        logger.info(f"Calibrator saved to: {path}")

    @classmethod
    def load(cls, path: Path) -> "ExactaCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(method=data["method"])
        instance.calibrator = data["calibrator"]
        instance.fitted = data["fitted"]
        logger.info(f"Calibrator loaded from: {path}")
        return instance


# =============================================================================
# Calibration Analysis
# =============================================================================

@dataclass
class CalibrationBin:
    """Statistics for a single calibration bin."""
    bin_start: float
    bin_end: float
    bin_center: float
    num_samples: int
    avg_predicted_prob: float
    actual_hit_rate: float
    num_hits: int
    confidence_error: float  # predicted - actual


class CalibrationAnalyzer:
    """Analyze probability calibration of predictions."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def analyze(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> List[CalibrationBin]:
        """Bin predictions and compare to actual hit rates."""
        if len(probs) == 0:
            return []

        bin_edges = np.linspace(0, probs.max() + 1e-6, self.n_bins + 1)
        bins = []

        for i in range(self.n_bins):
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= bin_start) & (probs < bin_end)
            n_samples = mask.sum()

            if n_samples == 0:
                continue

            bin_probs, bin_hits = probs[mask], labels[mask]
            avg_pred, actual_rate = bin_probs.mean(), bin_hits.mean()

            bins.append(CalibrationBin(
                bin_start=bin_start,
                bin_end=bin_end,
                bin_center=(bin_start + bin_end) / 2,
                num_samples=int(n_samples),
                avg_predicted_prob=avg_pred,
                actual_hit_rate=actual_rate,
                num_hits=int(bin_hits.sum()),
                confidence_error=avg_pred - actual_rate,
            ))

        return bins

    def brier_score(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Brier score (lower is better)."""
        return np.mean((probs - labels) ** 2)

    def expected_calibration_error(self, bins: List[CalibrationBin]) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if not bins:
            return 0
        total = sum(b.num_samples for b in bins)
        if total == 0:
            return 0
        return sum(b.num_samples / total * abs(b.confidence_error) for b in bins)

    def detect_issues(self, bins: List[CalibrationBin]) -> Dict[str, str]:
        """Detect over/under-confidence issues."""
        if not bins:
            return {"status": "No data for analysis"}

        issues = {}
        avg_error = np.mean([b.confidence_error for b in bins])

        if avg_error > 0.02:
            issues["overall"] = f"Overconfident: predictions average {avg_error:.1%} too high"
        elif avg_error < -0.02:
            issues["overall"] = f"Underconfident: predictions average {abs(avg_error):.1%} too low"
        else:
            issues["overall"] = "Well-calibrated overall"

        high_prob_bins = [b for b in bins if b.avg_predicted_prob > 0.1]
        if high_prob_bins:
            high_err = np.mean([b.confidence_error for b in high_prob_bins])
            if abs(high_err) > 0.03:
                direction = "overconfident" if high_err > 0 else "underconfident"
                issues["high_prob"] = f"High-confidence picks {direction} by {abs(high_err):.1%}"

        return issues


def compare_calibration_methods(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, Dict]:
    """Compare different calibration methods."""
    methods = {
        "uncalibrated": None,
        "platt": PlattScaling(),
        "isotonic": IsotonicCalibration(),
        "temperature": TemperatureScaling(),
        "binning": BinningCalibration(n_bins=15),
    }

    results = {}
    analyzer = CalibrationAnalyzer()

    for name, calibrator in methods.items():
        if calibrator is not None:
            calibrator.fit(train_probs, train_labels)
            calibrated = calibrator.calibrate(test_probs)
        else:
            calibrated = test_probs

        bins = analyzer.analyze(calibrated, test_labels)
        brier = analyzer.brier_score(calibrated, test_labels)
        ece = analyzer.expected_calibration_error(bins)

        eps = 1e-10
        clipped = np.clip(calibrated, eps, 1 - eps)
        log_loss = -np.mean(test_labels * np.log(clipped) + (1 - test_labels) * np.log(1 - clipped))

        results[name] = {
            "brier_score": brier,
            "log_loss": log_loss,
            "ece": ece,
            "calibration_error": calibrated.mean() - test_labels.mean(),
            "avg_predicted": calibrated.mean(),
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

    for method, m in results.items():
        print(f"{method:<15} {m['brier_score']:>10.4f} {m['log_loss']:>10.4f} "
              f"{m['ece']:>10.4f} {m['calibration_error']:>+11.2%}")

    print("=" * 70)
