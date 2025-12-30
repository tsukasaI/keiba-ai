"""
競馬AI予測システム - キャリブレーション分析

Probability calibration analysis for exacta predictions.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtester import BacktestResults, BetResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    """
    Analyze probability calibration of exacta predictions.

    Well-calibrated predictions should have:
    - Predicted 10% probabilities hitting ~10% of the time
    - Predicted 5% probabilities hitting ~5% of the time
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins

    def analyze_calibration(
        self, results: BacktestResults
    ) -> List[CalibrationBin]:
        """
        Bin predictions and compare to actual hit rates.

        Args:
            results: Backtest results with predictions and outcomes

        Returns:
            List of CalibrationBin with statistics
        """
        if not results.bets:
            return []

        # Extract predicted probs and outcomes
        probs = np.array([b.predicted_prob for b in results.bets])
        hits = np.array([1 if b.won else 0 for b in results.bets])

        # Create bins
        bin_edges = np.linspace(0, probs.max() + 1e-6, self.n_bins + 1)
        bins = []

        for i in range(self.n_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]

            mask = (probs >= bin_start) & (probs < bin_end)
            n_samples = mask.sum()

            if n_samples == 0:
                continue

            bin_probs = probs[mask]
            bin_hits = hits[mask]

            avg_pred = bin_probs.mean()
            actual_rate = bin_hits.mean()

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

    def calculate_brier_score(self, results: BacktestResults) -> float:
        """
        Calculate Brier score (lower is better).

        Brier = mean((predicted - actual)^2)
        """
        if not results.bets:
            return 0

        probs = np.array([b.predicted_prob for b in results.bets])
        hits = np.array([1 if b.won else 0 for b in results.bets])

        return np.mean((probs - hits) ** 2)

    def calculate_expected_calibration_error(
        self, bins: List[CalibrationBin]
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE = sum(|bin_samples| / total * |avg_pred - actual_rate|)
        """
        if not bins:
            return 0

        total_samples = sum(b.num_samples for b in bins)
        if total_samples == 0:
            return 0

        ece = 0
        for bin in bins:
            weight = bin.num_samples / total_samples
            ece += weight * abs(bin.confidence_error)

        return ece

    def detect_calibration_issues(
        self, bins: List[CalibrationBin]
    ) -> Dict[str, str]:
        """
        Detect over/under-confidence issues.

        Returns dict with diagnostic messages.
        """
        issues = {}

        if not bins:
            return {"status": "No data for analysis"}

        # Overall tendency
        avg_error = np.mean([b.confidence_error for b in bins])

        if avg_error > 0.02:
            issues["overall"] = f"Overconfident: predictions average {avg_error:.1%} too high"
        elif avg_error < -0.02:
            issues["overall"] = f"Underconfident: predictions average {abs(avg_error):.1%} too low"
        else:
            issues["overall"] = "Well-calibrated overall"

        # High probability bins
        high_prob_bins = [b for b in bins if b.avg_predicted_prob > 0.1]
        if high_prob_bins:
            high_avg_error = np.mean([b.confidence_error for b in high_prob_bins])
            if high_avg_error > 0.03:
                issues["high_prob"] = f"High-confidence picks overconfident by {high_avg_error:.1%}"
            elif high_avg_error < -0.03:
                issues["high_prob"] = f"High-confidence picks underconfident by {abs(high_avg_error):.1%}"

        # Low probability bins
        low_prob_bins = [b for b in bins if b.avg_predicted_prob < 0.05]
        if low_prob_bins:
            low_avg_error = np.mean([b.confidence_error for b in low_prob_bins])
            if low_avg_error > 0.02:
                issues["low_prob"] = f"Long-shot predictions overconfident by {low_avg_error:.1%}"

        return issues

    def get_calibration_curve_data(
        self, bins: List[CalibrationBin]
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Get data for plotting calibration curve.

        Returns:
            (predicted_probs, actual_rates, sample_counts)
        """
        if not bins:
            return [], [], []

        predicted = [b.avg_predicted_prob for b in bins]
        actual = [b.actual_hit_rate for b in bins]
        counts = [b.num_samples for b in bins]

        return predicted, actual, counts

    def analyze_by_ev_tier(
        self, results: BacktestResults
    ) -> Dict[str, Dict]:
        """
        Analyze calibration by expected value tier.

        Returns stats for each EV tier.
        """
        if not results.bets:
            return {}

        tiers = {
            "high_ev": {"min": 1.5, "max": float("inf")},
            "medium_ev": {"min": 1.2, "max": 1.5},
            "low_ev": {"min": 1.0, "max": 1.2},
        }

        tier_results = {}

        for tier_name, bounds in tiers.items():
            tier_bets = [
                b for b in results.bets
                if bounds["min"] <= b.expected_value < bounds["max"]
            ]

            if not tier_bets:
                continue

            probs = np.array([b.predicted_prob for b in tier_bets])
            hits = np.array([1 if b.won else 0 for b in tier_bets])
            evs = np.array([b.expected_value for b in tier_bets])

            tier_results[tier_name] = {
                "num_bets": len(tier_bets),
                "avg_predicted_prob": probs.mean(),
                "actual_hit_rate": hits.mean(),
                "avg_ev": evs.mean(),
                "num_hits": int(hits.sum()),
                "calibration_error": probs.mean() - hits.mean(),
            }

        return tier_results

    def print_calibration_report(
        self,
        bins: List[CalibrationBin],
        brier_score: float,
        ece: float,
        issues: Dict[str, str],
        ev_analysis: Dict[str, Dict],
    ) -> None:
        """Print formatted calibration report."""
        print("\n" + "=" * 60)
        print("CALIBRATION ANALYSIS REPORT")
        print("=" * 60)

        print("\n--- Overall Metrics ---")
        print(f"Brier Score:                    {brier_score:.4f}")
        print(f"Expected Calibration Error:     {ece:.4f}")

        print("\n--- Calibration by Probability Bin ---")
        print(f"{'Bin':>12} {'Samples':>10} {'Predicted':>12} {'Actual':>10} {'Error':>10}")
        print("-" * 56)
        for b in bins:
            print(
                f"{b.bin_start:.2%}-{b.bin_end:.2%}"
                f"{b.num_samples:>10}"
                f"{b.avg_predicted_prob:>12.2%}"
                f"{b.actual_hit_rate:>10.2%}"
                f"{b.confidence_error:>+10.2%}"
            )

        print("\n--- Calibration Issues ---")
        for category, message in issues.items():
            print(f"{category}: {message}")

        if ev_analysis:
            print("\n--- Calibration by EV Tier ---")
            print(f"{'Tier':>12} {'Bets':>8} {'Predicted':>12} {'Actual':>10} {'Error':>10}")
            print("-" * 54)
            for tier, stats in ev_analysis.items():
                print(
                    f"{tier:>12}"
                    f"{stats['num_bets']:>8}"
                    f"{stats['avg_predicted_prob']:>12.2%}"
                    f"{stats['actual_hit_rate']:>10.2%}"
                    f"{stats['calibration_error']:>+10.2%}"
                )

        print("=" * 60)


def run_calibration_analysis(results: BacktestResults) -> Dict:
    """
    Run full calibration analysis on backtest results.

    Args:
        results: BacktestResults from backtester

    Returns:
        Dict with all calibration metrics
    """
    analyzer = CalibrationAnalyzer(n_bins=10)

    bins = analyzer.analyze_calibration(results)
    brier_score = analyzer.calculate_brier_score(results)
    ece = analyzer.calculate_expected_calibration_error(bins)
    issues = analyzer.detect_calibration_issues(bins)
    ev_analysis = analyzer.analyze_by_ev_tier(results)

    # Print report
    analyzer.print_calibration_report(
        bins, brier_score, ece, issues, ev_analysis
    )

    # Return data for further processing
    return {
        "bins": bins,
        "brier_score": brier_score,
        "ece": ece,
        "issues": issues,
        "ev_analysis": ev_analysis,
    }


def main():
    """Run calibration analysis on backtest results."""
    from .data_loader import RaceDataLoader
    from .backtester import Backtester

    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    # Run backtest
    backtester = Backtester(ev_threshold=1.0)
    results, _ = backtester.run_walkforward_backtest(
        df,
        n_periods=6,
        train_months=18,
        test_months=3,
    )

    # Run calibration analysis
    run_calibration_analysis(results)


if __name__ == "__main__":
    main()
