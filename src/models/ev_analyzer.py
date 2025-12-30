"""
Keiba AI Prediction System - Expected Value Analyzer

Analyze EV characteristics of winning bets from backtest results.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .types import BacktestResults, BetResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EVAnalysisResult:
    """EV analysis results."""

    # Overall stats
    total_bets: int = 0
    winning_bets: int = 0
    hit_rate: float = 0.0
    roi: float = 0.0

    # EV analysis
    avg_predicted_prob: float = 0.0
    avg_actual_odds: float = 0.0  # Decimal odds (actual_odds / 100)
    avg_post_hoc_ev: float = 0.0  # P × actual_odds for wins

    # Probability bins
    prob_bin_results: Dict[str, Dict] = field(default_factory=dict)

    # Edge analysis
    avg_edge_vs_implied: float = 0.0  # Our prob vs 1/odds

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            "=" * 60,
            "EXPECTED VALUE ANALYSIS",
            "=" * 60,
            "",
            "Overall Results:",
            f"  Total Bets: {self.total_bets:,}",
            f"  Winning Bets: {self.winning_bets:,}",
            f"  Hit Rate: {self.hit_rate:.1%}",
            f"  ROI: {self.roi:+.1%}",
            "",
            "Probability Analysis:",
            f"  Avg Predicted Prob: {self.avg_predicted_prob:.2%}",
            f"  Avg Actual Odds (wins): {self.avg_actual_odds:.1f}x",
            f"  Avg Post-Hoc EV (wins): {self.avg_post_hoc_ev:.2f}",
            "",
            "Results by Probability Bin:",
        ]

        for bin_name, stats in sorted(self.prob_bin_results.items()):
            lines.append(f"  {bin_name}:")
            lines.append(f"    Bets: {stats['bets']:,}, Wins: {stats['wins']:,}")
            lines.append(f"    Hit Rate: {stats['hit_rate']:.1%}, ROI: {stats['roi']:+.1%}")

        return "\n".join(lines)


class EVAnalyzer:
    """
    Analyze expected value characteristics of backtest results.

    Since we don't have pre-race exacta odds for all combinations,
    we analyze:
    1. Post-hoc EV of winning bets
    2. Relationship between predicted probability and hit rate
    3. Edge vs implied probability from actual odds
    """

    def __init__(self):
        self.prob_bins = [
            (0.00, 0.03, "0-3%"),
            (0.03, 0.05, "3-5%"),
            (0.05, 0.08, "5-8%"),
            (0.08, 0.10, "8-10%"),
            (0.10, 0.15, "10-15%"),
            (0.15, 0.20, "15-20%"),
            (0.20, 1.00, "20%+"),
        ]

    def analyze(self, results: BacktestResults) -> EVAnalysisResult:
        """
        Analyze EV characteristics of backtest results.

        Args:
            results: BacktestResults from backtester

        Returns:
            EVAnalysisResult with detailed analysis
        """
        analysis = EVAnalysisResult()

        if not results.bets:
            return analysis

        # Overall stats
        analysis.total_bets = results.num_bets
        analysis.winning_bets = results.num_wins
        analysis.hit_rate = results.hit_rate / 100
        analysis.roi = results.roi / 100

        # Calculate probability stats
        all_probs = [b.predicted_prob for b in results.bets]
        analysis.avg_predicted_prob = np.mean(all_probs)

        # Analyze winning bets
        winning_bets = [b for b in results.bets if b.won]
        if winning_bets:
            # Convert from Japanese format (3560 = 35.6x) to decimal
            decimal_odds = [b.actual_odds / 100 for b in winning_bets]
            analysis.avg_actual_odds = np.mean(decimal_odds)

            # Post-hoc EV = predicted_prob × actual_decimal_odds
            post_hoc_evs = [
                b.predicted_prob * (b.actual_odds / 100)
                for b in winning_bets
            ]
            analysis.avg_post_hoc_ev = np.mean(post_hoc_evs)

        # Analyze by probability bins
        analysis.prob_bin_results = self._analyze_by_prob_bins(results.bets)

        return analysis

    def _analyze_by_prob_bins(
        self, bets: List[BetResult]
    ) -> Dict[str, Dict]:
        """Analyze results grouped by predicted probability."""
        bin_results = {}

        for low, high, name in self.prob_bins:
            bin_bets = [
                b for b in bets
                if low <= b.predicted_prob < high
            ]

            if not bin_bets:
                continue

            total_bet = sum(b.bet_amount for b in bin_bets)
            total_return = sum(b.payout for b in bin_bets)
            wins = sum(1 for b in bin_bets if b.won)

            bin_results[name] = {
                "bets": len(bin_bets),
                "wins": wins,
                "hit_rate": wins / len(bin_bets) if bin_bets else 0,
                "roi": (total_return - total_bet) / total_bet if total_bet > 0 else 0,
                "avg_prob": np.mean([b.predicted_prob for b in bin_bets]),
                "avg_odds_wins": (
                    np.mean([b.actual_odds / 100 for b in bin_bets if b.won])
                    if wins > 0 else 0
                ),
            }

        return bin_results

    def compare_to_random(
        self, results: BacktestResults, n_horses_avg: float = 14
    ) -> Dict:
        """
        Compare results to random betting baseline.

        Args:
            results: BacktestResults from backtester
            n_horses_avg: Average number of horses per race

        Returns:
            Dict with comparison metrics
        """
        # Random exacta probability = 1 / (n * (n-1))
        random_prob = 1 / (n_horses_avg * (n_horses_avg - 1))

        # Typical JRA takeout is ~25% for exacta
        # So break-even random odds = 1 / random_prob * 0.75
        breakeven_odds = 1 / random_prob * 0.75

        our_hit_rate = results.hit_rate / 100
        our_roi = results.roi / 100

        return {
            "random_prob": random_prob,
            "random_expected_hit_rate": random_prob,
            "our_hit_rate": our_hit_rate,
            "hit_rate_lift": our_hit_rate / random_prob if random_prob > 0 else 0,
            "random_roi": -0.25,  # -25% expected with takeout
            "our_roi": our_roi,
            "roi_improvement": our_roi - (-0.25),
        }


def main():
    """Run EV analysis on backtest results."""
    from .data_loader import RaceDataLoader
    from .backtester import Backtester

    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    # Run backtest with calibration
    backtester = Backtester(
        calibration_method='isotonic',
        filter_segments=True,
    )

    results, period_results = backtester.run_walkforward_backtest(
        df,
        n_periods=4,
        train_months=18,
        test_months=3,
        calibration_months=3,
    )

    # Analyze EV
    analyzer = EVAnalyzer()
    ev_analysis = analyzer.analyze(results)

    print(ev_analysis.summary())

    # Compare to random
    comparison = analyzer.compare_to_random(results)
    print("\n" + "=" * 60)
    print("COMPARISON TO RANDOM BETTING")
    print("=" * 60)
    print(f"Random exacta probability: {comparison['random_prob']:.4%}")
    print(f"Our hit rate: {comparison['our_hit_rate']:.2%}")
    print(f"Hit rate lift: {comparison['hit_rate_lift']:.1f}x")
    print(f"Random expected ROI: {comparison['random_roi']:.1%}")
    print(f"Our ROI: {comparison['our_roi']:.1%}")
    print(f"ROI improvement: +{comparison['roi_improvement']:.1%}")


if __name__ == "__main__":
    main()
