"""
Keiba AI Prediction System - Backtest Report

Generate comprehensive backtest summary reports.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from .backtester import Backtester
from .calibrator import CalibrationAnalyzer
from .data_loader import RaceDataLoader
from .types import BacktestResults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BacktestReporter:
    """Generate comprehensive backtest reports."""

    def __init__(self, results: BacktestResults, period_results: List[Dict]):
        self.results = results
        self.period_results = period_results
        self.calibration_analyzer = CalibrationAnalyzer()

    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics."""
        return {
            "total_bets": self.results.num_bets,
            "total_wins": self.results.num_wins,
            "hit_rate": self.results.hit_rate,
            "avg_odds_hit": self.results.avg_odds_hit,
            "total_wagered": self.results.total_bet,
            "total_return": self.results.total_return,
            "profit": self.results.profit,
            "roi": self.results.roi,
            "max_drawdown": self.results.get_max_drawdown(),
            "sharpe_ratio": self.results.get_sharpe_ratio(),
        }

    def generate_period_summary(self) -> pd.DataFrame:
        """Generate period-by-period summary."""
        if not self.period_results:
            return pd.DataFrame()

        df = pd.DataFrame(self.period_results)
        return df

    def generate_stratified_summary(
        self, backtester: Backtester
    ) -> Dict[str, pd.DataFrame]:
        """Generate stratified performance summaries."""
        stratified = backtester.get_stratified_results(self.results)
        summaries = {}

        for field, field_results in stratified.items():
            rows = []
            for value, res in field_results.items():
                rows.append({
                    "category": value,
                    "bets": res.num_bets,
                    "wins": res.num_wins,
                    "hit_rate": res.hit_rate,
                    "total_bet": res.total_bet,
                    "profit": res.profit,
                    "roi": res.roi,
                })
            summaries[field] = pd.DataFrame(rows).sort_values("bets", ascending=False)

        return summaries

    def generate_monthly_returns(self) -> pd.DataFrame:
        """Generate monthly return series."""
        if not self.results.bets:
            return pd.DataFrame()

        # Group bets by month
        monthly_data = {}
        for bet in self.results.bets:
            try:
                date = pd.to_datetime(bet.race_date)
                month_key = date.strftime("%Y-%m")
            except Exception:
                continue

            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    "bets": 0, "wins": 0, "wagered": 0, "returns": 0
                }

            monthly_data[month_key]["bets"] += 1
            monthly_data[month_key]["wins"] += 1 if bet.won else 0
            monthly_data[month_key]["wagered"] += bet.bet_amount
            monthly_data[month_key]["returns"] += bet.payout

        rows = []
        for month, data in sorted(monthly_data.items()):
            profit = data["returns"] - data["wagered"]
            roi = (profit / data["wagered"] * 100) if data["wagered"] > 0 else 0
            rows.append({
                "month": month,
                "bets": data["bets"],
                "wins": data["wins"],
                "wagered": data["wagered"],
                "returns": data["returns"],
                "profit": profit,
                "roi": roi,
            })

        return pd.DataFrame(rows)

    def print_full_report(self, backtester: Backtester) -> None:
        """Print comprehensive backtest report."""
        print("\n")
        print("=" * 70)
        print("       KEIBA-AI BACKTEST REPORT")
        print(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Summary stats
        stats = self.generate_summary_stats()
        print("\n" + "=" * 70)
        print("OVERALL PERFORMANCE")
        print("=" * 70)
        print(f"{'Total Bets:':<25} {stats['total_bets']:>15,}")
        print(f"{'Winning Bets:':<25} {stats['total_wins']:>15,}")
        print(f"{'Hit Rate:':<25} {stats['hit_rate']:>14.1f}%")
        print(f"{'Avg Winning Odds:':<25} {stats['avg_odds_hit']:>15.1f}")
        print()
        print(f"{'Total Wagered:':<25} {stats['total_wagered']:>12,.0f} yen")
        print(f"{'Total Return:':<25} {stats['total_return']:>12,.0f} yen")
        print(f"{'Profit/Loss:':<25} {stats['profit']:>+12,.0f} yen")
        print(f"{'ROI:':<25} {stats['roi']:>+14.1f}%")
        print()
        print(f"{'Max Drawdown:':<25} {stats['max_drawdown']:>12,.0f} yen")
        print(f"{'Sharpe Ratio:':<25} {stats['sharpe_ratio']:>15.2f}")

        # Period results
        print("\n" + "=" * 70)
        print("PERFORMANCE BY PERIOD")
        print("=" * 70)
        print(f"{'Period':<10} {'Test Range':<25} {'Bets':>8} {'Wins':>8} {'ROI':>10}")
        print("-" * 70)
        for p in self.period_results:
            test_range = f"{p['test_start']} - {p['test_end']}"
            print(
                f"{p['period']:<10} {test_range:<25} "
                f"{p['num_bets']:>8} {p['num_wins']:>8} {p['roi']:>+9.1f}%"
            )

        # Stratified results
        stratified = self.generate_stratified_summary(backtester)

        print("\n" + "=" * 70)
        print("PERFORMANCE BY SURFACE")
        print("=" * 70)
        if "surface" in stratified:
            self._print_strat_table(stratified["surface"])

        print("\n" + "=" * 70)
        print("PERFORMANCE BY RACE GRADE")
        print("=" * 70)
        if "race_grade" in stratified:
            self._print_strat_table(stratified["race_grade"])

        print("\n" + "=" * 70)
        print("PERFORMANCE BY DISTANCE")
        print("=" * 70)
        if "distance_category" in stratified:
            self._print_strat_table(stratified["distance_category"])

        print("\n" + "=" * 70)
        print("PERFORMANCE BY TRACK CONDITION")
        print("=" * 70)
        if "track_condition" in stratified:
            self._print_strat_table(stratified["track_condition"])

        print("\n" + "=" * 70)
        print("PERFORMANCE BY RACECOURSE")
        print("=" * 70)
        if "racecourse" in stratified:
            self._print_strat_table(stratified["racecourse"])

        # Monthly returns
        monthly = self.generate_monthly_returns()
        if not monthly.empty:
            print("\n" + "=" * 70)
            print("MONTHLY PERFORMANCE")
            print("=" * 70)
            print(f"{'Month':<10} {'Bets':>8} {'Wins':>8} {'Profit':>12} {'ROI':>10}")
            print("-" * 50)
            for _, row in monthly.iterrows():
                print(
                    f"{row['month']:<10} {row['bets']:>8} {row['wins']:>8} "
                    f"{row['profit']:>+11,.0f} {row['roi']:>+9.1f}%"
                )

        # Top wins
        print("\n" + "=" * 70)
        print("TOP 10 WINNING BETS")
        print("=" * 70)
        winning_bets = sorted(
            [b for b in self.results.bets if b.won],
            key=lambda x: x.payout,
            reverse=True
        )[:10]

        print(f"{'Date':<12} {'Pred':<10} {'Actual':<10} {'Odds':>10} {'Profit':>12}")
        print("-" * 60)
        for bet in winning_bets:
            pred = f"{bet.predicted_1st}-{bet.predicted_2nd}"
            actual = f"{bet.actual_1st}-{bet.actual_2nd}"
            print(
                f"{bet.race_date:<12} {pred:<10} {actual:<10} "
                f"{bet.actual_odds:>10.1f} {bet.profit:>+11,.0f}"
            )

        print("\n" + "=" * 70)

    def _print_strat_table(self, df: pd.DataFrame) -> None:
        """Print stratification table."""
        print(f"{'Category':<15} {'Bets':>8} {'Wins':>8} {'Profit':>12} {'ROI':>10}")
        print("-" * 55)
        for _, row in df.iterrows():
            print(
                f"{str(row['category']):<15} {row['bets']:>8} {row['wins']:>8} "
                f"{row['profit']:>+11,.0f} {row['roi']:>+9.1f}%"
            )


def run_full_backtest_report(
    ev_threshold: float = 1.0,
    n_periods: int = 6,
    train_months: int = 18,
    test_months: int = 3,
    calibration_method: Optional[str] = None,
    filter_segments: bool = False,
) -> Dict:
    """
    Run complete backtest and generate full report.

    Args:
        ev_threshold: Minimum EV for betting
        n_periods: Number of test periods
        train_months: Training window size
        test_months: Test window size
        calibration_method: Calibration method ("platt", "isotonic", "binning", None)
        filter_segments: Only bet on profitable segments (福島, yielding, long distance)

    Returns:
        Dict with all results and analysis
    """
    logger.info("Loading data...")
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    logger.info("Running walkforward backtest...")
    if calibration_method:
        logger.info(f"Using calibration: {calibration_method}")
    if filter_segments:
        logger.info("Filtering to profitable segments only")
    backtester = Backtester(
        ev_threshold=ev_threshold,
        calibration_method=calibration_method,
        filter_segments=filter_segments,
    )
    results, period_results = backtester.run_walkforward_backtest(
        df,
        n_periods=n_periods,
        train_months=train_months,
        test_months=test_months,
    )

    # Generate report
    reporter = BacktestReporter(results, period_results)
    reporter.print_full_report(backtester)

    # Run calibration analysis
    print("\n")
    calibration = _run_calibration_analysis(results)

    return {
        "results": results,
        "period_results": period_results,
        "calibration": calibration,
        "summary": reporter.generate_summary_stats(),
        "stratified": reporter.generate_stratified_summary(backtester),
        "monthly": reporter.generate_monthly_returns(),
    }


def _run_calibration_analysis(results: BacktestResults) -> Dict:
    """Run calibration analysis on backtest results."""

    if not results.bets:
        return {}

    probs = np.array([b.predicted_prob for b in results.bets])
    labels = np.array([1 if b.won else 0 for b in results.bets])

    analyzer = CalibrationAnalyzer(n_bins=10)
    bins = analyzer.analyze(probs, labels)
    brier = analyzer.brier_score(probs, labels)
    ece = analyzer.expected_calibration_error(bins)
    issues = analyzer.detect_issues(bins)

    # Print report
    print("=" * 60)
    print("CALIBRATION ANALYSIS REPORT")
    print("=" * 60)
    print("\n--- Overall Metrics ---")
    print(f"Brier Score:                    {brier:.4f}")
    print(f"Expected Calibration Error:     {ece:.4f}")

    print("\n--- Calibration by Probability Bin ---")
    print(f"{'Bin':>12} {'Samples':>10} {'Predicted':>12} {'Actual':>10} {'Error':>10}")
    print("-" * 56)
    for b in bins:
        print(f"{b.bin_start:.2%}-{b.bin_end:.2%}{b.num_samples:>10}"
              f"{b.avg_predicted_prob:>12.2%}{b.actual_hit_rate:>10.2%}{b.confidence_error:>+10.2%}")

    print("\n--- Calibration Issues ---")
    for category, message in issues.items():
        print(f"{category}: {message}")

    # EV tier analysis
    tiers = {"high_ev": (1.5, float("inf")), "medium_ev": (1.2, 1.5), "low_ev": (1.0, 1.2)}
    print("\n--- Calibration by EV Tier ---")
    print(f"{'Tier':>12} {'Bets':>8} {'Predicted':>12} {'Actual':>10} {'Error':>10}")
    print("-" * 54)

    for tier, (lo, hi) in tiers.items():
        tier_bets = [b for b in results.bets if lo <= b.expected_value < hi]
        if tier_bets:
            t_probs = np.array([b.predicted_prob for b in tier_bets])
            t_labels = np.array([1 if b.won else 0 for b in tier_bets])
            print(f"{tier:>12}{len(tier_bets):>8}{t_probs.mean():>12.2%}"
                  f"{t_labels.mean():>10.2%}{t_probs.mean() - t_labels.mean():>+10.2%}")

    print("=" * 60)

    return {"bins": bins, "brier_score": brier, "ece": ece, "issues": issues}


def main():
    """Run backtest with full report."""
    import sys

    # Parse command line args
    calibration_method = None
    filter_segments = False

    for arg in sys.argv[1:]:
        if arg == "--filter":
            filter_segments = True
        elif arg in ["platt", "isotonic", "binning"]:
            calibration_method = arg
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python -m src.models.backtest_report [platt|isotonic|binning] [--filter]")
            print("  platt/isotonic/binning: Calibration method")
            print("  --filter: Only bet on profitable segments (福島, yielding, long)")
            sys.exit(1)

    run_full_backtest_report(
        calibration_method=calibration_method,
        filter_segments=filter_segments,
    )


if __name__ == "__main__":
    main()
