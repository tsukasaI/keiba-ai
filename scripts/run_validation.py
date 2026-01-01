#!/usr/bin/env python
"""
Run validation backtest with the new 39-feature model.

Usage:
    python scripts/run_validation.py

This script:
1. Loads the features data
2. Runs a walk-forward backtest with isotonic calibration
3. Compares results against the baseline (23 features, +19.3% ROI)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backtester import Backtester
from src.models.data_loader import RaceDataLoader
from src.models.config import FEATURES, TARGET_COL


def main():
    print("=" * 60)
    print("MODEL VALIDATION - 39 FEATURES")
    print("=" * 60)
    print(f"\nUsing {len(FEATURES)} features:")
    for i, f in enumerate(FEATURES):
        print(f"  {i+1:2}. {f}")

    # Load data
    print("\n" + "-" * 60)
    print("Loading data...")
    loader = RaceDataLoader()
    df = loader.load_features()
    df = df[df[TARGET_COL].notna()]
    print(f"Loaded {len(df):,} rows")

    # Create backtester with isotonic calibration
    print("\n" + "-" * 60)
    print("Running walk-forward backtest with isotonic calibration...")
    backtester = Backtester(
        calibration_method="isotonic",
    )

    # Run walk-forward backtest
    results, period_results = backtester.run_walkforward_backtest(
        df=df,
        n_periods=6,
        train_months=18,
        test_months=3,
        calibration_months=3,
    )

    # Calculate summary
    total_bets = len(results.bets)
    total_wins = sum(1 for b in results.bets if b.won)
    total_wagered = sum(b.bet_amount for b in results.bets)
    total_return = sum(b.payout for b in results.bets)
    profit = total_return - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    hit_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (39 FEATURES + ISOTONIC CALIBRATION)")
    print("=" * 60)
    print(f"""
Total Bets: {total_bets:,}
Total Wins: {total_wins:,}
Hit Rate: {hit_rate:.2f}%

Total Wagered: ¥{total_wagered:,.0f}
Total Return: ¥{total_return:,.0f}
Profit/Loss: ¥{profit:,.0f}

ROI: {roi:.2f}%
""")

    print("Period results:")
    for i, period in enumerate(period_results):
        print(f"  Period {i+1}: ROI = {period.get('roi', 0):.1f}%, Bets = {period.get('bets', 0)}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    baseline_roi = 19.3
    print(f"  Baseline (23 features): +{baseline_roi:.1f}% ROI")
    print(f"  New (39 features):      +{roi:.1f}% ROI")
    print(f"  Improvement:            +{roi - baseline_roi:.1f} percentage points")

    if roi > baseline_roi:
        print("\n✓ Model improvement validated!")
    else:
        print("\n✗ No improvement detected")

    return roi


if __name__ == "__main__":
    main()
