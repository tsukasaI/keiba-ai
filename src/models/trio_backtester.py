"""
Keiba AI Prediction System - Trio Backtester

Walkforward backtesting for trio (三連複) betting with actual odds.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .calibrator import TrifectaCalibrator
from .config import (
    DATE_COL,
    FEATURES,
    PROFITABLE_SEGMENTS,
    RACE_ID_COL,
    TARGET_COL,
)
from .data_loader import RaceDataLoader
from .odds_loader import OddsLoader
from .position_model import PositionProbabilityModel
from .trio_calculator import TrioCalculator
from .types import TrioBetResult, TrioBacktestResults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrioBacktester:
    """
    Walkforward backtesting with actual trio odds.

    Trio = 3 horses in top 3, order doesn't matter.
    """

    def __init__(
        self,
        ev_threshold: float = 1.0,
        bet_unit: float = 100,
        max_bets_per_race: int = 10,
        min_probability: float = 0.005,
        calibration_method: Optional[str] = None,
        filter_segments: bool = False,
        custom_filters: Optional[Dict[str, List[str]]] = None,
    ):
        self.ev_threshold = ev_threshold
        self.bet_unit = bet_unit
        self.max_bets_per_race = max_bets_per_race
        self.min_probability = min_probability
        self.calibration_method = calibration_method
        self.filter_segments = filter_segments
        self.segment_filters = custom_filters or PROFITABLE_SEGMENTS

        self.trio_calc = TrioCalculator()
        self.odds_loader = OddsLoader()
        self.feature_cols = FEATURES.copy()
        self.calibrator: Optional[TrifectaCalibrator] = None

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        model: PositionProbabilityModel,
        trio_lookup: Dict[int, Tuple[int, int, int, float]],
    ) -> TrioBacktestResults:
        """
        Run backtest on test data.

        Args:
            test_df: Test data with features and race info
            model: Trained position prediction model
            trio_lookup: Dict mapping race_id to (h1, h2, h3, odds)

        Returns:
            TrioBacktestResults with all bet outcomes
        """
        results = TrioBacktestResults()

        race_groups = test_df.groupby(RACE_ID_COL)
        logger.info(f"Backtesting trio on {len(race_groups)} races...")

        for race_id, race_df in race_groups:
            if race_id not in trio_lookup:
                continue

            actual_h1, actual_h2, actual_h3, actual_odds = trio_lookup[race_id]

            if len(race_df) < 3:
                continue

            race_meta = self._get_race_metadata(race_df)

            bets = self._evaluate_race(
                race_df, model, actual_h1, actual_h2, actual_h3, actual_odds, race_meta
            )

            for bet in bets:
                results.bets.append(bet)
                results.total_bet += bet.bet_amount
                results.total_return += bet.payout
                results.num_bets += 1
                if bet.won:
                    results.num_wins += 1

        return results

    def _get_race_metadata(self, race_df: pd.DataFrame) -> Dict:
        """Extract race metadata for stratification."""
        row = race_df.iloc[0]

        surface = "turf" if row.get("is_turf", 0) == 1 else "dirt"

        distance = row.get("distance_num", 1600)
        if distance <= 1400:
            distance_cat = "sprint"
        elif distance <= 1800:
            distance_cat = "mile"
        elif distance <= 2200:
            distance_cat = "intermediate"
        else:
            distance_cat = "long"

        race_name = str(row.get("競争条件", ""))
        if "G1" in race_name or "G2" in race_name or "G3" in race_name:
            grade = "graded"
        elif "オープン" in race_name or "OP" in race_name:
            grade = "open"
        elif "新馬" in race_name or "未勝利" in race_name:
            grade = "maiden"
        else:
            grade = "conditions"

        condition_num = row.get("track_condition_num", 0)
        conditions = {0: "good", 1: "yielding", 2: "soft", 3: "heavy"}
        track_condition = conditions.get(condition_num, "good")

        racecourse = str(row.get("競馬場名", "unknown"))

        return {
            "surface": surface,
            "distance_category": distance_cat,
            "race_grade": grade,
            "track_condition": track_condition,
            "racecourse": racecourse,
            "race_date": str(row.get(DATE_COL, "")),
        }

    def _matches_segment_filter(self, race_meta: Dict) -> bool:
        """Check if race matches any profitable segment filter."""
        if not self.filter_segments:
            return True

        for field, allowed_values in self.segment_filters.items():
            if field in race_meta:
                if race_meta[field] in allowed_values:
                    return True

        return False

    def _evaluate_race(
        self,
        race_df: pd.DataFrame,
        model: PositionProbabilityModel,
        actual_h1: int,
        actual_h2: int,
        actual_h3: int,
        actual_odds: float,
        race_meta: Dict,
    ) -> List[TrioBetResult]:
        """Evaluate trio betting opportunities for a single race."""
        bets = []

        if not self._matches_segment_filter(race_meta):
            return bets

        X_race = race_df[self.feature_cols]
        horse_numbers = race_df["馬番"].tolist()

        position_probs = model.predict_race(
            X_race,
            horse_names=[str(h) for h in horse_numbers],
        )

        trio_probs = self.trio_calc.calculate_trio_probs(position_probs)

        top_trios = self.trio_calc.get_top_trios(
            trio_probs, n=self.max_bets_per_race
        )

        # Actual winning set
        actual_set = frozenset({actual_h1, actual_h2, actual_h3})

        for horse_set, pred_prob in top_trios:
            if pred_prob < self.min_probability:
                continue

            # Check if we won (order doesn't matter)
            pred_set_int = frozenset({int(h) for h in horse_set})
            won = pred_set_int == actual_set

            payout = actual_odds if won else 0
            profit = payout - self.bet_unit

            decimal_odds = actual_odds / 100
            post_hoc_ev = pred_prob * decimal_odds if won else 0

            h1, h2, h3 = sorted(horse_set)

            bet = TrioBetResult(
                race_id=race_df[RACE_ID_COL].iloc[0],
                race_date=race_meta["race_date"],
                horse1=str(h1),
                horse2=str(h2),
                horse3=str(h3),
                actual_1st=actual_h1,
                actual_2nd=actual_h2,
                actual_3rd=actual_h3,
                predicted_prob=pred_prob,
                actual_odds=actual_odds if won else 0,
                expected_value=post_hoc_ev,
                bet_amount=self.bet_unit,
                payout=payout,
                profit=profit,
                won=won,
                surface=race_meta["surface"],
                distance_category=race_meta["distance_category"],
                race_grade=race_meta["race_grade"],
                track_condition=race_meta["track_condition"],
                racecourse=race_meta["racecourse"],
            )
            bets.append(bet)

        return bets

    def run_walkforward_backtest(
        self,
        df: pd.DataFrame,
        n_periods: int = 4,
        train_months: int = 18,
        test_months: int = 3,
        calibration_months: int = 3,
    ) -> Tuple[TrioBacktestResults, List[Dict]]:
        """Run walkforward backtesting for trio."""
        logger.info("=" * 60)
        logger.info("TRIO WALKFORWARD BACKTEST")
        if self.calibration_method:
            logger.info(f"Calibration: {self.calibration_method}")
        logger.info("=" * 60)

        # Load odds
        self.odds_loader.load_odds()
        trio_df = self.odds_loader.get_trio_results()
        trio_lookup = {
            row["レースID"]: (
                row["trio_horse1"], row["trio_horse2"], row["trio_horse3"], row["trio_odds"]
            )
            for _, row in trio_df.iterrows()
        }

        # Parse dates
        df = df.copy()
        df["_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=["_date"]).sort_values("_date")

        min_date = df["_date"].min()
        max_date = df["_date"].max()
        logger.info(f"Data range: {min_date.date()} to {max_date.date()}")

        all_results = TrioBacktestResults()
        period_results = []

        current_start = min_date + pd.DateOffset(months=train_months)

        for period in range(n_periods):
            test_start = current_start
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > max_date:
                break

            train_end = test_start
            train_start = min_date

            if self.calibration_method:
                cal_start = train_end - pd.DateOffset(months=calibration_months)
                model_train_end = cal_start
            else:
                cal_start = None
                model_train_end = train_end

            logger.info(f"\n--- Period {period + 1}/{n_periods} ---")
            logger.info(f"Train: {train_start.date()} to {model_train_end.date()}")
            if self.calibration_method:
                logger.info(f"Calib: {cal_start.date()} to {train_end.date()}")
            logger.info(f"Test:  {test_start.date()} to {test_end.date()}")

            model_train_mask = (df["_date"] >= train_start) & (df["_date"] < model_train_end)
            test_mask = (df["_date"] >= test_start) & (df["_date"] < test_end)

            model_train_df = df[model_train_mask]
            test_df = df[test_mask]

            if len(model_train_df) < 1000 or len(test_df) < 100:
                logger.warning("Insufficient data, skipping period")
                current_start = test_end
                continue

            logger.info(f"Train samples: {len(model_train_df):,}")
            logger.info(f"Test samples: {len(test_df):,}")

            X_train = model_train_df[self.feature_cols]
            y_train = self._prepare_target(model_train_df[TARGET_COL])

            model = PositionProbabilityModel()
            model.train(X_train, y_train)

            period_result = self.run_backtest(test_df, model, trio_lookup)

            logger.info(f"Period ROI: {period_result.roi:+.1f}%")
            logger.info(f"Period bets: {period_result.num_bets}")
            logger.info(f"Period wins: {period_result.num_wins}")

            period_results.append({
                "period": period + 1,
                "train_start": train_start.date(),
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "num_bets": period_result.num_bets,
                "num_wins": period_result.num_wins,
                "total_bet": period_result.total_bet,
                "total_return": period_result.total_return,
                "roi": period_result.roi,
                "hit_rate": period_result.hit_rate,
            })

            all_results.bets.extend(period_result.bets)
            all_results.total_bet += period_result.total_bet
            all_results.total_return += period_result.total_return
            all_results.num_bets += period_result.num_bets
            all_results.num_wins += period_result.num_wins

            current_start = test_end

        return all_results, period_results

    def _prepare_target(self, target: pd.Series) -> pd.Series:
        """Convert position to 0-indexed class."""
        target = pd.to_numeric(target, errors="coerce")
        target = target.clip(lower=1, upper=18)
        return (target - 1).astype(int)


def main():
    """Run trio walkforward backtest."""
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    backtester = TrioBacktester(
        min_probability=0.005,
        max_bets_per_race=5,
        filter_segments=True,
    )
    results, period_results = backtester.run_walkforward_backtest(
        df,
        n_periods=4,
        train_months=18,
        test_months=3,
    )

    print("\n" + "=" * 60)
    print("TRIO BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nTotal Bets: {results.num_bets:,}")
    print(f"Winning Bets: {results.num_wins:,}")
    print(f"Hit Rate: {results.hit_rate:.2f}%")
    print(f"Avg Winning Odds: {results.avg_odds_hit:.1f}x")
    print(f"\nTotal Wagered: {results.total_bet:,.0f} yen")
    print(f"Total Return: {results.total_return:,.0f} yen")
    print(f"Profit/Loss: {results.profit:+,.0f} yen")
    print(f"ROI: {results.roi:+.1f}%")

    print("\n" + "-" * 60)
    print("RESULTS BY PERIOD")
    print("-" * 60)
    for p in period_results:
        print(f"Period {p['period']}: ROI={p['roi']:+.1f}%, Bets={p['num_bets']}, Wins={p['num_wins']}")

    # Random trio probability for 14 horses = 6 / (14 * 13 * 12) ≈ 0.27%
    random_prob = 6 / (14 * 13 * 12)
    our_hit_rate = results.hit_rate / 100
    print(f"\nRandom trio probability: {random_prob:.2%}")
    print(f"Our hit rate: {our_hit_rate:.2%}")
    print(f"Hit rate lift: {our_hit_rate / random_prob:.1f}x" if random_prob > 0 else "N/A")


if __name__ == "__main__":
    main()
