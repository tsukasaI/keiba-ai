"""
Keiba AI Prediction System - Trifecta Backtester

Walkforward backtesting for trifecta (三連単) betting with actual odds.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .calibrator import TrifectaCalibrator
from .config import (
    BETTING_CONFIG,
    DATE_COL,
    FEATURES,
    PROFITABLE_SEGMENTS,
    RACE_ID_COL,
    TARGET_COL,
)
from .data_loader import RaceDataLoader
from .odds_loader import OddsLoader
from .position_model import PositionProbabilityModel
from .trifecta_calculator import TrifectaCalculator
from .types import TrifectaBetResult, TrifectaBacktestResults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrifectaBacktester:
    """
    Walkforward backtesting with actual trifecta odds.

    Strategy: Bet on top trifecta predictions, optionally filtered to profitable segments.
    """

    def __init__(
        self,
        ev_threshold: float = BETTING_CONFIG["ev_threshold"],
        bet_unit: float = BETTING_CONFIG["bet_unit"],
        max_bets_per_race: int = 10,  # More combinations for trifecta
        min_probability: float = 0.005,  # Lower threshold for trifecta (0.5%)
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

        self.trifecta_calc = TrifectaCalculator()
        self.odds_loader = OddsLoader()
        self.feature_cols = FEATURES.copy()
        self.calibrator: Optional[TrifectaCalibrator] = None

    def run_backtest(
        self,
        test_df: pd.DataFrame,
        model: PositionProbabilityModel,
        odds_lookup: Dict[int, Dict],
    ) -> TrifectaBacktestResults:
        """
        Run backtest on test data.

        Args:
            test_df: Test data with features and race info
            model: Trained position prediction model
            odds_lookup: Dict mapping race_id to odds data

        Returns:
            TrifectaBacktestResults with all bet outcomes
        """
        results = TrifectaBacktestResults()

        race_groups = test_df.groupby(RACE_ID_COL)
        logger.info(f"Backtesting trifecta on {len(race_groups)} races...")

        for race_id, race_df in race_groups:
            # Skip if no odds data
            if race_id not in odds_lookup:
                continue

            odds_data = odds_lookup[race_id]
            if "trifecta" not in odds_data:
                continue

            actual_1st, actual_2nd, actual_3rd, actual_odds = odds_data["trifecta"]

            # Skip invalid races (need at least 3 horses)
            if len(race_df) < 3:
                continue

            # Get race metadata for stratification
            race_meta = self._get_race_metadata(race_df)

            # Make predictions
            bets = self._evaluate_race(
                race_df, model, actual_1st, actual_2nd, actual_3rd, actual_odds, race_meta
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

        # Surface
        surface = "turf" if row.get("is_turf", 0) == 1 else "dirt"

        # Distance category
        distance = row.get("distance_num", 1600)
        if distance <= 1400:
            distance_cat = "sprint"
        elif distance <= 1800:
            distance_cat = "mile"
        elif distance <= 2200:
            distance_cat = "intermediate"
        else:
            distance_cat = "long"

        # Race grade
        race_name = str(row.get("競争条件", ""))
        if "G1" in race_name or "G2" in race_name or "G3" in race_name:
            grade = "graded"
        elif "オープン" in race_name or "OP" in race_name:
            grade = "open"
        elif "新馬" in race_name or "未勝利" in race_name:
            grade = "maiden"
        else:
            grade = "conditions"

        # Track condition
        condition_num = row.get("track_condition_num", 0)
        conditions = {0: "good", 1: "yielding", 2: "soft", 3: "heavy"}
        track_condition = conditions.get(condition_num, "good")

        # Racecourse
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
        actual_1st: int,
        actual_2nd: int,
        actual_3rd: int,
        actual_odds: float,
        race_meta: Dict,
    ) -> List[TrifectaBetResult]:
        """
        Evaluate trifecta betting opportunities for a single race.
        """
        bets = []

        # Skip if race doesn't match segment filter
        if not self._matches_segment_filter(race_meta):
            return bets

        # Get features
        X_race = race_df[self.feature_cols]
        horse_numbers = race_df["馬番"].tolist()

        # Predict position probabilities
        position_probs = model.predict_race(
            X_race,
            horse_names=[str(h) for h in horse_numbers],
        )

        # Calculate trifecta probabilities
        trifecta_probs = self.trifecta_calc.calculate_trifecta_probs(position_probs)

        # Apply calibration if available
        if self.calibrator is not None and self.calibrator.fitted:
            trifecta_probs = self.calibrator.calibrate_dict(trifecta_probs)

        # Get top trifectas by probability
        top_trifectas = self.trifecta_calc.get_top_trifectas(
            trifecta_probs, n=self.max_bets_per_race
        )

        for (pred_1st_str, pred_2nd_str, pred_3rd_str), pred_prob in top_trifectas:
            pred_1st = int(pred_1st_str)
            pred_2nd = int(pred_2nd_str)
            pred_3rd = int(pred_3rd_str)

            # Only bet if predicted probability exceeds minimum threshold
            if pred_prob < self.min_probability:
                continue

            # Check if we won
            won = (
                pred_1st == actual_1st and
                pred_2nd == actual_2nd and
                pred_3rd == actual_3rd
            )

            # Trifecta odds are in Japanese format (payout per 100 yen bet)
            payout = actual_odds if won else 0
            profit = payout - self.bet_unit

            # Calculate post-hoc EV for analysis
            decimal_odds = actual_odds / 100
            post_hoc_ev = pred_prob * decimal_odds if won else 0

            bet = TrifectaBetResult(
                race_id=race_df[RACE_ID_COL].iloc[0],
                race_date=race_meta["race_date"],
                predicted_1st=pred_1st_str,
                predicted_2nd=pred_2nd_str,
                predicted_3rd=pred_3rd_str,
                actual_1st=actual_1st,
                actual_2nd=actual_2nd,
                actual_3rd=actual_3rd,
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
    ) -> Tuple[TrifectaBacktestResults, List[Dict]]:
        """
        Run walkforward backtesting for trifecta.

        Args:
            df: Full dataset with features
            n_periods: Number of test periods
            train_months: Months of training data
            test_months: Months of test data per period
            calibration_months: Months reserved for calibration (from end of train)

        Returns:
            (aggregate_results, period_results)
        """
        logger.info("=" * 60)
        logger.info("TRIFECTA WALKFORWARD BACKTEST")
        if self.calibration_method:
            logger.info(f"Calibration: {self.calibration_method}")
        if self.filter_segments:
            logger.info(f"Segment filter: {self.segment_filters}")
        logger.info("=" * 60)

        # Load odds (including trifecta)
        self.odds_loader.load_odds()
        odds_lookup = self.odds_loader.create_race_odds_lookup()

        # Parse dates
        df = df.copy()
        df["_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=["_date"]).sort_values("_date")

        min_date = df["_date"].min()
        max_date = df["_date"].max()
        logger.info(f"Data range: {min_date.date()} to {max_date.date()}")

        # Calculate period boundaries
        all_results = TrifectaBacktestResults()
        period_results = []

        # Start after initial training period
        current_start = min_date + pd.DateOffset(months=train_months)

        for period in range(n_periods):
            test_start = current_start
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > max_date:
                break

            train_end = test_start
            train_start = min_date

            # For calibration, split training data
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

            # Split data
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

            # Train model
            X_train = model_train_df[self.feature_cols]
            y_train = self._prepare_target(model_train_df[TARGET_COL])

            model = PositionProbabilityModel()
            model.train(X_train, y_train)

            # Train calibrator if specified
            if self.calibration_method:
                cal_mask = (df["_date"] >= cal_start) & (df["_date"] < train_end)
                cal_df = df[cal_mask]

                self.calibrator = TrifectaCalibrator(method=self.calibration_method)
                self._train_calibrator(cal_df, model, odds_lookup)
                logger.info(f"Calibrator trained on {len(cal_df):,} samples")

            # Run backtest
            period_result = self.run_backtest(test_df, model, odds_lookup)

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

            # Aggregate results
            all_results.bets.extend(period_result.bets)
            all_results.total_bet += period_result.total_bet
            all_results.total_return += period_result.total_return
            all_results.num_bets += period_result.num_bets
            all_results.num_wins += period_result.num_wins

            # Move to next period
            current_start = test_end

        return all_results, period_results

    def _prepare_target(self, target: pd.Series) -> pd.Series:
        """Convert position to 0-indexed class."""
        target = pd.to_numeric(target, errors="coerce")
        target = target.clip(lower=1, upper=18)
        return (target - 1).astype(int)

    def _train_calibrator(
        self,
        cal_df: pd.DataFrame,
        model: PositionProbabilityModel,
        odds_lookup: Dict,
    ) -> None:
        """Train calibrator on calibration data."""
        predictions = []
        race_groups = cal_df.groupby(RACE_ID_COL)

        for race_id, race_df in race_groups:
            if race_id not in odds_lookup:
                continue

            odds_data = odds_lookup[race_id]
            if "trifecta" not in odds_data:
                continue

            actual_1st, actual_2nd, actual_3rd, _ = odds_data["trifecta"]

            if len(race_df) < 3:
                continue

            X_race = race_df[self.feature_cols]
            horse_numbers = race_df["馬番"].tolist()

            position_probs = model.predict_race(
                X_race,
                horse_names=[str(h) for h in horse_numbers],
            )

            trifecta_probs = self.trifecta_calc.calculate_trifecta_probs(position_probs)

            # Collect all predictions (not just top ones) for calibration
            for (h1, h2, h3), prob in trifecta_probs.items():
                won = (int(h1) == actual_1st) and (int(h2) == actual_2nd) and (int(h3) == actual_3rd)
                predictions.append({
                    "predicted_prob": prob,
                    "won": won,
                })

        self.calibrator.fit_from_backtest(predictions)

    def get_stratified_results(
        self, results: TrifectaBacktestResults
    ) -> Dict[str, Dict[str, TrifectaBacktestResults]]:
        """Group results by stratification fields."""
        strat_fields = [
            "surface", "distance_category", "race_grade",
            "track_condition", "racecourse"
        ]

        stratified = {}

        for field in strat_fields:
            field_results = defaultdict(TrifectaBacktestResults)

            for bet in results.bets:
                value = getattr(bet, field, "unknown")
                field_results[value].bets.append(bet)
                field_results[value].total_bet += bet.bet_amount
                field_results[value].total_return += bet.payout
                field_results[value].num_bets += 1
                if bet.won:
                    field_results[value].num_wins += 1

            stratified[field] = dict(field_results)

        return stratified


def main():
    """Run trifecta walkforward backtest."""
    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    # Run backtest with calibration (lower threshold since calibrated probs are lower)
    backtester = TrifectaBacktester(
        min_probability=0.005,  # 0.5% threshold (calibrated probs are lower)
        max_bets_per_race=5,   # Top 5 predictions per race
        calibration_method='isotonic',  # Use isotonic calibration
        filter_segments=True,  # Use profitable segments
    )
    results, period_results = backtester.run_walkforward_backtest(
        df,
        n_periods=4,
        train_months=18,
        test_months=3,
        calibration_months=3,
    )

    # Print results
    print("\n" + "=" * 60)
    print("TRIFECTA BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nTotal Bets: {results.num_bets:,}")
    print(f"Winning Bets: {results.num_wins:,}")
    print(f"Hit Rate: {results.hit_rate:.2f}%")
    print(f"Avg Winning Odds: {results.avg_odds_hit:.1f}x")
    print(f"\nTotal Wagered: {results.total_bet:,.0f} yen")
    print(f"Total Return: {results.total_return:,.0f} yen")
    print(f"Profit/Loss: {results.profit:+,.0f} yen")
    print(f"ROI: {results.roi:+.1f}%")

    # Period results
    print("\n" + "-" * 60)
    print("RESULTS BY PERIOD")
    print("-" * 60)
    for p in period_results:
        print(f"Period {p['period']}: ROI={p['roi']:+.1f}%, Bets={p['num_bets']}, Wins={p['num_wins']}")

    # Compare to random
    print("\n" + "-" * 60)
    print("COMPARISON TO RANDOM")
    print("-" * 60)
    # Random trifecta probability for 14 horses = 1 / (14 * 13 * 12) ≈ 0.046%
    random_prob = 1 / (14 * 13 * 12)
    our_hit_rate = results.hit_rate / 100
    print(f"Random trifecta probability: {random_prob:.4%}")
    print(f"Our hit rate: {our_hit_rate:.4%}")
    print(f"Hit rate lift: {our_hit_rate / random_prob:.1f}x" if random_prob > 0 else "N/A")


if __name__ == "__main__":
    main()
