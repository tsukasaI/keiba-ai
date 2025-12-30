"""
競馬AI予測システム - モデル評価

Evaluation metrics, calibration analysis, and ROI simulation.
"""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    RACE_ID_COL,
    HORSE_NAME_COL,
    TARGET_COL,
    BETTING_CONFIG,
)
from .position_model import PositionProbabilityModel
from .exacta_calculator import ExactaCalculator
from .expected_value import ExpectedValueCalculator, ValueBet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate model performance with horse racing-specific metrics.
    """

    def __init__(
        self,
        ev_threshold: float = BETTING_CONFIG["ev_threshold"],
        bet_unit: int = BETTING_CONFIG["bet_unit"],
    ):
        self.ev_threshold = ev_threshold
        self.bet_unit = bet_unit
        self.exacta_calc = ExactaCalculator()
        self.ev_calc = ExpectedValueCalculator(ev_threshold=ev_threshold)

    def calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Calculate basic classification metrics.

        Args:
            y_true: True position labels (0-indexed)
            y_pred_proba: Predicted probabilities (n_samples, 18)

        Returns:
            Dict of metrics
        """
        # Top-1 accuracy (win prediction)
        preds = y_pred_proba.argmax(axis=1)
        top1_accuracy = (preds == y_true).mean()

        # Top-3 accuracy (place prediction)
        top3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]
        top3_accuracy = np.mean([
            y in top3 for y, top3 in zip(y_true, top3_preds)
        ])

        # Log loss
        eps = 1e-10
        probs_clipped = np.clip(y_pred_proba, eps, 1 - eps)
        log_loss = -np.mean([
            np.log(probs_clipped[i, y])
            for i, y in enumerate(y_true)
        ])

        # Average predicted probability for actual position
        avg_correct_prob = np.mean([
            y_pred_proba[i, y] for i, y in enumerate(y_true)
        ])

        return {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "log_loss": log_loss,
            "avg_correct_prob": avg_correct_prob,
        }

    def calculate_exacta_accuracy(
        self,
        test_df: pd.DataFrame,
        model: PositionProbabilityModel,
        feature_cols: List[str],
    ) -> Dict:
        """
        Calculate exacta prediction accuracy.

        Args:
            test_df: Test data with race groupings
            model: Trained model
            feature_cols: Feature column names

        Returns:
            Dict of exacta-related metrics
        """
        race_groups = test_df.groupby(RACE_ID_COL)
        n_races = len(race_groups)

        correct_exactas = 0
        correct_win = 0
        correct_place = 0
        total_races = 0

        for race_id, race_df in race_groups:
            if len(race_df) < 2:
                continue

            total_races += 1

            # Get actual results
            race_df = race_df.sort_values(TARGET_COL)
            actual_positions = race_df[TARGET_COL].values

            # Skip if missing position data
            if pd.isna(actual_positions[:2]).any():
                continue

            actual_1st_idx = race_df.index[0]
            actual_2nd_idx = race_df.index[1]
            actual_1st = race_df.loc[actual_1st_idx, HORSE_NAME_COL]
            actual_2nd = race_df.loc[actual_2nd_idx, HORSE_NAME_COL]

            # Get predictions
            X_race = race_df[feature_cols]
            horse_names = race_df[HORSE_NAME_COL].tolist()

            position_probs = model.predict_race(X_race, horse_names)
            exacta_probs = self.exacta_calc.calculate_exacta_probs(position_probs)

            if not exacta_probs:
                continue

            # Get predicted 1st and 2nd (highest probability exacta)
            top_exacta = self.exacta_calc.get_top_exactas(exacta_probs, n=1)
            if top_exacta:
                (pred_1st, pred_2nd), _ = top_exacta[0]

                # Check if correct
                if pred_1st == actual_1st and pred_2nd == actual_2nd:
                    correct_exactas += 1

                if pred_1st == actual_1st:
                    correct_win += 1

                if pred_1st == actual_1st or pred_1st == actual_2nd:
                    correct_place += 1

        return {
            "total_races": total_races,
            "exacta_accuracy": correct_exactas / total_races if total_races > 0 else 0,
            "win_accuracy": correct_win / total_races if total_races > 0 else 0,
            "place_accuracy": correct_place / total_races if total_races > 0 else 0,
            "correct_exactas": correct_exactas,
        }

    def simulate_roi(
        self,
        test_df: pd.DataFrame,
        model: PositionProbabilityModel,
        feature_cols: List[str],
        odds_col: str = "単勝",
    ) -> Dict:
        """
        Simulate ROI using EV > threshold betting strategy.

        Note: This simulation uses single-win odds as a proxy for exacta odds
        since exacta odds are not directly available in the dataset.
        For a more accurate simulation, exacta odds would be needed.

        Args:
            test_df: Test data
            model: Trained model
            feature_cols: Feature columns
            odds_col: Column containing odds

        Returns:
            Dict with ROI simulation results
        """
        race_groups = test_df.groupby(RACE_ID_COL)

        total_bet = 0
        total_return = 0
        num_bets = 0
        winning_bets = 0
        all_bets = []

        for race_id, race_df in race_groups:
            if len(race_df) < 2:
                continue

            # Get actual results
            race_df = race_df.sort_values(TARGET_COL)
            if pd.isna(race_df[TARGET_COL].values[:2]).any():
                continue

            actual_1st = race_df.iloc[0][HORSE_NAME_COL]
            actual_2nd = race_df.iloc[1][HORSE_NAME_COL]
            actual_exacta = (actual_1st, actual_2nd)

            # Get predictions
            X_race = race_df[feature_cols]
            horse_names = race_df[HORSE_NAME_COL].tolist()

            position_probs = model.predict_race(X_race, horse_names)
            exacta_probs = self.exacta_calc.calculate_exacta_probs(position_probs)

            if not exacta_probs:
                continue

            # Create synthetic exacta odds from win odds
            # Approximate: exacta_odds ≈ win_odds_1st * win_odds_2nd * 0.8 (takeout)
            horse_odds = dict(zip(race_df[HORSE_NAME_COL], race_df[odds_col]))
            exacta_odds = {}
            for (h1, h2), _ in exacta_probs.items():
                if h1 in horse_odds and h2 in horse_odds:
                    o1 = horse_odds[h1]
                    o2 = horse_odds[h2]
                    if pd.notna(o1) and pd.notna(o2):
                        # Synthetic exacta odds (rough approximation)
                        exacta_odds[(h1, h2)] = o1 * o2 * 0.7

            # Find value bets
            value_bets = self.ev_calc.find_value_bets(exacta_probs, exacta_odds)

            for bet in value_bets:
                total_bet += self.bet_unit
                num_bets += 1

                if bet.combination == actual_exacta:
                    payout = self.bet_unit * bet.odds
                    total_return += payout
                    winning_bets += 1

                all_bets.append({
                    "race_id": race_id,
                    "combination": bet.combination,
                    "predicted_prob": bet.predicted_prob,
                    "odds": bet.odds,
                    "ev": bet.expected_value,
                    "won": bet.combination == actual_exacta,
                    "payout": self.bet_unit * bet.odds if bet.combination == actual_exacta else 0,
                })

        # Calculate ROI
        roi = (total_return - total_bet) / total_bet * 100 if total_bet > 0 else 0
        hit_rate = winning_bets / num_bets * 100 if num_bets > 0 else 0

        return {
            "total_bet": total_bet,
            "total_return": total_return,
            "profit": total_return - total_bet,
            "roi_percent": roi,
            "num_bets": num_bets,
            "winning_bets": winning_bets,
            "hit_rate_percent": hit_rate,
            "avg_bet_ev": np.mean([b["ev"] for b in all_bets]) if all_bets else 0,
            "bets_detail": all_bets,
        }

    def full_evaluation(
        self,
        test_df: pd.DataFrame,
        model: PositionProbabilityModel,
        feature_cols: List[str],
    ) -> Dict:
        """
        Run full evaluation suite.

        Args:
            test_df: Test data
            model: Trained model
            feature_cols: Feature columns

        Returns:
            Dict with all evaluation results
        """
        logger.info("Running full evaluation...")

        # Basic metrics
        X_test = test_df[feature_cols]
        y_test = pd.to_numeric(test_df[TARGET_COL], errors="coerce")
        y_test = (y_test.clip(lower=1, upper=18) - 1).astype(int)

        probs = model.predict_proba(X_test)
        basic_metrics = self.calculate_basic_metrics(y_test.values, probs)

        # Exacta accuracy
        exacta_metrics = self.calculate_exacta_accuracy(test_df, model, feature_cols)

        # ROI simulation
        roi_results = self.simulate_roi(test_df, model, feature_cols)

        results = {
            **basic_metrics,
            **exacta_metrics,
            **{k: v for k, v in roi_results.items() if k != "bets_detail"},
        }

        return results

    def print_evaluation_report(self, results: Dict) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)

        print("\n--- Basic Metrics ---")
        print(f"Top-1 Accuracy (Win):   {results.get('top1_accuracy', 0):.2%}")
        print(f"Top-3 Accuracy (Place): {results.get('top3_accuracy', 0):.2%}")
        print(f"Log Loss:               {results.get('log_loss', 0):.4f}")
        print(f"Avg Correct Prob:       {results.get('avg_correct_prob', 0):.2%}")

        print("\n--- Exacta Metrics ---")
        print(f"Total Races:            {results.get('total_races', 0):,}")
        print(f"Exacta Accuracy:        {results.get('exacta_accuracy', 0):.2%}")
        print(f"Win Accuracy:           {results.get('win_accuracy', 0):.2%}")
        print(f"Place Accuracy:         {results.get('place_accuracy', 0):.2%}")

        print("\n--- ROI Simulation (EV > 1.0 Strategy) ---")
        print(f"Total Bets:             {results.get('num_bets', 0):,}")
        print(f"Total Wagered:          {results.get('total_bet', 0):,.0f} yen")
        print(f"Total Return:           {results.get('total_return', 0):,.0f} yen")
        print(f"Profit/Loss:            {results.get('profit', 0):+,.0f} yen")
        print(f"ROI:                    {results.get('roi_percent', 0):+.1f}%")
        print(f"Hit Rate:               {results.get('hit_rate_percent', 0):.1f}%")
        print(f"Avg Bet EV:             {results.get('avg_bet_ev', 0):.2f}")

        print("=" * 60)


def main():
    """Run evaluation on test data."""
    from .data_loader import RaceDataLoader
    from .trainer import ModelTrainer

    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = loader.filter_valid_races(df)
    df = loader.handle_missing_values(df)

    # Train final model
    trainer = ModelTrainer()
    model, test_df = trainer.train_final_model(df, loader.feature_cols)

    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.full_evaluation(test_df, model, loader.feature_cols)
    evaluator.print_evaluation_report(results)


if __name__ == "__main__":
    main()
