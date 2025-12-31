"""
Keiba AI Prediction System - Expected Value Calculation

Calculate expected value and identify value bets.
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .config import BETTING_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValueBet:
    """Represents a potential value bet."""

    combination: Tuple[str, str]  # (first_horse, second_horse)
    predicted_prob: float
    odds: float
    expected_value: float
    edge: float  # EV - 1.0 (profit margin)

    def __repr__(self):
        first, second = self.combination
        return (
            f"ValueBet({first}->{second}, "
            f"prob={self.predicted_prob:.2%}, "
            f"odds={self.odds:.1f}, "
            f"EV={self.expected_value:.2f}, "
            f"edge={self.edge:+.1%})"
        )


class ExpectedValueCalculator:
    """
    Calculate expected value for betting decisions.

    EV = predicted_probability * odds
    Bet only when EV > threshold (default: 1.0)
    """

    def __init__(
        self,
        ev_threshold: float = BETTING_CONFIG["ev_threshold"],
        bet_unit: int = BETTING_CONFIG["bet_unit"],
    ):
        self.ev_threshold = ev_threshold
        self.bet_unit = bet_unit

    def calculate_ev(self, probability: float, odds: float) -> float:
        """
        Calculate expected value.

        Args:
            probability: Predicted probability of outcome
            odds: Payout odds (e.g., 10.0 means 10x return)

        Returns:
            Expected value (EV > 1.0 indicates positive edge)
        """
        return probability * odds

    def find_value_bets(
        self,
        exacta_probs: Dict[Tuple[str, str], float],
        exacta_odds: Dict[Tuple[str, str], float],
    ) -> List[ValueBet]:
        """
        Find all bets with EV > threshold.

        Args:
            exacta_probs: Dict of (first, second) -> predicted probability
            exacta_odds: Dict of (first, second) -> payout odds

        Returns:
            List of ValueBet objects, sorted by EV descending
        """
        value_bets = []

        for combo, prob in exacta_probs.items():
            if combo not in exacta_odds:
                continue

            odds = exacta_odds[combo]
            ev = self.calculate_ev(prob, odds)

            if ev > self.ev_threshold:
                value_bets.append(ValueBet(
                    combination=combo,
                    predicted_prob=prob,
                    odds=odds,
                    expected_value=ev,
                    edge=ev - 1.0,
                ))

        # Sort by EV descending
        value_bets.sort(key=lambda x: x.expected_value, reverse=True)

        return value_bets

    def calculate_kelly_fraction(
        self, probability: float, odds: float
    ) -> float:
        """
        Calculate Kelly criterion bet fraction.

        Kelly fraction = (p * b - q) / b
        where:
            p = probability of winning
            b = odds - 1 (net odds)
            q = 1 - p (probability of losing)

        Returns:
            Optimal fraction of bankroll to bet (0 if negative EV)
        """
        if probability <= 0 or odds <= 1:
            return 0.0

        b = odds - 1  # Net odds
        q = 1 - probability

        kelly = (probability * b - q) / b

        # Don't bet if negative Kelly
        return max(0.0, kelly)

    def calculate_bet_size(
        self,
        probability: float,
        odds: float,
        bankroll: float,
        kelly_fraction: float = 0.25,  # Quarter Kelly for safety
    ) -> float:
        """
        Calculate recommended bet size using fractional Kelly.

        Args:
            probability: Predicted probability
            odds: Payout odds
            bankroll: Current bankroll
            kelly_fraction: Fraction of full Kelly to use (default: 0.25)

        Returns:
            Recommended bet amount
        """
        full_kelly = self.calculate_kelly_fraction(probability, odds)
        fraction = full_kelly * kelly_fraction

        bet_size = bankroll * fraction

        # Round to bet unit
        bet_size = max(self.bet_unit, round(bet_size / self.bet_unit) * self.bet_unit)

        return bet_size

    def summarize_value_bets(
        self, value_bets: List[ValueBet]
    ) -> Dict:
        """
        Summarize value bets for a race.

        Returns:
            Dict with summary statistics
        """
        if not value_bets:
            return {
                "num_bets": 0,
                "avg_ev": 0,
                "max_ev": 0,
                "avg_edge": 0,
                "total_probability": 0,
            }

        return {
            "num_bets": len(value_bets),
            "avg_ev": sum(b.expected_value for b in value_bets) / len(value_bets),
            "max_ev": max(b.expected_value for b in value_bets),
            "avg_edge": sum(b.edge for b in value_bets) / len(value_bets),
            "total_probability": sum(b.predicted_prob for b in value_bets),
            "best_bet": value_bets[0] if value_bets else None,
        }


def main():
    """Test expected value calculation."""
    # Example: Simulated exacta probabilities and odds
    exacta_probs = {
        ("Horse_1", "Horse_2"): 0.08,
        ("Horse_2", "Horse_1"): 0.06,
        ("Horse_1", "Horse_3"): 0.05,
        ("Horse_3", "Horse_1"): 0.04,
        ("Horse_2", "Horse_3"): 0.04,
        ("Horse_3", "Horse_2"): 0.03,
    }

    # Market odds (typically include ~25% takeout)
    exacta_odds = {
        ("Horse_1", "Horse_2"): 10.5,
        ("Horse_2", "Horse_1"): 14.2,
        ("Horse_1", "Horse_3"): 18.0,
        ("Horse_3", "Horse_1"): 22.5,
        ("Horse_2", "Horse_3"): 25.0,
        ("Horse_3", "Horse_2"): 30.0,
    }

    print("=" * 60)
    print("EXPECTED VALUE CALCULATION TEST")
    print("=" * 60)

    calculator = ExpectedValueCalculator()

    print("\nExacta Analysis:")
    print("-" * 60)
    print(f"{'Combination':<20} {'Prob':>8} {'Odds':>8} {'EV':>8} {'Edge':>8}")
    print("-" * 60)

    for combo, prob in exacta_probs.items():
        odds = exacta_odds.get(combo, 0)
        ev = calculator.calculate_ev(prob, odds)
        edge = ev - 1.0
        marker = " **" if ev > 1.0 else ""
        print(f"{str(combo):<20} {prob:>7.1%} {odds:>8.1f} {ev:>8.2f} {edge:>+7.1%}{marker}")

    print("-" * 60)

    # Find value bets
    value_bets = calculator.find_value_bets(exacta_probs, exacta_odds)

    print(f"\nValue Bets (EV > 1.0): {len(value_bets)}")
    for bet in value_bets:
        print(f"  {bet}")

    # Summary
    summary = calculator.summarize_value_bets(value_bets)
    print("\nSummary:")
    print(f"  Average EV: {summary['avg_ev']:.2f}")
    print(f"  Max EV: {summary['max_ev']:.2f}")
    print(f"  Average Edge: {summary['avg_edge']:.1%}")

    # Kelly sizing example
    print("\nKelly Bet Sizing (Bankroll = 100,000 yen):")
    bankroll = 100000
    for bet in value_bets[:3]:
        kelly = calculator.calculate_kelly_fraction(bet.predicted_prob, bet.odds)
        bet_size = calculator.calculate_bet_size(
            bet.predicted_prob, bet.odds, bankroll, kelly_fraction=0.25
        )
        print(f"  {bet.combination}: Kelly={kelly:.1%}, Bet={bet_size:,.0f} yen")


if __name__ == "__main__":
    main()
