"""
Keiba AI Prediction System - Quinella Probability Calculation

Calculate Umaren (quinella/馬連) probabilities from position distributions.
Quinella = 1st and 2nd place, order doesn't matter.
"""

import logging
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuinellaCalculator:
    """
    Calculate Umaren (馬連/quinella) probabilities from position distributions.

    Quinella(A, B) = Exacta(A, B) + Exacta(B, A)
                   = P(A=1st)*P(B=2nd|A=1st) + P(B=1st)*P(A=2nd|B=1st)

    Using Harville formula:
    P(B=2nd | A=1st) = P(B=1st) / (1 - P(A=1st))
    """

    def __init__(
        self,
        min_probability: float = 0.001,
        max_combinations: int = 50,
    ):
        self.min_probability = min_probability
        self.max_combinations = max_combinations

    def _calculate_exacta_prob(
        self,
        position_probs: Dict[str, np.ndarray],
        first: str,
        second: str,
    ) -> float:
        """Calculate exacta probability using Harville formula."""
        p_first_wins = position_probs[first][0]
        p_second_given_first = position_probs[second][0] / (1 - p_first_wins + 1e-10)
        return p_first_wins * p_second_given_first

    def calculate_quinella_probs(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[FrozenSet[str], float]:
        """
        Calculate all quinella combination probabilities.

        Args:
            position_probs: Dict mapping horse_id to probability array
                           where array[0] = P(1st), array[1] = P(2nd), etc.

        Returns:
            Dict mapping frozenset({horse_a, horse_b}) to probability
        """
        quinella_probs = {}
        horses = list(position_probs.keys())

        for horse_a, horse_b in combinations(horses, 2):
            # Quinella = Exacta(A,B) + Exacta(B,A)
            exacta_ab = self._calculate_exacta_prob(position_probs, horse_a, horse_b)
            exacta_ba = self._calculate_exacta_prob(position_probs, horse_b, horse_a)

            quinella_prob = exacta_ab + exacta_ba

            if quinella_prob >= self.min_probability:
                quinella_probs[frozenset({horse_a, horse_b})] = quinella_prob

        return quinella_probs

    def calculate_quinella_probs_tuple(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate quinella probabilities with sorted tuple keys.
        Useful for compatibility with existing code.

        Returns:
            Dict mapping (smaller_horse, larger_horse) to probability
        """
        quinella_probs = self.calculate_quinella_probs(position_probs)
        return {
            tuple(sorted(k)): v
            for k, v in quinella_probs.items()
        }

    def get_top_quinellas(
        self,
        quinella_probs: Dict[FrozenSet[str], float],
        n: Optional[int] = None,
    ) -> List[Tuple[FrozenSet[str], float]]:
        """
        Get top N most likely quinella combinations.

        Args:
            quinella_probs: Dict of quinella probabilities
            n: Number of combinations to return (default: max_combinations)

        Returns:
            List of (frozenset{horse_a, horse_b}, probability) tuples
        """
        if n is None:
            n = self.max_combinations

        sorted_quinellas = sorted(
            quinella_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_quinellas[:n]

    def normalize_quinella_probs(
        self, quinella_probs: Dict[FrozenSet[str], float]
    ) -> Dict[FrozenSet[str], float]:
        """Normalize quinella probabilities to sum to 1."""
        total = sum(quinella_probs.values())
        if total <= 0:
            return quinella_probs

        return {k: v / total for k, v in quinella_probs.items()}


def main():
    """Test quinella calculation."""
    np.random.seed(42)

    horse_names = [f"Horse_{i+1}" for i in range(10)]

    # Create mock position probabilities
    raw_probs = np.random.dirichlet(np.ones(10) * 2, size=10)

    for _ in range(10):
        raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        raw_probs = raw_probs / raw_probs.sum(axis=0, keepdims=True)

    position_matrix = np.zeros((10, 18))
    position_matrix[:, :10] = raw_probs

    position_probs = {name: position_matrix[i] for i, name in enumerate(horse_names)}

    print("=" * 60)
    print("QUINELLA CALCULATION TEST")
    print("=" * 60)

    print("\nWin Probabilities:")
    for horse, probs in position_probs.items():
        print(f"  {horse}: {probs[0]:.2%}")

    calculator = QuinellaCalculator()
    quinella_probs = calculator.calculate_quinella_probs(position_probs)

    print(f"\nTotal quinella combinations: {len(quinella_probs)}")

    print("\nTop 10 Quinella Combinations:")
    top_quinellas = calculator.get_top_quinellas(quinella_probs, n=10)
    for horses, prob in top_quinellas:
        h1, h2 = sorted(horses)
        print(f"  {h1} - {h2}: {prob:.2%}")

    total_prob = sum(quinella_probs.values())
    print(f"\nTotal quinella probability sum: {total_prob:.4f}")


if __name__ == "__main__":
    main()
