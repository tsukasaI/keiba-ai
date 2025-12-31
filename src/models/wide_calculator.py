"""
Keiba AI Prediction System - Wide Probability Calculation

Calculate Wide (ワイド) probabilities from position distributions.
Wide = 2 horses both finish in top 3, order doesn't matter.
"""

import logging
from itertools import combinations, permutations
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WideCalculator:
    """
    Calculate Wide (ワイド) probabilities from position distributions.

    Wide(A, B) = P(A and B both in top 3)
               = Sum over all C of Trio(A, B, C)
               = Sum over all C of [6 trifecta permutations of A, B, C]

    This is equivalent to:
    - Quinella(A,B) [both in top 2]
    - Plus: one in top 2, the other 3rd
    """

    def __init__(
        self,
        min_probability: float = 0.001,
        max_combinations: int = 50,
    ):
        self.min_probability = min_probability
        self.max_combinations = max_combinations

    def _calculate_trifecta_prob(
        self,
        position_probs: Dict[str, np.ndarray],
        first: str,
        second: str,
        third: str,
    ) -> float:
        """Calculate single trifecta probability using Harville formula."""
        p_first = position_probs[first][0]
        p_second = position_probs[second][0]
        p_third = position_probs[third][0]

        # P(B=2nd | A=1st) = P(B) / (1 - P(A))
        p_second_given_first = p_second / (1 - p_first + 1e-10)

        # P(C=3rd | A,B) = P(C) / (1 - P(A) - P(B))
        remaining_prob = 1 - p_first - p_second + 1e-10
        p_third_given_first_second = p_third / remaining_prob

        return p_first * p_second_given_first * p_third_given_first_second

    def calculate_wide_probs(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[FrozenSet[str], float]:
        """
        Calculate all wide combination probabilities.

        Wide(A, B) = Σ Trio(A, B, C) for all C ≠ A, B
                   = Σ [6 trifecta permutations] for each third horse C

        Args:
            position_probs: Dict mapping horse_id to probability array
                           where array[0] = P(1st), array[1] = P(2nd), etc.

        Returns:
            Dict mapping frozenset({horse_a, horse_b}) to probability
        """
        wide_probs = {}
        horses = list(position_probs.keys())
        n_horses = len(horses)

        if n_horses < 3:
            return wide_probs

        for horse_a, horse_b in combinations(horses, 2):
            wide_prob = 0.0

            # Sum over all possible third horses
            for horse_c in horses:
                if horse_c == horse_a or horse_c == horse_b:
                    continue

                # Sum all 6 permutations of (A, B, C) = Trio(A, B, C)
                for perm in permutations([horse_a, horse_b, horse_c]):
                    first, second, third = perm
                    wide_prob += self._calculate_trifecta_prob(
                        position_probs, first, second, third
                    )

            if wide_prob >= self.min_probability:
                wide_probs[frozenset({horse_a, horse_b})] = wide_prob

        return wide_probs

    def calculate_wide_probs_tuple(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate wide probabilities with sorted tuple keys.
        Useful for compatibility with existing code.

        Returns:
            Dict mapping (smaller_horse, larger_horse) to probability
        """
        wide_probs = self.calculate_wide_probs(position_probs)
        return {
            tuple(sorted(k)): v
            for k, v in wide_probs.items()
        }

    def get_top_wides(
        self,
        wide_probs: Dict[FrozenSet[str], float],
        n: Optional[int] = None,
    ) -> List[Tuple[FrozenSet[str], float]]:
        """
        Get top N most likely wide combinations.

        Args:
            wide_probs: Dict of wide probabilities
            n: Number of combinations to return (default: max_combinations)

        Returns:
            List of (frozenset{horse_a, horse_b}, probability) tuples
        """
        if n is None:
            n = self.max_combinations

        sorted_wides = sorted(
            wide_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_wides[:n]

    def normalize_wide_probs(
        self, wide_probs: Dict[FrozenSet[str], float]
    ) -> Dict[FrozenSet[str], float]:
        """Normalize wide probabilities to sum to 1."""
        total = sum(wide_probs.values())
        if total <= 0:
            return wide_probs

        return {k: v / total for k, v in wide_probs.items()}


def main():
    """Test wide calculation."""
    np.random.seed(42)

    horse_names = [f"Horse_{i+1}" for i in range(10)]

    # Create mock win probabilities (sum to 1)
    raw_probs = np.random.dirichlet(np.ones(10) * 2)

    position_probs = {
        name: np.array([raw_probs[i]] * 18)
        for i, name in enumerate(horse_names)
    }

    print("=" * 60)
    print("WIDE CALCULATION TEST")
    print("=" * 60)

    print("\nWin Probabilities:")
    for horse, probs in sorted(position_probs.items(), key=lambda x: -x[1][0]):
        print(f"  {horse}: {probs[0]:.2%}")

    calculator = WideCalculator()
    wide_probs = calculator.calculate_wide_probs(position_probs)

    print(f"\nTotal wide combinations: {len(wide_probs)}")
    print("Expected C(10,2) = 45")

    print("\nTop 10 Wide Combinations:")
    top_wides = calculator.get_top_wides(wide_probs, n=10)
    for horses, prob in top_wides:
        h1, h2 = sorted(horses)
        print(f"  {h1} - {h2}: {prob:.2%}")

    total_prob = sum(wide_probs.values())
    print(f"\nTotal wide probability sum: {total_prob:.4f}")
    print("(Note: Wide sums > 1 because each trio contributes to 3 wide combos)")


if __name__ == "__main__":
    main()
