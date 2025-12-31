"""
Keiba AI Prediction System - Trio Probability Calculation

Calculate Sanrenpuku (三連複/trio) probabilities from position distributions.
Trio = 3 horses in top 3, order doesn't matter.
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


class TrioCalculator:
    """
    Calculate Sanrenpuku (三連複/trio) probabilities from position distributions.

    Trio(A, B, C) = Sum of all 6 trifecta permutations:
        P(A-B-C) + P(A-C-B) + P(B-A-C) + P(B-C-A) + P(C-A-B) + P(C-B-A)

    Using extended Harville formula for each trifecta:
    P(1st, 2nd, 3rd) = P(1st) * P(2nd|1st) * P(3rd|1st,2nd)
    """

    def __init__(
        self,
        min_probability: float = 0.0001,
        max_combinations: int = 100,
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

    def calculate_trio_probs(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[FrozenSet[str], float]:
        """
        Calculate all trio combination probabilities.

        Args:
            position_probs: Dict mapping horse_id to probability array
                           where array[0] = P(1st), array[1] = P(2nd), etc.

        Returns:
            Dict mapping frozenset({horse_a, horse_b, horse_c}) to probability
        """
        trio_probs = {}
        horses = list(position_probs.keys())
        n_horses = len(horses)

        if n_horses < 3:
            return trio_probs

        for horse_combo in combinations(horses, 3):
            # Sum all 6 permutations
            trio_prob = 0.0
            for perm in permutations(horse_combo):
                first, second, third = perm
                trio_prob += self._calculate_trifecta_prob(
                    position_probs, first, second, third
                )

            if trio_prob >= self.min_probability:
                trio_probs[frozenset(horse_combo)] = trio_prob

        return trio_probs

    def calculate_trio_probs_tuple(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Calculate trio probabilities with sorted tuple keys.
        Useful for compatibility with existing code.

        Returns:
            Dict mapping (h1, h2, h3) sorted tuple to probability
        """
        trio_probs = self.calculate_trio_probs(position_probs)
        return {
            tuple(sorted(k)): v
            for k, v in trio_probs.items()
        }

    def get_top_trios(
        self,
        trio_probs: Dict[FrozenSet[str], float],
        n: Optional[int] = None,
    ) -> List[Tuple[FrozenSet[str], float]]:
        """
        Get top N most likely trio combinations.

        Args:
            trio_probs: Dict of trio probabilities
            n: Number of combinations to return (default: max_combinations)

        Returns:
            List of (frozenset{h1, h2, h3}, probability) tuples
        """
        if n is None:
            n = self.max_combinations

        sorted_trios = sorted(
            trio_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_trios[:n]

    def normalize_trio_probs(
        self, trio_probs: Dict[FrozenSet[str], float]
    ) -> Dict[FrozenSet[str], float]:
        """Normalize trio probabilities to sum to 1."""
        total = sum(trio_probs.values())
        if total <= 0:
            return trio_probs

        return {k: v / total for k, v in trio_probs.items()}


def main():
    """Test trio calculation."""
    np.random.seed(42)

    horse_names = [f"Horse_{i+1}" for i in range(10)]

    # Create mock win probabilities (sum to 1)
    raw_probs = np.random.dirichlet(np.ones(10) * 2)

    position_probs = {
        name: np.array([raw_probs[i]] * 18)
        for i, name in enumerate(horse_names)
    }

    print("=" * 60)
    print("TRIO CALCULATION TEST")
    print("=" * 60)

    print("\nWin Probabilities:")
    for horse, probs in sorted(position_probs.items(), key=lambda x: -x[1][0]):
        print(f"  {horse}: {probs[0]:.2%}")

    calculator = TrioCalculator()
    trio_probs = calculator.calculate_trio_probs(position_probs)

    print(f"\nTotal trio combinations: {len(trio_probs)}")
    print("Expected C(10,3) = 120")

    print("\nTop 10 Trio Combinations:")
    top_trios = calculator.get_top_trios(trio_probs, n=10)
    for horses, prob in top_trios:
        h1, h2, h3 = sorted(horses)
        print(f"  {h1} - {h2} - {h3}: {prob:.4%}")

    total_prob = sum(trio_probs.values())
    print(f"\nTotal trio probability sum: {total_prob:.4f}")


if __name__ == "__main__":
    main()
