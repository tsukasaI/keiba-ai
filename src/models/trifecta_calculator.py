"""
Keiba AI Prediction System - Trifecta Probability Calculation

Calculate Sanrentan (3連単/trifecta) probabilities from position distributions
using the extended Harville formula.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

from .config import BETTING_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrifectaCalculator:
    """
    Calculate Sanrentan (3連単/trifecta) probabilities from position distributions.

    Trifecta(A, B, C) = P(A=1st) * P(B=2nd | A=1st) * P(C=3rd | A,B finished)

    Using extended Harville formula:
    P(B=2nd | A=1st) = P(B=1st) / (1 - P(A=1st))
    P(C=3rd | A=1st, B=2nd) = P(C=1st) / (1 - P(A=1st) - P(B=1st))
    """

    def __init__(
        self,
        min_probability: float = BETTING_CONFIG["min_probability"],
        max_combinations: int = 100,
    ):
        self.min_probability = min_probability
        self.max_combinations = max_combinations

    def calculate_trifecta_probs(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Calculate all trifecta combination probabilities.

        Args:
            position_probs: Dict mapping horse_id to probability array
                           where array[0] = P(1st), array[1] = P(2nd), etc.

        Returns:
            Dict mapping (first_horse, second_horse, third_horse) to probability
        """
        trifecta_probs = {}
        horses = list(position_probs.keys())
        n_horses = len(horses)

        if n_horses < 3:
            return trifecta_probs

        for first in horses:
            p_first = position_probs[first][0]

            # Skip if probability too low
            if p_first < self.min_probability:
                continue

            for second in horses:
                if second == first:
                    continue

                p_second = position_probs[second][0]

                # Harville: P(B=2nd | A=1st) = P(B) / (1 - P(A))
                p_second_given_first = p_second / (1 - p_first + 1e-10)

                # Early pruning
                p_first_second = p_first * p_second_given_first
                if p_first_second < self.min_probability:
                    continue

                for third in horses:
                    if third == first or third == second:
                        continue

                    p_third = position_probs[third][0]

                    # Harville: P(C=3rd | A,B) = P(C) / (1 - P(A) - P(B))
                    remaining_prob = 1 - p_first - p_second + 1e-10
                    p_third_given_first_second = p_third / remaining_prob

                    # Trifecta probability
                    trifecta_prob = p_first_second * p_third_given_first_second

                    if trifecta_prob >= self.min_probability:
                        trifecta_probs[(first, second, third)] = trifecta_prob

        return trifecta_probs

    def calculate_trifecta_probs_from_matrix(
        self, prob_matrix: np.ndarray, horse_names: List[str]
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Calculate trifecta probabilities from a probability matrix.

        Args:
            prob_matrix: Array of shape (n_horses, n_positions) where
                        prob_matrix[i, j] = P(horse i finishes in position j+1)
            horse_names: List of horse identifiers

        Returns:
            Dict mapping (first, second, third) to probability
        """
        position_probs = {
            name: prob_matrix[i] for i, name in enumerate(horse_names)
        }
        return self.calculate_trifecta_probs(position_probs)

    def get_top_trifectas(
        self,
        trifecta_probs: Dict[Tuple[str, str, str], float],
        n: Optional[int] = None,
    ) -> List[Tuple[Tuple[str, str, str], float]]:
        """
        Get top N most likely trifecta combinations.

        Args:
            trifecta_probs: Dict of trifecta probabilities
            n: Number of combinations to return (default: max_combinations)

        Returns:
            List of ((first, second, third), probability) tuples
        """
        if n is None:
            n = self.max_combinations

        sorted_trifectas = sorted(
            trifecta_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_trifectas[:n]

    def normalize_trifecta_probs(
        self, trifecta_probs: Dict[Tuple[str, str, str], float]
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Normalize trifecta probabilities to sum to 1.
        """
        total = sum(trifecta_probs.values())
        if total <= 0:
            return trifecta_probs

        return {k: v / total for k, v in trifecta_probs.items()}


def main():
    """Test trifecta calculation."""
    import numpy as np

    # Simulate a 10-horse race with win probabilities
    np.random.seed(42)

    horse_names = [f"Horse_{i+1}" for i in range(10)]

    # Create mock win probabilities (sum to 1)
    raw_probs = np.random.dirichlet(np.ones(10) * 2)

    # Create position probability array (simplified: use win prob for all positions)
    position_probs = {
        name: np.array([raw_probs[i]] * 18)
        for i, name in enumerate(horse_names)
    }

    print("=" * 60)
    print("TRIFECTA CALCULATION TEST")
    print("=" * 60)

    print("\nWin Probabilities:")
    for horse, probs in sorted(position_probs.items(), key=lambda x: -x[1][0]):
        print(f"  {horse}: {probs[0]:.2%}")

    # Calculate trifectas
    calculator = TrifectaCalculator()
    trifecta_probs = calculator.calculate_trifecta_probs(position_probs)

    print(f"\nTotal trifecta combinations: {len(trifecta_probs)}")

    print("\nTop 10 Trifecta Combinations:")
    top_trifectas = calculator.get_top_trifectas(trifecta_probs, n=10)
    for (first, second, third), prob in top_trifectas:
        print(f"  {first} -> {second} -> {third}: {prob:.4%}")

    # Verify probabilities sum reasonably
    total_prob = sum(trifecta_probs.values())
    print(f"\nTotal trifecta probability sum: {total_prob:.4f}")
    print("Expected (for 10 horses): ~1.0")


if __name__ == "__main__":
    main()
