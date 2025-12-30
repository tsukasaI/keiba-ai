"""
Keiba AI Prediction System - Exacta Probability Calculation

Calculate Umatan (exacta) probabilities from position distributions
using the Harville formula.
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


class ExactaCalculator:
    """
    Calculate Umatan (馬単/exacta) probabilities from position distributions.

    Exacta(A, B) = P(A=1st) * P(B=2nd | A=1st)

    Using Harville formula:
    P(B=2nd | A=1st) = P(B=1st) / (1 - P(A=1st))
    """

    def __init__(
        self,
        min_probability: float = BETTING_CONFIG["min_probability"],
        max_combinations: int = BETTING_CONFIG["max_combinations"],
    ):
        self.min_probability = min_probability
        self.max_combinations = max_combinations

    def calculate_exacta_probs(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate all exacta combination probabilities.

        Args:
            position_probs: Dict mapping horse_id to probability array
                           where array[0] = P(1st), array[1] = P(2nd), etc.

        Returns:
            Dict mapping (first_horse, second_horse) to probability
        """
        exacta_probs = {}
        horses = list(position_probs.keys())

        for first in horses:
            # P(first wins)
            p_first_wins = position_probs[first][0]

            # Skip if probability too low
            if p_first_wins < self.min_probability:
                continue

            for second in horses:
                if first == second:
                    continue

                # Harville formula for conditional probability
                # P(B=2nd | A=1st) = P(B=1st) / (1 - P(A=1st))
                p_second_given_first = position_probs[second][0] / (1 - p_first_wins + 1e-10)

                # Exacta probability
                exacta_prob = p_first_wins * p_second_given_first

                # Skip very low probabilities
                if exacta_prob >= self.min_probability:
                    exacta_probs[(first, second)] = exacta_prob

        return exacta_probs

    def calculate_exacta_probs_from_matrix(
        self, prob_matrix: np.ndarray, horse_names: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate exacta probabilities from a probability matrix.

        Args:
            prob_matrix: Array of shape (n_horses, n_positions) where
                        prob_matrix[i, j] = P(horse i finishes in position j+1)
            horse_names: List of horse identifiers

        Returns:
            Dict mapping (first_horse, second_horse) to probability
        """
        position_probs = {
            name: prob_matrix[i] for i, name in enumerate(horse_names)
        }
        return self.calculate_exacta_probs(position_probs)

    def get_top_exactas(
        self,
        exacta_probs: Dict[Tuple[str, str], float],
        n: Optional[int] = None,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get top N most likely exacta combinations.

        Args:
            exacta_probs: Dict of exacta probabilities
            n: Number of combinations to return (default: max_combinations)

        Returns:
            List of ((first, second), probability) tuples, sorted by probability
        """
        if n is None:
            n = self.max_combinations

        sorted_exactas = sorted(
            exacta_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_exactas[:n]

    def normalize_exacta_probs(
        self, exacta_probs: Dict[Tuple[str, str], float]
    ) -> Dict[Tuple[str, str], float]:
        """
        Normalize exacta probabilities to sum to 1.

        Useful for evaluation metrics.
        """
        total = sum(exacta_probs.values())
        if total <= 0:
            return exacta_probs

        return {k: v / total for k, v in exacta_probs.items()}

    def get_win_probs(
        self, position_probs: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Extract win probabilities (1st place) from position distributions.

        Args:
            position_probs: Dict mapping horse_id to probability array

        Returns:
            Dict mapping horse_id to win probability
        """
        return {horse: probs[0] for horse, probs in position_probs.items()}

    def get_place_probs(
        self, position_probs: Dict[str, np.ndarray], top_n: int = 3
    ) -> Dict[str, float]:
        """
        Calculate place probabilities (top N finish).

        Args:
            position_probs: Dict mapping horse_id to probability array
            top_n: Number of top positions to consider (default: 3 for show)

        Returns:
            Dict mapping horse_id to place probability
        """
        return {
            horse: probs[:top_n].sum()
            for horse, probs in position_probs.items()
        }


def main():
    """Test exacta calculation."""
    import numpy as np

    # Simulate a 10-horse race with position probabilities
    np.random.seed(42)

    horse_names = [f"Horse_{i+1}" for i in range(10)]

    # Create mock position probabilities (each row sums to ~1)
    raw_probs = np.random.dirichlet(np.ones(10) * 2, size=10)

    # Normalize columns (each position sums to 1)
    for _ in range(10):
        raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        raw_probs = raw_probs / raw_probs.sum(axis=0, keepdims=True)

    # Pad to 18 positions
    position_matrix = np.zeros((10, 18))
    position_matrix[:, :10] = raw_probs

    position_probs = {name: position_matrix[i] for i, name in enumerate(horse_names)}

    print("=" * 60)
    print("EXACTA CALCULATION TEST")
    print("=" * 60)

    print("\nWin Probabilities:")
    for horse, probs in position_probs.items():
        print(f"  {horse}: {probs[0]:.2%}")

    # Calculate exactas
    calculator = ExactaCalculator()
    exacta_probs = calculator.calculate_exacta_probs(position_probs)

    print(f"\nTotal exacta combinations: {len(exacta_probs)}")

    print("\nTop 10 Exacta Combinations:")
    top_exactas = calculator.get_top_exactas(exacta_probs, n=10)
    for (first, second), prob in top_exactas:
        print(f"  {first} -> {second}: {prob:.2%}")

    # Verify probabilities sum to ~1
    total_prob = sum(exacta_probs.values())
    print(f"\nTotal exacta probability sum: {total_prob:.4f}")


if __name__ == "__main__":
    main()
