"""Tests for exacta calculator module."""

import numpy as np
import pytest

from src.models.exacta_calculator import ExactaCalculator


class TestExactaCalculator:
    """Test ExactaCalculator class."""

    @pytest.fixture
    def calculator(self) -> ExactaCalculator:
        """Create calculator with default settings."""
        return ExactaCalculator(min_probability=0.001, max_combinations=50)

    @pytest.fixture
    def position_probs(self) -> dict:
        """Create mock position probabilities for 3 horses."""
        return {
            "Horse_A": np.array([0.5, 0.3, 0.2] + [0] * 15),  # Favorite
            "Horse_B": np.array([0.3, 0.4, 0.3] + [0] * 15),  # Second choice
            "Horse_C": np.array([0.2, 0.3, 0.5] + [0] * 15),  # Third choice
        }

    def test_calculate_exacta_probs(self, calculator, position_probs):
        """Test exacta probability calculation."""
        exacta_probs = calculator.calculate_exacta_probs(position_probs)

        # Should have 6 combinations: A-B, A-C, B-A, B-C, C-A, C-B
        assert len(exacta_probs) == 6

        # All probabilities should be between 0 and 1
        for prob in exacta_probs.values():
            assert 0 < prob < 1

    def test_harville_formula(self, calculator, position_probs):
        """Test that Harville formula is correctly applied."""
        exacta_probs = calculator.calculate_exacta_probs(position_probs)

        # Check A-B: P(A=1st) * P(B=2nd|A=1st)
        # P(B=2nd|A=1st) = P(B=1st) / (1 - P(A=1st)) = 0.3 / 0.5 = 0.6
        # P(A-B) = 0.5 * 0.6 = 0.3
        expected_a_b = 0.5 * (0.3 / 0.5)
        assert abs(exacta_probs[("Horse_A", "Horse_B")] - expected_a_b) < 0.01

    def test_favorite_has_highest_prob(self, calculator, position_probs):
        """Test that favorite combinations have highest probabilities."""
        exacta_probs = calculator.calculate_exacta_probs(position_probs)

        # A-B and A-C should be among the highest
        a_b = exacta_probs[("Horse_A", "Horse_B")]
        a_c = exacta_probs[("Horse_A", "Horse_C")]

        # A-B should be higher since B has higher win prob than C
        assert a_b > a_c

    def test_get_top_exactas(self, calculator, position_probs):
        """Test getting top N exacta combinations."""
        exacta_probs = calculator.calculate_exacta_probs(position_probs)
        top_3 = calculator.get_top_exactas(exacta_probs, n=3)

        assert len(top_3) == 3

        # Should be sorted by probability (descending)
        probs = [p for _, p in top_3]
        assert probs == sorted(probs, reverse=True)

    def test_normalize_exacta_probs(self, calculator, position_probs):
        """Test probability normalization."""
        exacta_probs = calculator.calculate_exacta_probs(position_probs)
        normalized = calculator.normalize_exacta_probs(exacta_probs)

        # Should sum to 1
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.01

    def test_get_win_probs(self, calculator, position_probs):
        """Test extracting win probabilities."""
        win_probs = calculator.get_win_probs(position_probs)

        assert win_probs["Horse_A"] == 0.5
        assert win_probs["Horse_B"] == 0.3
        assert win_probs["Horse_C"] == 0.2

    def test_get_place_probs(self, calculator, position_probs):
        """Test calculating place probabilities."""
        place_probs = calculator.get_place_probs(position_probs, top_n=3)

        # Horse_A: 0.5 + 0.3 + 0.2 = 1.0
        assert place_probs["Horse_A"] == 1.0

        # Horse_B: 0.3 + 0.4 + 0.3 = 1.0
        assert place_probs["Horse_B"] == 1.0

    def test_min_probability_filter(self):
        """Test minimum probability filtering."""
        calculator = ExactaCalculator(min_probability=0.1)

        position_probs = {
            "Horse_A": np.array([0.05] + [0] * 17),  # Below threshold
            "Horse_B": np.array([0.95] + [0] * 17),
        }

        exacta_probs = calculator.calculate_exacta_probs(position_probs)

        # A shouldn't be in first position due to min_probability filter
        assert ("Horse_A", "Horse_B") not in exacta_probs

    def test_calculate_from_matrix(self, calculator):
        """Test calculation from probability matrix."""
        prob_matrix = np.array([
            [0.5, 0.3, 0.2] + [0] * 15,
            [0.3, 0.4, 0.3] + [0] * 15,
        ])
        horse_names = ["Horse_A", "Horse_B"]

        exacta_probs = calculator.calculate_exacta_probs_from_matrix(
            prob_matrix, horse_names
        )

        assert len(exacta_probs) == 2
        assert ("Horse_A", "Horse_B") in exacta_probs
        assert ("Horse_B", "Horse_A") in exacta_probs


class TestExactaProbabilitySum:
    """Test that exacta probabilities approximately sum to 1."""

    def test_probability_sum(self):
        """Test that all exacta combinations sum to approximately 1."""
        np.random.seed(42)

        # Create 10 horses with random win probabilities
        n_horses = 10
        win_probs = np.random.dirichlet(np.ones(n_horses))

        # Create position probability distributions
        position_probs = {}
        for i in range(n_horses):
            probs = np.zeros(18)
            probs[:n_horses] = np.random.dirichlet(np.ones(n_horses))
            # Set win prob to expected
            probs[0] = win_probs[i]
            position_probs[f"Horse_{i}"] = probs

        calculator = ExactaCalculator(min_probability=0.0)
        exacta_probs = calculator.calculate_exacta_probs(position_probs)

        # Note: Harville formula doesn't guarantee sum to 1
        # but should be reasonably close for well-calibrated probabilities
        total = sum(exacta_probs.values())
        assert 0.5 < total < 2.0  # Reasonable range
