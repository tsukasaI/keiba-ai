"""Tests for expected value calculator module."""

import pytest

from src.models.expected_value import ExpectedValueCalculator, ValueBet


class TestExpectedValueCalculator:
    """Test ExpectedValueCalculator class."""

    @pytest.fixture
    def calculator(self) -> ExpectedValueCalculator:
        """Create calculator with default settings."""
        return ExpectedValueCalculator(ev_threshold=1.0, bet_unit=100)

    def test_calculate_ev(self, calculator):
        """Test EV calculation."""
        # 10% probability at 15x odds = 1.5 EV
        ev = calculator.calculate_ev(0.10, 15.0)
        assert ev == 1.5

    def test_calculate_ev_edge_case(self, calculator):
        """Test EV calculation at breakeven."""
        # 10% at 10x = 1.0 EV (breakeven)
        ev = calculator.calculate_ev(0.10, 10.0)
        assert ev == 1.0

    def test_find_value_bets(self, calculator):
        """Test finding value bets."""
        exacta_probs = {
            ("A", "B"): 0.10,  # Value bet at odds > 10
            ("A", "C"): 0.05,  # Value bet at odds > 20
            ("B", "A"): 0.08,  # Not value at odds < 12.5
        }
        exacta_odds = {
            ("A", "B"): 15.0,  # EV = 1.5
            ("A", "C"): 25.0,  # EV = 1.25
            ("B", "A"): 10.0,  # EV = 0.8
        }

        value_bets = calculator.find_value_bets(exacta_probs, exacta_odds)

        assert len(value_bets) == 2
        assert all(b.expected_value > 1.0 for b in value_bets)

        # Should be sorted by EV descending
        assert value_bets[0].expected_value > value_bets[1].expected_value

    def test_find_value_bets_no_odds(self, calculator):
        """Test that missing odds are handled."""
        exacta_probs = {("A", "B"): 0.10}
        exacta_odds = {}  # No odds data

        value_bets = calculator.find_value_bets(exacta_probs, exacta_odds)
        assert len(value_bets) == 0

    def test_value_bet_dataclass(self):
        """Test ValueBet dataclass."""
        bet = ValueBet(
            combination=("A", "B"),
            predicted_prob=0.10,
            odds=15.0,
            expected_value=1.5,
            edge=0.5,
        )
        assert bet.combination == ("A", "B")
        assert bet.edge == 0.5
        assert "A->B" in repr(bet)


class TestKellyCriterion:
    """Test Kelly criterion bet sizing."""

    @pytest.fixture
    def calculator(self) -> ExpectedValueCalculator:
        """Create calculator."""
        return ExpectedValueCalculator(bet_unit=100)

    def test_calculate_kelly_fraction(self, calculator):
        """Test Kelly fraction calculation."""
        # 50% win at 3x odds
        # Kelly = (0.5 * 2 - 0.5) / 2 = 0.25
        kelly = calculator.calculate_kelly_fraction(0.5, 3.0)
        assert abs(kelly - 0.25) < 0.01

    def test_kelly_fraction_zero_for_negative_ev(self, calculator):
        """Test that Kelly is 0 for negative EV."""
        # 10% at 5x = 0.5 EV (negative)
        kelly = calculator.calculate_kelly_fraction(0.10, 5.0)
        assert kelly == 0.0

    def test_kelly_fraction_edge_cases(self, calculator):
        """Test Kelly with edge cases."""
        assert calculator.calculate_kelly_fraction(0, 10) == 0
        assert calculator.calculate_kelly_fraction(0.5, 1) == 0
        assert calculator.calculate_kelly_fraction(0.5, 0) == 0

    def test_calculate_bet_size(self, calculator):
        """Test bet size calculation."""
        # 20% at 10x, bankroll 100k, quarter Kelly
        # Full Kelly = (0.2 * 9 - 0.8) / 9 = 0.111
        # Quarter Kelly = 0.028, Bet = 2800
        bet_size = calculator.calculate_bet_size(
            probability=0.2,
            odds=10.0,
            bankroll=100000,
            kelly_fraction=0.25,
        )

        # Should be rounded to bet unit (100)
        assert bet_size % 100 == 0
        assert bet_size > 0

    def test_bet_size_minimum(self, calculator):
        """Test that bet size has minimum of bet_unit."""
        bet_size = calculator.calculate_bet_size(
            probability=0.01,  # Very low
            odds=2.0,
            bankroll=1000,
            kelly_fraction=0.25,
        )

        assert bet_size >= calculator.bet_unit


class TestSummarizeValueBets:
    """Test value bet summarization."""

    @pytest.fixture
    def calculator(self) -> ExpectedValueCalculator:
        """Create calculator."""
        return ExpectedValueCalculator()

    def test_summarize_empty(self, calculator):
        """Test summarizing empty bets."""
        summary = calculator.summarize_value_bets([])

        assert summary["num_bets"] == 0
        assert summary["avg_ev"] == 0

    def test_summarize_value_bets(self, calculator):
        """Test summarizing value bets."""
        bets = [
            ValueBet(("A", "B"), 0.1, 15.0, 1.5, 0.5),
            ValueBet(("A", "C"), 0.05, 25.0, 1.25, 0.25),
        ]

        summary = calculator.summarize_value_bets(bets)

        assert summary["num_bets"] == 2
        assert summary["max_ev"] == 1.5
        assert summary["avg_ev"] == 1.375
        assert summary["avg_edge"] == 0.375
        assert abs(summary["total_probability"] - 0.15) < 1e-10
        assert summary["best_bet"] == bets[0]
