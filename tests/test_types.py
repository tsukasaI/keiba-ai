"""Tests for types module."""

import pytest

from src.models.types import BetResult, BacktestResults


class TestBetResult:
    """Test BetResult dataclass."""

    def test_create_bet_result(self):
        """Test creating a BetResult."""
        bet = BetResult(
            race_id=123,
            race_date="2021-01-01",
            predicted_1st="Horse1",
            predicted_2nd="Horse2",
            actual_1st=1,
            actual_2nd=2,
            predicted_prob=0.05,
            actual_odds=3560.0,
            expected_value=1.78,
            bet_amount=100,
            payout=3560.0,
            profit=3460.0,
            won=True,
        )
        assert bet.race_id == 123
        assert bet.won is True
        assert bet.profit == 3460.0

    def test_bet_result_stratification_fields(self):
        """Test stratification fields have defaults."""
        bet = BetResult(
            race_id=1,
            race_date="2021-01-01",
            predicted_1st="A",
            predicted_2nd="B",
            actual_1st=1,
            actual_2nd=2,
            predicted_prob=0.1,
            actual_odds=1000.0,
            expected_value=1.0,
            bet_amount=100,
            payout=0,
            profit=-100,
            won=False,
        )
        assert bet.surface == ""
        assert bet.distance_category == ""
        assert bet.racecourse == ""


class TestBacktestResults:
    """Test BacktestResults dataclass."""

    @pytest.fixture
    def winning_bet(self) -> BetResult:
        """Create a winning bet."""
        return BetResult(
            race_id=1,
            race_date="2021-01-01",
            predicted_1st="A",
            predicted_2nd="B",
            actual_1st=1,
            actual_2nd=2,
            predicted_prob=0.05,
            actual_odds=2000.0,
            expected_value=1.0,
            bet_amount=100,
            payout=2000.0,
            profit=1900.0,
            won=True,
        )

    @pytest.fixture
    def losing_bet(self) -> BetResult:
        """Create a losing bet."""
        return BetResult(
            race_id=2,
            race_date="2021-01-02",
            predicted_1st="C",
            predicted_2nd="D",
            actual_1st=3,
            actual_2nd=4,
            predicted_prob=0.05,
            actual_odds=1500.0,
            expected_value=0.75,
            bet_amount=100,
            payout=0.0,
            profit=-100.0,
            won=False,
        )

    def test_empty_results(self):
        """Test empty results."""
        results = BacktestResults()
        assert results.profit == 0
        assert results.roi == 0
        assert results.hit_rate == 0
        assert results.avg_odds_hit == 0

    def test_roi_calculation(self, winning_bet, losing_bet):
        """Test ROI calculation."""
        results = BacktestResults(
            bets=[winning_bet, losing_bet],
            total_bet=200,
            total_return=2000,
            num_bets=2,
            num_wins=1,
        )
        assert results.profit == 1800
        assert results.roi == 900.0  # 1800/200 * 100

    def test_hit_rate_calculation(self, winning_bet, losing_bet):
        """Test hit rate calculation."""
        results = BacktestResults(
            bets=[winning_bet, losing_bet],
            total_bet=200,
            total_return=2000,
            num_bets=2,
            num_wins=1,
        )
        assert results.hit_rate == 50.0

    def test_cumulative_profit(self, winning_bet, losing_bet):
        """Test cumulative profit calculation."""
        results = BacktestResults(
            bets=[winning_bet, losing_bet],
            total_bet=200,
            total_return=2000,
            num_bets=2,
            num_wins=1,
        )
        cumulative = results.get_cumulative_profit()
        assert cumulative == [1900.0, 1800.0]

    def test_max_drawdown(self, winning_bet, losing_bet):
        """Test max drawdown calculation."""
        # Win then lose: peak at 1900, drops to 1800
        results = BacktestResults(
            bets=[winning_bet, losing_bet],
            total_bet=200,
            total_return=2000,
            num_bets=2,
            num_wins=1,
        )
        drawdown = results.get_max_drawdown()
        assert drawdown == 100.0  # Peak 1900, low 1800

    def test_sharpe_ratio_calculation(self, winning_bet, losing_bet):
        """Test Sharpe ratio calculation."""
        results = BacktestResults(
            bets=[winning_bet, losing_bet],
            total_bet=200,
            total_return=2000,
            num_bets=2,
            num_wins=1,
        )
        sharpe = results.get_sharpe_ratio()
        # Sharpe should be positive since mean return is positive
        assert sharpe > 0

    def test_sharpe_ratio_empty(self):
        """Test Sharpe ratio with no bets."""
        results = BacktestResults()
        assert results.get_sharpe_ratio() == 0

    def test_avg_odds_hit(self, winning_bet):
        """Test average odds for winning bets."""
        results = BacktestResults(
            bets=[winning_bet],
            total_bet=100,
            total_return=2000,
            num_bets=1,
            num_wins=1,
        )
        # Japanese format: odds 2000 -> decimal 20.0
        assert results.avg_odds_hit == 20.0
