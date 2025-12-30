"""
Keiba AI Prediction System - Data Type Definitions

Shared dataclasses and type definitions.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BetResult:
    """Result of a single bet."""
    race_id: int
    race_date: str
    predicted_1st: str
    predicted_2nd: str
    actual_1st: int
    actual_2nd: int
    predicted_prob: float
    actual_odds: float
    expected_value: float
    bet_amount: float
    payout: float
    profit: float
    won: bool
    # Stratification fields
    surface: str = ""
    distance_category: str = ""
    race_grade: str = ""
    track_condition: str = ""
    racecourse: str = ""


@dataclass
class TrifectaBetResult:
    """Result of a single trifecta bet."""
    race_id: int
    race_date: str
    predicted_1st: str
    predicted_2nd: str
    predicted_3rd: str
    actual_1st: int
    actual_2nd: int
    actual_3rd: int
    predicted_prob: float
    actual_odds: float
    expected_value: float
    bet_amount: float
    payout: float
    profit: float
    won: bool
    # Stratification fields
    surface: str = ""
    distance_category: str = ""
    race_grade: str = ""
    track_condition: str = ""
    racecourse: str = ""


@dataclass
class BacktestResults:
    """Aggregate backtest results."""
    bets: List[BetResult] = field(default_factory=list)
    total_bet: float = 0
    total_return: float = 0
    num_bets: int = 0
    num_wins: int = 0

    @property
    def profit(self) -> float:
        return self.total_return - self.total_bet

    @property
    def roi(self) -> float:
        if self.total_bet == 0:
            return 0
        return (self.total_return - self.total_bet) / self.total_bet * 100

    @property
    def hit_rate(self) -> float:
        if self.num_bets == 0:
            return 0
        return self.num_wins / self.num_bets * 100

    @property
    def avg_odds_hit(self) -> float:
        """Average decimal odds of winning bets (convert from Japanese format)."""
        winning_bets = [b for b in self.bets if b.won]
        if not winning_bets:
            return 0
        return np.mean([b.actual_odds / 100 for b in winning_bets])

    def get_cumulative_profit(self) -> List[float]:
        """Get cumulative profit over time."""
        cumulative = []
        total = 0
        for bet in self.bets:
            total += bet.profit
            cumulative.append(total)
        return cumulative

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = self.get_cumulative_profit()
        if not cumulative:
            return 0

        peak = cumulative[0]
        max_dd = 0

        for value in cumulative:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def get_sharpe_ratio(self, risk_free_rate: float = 0) -> float:
        """Calculate Sharpe ratio of returns."""
        if not self.bets:
            return 0

        returns = [b.profit / b.bet_amount for b in self.bets]
        if len(returns) < 2:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        return (mean_return - risk_free_rate) / std_return


@dataclass
class TrifectaBacktestResults:
    """Aggregate trifecta backtest results."""
    bets: List[TrifectaBetResult] = field(default_factory=list)
    total_bet: float = 0
    total_return: float = 0
    num_bets: int = 0
    num_wins: int = 0

    @property
    def profit(self) -> float:
        return self.total_return - self.total_bet

    @property
    def roi(self) -> float:
        if self.total_bet == 0:
            return 0
        return (self.total_return - self.total_bet) / self.total_bet * 100

    @property
    def hit_rate(self) -> float:
        if self.num_bets == 0:
            return 0
        return self.num_wins / self.num_bets * 100

    @property
    def avg_odds_hit(self) -> float:
        """Average decimal odds of winning bets (convert from Japanese format)."""
        winning_bets = [b for b in self.bets if b.won]
        if not winning_bets:
            return 0
        return np.mean([b.actual_odds / 100 for b in winning_bets])

    def get_cumulative_profit(self) -> List[float]:
        """Get cumulative profit over time."""
        cumulative = []
        total = 0
        for bet in self.bets:
            total += bet.profit
            cumulative.append(total)
        return cumulative

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = self.get_cumulative_profit()
        if not cumulative:
            return 0

        peak = cumulative[0]
        max_dd = 0

        for value in cumulative:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def get_sharpe_ratio(self, risk_free_rate: float = 0) -> float:
        """Calculate Sharpe ratio of returns."""
        if not self.bets:
            return 0

        returns = [b.profit / b.bet_amount for b in self.bets]
        if len(returns) < 2:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        return (mean_return - risk_free_rate) / std_return


@dataclass
class QuinellaBetResult:
    """Result of a single quinella bet."""
    race_id: int
    race_date: str
    horse1: str
    horse2: str
    actual_1st: int
    actual_2nd: int
    predicted_prob: float
    actual_odds: float
    expected_value: float
    bet_amount: float
    payout: float
    profit: float
    won: bool
    # Stratification fields
    surface: str = ""
    distance_category: str = ""
    race_grade: str = ""
    track_condition: str = ""
    racecourse: str = ""


@dataclass
class QuinellaBacktestResults:
    """Aggregate quinella backtest results."""
    bets: List[QuinellaBetResult] = field(default_factory=list)
    total_bet: float = 0
    total_return: float = 0
    num_bets: int = 0
    num_wins: int = 0

    @property
    def profit(self) -> float:
        return self.total_return - self.total_bet

    @property
    def roi(self) -> float:
        if self.total_bet == 0:
            return 0
        return (self.total_return - self.total_bet) / self.total_bet * 100

    @property
    def hit_rate(self) -> float:
        if self.num_bets == 0:
            return 0
        return self.num_wins / self.num_bets * 100

    @property
    def avg_odds_hit(self) -> float:
        winning_bets = [b for b in self.bets if b.won]
        if not winning_bets:
            return 0
        return np.mean([b.actual_odds / 100 for b in winning_bets])

    def get_cumulative_profit(self) -> List[float]:
        cumulative = []
        total = 0
        for bet in self.bets:
            total += bet.profit
            cumulative.append(total)
        return cumulative

    def get_max_drawdown(self) -> float:
        cumulative = self.get_cumulative_profit()
        if not cumulative:
            return 0
        peak = cumulative[0]
        max_dd = 0
        for value in cumulative:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def get_sharpe_ratio(self, risk_free_rate: float = 0) -> float:
        if not self.bets:
            return 0
        returns = [b.profit / b.bet_amount for b in self.bets]
        if len(returns) < 2:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return (mean_return - risk_free_rate) / std_return


@dataclass
class TrioBetResult:
    """Result of a single trio bet."""
    race_id: int
    race_date: str
    horse1: str
    horse2: str
    horse3: str
    actual_1st: int
    actual_2nd: int
    actual_3rd: int
    predicted_prob: float
    actual_odds: float
    expected_value: float
    bet_amount: float
    payout: float
    profit: float
    won: bool
    # Stratification fields
    surface: str = ""
    distance_category: str = ""
    race_grade: str = ""
    track_condition: str = ""
    racecourse: str = ""


@dataclass
class TrioBacktestResults:
    """Aggregate trio backtest results."""
    bets: List[TrioBetResult] = field(default_factory=list)
    total_bet: float = 0
    total_return: float = 0
    num_bets: int = 0
    num_wins: int = 0

    @property
    def profit(self) -> float:
        return self.total_return - self.total_bet

    @property
    def roi(self) -> float:
        if self.total_bet == 0:
            return 0
        return (self.total_return - self.total_bet) / self.total_bet * 100

    @property
    def hit_rate(self) -> float:
        if self.num_bets == 0:
            return 0
        return self.num_wins / self.num_bets * 100

    @property
    def avg_odds_hit(self) -> float:
        winning_bets = [b for b in self.bets if b.won]
        if not winning_bets:
            return 0
        return np.mean([b.actual_odds / 100 for b in winning_bets])

    def get_cumulative_profit(self) -> List[float]:
        cumulative = []
        total = 0
        for bet in self.bets:
            total += bet.profit
            cumulative.append(total)
        return cumulative

    def get_max_drawdown(self) -> float:
        cumulative = self.get_cumulative_profit()
        if not cumulative:
            return 0
        peak = cumulative[0]
        max_dd = 0
        for value in cumulative:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def get_sharpe_ratio(self, risk_free_rate: float = 0) -> float:
        if not self.bets:
            return 0
        returns = [b.profit / b.bet_amount for b in self.bets]
        if len(returns) < 2:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return (mean_return - risk_free_rate) / std_return


@dataclass
class WideBetResult:
    """Result of a single wide bet."""
    race_id: int
    race_date: str
    horse1: str
    horse2: str
    actual_1st: int
    actual_2nd: int
    actual_3rd: int
    predicted_prob: float
    actual_odds: float
    expected_value: float
    bet_amount: float
    payout: float
    profit: float
    won: bool
    # Stratification fields
    surface: str = ""
    distance_category: str = ""
    race_grade: str = ""
    track_condition: str = ""
    racecourse: str = ""


@dataclass
class WideBacktestResults:
    """Aggregate wide backtest results."""
    bets: List[WideBetResult] = field(default_factory=list)
    total_bet: float = 0
    total_return: float = 0
    num_bets: int = 0
    num_wins: int = 0

    @property
    def profit(self) -> float:
        return self.total_return - self.total_bet

    @property
    def roi(self) -> float:
        if self.total_bet == 0:
            return 0
        return (self.total_return - self.total_bet) / self.total_bet * 100

    @property
    def hit_rate(self) -> float:
        if self.num_bets == 0:
            return 0
        return self.num_wins / self.num_bets * 100

    @property
    def avg_odds_hit(self) -> float:
        winning_bets = [b for b in self.bets if b.won]
        if not winning_bets:
            return 0
        return np.mean([b.actual_odds / 100 for b in winning_bets])

    def get_cumulative_profit(self) -> List[float]:
        cumulative = []
        total = 0
        for bet in self.bets:
            total += bet.profit
            cumulative.append(total)
        return cumulative

    def get_max_drawdown(self) -> float:
        cumulative = self.get_cumulative_profit()
        if not cumulative:
            return 0
        peak = cumulative[0]
        max_dd = 0
        for value in cumulative:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def get_sharpe_ratio(self, risk_free_rate: float = 0) -> float:
        if not self.bets:
            return 0
        returns = [b.profit / b.bet_amount for b in self.bets]
        if len(returns) < 2:
            return 0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return (mean_return - risk_free_rate) / std_return
