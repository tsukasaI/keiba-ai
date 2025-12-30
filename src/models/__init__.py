"""
Keiba AI Prediction System - Model Module

Models for horse racing position prediction and exacta betting.
"""

from .config import (
    BACKTEST_CONFIG,
    BETTING_CONFIG,
    FEATURES,
    FEATURE_GROUPS,
    LGBM_PARAMS,
    MISSING_DEFAULTS,
    PROFITABLE_SEGMENTS,
    TRAINING_CONFIG,
)
from .types import BacktestResults, BetResult
from .data_loader import RaceDataLoader
from .odds_loader import OddsLoader
from .position_model import PositionProbabilityModel
from .exacta_calculator import ExactaCalculator
from .expected_value import ExpectedValueCalculator, ValueBet
from .calibrator import (
    BinningCalibration,
    CalibrationAnalyzer,
    CalibrationBin,
    ExactaCalibrator,
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
    compare_calibration_methods,
)
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .backtester import Backtester
from .backtest_report import BacktestReporter, run_full_backtest_report

__all__ = [
    # Config
    "BACKTEST_CONFIG",
    "BETTING_CONFIG",
    "FEATURES",
    "FEATURE_GROUPS",
    "LGBM_PARAMS",
    "MISSING_DEFAULTS",
    "PROFITABLE_SEGMENTS",
    "TRAINING_CONFIG",
    # Types
    "BacktestResults",
    "BetResult",
    # Data loading
    "RaceDataLoader",
    "OddsLoader",
    # Models
    "PositionProbabilityModel",
    "ExactaCalculator",
    "ExpectedValueCalculator",
    "ValueBet",
    # Calibration
    "BinningCalibration",
    "CalibrationAnalyzer",
    "CalibrationBin",
    "ExactaCalibrator",
    "IsotonicCalibration",
    "PlattScaling",
    "TemperatureScaling",
    "compare_calibration_methods",
    # Training & Evaluation
    "ModelTrainer",
    "ModelEvaluator",
    # Backtesting
    "Backtester",
    "BacktestReporter",
    "run_full_backtest_report",
]
