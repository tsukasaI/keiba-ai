"""
Keiba AI Prediction System - Model Module

Models for horse racing position prediction and multi-bet type betting.
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
from .types import (
    BacktestResults,
    BetResult,
    QuinellaBetResult,
    QuinellaBacktestResults,
    TrioBetResult,
    TrioBacktestResults,
    TrifectaBetResult,
    TrifectaBacktestResults,
    WideBetResult,
    WideBacktestResults,
)
from .data_loader import RaceDataLoader
from .odds_loader import OddsLoader
from .position_model import PositionProbabilityModel
from .exacta_calculator import ExactaCalculator
from .trifecta_calculator import TrifectaCalculator
from .quinella_calculator import QuinellaCalculator
from .trio_calculator import TrioCalculator
from .wide_calculator import WideCalculator
from .expected_value import ExpectedValueCalculator, ValueBet
from .calibrator import (
    BinningCalibration,
    CalibrationAnalyzer,
    CalibrationBin,
    ExactaCalibrator,
    IsotonicCalibration,
    PlattScaling,
    QuinellaCalibrator,
    TemperatureScaling,
    TrioCalibrator,
    TrifectaCalibrator,
    WideCalibrator,
    compare_calibration_methods,
)
from .export_calibration import (
    export_calibrator_to_json,
    create_temperature_calibrator,
    create_binning_calibrator,
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
    "QuinellaBetResult",
    "QuinellaBacktestResults",
    "TrioBetResult",
    "TrioBacktestResults",
    "TrifectaBetResult",
    "TrifectaBacktestResults",
    "WideBetResult",
    "WideBacktestResults",
    # Data loading
    "RaceDataLoader",
    "OddsLoader",
    # Models
    "PositionProbabilityModel",
    "ExactaCalculator",
    "TrifectaCalculator",
    "QuinellaCalculator",
    "TrioCalculator",
    "WideCalculator",
    "ExpectedValueCalculator",
    "ValueBet",
    # Calibration
    "BinningCalibration",
    "CalibrationAnalyzer",
    "CalibrationBin",
    "ExactaCalibrator",
    "IsotonicCalibration",
    "PlattScaling",
    "QuinellaCalibrator",
    "TemperatureScaling",
    "TrioCalibrator",
    "TrifectaCalibrator",
    "WideCalibrator",
    "compare_calibration_methods",
    "export_calibrator_to_json",
    "create_temperature_calibrator",
    "create_binning_calibrator",
    # Training & Evaluation
    "ModelTrainer",
    "ModelEvaluator",
    # Backtesting
    "Backtester",
    "BacktestReporter",
    "run_full_backtest_report",
]
