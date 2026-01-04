#!/usr/bin/env python
"""
Model retraining pipeline for Keiba-AI.

Orchestrates: Feature engineering → Training → Validation → Export

Usage:
    python scripts/retrain.py                    # Full pipeline (LightGBM)
    python scripts/retrain.py --model-type catboost   # Use CatBoost model
    python scripts/retrain.py --model-type ensemble   # Use ensemble model
    python scripts/retrain.py --model-type ensemble --stacking  # Ensemble with stacking
    python scripts/retrain.py --optimize         # Run hyperparameter optimization
    python scripts/retrain.py --skip-features    # Skip feature engineering
    python scripts/retrain.py --validate-only    # Only run validation
    python scripts/retrain.py --export-only      # Only run export (ONNX + calibration)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backtester import Backtester
from src.models.calibrator import TemperatureScaling, BinningCalibration
from src.models.config import FEATURES, TARGET_COL, DATE_COL, RACE_ID_COL
from src.models.data_loader import RaceDataLoader
from src.models.exacta_calculator import ExactaCalculator
from src.models.position_model import PositionProbabilityModel
from src.models.xgboost_model import XGBoostPositionModel
from src.models.catboost_model import CatBoostPositionModel
from src.models.ensemble import EnsembleModel
from src.models.trainer import ModelTrainer
from src.preprocessing.feature_engineering import FeatureEngineer

# Type alias for all model types
ModelType = Union[PositionProbabilityModel, XGBoostPositionModel, CatBoostPositionModel, EnsembleModel]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"
MODEL_PKL_PATH = MODELS_DIR / "position_model_39features.pkl"
ONNX_PATH = MODELS_DIR / "position_model.onnx"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"

# Baseline for comparison
BASELINE_ROI = 19.3

# Global model type setting
CURRENT_MODEL_TYPE = "lgbm"


def get_model_class(model_type: str):
    """Get the model class for the specified type.

    Args:
        model_type: One of 'lgbm', 'xgb', 'catboost', 'ensemble'

    Returns:
        Model class
    """
    if model_type == "lgbm":
        return PositionProbabilityModel
    elif model_type == "xgb":
        return XGBoostPositionModel
    elif model_type == "catboost":
        return CatBoostPositionModel
    elif model_type == "ensemble":
        return EnsembleModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model(model_type: str, params: dict = None) -> ModelType:
    """Create a model instance of the specified type.

    Args:
        model_type: One of 'lgbm', 'xgb', 'catboost', 'ensemble'
        params: Optional hyperparameters

    Returns:
        Model instance
    """
    model_class = get_model_class(model_type)
    if params:
        return model_class(params=params)
    return model_class()


def step_banner(step_num: int, title: str) -> None:
    """Print a step banner."""
    print("\n" + "=" * 60)
    print(f"STEP {step_num}: {title}")
    print("=" * 60)


def run_feature_engineering() -> bool:
    """Step 1: Create features from raw data."""
    step_banner(1, "FEATURE ENGINEERING")

    start = datetime.now()
    logger.info(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    engineer = FeatureEngineer()
    result = engineer.run()

    elapsed = (datetime.now() - start).total_seconds()

    if result is None:
        logger.error("Feature engineering failed")
        return False

    logger.info(f"Completed in {elapsed:.1f} seconds")
    logger.info(f"Output: {result}")
    return True


def run_training(model_type: str = "lgbm", params: dict = None, use_stacking: bool = False) -> bool:
    """Step 2: Train model with cross-validation.

    Args:
        model_type: Model type ('lgbm', 'xgb', 'catboost', 'ensemble')
        params: Optional hyperparameters from optimization
        use_stacking: Use stacking strategy for ensemble (default: False)
    """
    strategy_str = " with STACKING" if (model_type == "ensemble" and use_stacking) else ""
    step_banner(2, f"MODEL TRAINING ({model_type.upper()}{strategy_str})")

    start = datetime.now()
    logger.info(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using {len(FEATURES)} features")
    logger.info(f"Model type: {model_type}")
    if model_type == "ensemble":
        logger.info(f"Ensemble strategy: {'stacking' if use_stacking else 'weighted_average'}")

    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = df[df[TARGET_COL].notna()]
    logger.info(f"Loaded {len(df):,} samples")

    # Prepare data
    df = df.copy()
    df["_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=["_date"])
    df = df.sort_values("_date")

    # Split: last 3 months for test
    max_date = df["_date"].max()
    test_start = max_date - pd.DateOffset(months=3)
    train_df = df[df["_date"] < test_start].copy()
    test_df = df[df["_date"] >= test_start].copy()

    # Use last 20% of training data for validation (early stopping)
    split_idx = int(len(train_df) * 0.8)
    train_fit_df = train_df.iloc[:split_idx]
    val_df = train_df.iloc[split_idx:]

    X_train = train_fit_df[FEATURES].values
    y_train = (train_fit_df[TARGET_COL].clip(1, 18) - 1).astype(int).values
    X_val = val_df[FEATURES].values
    y_val = (val_df[TARGET_COL].clip(1, 18) - 1).astype(int).values

    logger.info(f"Train samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Test samples: {len(test_df):,}")

    # Create and train model
    if model_type == "ensemble":
        # Ensemble trains multiple models internally
        strategy = "stacking" if use_stacking else "weighted_average"
        final_model = EnsembleModel(strategy=strategy)
        final_model.train(
            X_train, y_train,
            X_val, y_val,
            feature_names=FEATURES,
        )
        logger.info(f"Ensemble info: {final_model.get_model_info()}")
    else:
        # Single model training
        final_model = create_model(model_type, params)
        final_model.train(
            X_train, y_train,
            X_val, y_val,
            feature_names=FEATURES,
        )

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"position_model_{model_type}.pkl"
    final_model.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Also save as default path for compatibility
    final_model.save(MODEL_PKL_PATH)
    logger.info(f"Model also saved to: {MODEL_PKL_PATH}")

    # Export and print feature importance
    print("\n" + "-" * 60)
    print("FEATURE IMPORTANCE")
    print("-" * 60)
    try:
        if hasattr(final_model, 'get_feature_importance'):
            importance = final_model.get_feature_importance()
        else:
            importance = final_model.feature_importance()

        # Save to CSV
        importance_path = MODELS_DIR / f"feature_importance_{model_type}.csv"
        importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to: {importance_path}")

        # Also save as default name for compatibility
        default_importance_path = MODELS_DIR / "feature_importance.csv"
        importance.to_csv(default_importance_path, index=False)

        # Print top 15
        print(f"\nTop 15 features (full list in {importance_path.name}):")
        print(importance.head(15).to_string(index=False))

        # Print summary stats
        total_importance = importance['importance'].sum()
        top_10_importance = importance.head(10)['importance'].sum()
        print(f"\nTop 10 features account for {top_10_importance/total_importance*100:.1f}% of total importance")
    except Exception as e:
        logger.warning(f"Could not get feature importance: {e}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Completed in {elapsed:.1f} seconds")

    return True


def run_optimization(model_type: str = "lgbm", n_trials: int = 50) -> dict:
    """Run hyperparameter optimization.

    Args:
        model_type: Model type to optimize
        n_trials: Number of optimization trials

    Returns:
        Best hyperparameters found
    """
    step_banner(0, f"HYPERPARAMETER OPTIMIZATION ({model_type.upper()})")

    start = datetime.now()
    logger.info(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Running {n_trials} optimization trials")

    from src.models.optimizer import ModelOptimizer

    optimizer = ModelOptimizer(
        model_type=model_type,
        n_trials=n_trials,
    )

    best_params = optimizer.optimize()

    # Save best params
    params_path = MODELS_DIR / f"best_params_{model_type}.json"
    optimizer.save_best_params(params_path)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Optimization completed in {elapsed:.1f} seconds")

    print("\n" + "-" * 60)
    print("BEST HYPERPARAMETERS")
    print("-" * 60)
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return optimizer.get_best_params()


def run_validation() -> Tuple[dict, List[Dict]]:
    """Step 3: Walk-forward backtest with calibration.

    Returns:
        Tuple of (results_dict, calibration_predictions)
        - results_dict: ROI, hit rate, etc.
        - calibration_predictions: List of {predicted_prob, won} for fitting calibrator
    """
    step_banner(3, "VALIDATION (WALK-FORWARD BACKTEST)")

    start = datetime.now()
    logger.info(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = df[df[TARGET_COL].notna()]
    logger.info(f"Loaded {len(df):,} samples")

    # Create backtester with isotonic calibration
    backtester = Backtester(calibration_method="isotonic")

    # Run walk-forward backtest
    results, period_results = backtester.run_walkforward_backtest(
        df=df,
        n_periods=6,
        train_months=18,
        test_months=3,
        calibration_months=3,
    )

    # Collect calibration data from all bets
    calibration_predictions = [
        {"predicted_prob": bet.predicted_prob, "won": bet.won}
        for bet in results.bets
    ]

    # Calculate summary
    total_bets = len(results.bets)
    total_wins = sum(1 for b in results.bets if b.won)
    total_wagered = sum(b.bet_amount for b in results.bets)
    total_return = sum(b.payout for b in results.bets)
    profit = total_return - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    hit_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0

    # Print results
    print("\n" + "-" * 60)
    print("BACKTEST RESULTS")
    print("-" * 60)
    print(f"""
Total Bets: {total_bets:,}
Total Wins: {total_wins:,}
Hit Rate: {hit_rate:.2f}%

Total Wagered: ¥{total_wagered:,.0f}
Total Return: ¥{total_return:,.0f}
Profit/Loss: ¥{profit:,.0f}

ROI: {roi:.2f}%
""")

    print("Period results:")
    for i, period in enumerate(period_results):
        print(f"  Period {i+1}: ROI = {period.get('roi', 0):.1f}%, Bets = {period.get('bets', 0)}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Completed in {elapsed:.1f} seconds")

    return {
        "roi": roi,
        "hit_rate": hit_rate,
        "total_bets": total_bets,
        "total_wins": total_wins,
        "profit": profit,
    }, calibration_predictions


def fit_calibration(predictions: List[Dict], method: str = "temperature") -> dict:
    """Fit calibration from validation predictions.

    Args:
        predictions: List of {predicted_prob, won} from validation
        method: 'temperature' or 'binning'

    Returns:
        Calibration config dict for JSON export
    """
    if not predictions:
        logger.warning("No predictions for calibration, using default")
        return {"type": "temperature", "temperature": 1.0}

    probs = np.array([p["predicted_prob"] for p in predictions])
    labels = np.array([1 if p["won"] else 0 for p in predictions])

    logger.info(f"Fitting {method} calibration on {len(probs):,} predictions")
    logger.info(f"  Hit rate: {labels.mean():.2%}")
    logger.info(f"  Avg predicted prob: {probs.mean():.4f}")

    if method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(probs, labels)
        return {
            "type": "temperature",
            "temperature": float(calibrator.temperature),
        }
    elif method == "binning":
        calibrator = BinningCalibration(n_bins=15)
        calibrator.fit(probs, labels)
        return {
            "type": "binning",
            "n_bins": int(calibrator.n_bins),
            "bin_edges": [float(x) for x in calibrator.bin_edges],
            "bin_values": [float(x) for x in calibrator.bin_values],
        }
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def run_export(calibration_predictions: List[Dict] = None) -> bool:
    """Step 4: Export ONNX model and calibration.

    Args:
        calibration_predictions: Optional list of {predicted_prob, won} for fitting calibrator.
                                 If None, uses default temperature=1.0.
    """
    step_banner(4, "EXPORT (ONNX + CALIBRATION)")

    start = datetime.now()
    logger.info(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # Import ONNX tools
    try:
        import onnx
        from onnx import helper
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as e:
        logger.error(f"Missing ONNX dependencies: {e}")
        logger.error("Install with: uv pip install onnx onnxmltools")
        return False

    # Load model
    if not MODEL_PKL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PKL_PATH}")
        return False

    logger.info(f"Loading model from {MODEL_PKL_PATH}")
    model = PositionProbabilityModel.load(MODEL_PKL_PATH)
    lgb_model = model.model

    # Convert to ONNX
    n_features = len(FEATURES)
    initial_types = [("input", FloatTensorType([None, n_features]))]

    logger.info(f"Converting to ONNX ({n_features} features)")
    onnx_model = convert_lightgbm(
        lgb_model,
        initial_types=initial_types,
        target_opset=12,
    )

    # Remove ZipMap operator for simpler tensor output
    graph = onnx_model.graph
    nodes_to_remove = []
    zipmap_input = None

    for node in graph.node:
        if node.op_type == "ZipMap":
            nodes_to_remove.append(node)
            zipmap_input = node.input[0]

    if zipmap_input:
        for node in nodes_to_remove:
            graph.node.remove(node)

        while len(graph.output) > 0:
            graph.output.pop()

        label_output = helper.make_tensor_value_info(
            "label", onnx.TensorProto.INT64, [None]
        )
        graph.output.append(label_output)

        prob_output = helper.make_tensor_value_info(
            zipmap_input, onnx.TensorProto.FLOAT, [None, 18]
        )
        graph.output.append(prob_output)

        logger.info("Removed ZipMap operator for simpler tensor output")

    # Save ONNX model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(ONNX_PATH))
    logger.info(f"ONNX model saved to: {ONNX_PATH}")

    # Verify ONNX model
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed")

    # Fit and export calibration
    if calibration_predictions:
        calibration_config = fit_calibration(calibration_predictions, method="temperature")
    else:
        logger.warning("No calibration predictions provided, using default temperature=1.0")
        calibration_config = {"type": "temperature", "temperature": 1.0}

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calibration_config, f, indent=2)
    logger.info(f"Calibration config saved to: {CALIBRATION_PATH}")

    # Print summary
    print("\n" + "-" * 60)
    print("EXPORT SUMMARY")
    print("-" * 60)
    print(f"ONNX Model: {ONNX_PATH}")
    print(f"  Input: [batch_size, {n_features}]")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    print(f"\nCalibration: {CALIBRATION_PATH}")
    print(f"  Type: {calibration_config['type']}")
    if calibration_config['type'] == 'temperature':
        print(f"  Temperature: {calibration_config['temperature']:.4f}")
    elif calibration_config['type'] == 'binning':
        print(f"  Bins: {calibration_config['n_bins']}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Completed in {elapsed:.1f} seconds")

    return True


def print_summary(validation_results: dict) -> None:
    """Print final summary and comparison."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    roi = validation_results.get("roi", 0)
    improvement = roi - BASELINE_ROI

    print(f"""
Model: {MODEL_PKL_PATH.name}
ONNX:  {ONNX_PATH.name}
Calibration: {CALIBRATION_PATH.name}

Validation Results:
  ROI: +{roi:.1f}%
  Hit Rate: {validation_results.get('hit_rate', 0):.2f}%
  Total Bets: {validation_results.get('total_bets', 0):,}
  Profit: ¥{validation_results.get('profit', 0):,.0f}

Comparison to Baseline:
  Baseline (23 features): +{BASELINE_ROI:.1f}% ROI
  Current (39 features):  +{roi:.1f}% ROI
  Improvement: {'+' if improvement >= 0 else ''}{improvement:.1f} percentage points
""")

    if roi > BASELINE_ROI:
        print("✓ Model improvement validated!")
    else:
        print("✗ Model did not improve over baseline")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Keiba-AI Model Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/retrain.py                         # Full pipeline (LightGBM)
  python scripts/retrain.py --model-type catboost   # Use CatBoost model
  python scripts/retrain.py --model-type xgb        # Use XGBoost model
  python scripts/retrain.py --model-type ensemble   # Use ensemble (LightGBM+XGBoost+CatBoost)
  python scripts/retrain.py --model-type ensemble --stacking  # Ensemble with stacking
  python scripts/retrain.py --optimize              # Run hyperparameter optimization first
  python scripts/retrain.py --optimize --n-trials 100  # More optimization trials
  python scripts/retrain.py --skip-features         # Skip feature engineering
  python scripts/retrain.py --validate-only         # Only run validation
  python scripts/retrain.py --export-only           # Only run ONNX/calibration export
        """,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lgbm", "xgb", "catboost", "ensemble"],
        default="lgbm",
        help="Model type to train (default: lgbm)",
    )
    parser.add_argument(
        "--stacking",
        action="store_true",
        help="Use stacking strategy for ensemble (trains meta-learner on CV predictions)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization before training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering step (use existing features.parquet)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation (requires trained model)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export ONNX and calibration (requires trained model)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("KEIBA-AI MODEL RETRAINING PIPELINE")
    print("=" * 60)
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {args.model_type}")
    if args.model_type == "ensemble":
        print(f"Ensemble strategy: {'stacking' if args.stacking else 'weighted_average'}")
    print(f"Features: {len(FEATURES)}")
    if args.optimize:
        print(f"Optimization: {args.n_trials} trials")

    start_time = datetime.now()
    validation_results = {}
    calibration_predictions = []
    optimized_params = None

    try:
        if args.validate_only:
            # Only validation
            validation_results, calibration_predictions = run_validation()
        elif args.export_only:
            # Only export (no calibration data available)
            if not run_export():
                sys.exit(1)
        else:
            # Full or partial pipeline

            # Step 0: Hyperparameter optimization (optional)
            if args.optimize and args.model_type != "ensemble":
                optimized_params = run_optimization(args.model_type, args.n_trials)

            # Step 1: Feature engineering
            if not args.skip_features:
                if not run_feature_engineering():
                    sys.exit(1)
            else:
                logger.info("Skipping feature engineering (--skip-features)")

            # Step 2: Training
            if not run_training(args.model_type, optimized_params, args.stacking):
                sys.exit(1)

            # Step 3: Validation (also collects calibration data)
            validation_results, calibration_predictions = run_validation()

            # Step 4: Export with fitted calibration
            # Note: ONNX export only works for LightGBM
            if args.model_type == "lgbm":
                if not run_export(calibration_predictions):
                    sys.exit(1)
            else:
                logger.warning(f"ONNX export not supported for {args.model_type}, skipping")
                # Still export calibration
                if calibration_predictions:
                    calibration_config = fit_calibration(calibration_predictions, method="temperature")
                    with open(CALIBRATION_PATH, "w") as f:
                        json.dump(calibration_config, f, indent=2)
                    logger.info(f"Calibration config saved to: {CALIBRATION_PATH}")

        # Print summary
        if validation_results:
            print_summary(validation_results)

        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nTotal time: {total_elapsed:.1f} seconds")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
