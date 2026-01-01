#!/usr/bin/env python
"""
Model retraining pipeline for Keiba-AI.

Orchestrates: Feature engineering → Training → Validation → Export

Usage:
    python scripts/retrain.py                    # Full pipeline
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

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backtester import Backtester
from src.models.config import FEATURES, TARGET_COL
from src.models.data_loader import RaceDataLoader
from src.models.position_model import PositionProbabilityModel
from src.models.trainer import ModelTrainer
from src.preprocessing.feature_engineering import FeatureEngineer

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


def run_training() -> bool:
    """Step 2: Train model with cross-validation."""
    step_banner(2, "MODEL TRAINING")

    start = datetime.now()
    logger.info(f"Started at {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Using {len(FEATURES)} features")

    # Load data
    loader = RaceDataLoader()
    df = loader.load_features()
    df = df[df[TARGET_COL].notna()]
    logger.info(f"Loaded {len(df):,} samples")

    # Train with cross-validation
    trainer = ModelTrainer(n_splits=5)
    models, cv_results = trainer.train_with_cv(
        df,
        feature_cols=FEATURES,
        target_col=TARGET_COL,
    )

    # Print CV summary
    trainer.summarize_cv_results()

    # Train final model
    final_model, test_df = trainer.train_final_model(
        df,
        feature_cols=FEATURES,
        target_col=TARGET_COL,
    )

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save(MODEL_PKL_PATH)
    logger.info(f"Model saved to: {MODEL_PKL_PATH}")

    # Print feature importance
    print("\n" + "-" * 60)
    print("TOP 15 FEATURES")
    print("-" * 60)
    importance = final_model.get_feature_importance()
    print(importance.head(15).to_string(index=False))

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Completed in {elapsed:.1f} seconds")

    return True


def run_validation() -> dict:
    """Step 3: Walk-forward backtest with calibration."""
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
    }


def run_export() -> bool:
    """Step 4: Export ONNX model and calibration."""
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

    # Create default calibration config (temperature scaling)
    # In production, this should be fitted from validation data
    calibration_config = {
        "type": "temperature",
        "temperature": 1.0,  # Default no-op calibration
    }

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
    print(f"  Temperature: {calibration_config['temperature']}")

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
  python scripts/retrain.py                    # Full pipeline
  python scripts/retrain.py --skip-features    # Skip feature engineering
  python scripts/retrain.py --validate-only    # Only run validation
  python scripts/retrain.py --export-only      # Only run ONNX/calibration export
        """,
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
    print(f"Features: {len(FEATURES)}")

    start_time = datetime.now()
    validation_results = {}

    try:
        if args.validate_only:
            # Only validation
            validation_results = run_validation()
        elif args.export_only:
            # Only export
            if not run_export():
                sys.exit(1)
        else:
            # Full or partial pipeline

            # Step 1: Feature engineering
            if not args.skip_features:
                if not run_feature_engineering():
                    sys.exit(1)
            else:
                logger.info("Skipping feature engineering (--skip-features)")

            # Step 2: Training
            if not run_training():
                sys.exit(1)

            # Step 3: Validation
            validation_results = run_validation()

            # Step 4: Export
            if not run_export():
                sys.exit(1)

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
