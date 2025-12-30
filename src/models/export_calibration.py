"""
Export Python calibrators to JSON format for Rust API consumption.

Usage:
    python export_calibration.py <calibrator.pkl> <output.json>

Example:
    python export_calibration.py data/models/exacta_calibrator.pkl data/models/calibration.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Union

from calibrator import (
    BinningCalibration,
    ExactaCalibrator,
    QuinellaCalibrator,
    TemperatureScaling,
    TrifectaCalibrator,
    TrioCalibrator,
    WideCalibrator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_temperature_scaling(calibrator: TemperatureScaling) -> Dict[str, Any]:
    """Export TemperatureScaling to JSON-compatible dict."""
    return {
        "type": "temperature",
        "temperature": float(calibrator.temperature),
    }


def export_binning_calibration(calibrator: BinningCalibration) -> Dict[str, Any]:
    """Export BinningCalibration to JSON-compatible dict."""
    return {
        "type": "binning",
        "n_bins": int(calibrator.n_bins),
        "bin_edges": [float(x) for x in calibrator.bin_edges],
        "bin_values": [float(x) for x in calibrator.bin_values],
    }


def export_calibrator(
    calibrator: Union[
        TemperatureScaling,
        BinningCalibration,
        ExactaCalibrator,
        TrifectaCalibrator,
        QuinellaCalibrator,
        TrioCalibrator,
        WideCalibrator,
    ]
) -> Dict[str, Any]:
    """
    Export a calibrator to JSON-compatible dict.

    Supported calibrators:
    - TemperatureScaling -> {"type": "temperature", "temperature": 1.15}
    - BinningCalibration -> {"type": "binning", "n_bins": 15, "bin_edges": [...], "bin_values": [...]}
    - ExactaCalibrator (and other wrappers) -> extracts inner calibrator

    Note: PlattScaling and IsotonicCalibration are not directly supported
    as they require sklearn model serialization. For these, use binning
    approximation or temperature scaling as an alternative.
    """
    # Handle wrapper classes (ExactaCalibrator, etc.)
    if hasattr(calibrator, "calibrator"):
        inner = calibrator.calibrator
        method = getattr(calibrator, "method", "unknown")
        logger.info(f"Extracting inner calibrator (method: {method})")
        return export_calibrator(inner)

    # Handle direct calibrator types
    if isinstance(calibrator, TemperatureScaling):
        if not calibrator.fitted:
            raise ValueError("TemperatureScaling calibrator is not fitted")
        return export_temperature_scaling(calibrator)

    elif isinstance(calibrator, BinningCalibration):
        if not calibrator.fitted:
            raise ValueError("BinningCalibration calibrator is not fitted")
        return export_binning_calibration(calibrator)

    else:
        raise ValueError(
            f"Unsupported calibrator type: {type(calibrator).__name__}. "
            "Only TemperatureScaling and BinningCalibration can be exported to JSON. "
            "For PlattScaling or IsotonicCalibration, consider retraining with "
            "method='temperature' or method='binning'."
        )


def export_calibrator_to_json(
    calibrator: Any,
    output_path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Export a calibrator to JSON file.

    Args:
        calibrator: The calibrator to export (TemperatureScaling, BinningCalibration,
                   or wrapper like ExactaCalibrator)
        output_path: Path to save the JSON file
        indent: JSON indentation level (default: 2)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = export_calibrator(calibrator)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=indent)

    logger.info(f"Calibrator exported to: {output_path}")
    logger.info(f"Config: {json.dumps(config, indent=indent)}")


def load_and_export(pickle_path: Union[str, Path], json_path: Union[str, Path]) -> None:
    """
    Load a pickled calibrator and export to JSON.

    Args:
        pickle_path: Path to the pickled calibrator file
        json_path: Path to save the JSON file
    """
    pickle_path = Path(pickle_path)

    # Try loading as a wrapper calibrator first
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        # Handle pickle format used by save() method
        if isinstance(data, dict) and "calibrator" in data:
            calibrator = data["calibrator"]
            method = data.get("method", "unknown")
            logger.info(f"Loaded calibrator from pickle (method: {method})")
        else:
            calibrator = data
            logger.info(f"Loaded calibrator from pickle (type: {type(calibrator).__name__})")

        export_calibrator_to_json(calibrator, json_path)

    except Exception as e:
        logger.error(f"Failed to load/export calibrator: {e}")
        raise


def create_temperature_calibrator(temperature: float, output_path: Union[str, Path]) -> None:
    """
    Create a temperature scaling JSON config directly.

    Args:
        temperature: Temperature value (typically 0.5-2.0)
        output_path: Path to save the JSON file
    """
    config = {
        "type": "temperature",
        "temperature": float(temperature),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Temperature calibrator created: {output_path}")
    logger.info(f"Temperature: {temperature}")


def create_binning_calibrator(
    n_bins: int,
    bin_values: list,
    output_path: Union[str, Path],
) -> None:
    """
    Create a binning calibration JSON config directly.

    Args:
        n_bins: Number of bins
        bin_values: List of calibrated values for each bin
        output_path: Path to save the JSON file
    """
    if len(bin_values) != n_bins:
        raise ValueError(f"bin_values length ({len(bin_values)}) must match n_bins ({n_bins})")

    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    config = {
        "type": "binning",
        "n_bins": n_bins,
        "bin_edges": bin_edges,
        "bin_values": [float(v) for v in bin_values],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Binning calibrator created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Python calibrators to JSON for Rust API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a pickled calibrator to JSON
  python export_calibration.py calibrator.pkl calibration.json

  # Create a temperature scaling config directly
  python export_calibration.py --temperature 1.15 calibration.json

  # Create a binning config directly
  python export_calibration.py --binning 10 --values 0.01,0.03,0.05,0.08,0.12,0.18,0.25,0.35,0.50,0.70 calibration.json
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Path to pickled calibrator file (required unless using --temperature or --binning)",
    )
    parser.add_argument(
        "output",
        help="Path to save JSON config file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Create temperature scaling config with this value",
    )
    parser.add_argument(
        "--binning",
        type=int,
        help="Create binning config with this many bins",
    )
    parser.add_argument(
        "--values",
        type=str,
        help="Comma-separated bin values (required with --binning)",
    )

    args = parser.parse_args()

    if args.temperature is not None:
        create_temperature_calibrator(args.temperature, args.output)
    elif args.binning is not None:
        if not args.values:
            parser.error("--values is required when using --binning")
        values = [float(v.strip()) for v in args.values.split(",")]
        create_binning_calibrator(args.binning, values, args.output)
    elif args.input:
        load_and_export(args.input, args.output)
    else:
        parser.error("Either input pickle file, --temperature, or --binning is required")


if __name__ == "__main__":
    main()
