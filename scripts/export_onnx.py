"""
Export the trained position model to ONNX for Rust inference.

Supports both CatBoost (default — the deployed model, which weights market odds
heavily) and LightGBM. Both export a graph whose output index 1 is a raw
[N, 18] float probability tensor (ZipMap stripped), matching what the Rust
consumer expects (`src/api/src/model.rs::predict` reads `outputs[1]` by index).

Usage:
    uv run python scripts/export_onnx.py                 # auto-detect from default pkl
    uv run python scripts/export_onnx.py --model-type catboost
    uv run python scripts/export_onnx.py --pkl data/models/position_model_lgbm.pkl
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

from src.models.config import FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ONNX_PATH = Path("data/models/position_model.onnx")
DEFAULT_PKL = {
    "catboost": Path("data/models/position_model_catboost.pkl"),
    "lgbm": Path("data/models/position_model_39features.pkl"),
}
NUM_CLASSES = 18


def _load_model(pkl_path: Path):
    """Return the underlying estimator from a saved model dict."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["model"] if isinstance(data, dict) else data


def _strip_zipmap(onnx_model):
    """Replace the ZipMap (sequence-of-maps) probability output with a raw
    [None, NUM_CLASSES] float tensor at output index 1.

    Both CatBoost's native ONNX export and onnxmltools' LightGBM conversion emit
    a ZipMap that turns class probabilities into a sequence of {class: prob}
    maps. The Rust side reads `outputs[1]` as a 2D float tensor, so we drop the
    ZipMap node and expose its input tensor directly.
    """
    graph = onnx_model.graph
    zipmap_input = None
    for node in list(graph.node):
        if node.op_type == "ZipMap":
            zipmap_input = node.input[0]
            graph.node.remove(node)
    if zipmap_input is None:
        raise RuntimeError("No ZipMap node found; cannot locate the probability tensor")

    logger.info("Stripping ZipMap; probability tensor = %s", zipmap_input)
    while len(graph.output) > 0:
        graph.output.pop()
    graph.output.append(
        helper.make_tensor_value_info("label", onnx.TensorProto.INT64, [None])
    )
    graph.output.append(
        helper.make_tensor_value_info(
            zipmap_input, onnx.TensorProto.FLOAT, [None, NUM_CLASSES]
        )
    )
    return onnx_model


def export_catboost(pkl_path: Path):
    """Export a CatBoostClassifier to ONNX via its native exporter + ZipMap strip."""
    model = _load_model(pkl_path)
    logger.info("Loaded %s from %s", type(model).__name__, pkl_path)

    raw_path = ONNX_PATH.with_suffix(".catboost_raw.onnx")
    model.save_model(str(raw_path), format="onnx")
    onnx_model = _strip_zipmap(onnx.load(str(raw_path)))
    onnx.checker.check_model(onnx_model)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(ONNX_PATH))
    raw_path.unlink(missing_ok=True)
    logger.info("ONNX model saved to %s", ONNX_PATH)
    return model


def export_lightgbm(pkl_path: Path):
    """Export a LightGBM Booster to ONNX via onnxmltools + ZipMap strip."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    model = _load_model(pkl_path)
    logger.info("Loaded %s from %s", type(model).__name__, pkl_path)

    initial_types = [("input", FloatTensorType([None, len(FEATURES)]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=12)
    onnx_model = _strip_zipmap(onnx_model)
    onnx.checker.check_model(onnx_model)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(ONNX_PATH))
    logger.info("ONNX model saved to %s", ONNX_PATH)
    return model


def _predict_proba(model, X):
    """Class probabilities for either estimator type."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # LightGBM Booster: predict() already returns class probabilities for multiclass
    return model.predict(X)


def verify(model):
    """Parity gate: ONNX output[1] must match the estimator's probabilities and
    be a [N, NUM_CLASSES] float tensor."""
    import onnxruntime as ort
    import pandas as pd

    df = pd.read_parquet("data/processed/features.parquet")
    X = df[FEATURES].head(16).to_numpy(dtype=np.float32)

    sess = ort.InferenceSession(str(ONNX_PATH))
    outs = sess.run(None, {sess.get_inputs()[0].name: X})
    if len(outs) < 2:
        raise RuntimeError(f"Expected >=2 ONNX outputs, got {len(outs)}")
    probs = np.asarray(outs[1])
    expected = np.asarray(_predict_proba(model, X))

    print("\n" + "=" * 50)
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
    print("onnx probs shape/dtype:", probs.shape, probs.dtype)
    print("estimator probs shape:", expected.shape)
    if probs.shape != expected.shape:
        raise RuntimeError(f"Shape mismatch: onnx {probs.shape} vs model {expected.shape}")
    max_diff = float(np.max(np.abs(probs - expected)))
    print(f"max abs diff: {max_diff:.2e}")
    print("PARITY PASS" if max_diff < 1e-4 else "PARITY FAIL")
    print("=" * 50)
    if max_diff >= 1e-4:
        raise RuntimeError(f"Parity failed: max diff {max_diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["catboost", "lgbm"], default="catboost")
    ap.add_argument("--pkl", type=Path, default=None, help="Override source pkl path")
    args = ap.parse_args()

    pkl = args.pkl or DEFAULT_PKL[args.model_type]
    model = (
        export_catboost(pkl) if args.model_type == "catboost" else export_lightgbm(pkl)
    )
    verify(model)


if __name__ == "__main__":
    main()
