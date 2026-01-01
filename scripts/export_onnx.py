"""
Export LightGBM model to ONNX format for Rust inference.

Usage:
    uv run python scripts/export_onnx.py
"""

import logging
from pathlib import Path

import numpy as np
import onnx
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from src.models import PositionProbabilityModel
from src.models.config import FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths (updated to 39-feature model)
MODEL_PATH = Path("data/models/position_model_39features.pkl")
ONNX_PATH = Path("data/models/position_model.onnx")


def export_to_onnx():
    """Export LightGBM model to ONNX format."""
    logger.info(f"Loading model from {MODEL_PATH}")
    model = PositionProbabilityModel.load(MODEL_PATH)

    # Get the LightGBM booster
    lgb_model = model.model

    # Define input shape
    n_features = len(FEATURES)
    initial_types = [("input", FloatTensorType([None, n_features]))]

    logger.info(f"Converting to ONNX (features: {n_features})")

    # Convert to ONNX
    # Note: zipmap option is handled via post-processing or onnxmltools convert options
    onnx_model = convert_lightgbm(
        lgb_model,
        initial_types=initial_types,
        target_opset=12,
    )

    # Remove ZipMap operator from the model to get raw tensor output
    # LightGBM classifiers output ZipMap by default which is hard to handle in Rust
    from onnx import helper
    import onnx

    # Find and remove ZipMap nodes
    graph = onnx_model.graph
    nodes_to_remove = []
    zipmap_input = None

    for node in graph.node:
        if node.op_type == "ZipMap":
            nodes_to_remove.append(node)
            zipmap_input = node.input[0]

    if zipmap_input:
        # Remove ZipMap nodes
        for node in nodes_to_remove:
            graph.node.remove(node)

        # Print current outputs for debugging
        logger.info(f"Current outputs: {[o.name for o in graph.output]}")
        logger.info(f"ZipMap input was: {zipmap_input}")

        # Update graph outputs to use ZipMap input directly
        # Clear all outputs and create new ones
        while len(graph.output) > 0:
            graph.output.pop()

        # Add label output (keep original name 'label')
        label_output = helper.make_tensor_value_info(
            "label",
            onnx.TensorProto.INT64,
            [None],
        )
        graph.output.append(label_output)

        # Add probability tensor output using the ZipMap input name (lgbmprobabilities)
        prob_output = helper.make_tensor_value_info(
            zipmap_input,
            onnx.TensorProto.FLOAT,
            [None, 18],  # [batch_size, num_classes]
        )
        graph.output.append(prob_output)

        logger.info("Removed ZipMap operator for simpler tensor output")

    # Save the model
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, str(ONNX_PATH))
    logger.info(f"ONNX model saved to {ONNX_PATH}")

    # Verify the model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed")

    # Print model info
    print("\n" + "=" * 50)
    print("ONNX MODEL INFO")
    print("=" * 50)
    print(f"Input: {onnx_model.graph.input[0].name}")
    print(f"  Shape: [batch_size, {n_features}]")
    print(f"  Features: {FEATURES}")
    print(f"\nOutputs:")
    for output in onnx_model.graph.output:
        print(f"  - {output.name}")
    print("=" * 50)

    return onnx_model


def verify_inference():
    """Verify ONNX inference matches Python inference."""
    import onnxruntime as ort

    logger.info("Verifying inference consistency...")

    # Load both models
    py_model = PositionProbabilityModel.load(MODEL_PATH)
    onnx_session = ort.InferenceSession(str(ONNX_PATH))

    # Print model outputs info
    print("\nONNX Model Outputs:")
    for output in onnx_session.get_outputs():
        print(f"  - {output.name}: shape={output.shape}, type={output.type}")

    # Create sample input
    n_horses = 5
    n_features = len(FEATURES)
    sample_input = np.random.rand(n_horses, n_features).astype(np.float32)

    # Python inference
    py_output = py_model.predict_proba(sample_input)

    # ONNX inference
    input_name = onnx_session.get_inputs()[0].name
    onnx_outputs = onnx_session.run(None, {input_name: sample_input})

    # Check output format
    n_classes = 18

    # After removing ZipMap, output should be a tensor directly
    # Output 0 is labels, Output 1 is probabilities tensor
    if len(onnx_outputs) >= 2:
        onnx_probs_raw = onnx_outputs[1]

        # Check if it's already a numpy array (tensor output after ZipMap removal)
        if isinstance(onnx_probs_raw, np.ndarray):
            onnx_probs = onnx_probs_raw
            logger.info("ONNX output is tensor format (ZipMap removed)")
        elif isinstance(onnx_probs_raw, list):
            # Old ZipMap format: list of dicts
            logger.info("ONNX output is ZipMap format (list of dicts)")
            onnx_probs = np.zeros((n_horses, n_classes))
            for i, prob_dict in enumerate(onnx_probs_raw):
                for class_idx, prob in prob_dict.items():
                    if class_idx < n_classes:
                        onnx_probs[i, class_idx] = prob
        else:
            logger.warning(f"Unexpected ONNX output type: {type(onnx_probs_raw)}")
            return False
    else:
        logger.error(f"Expected at least 2 outputs, got {len(onnx_outputs)}")
        return False

    # Check if shapes match
    logger.info(f"Python output shape: {py_output.shape}")
    logger.info(f"ONNX output shape: {onnx_probs.shape}")

    # Compare values
    max_diff = np.abs(py_output - onnx_probs).max()
    logger.info(f"Max difference: {max_diff:.6f}")

    if max_diff < 1e-5:
        logger.info("Inference verification PASSED")
    else:
        logger.warning(f"Inference verification WARNING: max diff = {max_diff}")

    return max_diff < 1e-3


if __name__ == "__main__":
    export_to_onnx()

    # Install onnxruntime if not available
    try:
        verify_inference()
    except ImportError:
        logger.info("Install onnxruntime to verify inference: uv pip install onnxruntime")
