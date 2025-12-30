//! ONNX model loading and inference.

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::config::FEATURE_NAMES;

/// Number of position classes (1st through 18th)
pub const NUM_CLASSES: usize = 18;

/// Number of input features
pub const NUM_FEATURES: usize = 23;

/// ONNX model wrapper for position probability prediction.
pub struct PositionModel {
    session: Mutex<Session>,
}

impl PositionModel {
    /// Load ONNX model from file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path.as_ref())
            .context("Failed to load ONNX model")?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Predict position probabilities for a batch of horses.
    ///
    /// # Arguments
    /// * `features` - 2D array of shape (n_horses, NUM_FEATURES)
    ///
    /// # Returns
    /// Vector of position probability vectors per horse
    pub fn predict(&self, features: Array2<f32>) -> Result<Vec<Vec<f64>>> {
        let n_horses = features.nrows();

        // Create input tensor from ndarray
        let input_tensor = Tensor::from_array(features)?;

        // Lock the session for inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock session: {}", e))?;

        // Run inference
        let outputs = session.run(ort::inputs![input_tensor])?;

        // Get probabilities output (output index 1: "lgbmprobabilities")
        // Output 0 is the predicted labels, output 1 is probabilities
        if outputs.len() < 2 {
            anyhow::bail!("Expected at least 2 outputs from model");
        }

        // Extract probabilities tensor
        // Output is a 2D tensor of shape [n_horses, 18]
        // try_extract_tensor returns (&Shape, &[f32])
        let (shape, probs_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .context("Failed to extract probability tensor")?;

        let shape_dims: Vec<i64> = shape.iter().copied().collect();
        if shape_dims.len() != 2 || (shape_dims[1] as usize) < NUM_CLASSES {
            anyhow::bail!(
                "Unexpected output shape: {:?}, expected [{}, {}]",
                shape_dims,
                n_horses,
                NUM_CLASSES
            );
        }

        let n_cols = shape_dims[1] as usize;

        // Convert to Vec<Vec<f64>>
        let mut result = Vec::with_capacity(n_horses);
        for i in 0..n_horses {
            let mut probs = Vec::with_capacity(NUM_CLASSES);
            for j in 0..NUM_CLASSES {
                probs.push(probs_data[i * n_cols + j] as f64);
            }
            result.push(probs);
        }

        Ok(result)
    }

    /// Get feature names.
    #[allow(dead_code)]
    pub fn feature_names(&self) -> &[&str] {
        &FEATURE_NAMES
    }
}

/// Thread-safe model wrapper for use in web handlers.
pub type SharedModel = Arc<PositionModel>;

/// Create a shared model instance.
pub fn create_shared_model<P: AsRef<Path>>(path: P) -> Result<SharedModel> {
    let model = PositionModel::load(path)?;
    Ok(Arc::new(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_names() {
        assert_eq!(FEATURE_NAMES.len(), NUM_FEATURES);
        assert_eq!(FEATURE_NAMES[0], "horse_age_num");
        assert_eq!(FEATURE_NAMES[22], "odds_log");
    }
}
