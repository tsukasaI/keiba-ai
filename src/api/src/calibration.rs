//! Probability calibration methods.
//!
//! Calibration adjusts model probabilities to better match actual frequencies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Temperature scaling calibration.
///
/// Applies a learned temperature to scale logits:
/// calibrated_prob = sigmoid(logit / temperature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScaling {
    pub temperature: f64,
}

impl Default for TemperatureScaling {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl TemperatureScaling {
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature: temperature.clamp(0.1, 10.0),
        }
    }

    /// Calibrate a single probability.
    pub fn calibrate(&self, prob: f64) -> f64 {
        let eps = 1e-10;
        let prob_clipped = prob.clamp(eps, 1.0 - eps);

        // Convert to logit
        let logit = (prob_clipped / (1.0 - prob_clipped)).ln();

        // Scale by temperature
        let scaled_logit = logit / self.temperature;

        // Convert back to probability
        1.0 / (1.0 + (-scaled_logit).exp())
    }

    /// Calibrate multiple probabilities.
    pub fn calibrate_vec(&self, probs: &[f64]) -> Vec<f64> {
        probs.iter().map(|p| self.calibrate(*p)).collect()
    }

    /// Calibrate a probability map.
    pub fn calibrate_map<K: Clone + std::hash::Hash + Eq>(
        &self,
        probs: &HashMap<K, f64>,
    ) -> HashMap<K, f64> {
        probs
            .iter()
            .map(|(k, v)| (k.clone(), self.calibrate(*v)))
            .collect()
    }
}

/// Binning calibration (histogram binning).
///
/// Maps probabilities to bin averages learned from calibration data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinningCalibration {
    pub n_bins: usize,
    pub bin_edges: Vec<f64>,
    pub bin_values: Vec<f64>,
}

impl Default for BinningCalibration {
    fn default() -> Self {
        Self::new(10)
    }
}

impl BinningCalibration {
    pub fn new(n_bins: usize) -> Self {
        let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();
        let bin_values: Vec<f64> = (0..n_bins)
            .map(|i| (bin_edges[i] + bin_edges[i + 1]) / 2.0)
            .collect();

        Self {
            n_bins,
            bin_edges,
            bin_values,
        }
    }

    /// Create from pre-fitted bin values.
    pub fn from_values(bin_values: Vec<f64>) -> Self {
        let n_bins = bin_values.len();
        let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();

        Self {
            n_bins,
            bin_edges,
            bin_values,
        }
    }

    /// Find which bin a probability falls into.
    fn find_bin(&self, prob: f64) -> usize {
        for i in 0..self.n_bins {
            if prob >= self.bin_edges[i] && prob < self.bin_edges[i + 1] {
                return i;
            }
        }
        // Edge case: prob == 1.0
        self.n_bins - 1
    }

    /// Calibrate a single probability.
    pub fn calibrate(&self, prob: f64) -> f64 {
        let bin_idx = self.find_bin(prob);
        self.bin_values[bin_idx]
    }

    /// Calibrate multiple probabilities.
    pub fn calibrate_vec(&self, probs: &[f64]) -> Vec<f64> {
        probs.iter().map(|p| self.calibrate(*p)).collect()
    }

    /// Calibrate a probability map.
    pub fn calibrate_map<K: Clone + std::hash::Hash + Eq>(
        &self,
        probs: &HashMap<K, f64>,
    ) -> HashMap<K, f64> {
        probs
            .iter()
            .map(|(k, v)| (k.clone(), self.calibrate(*v)))
            .collect()
    }
}

/// Calibrator enum for runtime selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Calibrator {
    Temperature(TemperatureScaling),
    Binning(BinningCalibration),
    None,
}

impl Default for Calibrator {
    fn default() -> Self {
        Calibrator::None
    }
}

impl Calibrator {
    /// Calibrate a single probability.
    pub fn calibrate(&self, prob: f64) -> f64 {
        match self {
            Calibrator::Temperature(t) => t.calibrate(prob),
            Calibrator::Binning(b) => b.calibrate(prob),
            Calibrator::None => prob,
        }
    }

    /// Calibrate multiple probabilities.
    pub fn calibrate_vec(&self, probs: &[f64]) -> Vec<f64> {
        match self {
            Calibrator::Temperature(t) => t.calibrate_vec(probs),
            Calibrator::Binning(b) => b.calibrate_vec(probs),
            Calibrator::None => probs.to_vec(),
        }
    }

    /// Calibrate a probability map.
    pub fn calibrate_map<K: Clone + std::hash::Hash + Eq>(
        &self,
        probs: &HashMap<K, f64>,
    ) -> HashMap<K, f64> {
        match self {
            Calibrator::Temperature(t) => t.calibrate_map(probs),
            Calibrator::Binning(b) => b.calibrate_map(probs),
            Calibrator::None => probs.clone(),
        }
    }

    /// Check if calibration is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Calibrator::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scaling_identity() {
        let ts = TemperatureScaling::new(1.0);
        let prob = 0.5;
        let calibrated = ts.calibrate(prob);
        assert!((calibrated - prob).abs() < 0.01);
    }

    #[test]
    fn test_temperature_scaling_low_temp() {
        // Low temperature should push probabilities toward extremes
        let ts = TemperatureScaling::new(0.5);
        let high_prob = 0.7;
        let calibrated = ts.calibrate(high_prob);
        assert!(calibrated > high_prob);
    }

    #[test]
    fn test_temperature_scaling_high_temp() {
        // High temperature should push probabilities toward 0.5
        let ts = TemperatureScaling::new(2.0);
        let high_prob = 0.9;
        let calibrated = ts.calibrate(high_prob);
        assert!(calibrated < high_prob);
    }

    #[test]
    fn test_binning_calibration() {
        let bc = BinningCalibration::new(10);

        // Probability 0.05 should fall in bin 0 (0.0-0.1)
        let prob = 0.05;
        let calibrated = bc.calibrate(prob);
        assert!((calibrated - 0.05).abs() < 0.01);

        // Probability 0.95 should fall in bin 9 (0.9-1.0)
        let prob = 0.95;
        let calibrated = bc.calibrate(prob);
        assert!((calibrated - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_binning_from_values() {
        let bin_values = vec![0.02, 0.08, 0.15, 0.25, 0.4, 0.55, 0.65, 0.75, 0.85, 0.95];
        let bc = BinningCalibration::from_values(bin_values.clone());

        // Probability 0.05 should return bin 0's value (0.02)
        let prob = 0.05;
        let calibrated = bc.calibrate(prob);
        assert!((calibrated - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_calibrator_enum() {
        let ts = Calibrator::Temperature(TemperatureScaling::new(1.0));
        assert!(ts.is_enabled());

        let none = Calibrator::None;
        assert!(!none.is_enabled());
    }

    #[test]
    fn test_calibrate_map() {
        let ts = TemperatureScaling::new(1.0);
        let mut probs = HashMap::new();
        probs.insert("A".to_string(), 0.5);
        probs.insert("B".to_string(), 0.3);

        let calibrated = ts.calibrate_map(&probs);
        assert_eq!(calibrated.len(), 2);
        assert!((calibrated["A"] - 0.5).abs() < 0.01);
    }
}
