//! Request and response types for the Keiba API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Horse features for prediction
#[derive(Debug, Clone, Deserialize)]
pub struct HorseFeatures {
    pub horse_age_num: f32,
    pub horse_sex_encoded: f32,
    pub post_position_num: f32,
    pub weight_carried: f32,
    pub horse_weight: f32,
    pub jockey_win_rate: f32,
    pub jockey_place_rate: f32,
    pub trainer_win_rate: f32,
    pub jockey_races: f32,
    pub trainer_races: f32,
    pub distance_num: f32,
    pub is_turf: f32,
    pub is_dirt: f32,
    pub track_condition_num: f32,
    pub avg_position_last_3: f32,
    pub avg_position_last_5: f32,
    pub win_rate_last_3: f32,
    pub win_rate_last_5: f32,
    pub place_rate_last_3: f32,
    pub place_rate_last_5: f32,
    pub last_position: f32,
    pub career_races: f32,
    pub odds_log: f32,
}

impl HorseFeatures {
    /// Convert features to array in model input order
    pub fn to_array(&self) -> [f32; 23] {
        [
            self.horse_age_num,
            self.horse_sex_encoded,
            self.post_position_num,
            self.weight_carried,
            self.horse_weight,
            self.jockey_win_rate,
            self.jockey_place_rate,
            self.trainer_win_rate,
            self.jockey_races,
            self.trainer_races,
            self.distance_num,
            self.is_turf,
            self.is_dirt,
            self.track_condition_num,
            self.avg_position_last_3,
            self.avg_position_last_5,
            self.win_rate_last_3,
            self.win_rate_last_5,
            self.place_rate_last_3,
            self.place_rate_last_5,
            self.last_position,
            self.career_races,
            self.odds_log,
        ]
    }
}

/// Horse entry in prediction request
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct HorseEntry {
    pub horse_id: String,
    pub horse_name: Option<String>,
    pub features: HorseFeatures,
}

/// Prediction request for all bet types
#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub race_id: String,
    pub horses: Vec<HorseEntry>,
    /// Optional exacta odds: "horse1-horse2" -> odds (Japanese format, e.g., 1520)
    #[serde(default)]
    pub exacta_odds: HashMap<String, f64>,
    /// Optional trifecta odds: "horse1-horse2-horse3" -> odds
    #[serde(default)]
    pub trifecta_odds: HashMap<String, f64>,
    /// Optional quinella odds: "horse1-horse2" -> odds (unordered)
    #[serde(default)]
    pub quinella_odds: HashMap<String, f64>,
    /// Optional trio odds: "horse1-horse2-horse3" -> odds (unordered)
    #[serde(default)]
    pub trio_odds: HashMap<String, f64>,
    /// Optional wide odds: "horse1-horse2" -> odds (unordered, top 3)
    #[serde(default)]
    pub wide_odds: HashMap<String, f64>,
    /// Which bet types to calculate (default: all available)
    #[serde(default)]
    pub bet_types: Vec<String>,
}

/// Exacta prediction result
#[derive(Debug, Clone, Serialize)]
pub struct ExactaPrediction {
    pub first: String,
    pub second: String,
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub odds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge: Option<f64>,
    pub recommended: bool,
}

/// Trifecta prediction result
#[derive(Debug, Clone, Serialize)]
pub struct TrifectaPrediction {
    pub first: String,
    pub second: String,
    pub third: String,
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub odds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge: Option<f64>,
    pub recommended: bool,
}

/// Quinella prediction result (unordered pair)
#[derive(Debug, Clone, Serialize)]
pub struct QuinellaPrediction {
    pub horses: (String, String),
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub odds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge: Option<f64>,
    pub recommended: bool,
}

/// Trio prediction result (unordered triple)
#[derive(Debug, Clone, Serialize)]
pub struct TrioPrediction {
    pub horses: (String, String, String),
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub odds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge: Option<f64>,
    pub recommended: bool,
}

/// Wide prediction result (unordered pair, both in top 3)
#[derive(Debug, Clone, Serialize)]
pub struct WidePrediction {
    pub horses: (String, String),
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub odds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge: Option<f64>,
    pub recommended: bool,
}

/// Betting signal
#[derive(Debug, Clone, Serialize)]
pub struct BettingSignal {
    pub combination: (String, String),
    pub bet_type: String,
    pub probability: f64,
    pub odds: f64,
    pub expected_value: f64,
    pub kelly_fraction: f64,
    pub recommended_bet: u32,
}

/// Prediction results for all bet types
#[derive(Debug, Serialize, Default)]
pub struct Predictions {
    pub win_probabilities: HashMap<String, f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_exactas: Vec<ExactaPrediction>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_trifectas: Vec<TrifectaPrediction>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_quinellas: Vec<QuinellaPrediction>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_trios: Vec<TrioPrediction>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_wides: Vec<WidePrediction>,
}

/// Betting signals grouped by bet type
#[derive(Debug, Serialize, Default)]
pub struct BettingSignals {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub exacta: Vec<BettingSignal>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub trifecta: Vec<BettingSignal>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub quinella: Vec<BettingSignal>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub trio: Vec<BettingSignal>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub wide: Vec<BettingSignal>,
}

/// Full prediction response
#[derive(Debug, Serialize)]
pub struct PredictResponse {
    pub race_id: String,
    pub predictions: Predictions,
    pub betting_signals: BettingSignals,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Model info response
#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    pub model_path: String,
    pub num_features: usize,
    pub feature_names: Vec<String>,
}

/// API error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}
