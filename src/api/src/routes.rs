//! API route handlers.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use ndarray::Array2;
use std::sync::Arc;

use crate::betting::find_value_bets;
use crate::config::{AppConfig, FEATURE_NAMES};
use crate::exacta::{calculate_exacta_probs, extract_win_probs, get_top_exactas};
use crate::model::{SharedModel, NUM_FEATURES};
use crate::types::{
    ErrorResponse, ExactaPrediction, HealthResponse, ModelInfoResponse, PredictRequest,
    PredictResponse, Predictions,
};

/// Application state shared across handlers.
pub struct AppState {
    pub model: SharedModel,
    pub config: AppConfig,
}

/// Error type for API handlers.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            error: self.status.to_string(),
            message: self.message,
        });
        (self.status, body).into_response()
    }
}

/// Health check endpoint.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Model info endpoint.
pub async fn model_info(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    Json(ModelInfoResponse {
        model_path: state.config.model.path.clone(),
        num_features: NUM_FEATURES,
        feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
    })
}

/// Prediction endpoint.
pub async fn predict(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    // Validate request
    if req.horses.is_empty() {
        return Err(ApiError::bad_request("No horses provided"));
    }

    if req.horses.len() < 2 {
        return Err(ApiError::bad_request("At least 2 horses required for exacta"));
    }

    // Build feature matrix
    let n_horses = req.horses.len();
    let mut features = Array2::<f32>::zeros((n_horses, NUM_FEATURES));

    let horse_ids: Vec<String> = req.horses.iter().map(|h| h.horse_id.clone()).collect();

    for (i, horse) in req.horses.iter().enumerate() {
        let feature_array = horse.features.to_array();
        for (j, &val) in feature_array.iter().enumerate() {
            features[[i, j]] = val;
        }
    }

    // Run model inference
    let position_probs = state
        .model
        .predict(features)
        .map_err(|e| ApiError::internal(format!("Model inference failed: {}", e)))?;

    // Extract win probabilities
    let win_probs = extract_win_probs(&position_probs, &horse_ids);

    // Calculate exacta probabilities
    let exacta_probs = calculate_exacta_probs(&win_probs, state.config.betting.min_probability);

    // Get top exactas
    let top_exactas = get_top_exactas(&exacta_probs, state.config.betting.max_combinations);

    // Build exacta predictions with odds if available
    let exacta_predictions: Vec<ExactaPrediction> = top_exactas
        .iter()
        .map(|((first, second), prob)| {
            let odds_key = format!("{}-{}", first, second);
            let odds = req.exacta_odds.get(&odds_key).copied();
            let (ev, edge) = match odds {
                Some(o) => {
                    let ev = prob * (o / 100.0);
                    (Some(ev), Some(ev - 1.0))
                }
                None => (None, None),
            };

            ExactaPrediction {
                first: first.clone(),
                second: second.clone(),
                probability: *prob,
                odds,
                expected_value: ev,
                edge,
                recommended: ev.map(|e| e > state.config.betting.ev_threshold).unwrap_or(false),
            }
        })
        .collect();

    // Find value bets if odds provided
    let betting_signals = if !req.exacta_odds.is_empty() {
        find_value_bets(&exacta_probs, &req.exacta_odds, &state.config.betting)
    } else {
        Vec::new()
    };

    Ok(Json(PredictResponse {
        race_id: req.race_id,
        predictions: Predictions {
            win_probabilities: win_probs,
            top_exactas: exacta_predictions,
        },
        betting_signals,
    }))
}
