//! API route handlers.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use ndarray::Array2;
use std::sync::Arc;

use crate::betting::{
    find_value_bets, find_value_bets_quinella, find_value_bets_trifecta, find_value_bets_trio,
    find_value_bets_wide,
};
use crate::calibration::Calibrator;
use crate::config::{AppConfig, FEATURE_NAMES};
use crate::exacta::{calculate_exacta_probs, extract_win_probs, get_top_exactas};
use crate::model::{SharedModel, NUM_FEATURES};
use crate::quinella::{calculate_quinella_probs, get_top_quinellas};
use crate::trifecta::{calculate_trifecta_probs, get_top_trifectas};
use crate::trio::{calculate_trio_probs, get_top_trios};
use crate::wide::{calculate_wide_probs, get_top_wides};
use crate::types::{
    BettingSignals, ErrorResponse, ExactaPrediction, HealthResponse, ModelInfoResponse,
    PredictRequest, PredictResponse, Predictions, QuinellaPrediction, TrifectaPrediction,
    TrioPrediction, WidePrediction,
};

/// Application state shared across handlers.
pub struct AppState {
    pub model: SharedModel,
    pub config: AppConfig,
    pub calibrator: Calibrator,
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

/// Prediction endpoint supporting all bet types.
pub async fn predict(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    // Validate request
    if req.horses.is_empty() {
        return Err(ApiError::bad_request("No horses provided"));
    }

    if req.horses.len() < 2 {
        return Err(ApiError::bad_request("At least 2 horses required"));
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

    // Apply calibration if enabled
    let win_probs = if state.calibrator.is_enabled() {
        state.calibrator.calibrate_map(&win_probs)
    } else {
        win_probs
    };

    let min_prob = state.config.betting.min_probability;
    let max_combos = state.config.betting.max_combinations;
    let ev_threshold = state.config.betting.ev_threshold;

    // Initialize predictions
    let mut predictions = Predictions {
        win_probabilities: win_probs.clone(),
        ..Default::default()
    };

    // Initialize betting signals
    let mut betting_signals = BettingSignals::default();

    // Determine which bet types to calculate
    let bet_types: Vec<&str> = if req.bet_types.is_empty() {
        vec!["exacta", "trifecta", "quinella", "trio", "wide"]
    } else {
        req.bet_types.iter().map(|s| s.as_str()).collect()
    };

    // Calculate Exacta
    if bet_types.contains(&"exacta") || bet_types.contains(&"all") {
        let exacta_probs = calculate_exacta_probs(&win_probs, min_prob);
        let top_exactas = get_top_exactas(&exacta_probs, max_combos);

        predictions.top_exactas = top_exactas
            .iter()
            .map(|((first, second), prob)| {
                let odds_key = format!("{}-{}", first, second);
                let odds = req.exacta_odds.get(&odds_key).copied();
                let (ev, edge) = calculate_ev_edge(*prob, odds);

                ExactaPrediction {
                    first: first.clone(),
                    second: second.clone(),
                    probability: *prob,
                    odds,
                    expected_value: ev,
                    edge,
                    recommended: ev.map(|e| e > ev_threshold).unwrap_or(false),
                }
            })
            .collect();

        if !req.exacta_odds.is_empty() {
            betting_signals.exacta = find_value_bets(&exacta_probs, &req.exacta_odds, &state.config.betting);
        }
    }

    // Calculate Trifecta
    if (bet_types.contains(&"trifecta") || bet_types.contains(&"all")) && n_horses >= 3 {
        let trifecta_probs = calculate_trifecta_probs(&win_probs, min_prob);
        let top_trifectas = get_top_trifectas(&trifecta_probs, max_combos);

        predictions.top_trifectas = top_trifectas
            .iter()
            .map(|((first, second, third), prob)| {
                let odds_key = format!("{}-{}-{}", first, second, third);
                let odds = req.trifecta_odds.get(&odds_key).copied();
                let (ev, edge) = calculate_ev_edge(*prob, odds);

                TrifectaPrediction {
                    first: first.clone(),
                    second: second.clone(),
                    third: third.clone(),
                    probability: *prob,
                    odds,
                    expected_value: ev,
                    edge,
                    recommended: ev.map(|e| e > ev_threshold).unwrap_or(false),
                }
            })
            .collect();

        if !req.trifecta_odds.is_empty() {
            betting_signals.trifecta = find_value_bets_trifecta(
                &trifecta_probs,
                &req.trifecta_odds,
                &state.config.betting,
            );
        }
    }

    // Calculate Quinella
    if bet_types.contains(&"quinella") || bet_types.contains(&"all") {
        let quinella_probs = calculate_quinella_probs(&win_probs, min_prob);
        let top_quinellas = get_top_quinellas(&quinella_probs, max_combos);

        predictions.top_quinellas = top_quinellas
            .iter()
            .map(|(set, prob)| {
                let horses: Vec<_> = set.iter().cloned().collect();
                let odds_key = format!("{}-{}", horses[0], horses[1]);
                let odds = req.quinella_odds.get(&odds_key).copied();
                let (ev, edge) = calculate_ev_edge(*prob, odds);

                QuinellaPrediction {
                    horses: (horses[0].clone(), horses[1].clone()),
                    probability: *prob,
                    odds,
                    expected_value: ev,
                    edge,
                    recommended: ev.map(|e| e > ev_threshold).unwrap_or(false),
                }
            })
            .collect();

        if !req.quinella_odds.is_empty() {
            betting_signals.quinella = find_value_bets_quinella(
                &quinella_probs,
                &req.quinella_odds,
                &state.config.betting,
            );
        }
    }

    // Calculate Trio
    if (bet_types.contains(&"trio") || bet_types.contains(&"all")) && n_horses >= 3 {
        let trio_probs = calculate_trio_probs(&win_probs, min_prob);
        let top_trios = get_top_trios(&trio_probs, max_combos);

        predictions.top_trios = top_trios
            .iter()
            .map(|(set, prob)| {
                let horses: Vec<_> = set.iter().cloned().collect();
                let odds_key = format!("{}-{}-{}", horses[0], horses[1], horses[2]);
                let odds = req.trio_odds.get(&odds_key).copied();
                let (ev, edge) = calculate_ev_edge(*prob, odds);

                TrioPrediction {
                    horses: (horses[0].clone(), horses[1].clone(), horses[2].clone()),
                    probability: *prob,
                    odds,
                    expected_value: ev,
                    edge,
                    recommended: ev.map(|e| e > ev_threshold).unwrap_or(false),
                }
            })
            .collect();

        if !req.trio_odds.is_empty() {
            betting_signals.trio = find_value_bets_trio(
                &trio_probs,
                &req.trio_odds,
                &state.config.betting,
            );
        }
    }

    // Calculate Wide
    if (bet_types.contains(&"wide") || bet_types.contains(&"all")) && n_horses >= 3 {
        let wide_probs = calculate_wide_probs(&win_probs, min_prob);
        let top_wides = get_top_wides(&wide_probs, max_combos);

        predictions.top_wides = top_wides
            .iter()
            .map(|(set, prob)| {
                let horses: Vec<_> = set.iter().cloned().collect();
                let odds_key = format!("{}-{}", horses[0], horses[1]);
                let odds = req.wide_odds.get(&odds_key).copied();
                let (ev, edge) = calculate_ev_edge(*prob, odds);

                WidePrediction {
                    horses: (horses[0].clone(), horses[1].clone()),
                    probability: *prob,
                    odds,
                    expected_value: ev,
                    edge,
                    recommended: ev.map(|e| e > ev_threshold).unwrap_or(false),
                }
            })
            .collect();

        if !req.wide_odds.is_empty() {
            betting_signals.wide = find_value_bets_wide(
                &wide_probs,
                &req.wide_odds,
                &state.config.betting,
            );
        }
    }

    Ok(Json(PredictResponse {
        race_id: req.race_id,
        predictions,
        betting_signals,
    }))
}

/// Calculate expected value and edge from probability and odds.
fn calculate_ev_edge(prob: f64, odds: Option<f64>) -> (Option<f64>, Option<f64>) {
    match odds {
        Some(o) => {
            let ev = prob * (o / 100.0);
            (Some(ev), Some(ev - 1.0))
        }
        None => (None, None),
    }
}
