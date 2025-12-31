//! Keiba-AI Inference API
//!
//! REST API and CLI for horse racing predictions with full betting signals.

mod backtest;
mod betting;
mod calibration;
mod cli;
mod config;
mod exacta;
mod model;
mod quinella;
mod routes;
mod scraper;
mod trifecta;
mod trio;
mod types;
mod wide;

use axum::{routing::get, routing::post, Router};
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::calibration::Calibrator;
use crate::cli::{Cli, Commands};
use crate::config::AppConfig;
use crate::model::create_shared_model;
use crate::routes::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { host, port } => run_server(Some(host), Some(port)).await,
        Commands::Predict {
            input,
            bet_types,
            format,
            model,
        } => cli::run_predict(input, bet_types, format, model).await,
        Commands::Backtest {
            features,
            odds,
            bet_type,
            model,
            calibration,
            periods,
            train_months,
            test_months,
            ev_threshold,
            format,
        } => {
            cli::run_backtest(
                features,
                odds,
                bet_type,
                model,
                calibration,
                periods,
                train_months,
                test_months,
                ev_threshold,
                format,
            )
            .await
        }
        Commands::Live {
            race_id,
            bet_type,
            ev_threshold,
            output,
            force,
            verbose,
        } => cli::run_live(race_id, bet_type, ev_threshold, output, force, verbose).await,
    }
}

/// Run the API server.
async fn run_server(host: Option<String>, port: Option<u16>) -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "keiba_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let mut config = AppConfig::load()?;

    // Override with CLI args
    if let Some(h) = host {
        config.server.host = h;
    }
    if let Some(p) = port {
        config.server.port = p;
    }

    tracing::info!("Configuration loaded");
    tracing::info!("Model path: {}", config.model.path);

    // Load model
    tracing::info!("Loading ONNX model...");
    let model = create_shared_model(&config.model.path)?;
    tracing::info!("Model loaded successfully");

    // Load calibrator
    let calibrator = if config.calibration.enabled {
        if let Some(ref path) = config.calibration.config_file {
            tracing::info!("Loading calibrator from: {}", path);
            match Calibrator::from_file(path) {
                Ok(cal) => {
                    tracing::info!("Calibrator loaded: {:?}", cal);
                    cal
                }
                Err(e) => {
                    tracing::warn!("Failed to load calibrator: {}, using None", e);
                    Calibrator::None
                }
            }
        } else {
            tracing::info!("Calibration enabled but no config_file specified");
            Calibrator::None
        }
    } else {
        tracing::info!("Calibration disabled");
        Calibrator::None
    };

    // Create application state
    let state = Arc::new(AppState {
        model,
        config: config.clone(),
        calibrator,
    });

    // Build router
    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/model/info", get(routes::model_info))
        .route("/predict", post(routes::predict))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = SocketAddr::new(config.server.host.parse()?, config.server.port);
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
