//! Keiba-AI Inference API
//!
//! REST API for horse racing predictions with full betting signals.

mod betting;
mod config;
mod exacta;
mod model;
mod routes;
mod types;

use axum::{routing::get, routing::post, Router};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config::AppConfig;
use crate::model::create_shared_model;
use crate::routes::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "keiba_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = AppConfig::load()?;
    tracing::info!("Configuration loaded");
    tracing::info!("Model path: {}", config.model.path);

    // Load model
    tracing::info!("Loading ONNX model...");
    let model = create_shared_model(&config.model.path)?;
    tracing::info!("Model loaded successfully");

    // Create application state
    let state = Arc::new(AppState {
        model,
        config: config.clone(),
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
    let addr = SocketAddr::new(
        config.server.host.parse()?,
        config.server.port,
    );
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
