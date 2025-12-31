//! Configuration for the Keiba API.

use serde::{Deserialize, Serialize};

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_model_path")]
    pub path: String,
}

fn default_model_path() -> String {
    "data/models/position_model.onnx".to_string()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: default_model_path(),
        }
    }
}

/// Betting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettingConfig {
    #[serde(default = "default_ev_threshold")]
    pub ev_threshold: f64,
    #[serde(default = "default_min_probability")]
    pub min_probability: f64,
    #[serde(default = "default_max_combinations")]
    pub max_combinations: usize,
    #[serde(default = "default_bet_unit")]
    pub bet_unit: u32,
    #[serde(default = "default_kelly_fraction")]
    pub kelly_fraction: f64,
}

fn default_ev_threshold() -> f64 {
    1.0
}

fn default_min_probability() -> f64 {
    0.001
}

fn default_max_combinations() -> usize {
    50
}

fn default_bet_unit() -> u32 {
    100
}

fn default_kelly_fraction() -> f64 {
    0.25
}

impl Default for BettingConfig {
    fn default() -> Self {
        Self {
            ev_threshold: default_ev_threshold(),
            min_probability: default_min_probability(),
            max_combinations: default_max_combinations(),
            bet_unit: default_bet_unit(),
            kelly_fraction: default_kelly_fraction(),
        }
    }
}

/// Calibration configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Whether calibration is enabled
    #[serde(default)]
    pub enabled: bool,
    /// Path to calibration JSON config file
    #[serde(default)]
    pub config_file: Option<String>,
}

/// Application configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub betting: BettingConfig,
    #[serde(default)]
    pub calibration: CalibrationConfig,
}

impl AppConfig {
    /// Load configuration from environment and config file
    pub fn load() -> anyhow::Result<Self> {
        let config = config::Config::builder()
            // Start with defaults
            .add_source(config::Config::try_from(&AppConfig::default())?)
            // Add config file if exists
            .add_source(config::File::with_name("config").required(false))
            // Override with environment variables (KEIBA_SERVER_PORT, etc.)
            .add_source(
                config::Environment::with_prefix("KEIBA")
                    .separator("_")
                    .try_parsing(true),
            )
            .build()?;

        Ok(config.try_deserialize()?)
    }
}

/// Feature names in model input order
pub const FEATURE_NAMES: [&str; 23] = [
    "horse_age_num",
    "horse_sex_encoded",
    "post_position_num",
    "weight_carried",
    "horse_weight",
    "jockey_win_rate",
    "jockey_place_rate",
    "trainer_win_rate",
    "jockey_races",
    "trainer_races",
    "distance_num",
    "is_turf",
    "is_dirt",
    "track_condition_num",
    "avg_position_last_3",
    "avg_position_last_5",
    "win_rate_last_3",
    "win_rate_last_5",
    "place_rate_last_3",
    "place_rate_last_5",
    "last_position",
    "career_races",
    "odds_log",
];
