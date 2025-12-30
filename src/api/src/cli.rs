//! CLI commands for keiba-api.
//!
//! Supports both API server mode, CLI prediction mode, and backtesting.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::backtest::{load_race_data, print_backtest_table, Backtester, OddsLookup};
use crate::calibration::Calibrator;
use crate::config::AppConfig;
use crate::exacta::{calculate_exacta_probs, extract_win_probs, get_top_exactas};
use crate::model::create_shared_model;
use crate::quinella::{calculate_quinella_probs, get_top_quinellas};
use crate::trifecta::{calculate_trifecta_probs, get_top_trifectas};
use crate::trio::{calculate_trio_probs, get_top_trios};
use crate::types::{PredictRequest, PredictResponse, Predictions};
use crate::wide::{calculate_wide_probs, get_top_wides};

#[derive(Parser)]
#[command(name = "keiba-api")]
#[command(version, about = "Keiba-AI: Horse racing prediction API and CLI", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the API server
    Serve {
        /// Host to bind to
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,

        /// Port to bind to
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
    },

    /// Run prediction on a race JSON file
    Predict {
        /// Path to race data JSON file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Bet types to calculate (exacta, trifecta, quinella, trio, wide, all)
        #[arg(short, long, value_delimiter = ',', default_value = "all")]
        bet_types: Vec<String>,

        /// Output format (json, table)
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Model path override
        #[arg(short, long)]
        model: Option<PathBuf>,
    },

    /// Run backtest on historical data
    Backtest {
        /// Path to features parquet file
        #[arg(value_name = "FEATURES")]
        features: PathBuf,

        /// Path to exacta odds CSV file
        #[arg(short, long)]
        odds: PathBuf,

        /// Model path override
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Calibration config JSON file
        #[arg(short, long)]
        calibration: Option<PathBuf>,

        /// Number of walk-forward periods
        #[arg(long, default_value_t = 6)]
        periods: usize,

        /// Training window in months
        #[arg(long, default_value_t = 18)]
        train_months: usize,

        /// Test window in months
        #[arg(long, default_value_t = 3)]
        test_months: usize,

        /// EV threshold for betting
        #[arg(long, default_value_t = 1.0)]
        ev_threshold: f64,

        /// Output format (json, table)
        #[arg(short, long, default_value = "table")]
        format: String,
    },
}

/// Run CLI prediction from file.
pub async fn run_predict(
    input: PathBuf,
    bet_types: Vec<String>,
    format: String,
    model_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    // Load configuration
    let mut config = AppConfig::load()?;

    // Override model path if provided
    if let Some(path) = model_path {
        config.model.path = path.to_string_lossy().to_string();
    }

    // Load model
    eprintln!("Loading model from: {}", config.model.path);
    let model = create_shared_model(&config.model.path)?;
    eprintln!("Model loaded successfully");

    // Read input file
    let input_json = std::fs::read_to_string(&input)?;
    let req: PredictRequest = serde_json::from_str(&input_json)?;

    eprintln!("Processing race: {}", req.race_id);
    eprintln!("Horses: {}", req.horses.len());

    // Build feature matrix
    let n_horses = req.horses.len();
    let mut features = ndarray::Array2::<f32>::zeros((n_horses, crate::model::NUM_FEATURES));

    let horse_ids: Vec<String> = req.horses.iter().map(|h| h.horse_id.clone()).collect();

    for (i, horse) in req.horses.iter().enumerate() {
        let feature_array = horse.features.to_array();
        for (j, &val) in feature_array.iter().enumerate() {
            features[[i, j]] = val;
        }
    }

    // Run model inference
    let position_probs = model.predict(features)?;
    let win_probs = extract_win_probs(&position_probs, &horse_ids);

    let min_prob = config.betting.min_probability;
    let max_combos = config.betting.max_combinations;

    // Initialize predictions
    let mut predictions = Predictions {
        win_probabilities: win_probs.clone(),
        ..Default::default()
    };

    // Determine which bet types to calculate
    let should_calc_all = bet_types.contains(&"all".to_string());

    // Calculate requested bet types
    if should_calc_all || bet_types.contains(&"exacta".to_string()) {
        let exacta_probs = calculate_exacta_probs(&win_probs, min_prob);
        let top_exactas = get_top_exactas(&exacta_probs, max_combos);

        predictions.top_exactas = top_exactas
            .iter()
            .map(|((first, second), prob)| crate::types::ExactaPrediction {
                first: first.clone(),
                second: second.clone(),
                probability: *prob,
                odds: None,
                expected_value: None,
                edge: None,
                recommended: false,
            })
            .collect();
    }

    if should_calc_all || bet_types.contains(&"trifecta".to_string()) {
        if n_horses >= 3 {
            let trifecta_probs = calculate_trifecta_probs(&win_probs, min_prob);
            let top_trifectas = get_top_trifectas(&trifecta_probs, max_combos);

            predictions.top_trifectas = top_trifectas
                .iter()
                .map(|((first, second, third), prob)| crate::types::TrifectaPrediction {
                    first: first.clone(),
                    second: second.clone(),
                    third: third.clone(),
                    probability: *prob,
                    odds: None,
                    expected_value: None,
                    edge: None,
                    recommended: false,
                })
                .collect();
        }
    }

    if should_calc_all || bet_types.contains(&"quinella".to_string()) {
        let quinella_probs = calculate_quinella_probs(&win_probs, min_prob);
        let top_quinellas = get_top_quinellas(&quinella_probs, max_combos);

        predictions.top_quinellas = top_quinellas
            .iter()
            .map(|(set, prob)| {
                let horses: Vec<_> = set.iter().cloned().collect();
                crate::types::QuinellaPrediction {
                    horses: (horses[0].clone(), horses[1].clone()),
                    probability: *prob,
                    odds: None,
                    expected_value: None,
                    edge: None,
                    recommended: false,
                }
            })
            .collect();
    }

    if should_calc_all || bet_types.contains(&"trio".to_string()) {
        if n_horses >= 3 {
            let trio_probs = calculate_trio_probs(&win_probs, min_prob);
            let top_trios = get_top_trios(&trio_probs, max_combos);

            predictions.top_trios = top_trios
                .iter()
                .map(|(set, prob)| {
                    let horses: Vec<_> = set.iter().cloned().collect();
                    crate::types::TrioPrediction {
                        horses: (horses[0].clone(), horses[1].clone(), horses[2].clone()),
                        probability: *prob,
                        odds: None,
                        expected_value: None,
                        edge: None,
                        recommended: false,
                    }
                })
                .collect();
        }
    }

    if should_calc_all || bet_types.contains(&"wide".to_string()) {
        if n_horses >= 3 {
            let wide_probs = calculate_wide_probs(&win_probs, min_prob);
            let top_wides = get_top_wides(&wide_probs, max_combos);

            predictions.top_wides = top_wides
                .iter()
                .map(|(set, prob)| {
                    let horses: Vec<_> = set.iter().cloned().collect();
                    crate::types::WidePrediction {
                        horses: (horses[0].clone(), horses[1].clone()),
                        probability: *prob,
                        odds: None,
                        expected_value: None,
                        edge: None,
                        recommended: false,
                    }
                })
                .collect();
        }
    }

    // Build response
    let response = PredictResponse {
        race_id: req.race_id,
        predictions,
        betting_signals: Default::default(),
    };

    // Output
    match format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        "table" => {
            print_table(&response);
        }
        _ => {
            eprintln!("Unknown format: {}. Using JSON.", format);
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
    }

    Ok(())
}

/// Print prediction results in table format.
fn print_table(response: &PredictResponse) {
    println!("Race: {}", response.race_id);
    println!();

    // Win probabilities
    println!("=== Win Probabilities ===");
    let mut sorted: Vec<_> = response.predictions.win_probabilities.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (horse, prob) in sorted.iter().take(10) {
        println!("  {:>6}: {:.2}%", horse, *prob * 100.0);
    }
    println!();

    // Exacta
    if !response.predictions.top_exactas.is_empty() {
        println!("=== Top Exactas ===");
        for (i, e) in response.predictions.top_exactas.iter().take(10).enumerate() {
            println!(
                "  {:2}. {}-{}: {:.4}%",
                i + 1,
                e.first,
                e.second,
                e.probability * 100.0
            );
        }
        println!();
    }

    // Trifecta
    if !response.predictions.top_trifectas.is_empty() {
        println!("=== Top Trifectas ===");
        for (i, t) in response.predictions.top_trifectas.iter().take(10).enumerate() {
            println!(
                "  {:2}. {}-{}-{}: {:.4}%",
                i + 1,
                t.first,
                t.second,
                t.third,
                t.probability * 100.0
            );
        }
        println!();
    }

    // Quinella
    if !response.predictions.top_quinellas.is_empty() {
        println!("=== Top Quinellas ===");
        for (i, q) in response.predictions.top_quinellas.iter().take(10).enumerate() {
            println!(
                "  {:2}. {}-{}: {:.4}%",
                i + 1,
                q.horses.0,
                q.horses.1,
                q.probability * 100.0
            );
        }
        println!();
    }

    // Trio
    if !response.predictions.top_trios.is_empty() {
        println!("=== Top Trios ===");
        for (i, t) in response.predictions.top_trios.iter().take(10).enumerate() {
            println!(
                "  {:2}. {}-{}-{}: {:.4}%",
                i + 1,
                t.horses.0,
                t.horses.1,
                t.horses.2,
                t.probability * 100.0
            );
        }
        println!();
    }

    // Wide
    if !response.predictions.top_wides.is_empty() {
        println!("=== Top Wides ===");
        for (i, w) in response.predictions.top_wides.iter().take(10).enumerate() {
            println!(
                "  {:2}. {}-{}: {:.4}%",
                i + 1,
                w.horses.0,
                w.horses.1,
                w.probability * 100.0
            );
        }
        println!();
    }
}

/// Run backtest on historical data.
pub async fn run_backtest(
    features_path: PathBuf,
    odds_path: PathBuf,
    model_path: Option<PathBuf>,
    calibration_path: Option<PathBuf>,
    periods: usize,
    train_months: usize,
    test_months: usize,
    ev_threshold: f64,
    format: String,
) -> anyhow::Result<()> {
    // Load configuration
    let mut config = AppConfig::load()?;

    // Override model path if provided
    if let Some(path) = model_path {
        config.model.path = path.to_string_lossy().to_string();
    }

    // Override EV threshold
    config.betting.ev_threshold = ev_threshold;

    // Load model
    eprintln!("Loading model from: {}", config.model.path);
    let model = create_shared_model(&config.model.path)?;
    eprintln!("Model loaded successfully");

    // Load calibrator
    let calibrator = if let Some(ref path) = calibration_path {
        eprintln!("Loading calibrator from: {}", path.display());
        match Calibrator::from_file(path) {
            Ok(cal) => {
                eprintln!("Calibrator loaded: {:?}", cal);
                cal
            }
            Err(e) => {
                eprintln!("Failed to load calibrator: {}, using None", e);
                Calibrator::None
            }
        }
    } else {
        Calibrator::None
    };

    // Load race data
    eprintln!("Loading race data from: {}", features_path.display());
    let races = load_race_data(&features_path)?;
    eprintln!("Loaded {} races", races.len());

    // Load odds data
    eprintln!("Loading odds data from: {}", odds_path.display());
    let odds_lookup = OddsLookup::from_csv(&odds_path)?;
    eprintln!("Odds data loaded");

    // Create backtester
    let backtester = Backtester::new(model, calibrator, config.betting.clone());

    // Run walk-forward backtest
    eprintln!("Running walk-forward backtest with {} periods...", periods);
    let results = backtester.run_walkforward(&races, &odds_lookup, periods, train_months, test_months);

    // Output results
    match format.as_str() {
        "json" => {
            // Create JSON output
            let json_output = serde_json::json!({
                "num_bets": results.num_bets,
                "num_wins": results.num_wins,
                "hit_rate": results.hit_rate(),
                "total_bet": results.total_bet,
                "total_return": results.total_return,
                "profit": results.profit(),
                "roi": results.roi(),
                "max_drawdown": results.max_drawdown(),
                "periods": results.periods.iter().map(|p| serde_json::json!({
                    "name": p.period_name,
                    "start_date": p.start_date.to_string(),
                    "end_date": p.end_date.to_string(),
                    "num_bets": p.num_bets,
                    "num_wins": p.num_wins,
                    "hit_rate": p.hit_rate(),
                    "roi": p.roi(),
                    "profit": p.profit(),
                })).collect::<Vec<_>>(),
            });
            println!("{}", serde_json::to_string_pretty(&json_output)?);
        }
        "table" | _ => {
            print_backtest_table(&results);
        }
    }

    Ok(())
}
