//! CLI commands for keiba-api.
//!
//! Supports both API server mode, CLI prediction mode, and backtesting.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Validate race ID format (YYYYRRCCNNDD)
///
/// Format: 12 digits where:
/// - YYYY: Year (2019-2030)
/// - RR: Racecourse code (01-10)
/// - CC: Meeting number (01-12)
/// - NN: Day number (01-12)
/// - DD: Race number (01-12)
pub fn validate_race_id(race_id: &str) -> Result<(), String> {
    // Check length
    if race_id.len() != 12 {
        return Err(format!(
            "Race ID must be 12 digits, got {} digits: '{}'",
            race_id.len(),
            race_id
        ));
    }

    // Check all digits
    if !race_id.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!(
            "Race ID must contain only digits: '{}'",
            race_id
        ));
    }

    // Parse and validate components
    let year: u32 = race_id[0..4].parse().unwrap();
    let racecourse: u32 = race_id[4..6].parse().unwrap();
    let meeting: u32 = race_id[6..8].parse().unwrap();
    let day: u32 = race_id[8..10].parse().unwrap();
    let race: u32 = race_id[10..12].parse().unwrap();

    if year < 2019 || year > 2030 {
        return Err(format!(
            "Year must be 2019-2030, got {}: '{}'",
            year, race_id
        ));
    }

    if racecourse < 1 || racecourse > 10 {
        return Err(format!(
            "Racecourse code must be 01-10, got {:02}: '{}'\n\
             Codes: 01=Sapporo, 02=Hakodate, 03=Fukushima, 04=Niigata, 05=Tokyo, \
             06=Nakayama, 07=Chukyo, 08=Kyoto, 09=Hanshin, 10=Kokura",
            racecourse, race_id
        ));
    }

    if meeting < 1 || meeting > 12 {
        return Err(format!(
            "Meeting number must be 01-12, got {:02}: '{}'",
            meeting, race_id
        ));
    }

    if day < 1 || day > 12 {
        return Err(format!(
            "Day number must be 01-12, got {:02}: '{}'",
            day, race_id
        ));
    }

    if race < 1 || race > 12 {
        return Err(format!(
            "Race number must be 01-12, got {:02}: '{}'",
            race, race_id
        ));
    }

    Ok(())
}

/// Validate bet type
pub fn validate_bet_type(bet_type: &str) -> Result<(), String> {
    let valid_types = ["exacta", "trifecta", "quinella", "trio", "wide", "all"];
    if !valid_types.contains(&bet_type.to_lowercase().as_str()) {
        return Err(format!(
            "Invalid bet type '{}'. Valid types: {}",
            bet_type,
            valid_types.join(", ")
        ));
    }
    Ok(())
}

use crate::backtest::{load_race_data, print_backtest_table, Backtester, BetType, OddsLookup};
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

        /// Calibration config JSON file
        #[arg(short, long)]
        calibration: Option<PathBuf>,
    },

    /// Run backtest on historical data
    Backtest {
        /// Path to features parquet file
        #[arg(value_name = "FEATURES")]
        features: PathBuf,

        /// Path to odds CSV file
        #[arg(short, long)]
        odds: PathBuf,

        /// Bet type to backtest (exacta, trifecta, quinella, trio, wide)
        #[arg(short, long, default_value = "exacta")]
        bet_type: String,

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

    /// Scrape live race data and run prediction in one command
    Live {
        /// Race ID (e.g., 202506050811)
        #[arg(value_name = "RACE_ID")]
        race_id: String,

        /// Bet type (exacta, trifecta)
        #[arg(short, long, default_value = "exacta")]
        bet_type: String,

        /// EV threshold for betting recommendations
        #[arg(long, default_value_t = 1.0)]
        ev_threshold: f64,

        /// Output file path (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Calibration config JSON file
        #[arg(long)]
        calibration: Option<PathBuf>,

        /// Force refresh cache
        #[arg(long)]
        force: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Scrape historical race data from db.netkeiba.com
    ScrapeHistorical {
        /// Specific date to scrape (YYYY-MM-DD)
        #[arg(long)]
        date: Option<String>,

        /// Start date for range (YYYY-MM-DD)
        #[arg(long)]
        start: Option<String>,

        /// End date for range (YYYY-MM-DD)
        #[arg(long)]
        end: Option<String>,

        /// SQLite database path
        #[arg(long, default_value = "data/historical/keiba.db")]
        db: PathBuf,

        /// Include all bet type odds (slower)
        #[arg(long)]
        include_odds: bool,

        /// Force re-scrape existing races
        #[arg(long)]
        force: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run backtest using historical SQLite data
    BacktestHistorical {
        /// SQLite database path
        #[arg(long, default_value = "data/historical/keiba.db")]
        db: PathBuf,

        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        start: String,

        /// End date (YYYY-MM-DD)
        #[arg(long)]
        end: String,

        /// Bet type (exacta, trifecta, quinella, trio, wide)
        #[arg(short, long, default_value = "exacta")]
        bet_type: String,

        /// Model path override
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Calibration config JSON file
        #[arg(short, long)]
        calibration: Option<PathBuf>,

        /// EV threshold for betting
        #[arg(long, default_value_t = 1.0)]
        ev_threshold: f64,

        /// Output format (json, table)
        #[arg(short, long, default_value = "table")]
        format: String,
    },
}

/// Load calibrator from path or default location.
fn load_calibrator(path: &Option<PathBuf>) -> Calibrator {
    match path {
        Some(p) => {
            eprintln!("Loading calibrator from: {}", p.display());
            match Calibrator::from_file(p) {
                Ok(cal) => {
                    eprintln!("Calibrator loaded: {:?}", cal);
                    cal
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load calibrator: {}", e);
                    Calibrator::None
                }
            }
        }
        None => {
            // Try default path
            let default = PathBuf::from("data/models/calibration.json");
            if default.exists() {
                eprintln!("Loading calibrator from default: {}", default.display());
                match Calibrator::from_file(&default) {
                    Ok(cal) => {
                        eprintln!("Calibrator loaded: {:?}", cal);
                        cal
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load default calibrator: {}", e);
                        Calibrator::None
                    }
                }
            } else {
                Calibrator::None
            }
        }
    }
}

/// Run CLI prediction from file.
pub async fn run_predict(
    input: PathBuf,
    bet_types: Vec<String>,
    format: String,
    model_path: Option<PathBuf>,
    calibration_path: Option<PathBuf>,
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
    let raw_win_probs = extract_win_probs(&position_probs, &horse_ids);

    // Load and apply calibration
    let calibrator = load_calibrator(&calibration_path);
    let win_probs = if calibrator.is_enabled() {
        eprintln!("Applying calibration...");
        calibrator.calibrate_map(&raw_win_probs)
    } else {
        raw_win_probs
    };

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

    if (should_calc_all || bet_types.contains(&"trifecta".to_string())) && n_horses >= 3 {
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

    if (should_calc_all || bet_types.contains(&"trio".to_string())) && n_horses >= 3 {
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

    if (should_calc_all || bet_types.contains(&"wide".to_string())) && n_horses >= 3 {
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
#[allow(clippy::too_many_arguments)]
pub async fn run_backtest(
    features_path: PathBuf,
    odds_path: PathBuf,
    bet_type_str: String,
    model_path: Option<PathBuf>,
    calibration_path: Option<PathBuf>,
    periods: usize,
    train_months: usize,
    test_months: usize,
    ev_threshold: f64,
    format: String,
) -> anyhow::Result<()> {
    // Parse bet type
    let bet_type = BetType::from_str(&bet_type_str).ok_or_else(|| {
        anyhow::anyhow!(
            "Invalid bet type: {}. Valid types: exacta, trifecta, quinella, trio, wide",
            bet_type_str
        )
    })?;

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
    eprintln!(
        "Loading {} odds data from: {}",
        bet_type.name(),
        odds_path.display()
    );
    let odds_lookup = OddsLookup::from_csv(&odds_path, bet_type)?;
    eprintln!("Odds data loaded");

    // Create backtester
    let backtester = Backtester::new(model, calibrator, config.betting.clone(), bet_type);

    // Run walk-forward backtest
    eprintln!(
        "Running walk-forward backtest for {} with {} periods...",
        bet_type.name(),
        periods
    );
    let results =
        backtester.run_walkforward(&races, &odds_lookup, periods, train_months, test_months);

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
        _ => {
            print_backtest_table(&results);
        }
    }

    Ok(())
}

/// Run live race prediction (scrape + predict in one command).
pub async fn run_live(
    race_id: String,
    bet_type: String,
    ev_threshold: f64,
    output: Option<PathBuf>,
    calibration_path: Option<PathBuf>,
    force: bool,
    verbose: bool,
) -> anyhow::Result<()> {
    use crate::scraper::{
        cache::{Cache, CacheCategory},
        feature_builder::FeatureBuilder,
        parsers::{
            HorseParser, HorseProfile, JockeyParser, JockeyProfile, RaceCardParser,
            TrainerParser, TrainerProfile,
        },
        Browser, RateLimiter,
    };
    use colored::Colorize;
    use futures::stream::{self, StreamExt};
    use indicatif::{ProgressBar, ProgressStyle};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    // Input validation
    if let Err(e) = validate_race_id(&race_id) {
        anyhow::bail!("Invalid race ID: {}", e);
    }
    if let Err(e) = validate_bet_type(&bet_type) {
        anyhow::bail!("Invalid bet type: {}", e);
    }

    eprintln!("{}", "=== Keiba-AI Live Prediction ===".cyan().bold());
    eprintln!("Race ID: {}", race_id.yellow());
    eprintln!("Bet type: {}", bet_type.yellow());
    eprintln!("EV threshold: {}", format!("{}", ev_threshold).yellow());
    eprintln!();

    // Initialize components
    let cache = Cache::default_cache();
    let rate_limiter = RateLimiter::default_limiter();

    // Launch browser once and reuse for all fetches (performance optimization)
    eprintln!("{}", "Launching browser...".dimmed());
    let browser = Browser::launch().await?;

    // Step 1: Fetch race card
    eprintln!("{}", "Step 1: Fetching race card...".green());
    let race_card_html = if !force {
        if let Some(cached) = cache.get::<String>(CacheCategory::RaceCard, &race_id) {
            if verbose {
                eprintln!("  Using cached race card");
            }
            cached
        } else {
            let html = fetch_race_card_with_browser(&race_id, &browser, &rate_limiter).await?;
            let _ = cache.set(CacheCategory::RaceCard, &race_id, &html);
            html
        }
    } else {
        fetch_race_card_with_browser(&race_id, &browser, &rate_limiter).await?
    };

    // Parse race card
    let (race_info, entries) = RaceCardParser::parse(&race_card_html, &race_id)?;
    eprintln!(
        "  Race: {} ({} {}m)",
        race_info.race_name.cyan(),
        race_info.surface,
        race_info.distance
    );
    eprintln!("  Entries: {} horses", format!("{}", entries.len()).yellow());

    if entries.is_empty() {
        let _ = browser.close().await;
        anyhow::bail!("No entries found in race card");
    }

    // Step 2: Fetch profiles (parallel fetching for performance)
    eprintln!("{}", "Step 2: Fetching horse/jockey/trainer profiles...".green());

    let mut horses: HashMap<String, HorseProfile> = HashMap::new();
    let mut jockeys: HashMap<String, JockeyProfile> = HashMap::new();
    let mut trainers: HashMap<String, TrainerProfile> = HashMap::new();

    // Collect unique IDs to fetch, checking cache first
    #[derive(Debug, Clone)]
    enum FetchTask {
        Horse(String),
        Jockey(String),
        Trainer(String),
    }

    let mut fetch_tasks: Vec<FetchTask> = Vec::new();
    let mut seen_jockeys: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut seen_trainers: std::collections::HashSet<String> = std::collections::HashSet::new();

    for entry in &entries {
        // Check horse cache
        if !entry.horse_id.is_empty() {
            if !force {
                if let Some(cached) = cache.get::<HorseProfile>(CacheCategory::Horse, &entry.horse_id) {
                    horses.insert(entry.horse_id.clone(), cached);
                } else {
                    fetch_tasks.push(FetchTask::Horse(entry.horse_id.clone()));
                }
            } else {
                fetch_tasks.push(FetchTask::Horse(entry.horse_id.clone()));
            }
        }

        // Check jockey cache (deduplicate)
        if !entry.jockey_id.is_empty() && !seen_jockeys.contains(&entry.jockey_id) {
            seen_jockeys.insert(entry.jockey_id.clone());
            if !force {
                if let Some(cached) = cache.get::<JockeyProfile>(CacheCategory::Jockey, &entry.jockey_id) {
                    jockeys.insert(entry.jockey_id.clone(), cached);
                } else {
                    fetch_tasks.push(FetchTask::Jockey(entry.jockey_id.clone()));
                }
            } else {
                fetch_tasks.push(FetchTask::Jockey(entry.jockey_id.clone()));
            }
        }

        // Check trainer cache (deduplicate)
        if !entry.trainer_id.is_empty() && !seen_trainers.contains(&entry.trainer_id) {
            seen_trainers.insert(entry.trainer_id.clone());
            if !force {
                if let Some(cached) = cache.get::<TrainerProfile>(CacheCategory::Trainer, &entry.trainer_id) {
                    trainers.insert(entry.trainer_id.clone(), cached);
                } else {
                    fetch_tasks.push(FetchTask::Trainer(entry.trainer_id.clone()));
                }
            } else {
                fetch_tasks.push(FetchTask::Trainer(entry.trainer_id.clone()));
            }
        }
    }

    let cached_count = horses.len() + jockeys.len() + trainers.len();
    let fetch_count = fetch_tasks.len();
    if verbose {
        eprintln!("  Cached: {}, To fetch: {}", cached_count, fetch_count);
    }

    // Parallel fetching with concurrency limit
    if !fetch_tasks.is_empty() {
        let pb = ProgressBar::new(fetch_tasks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        // Use Arc for shared state
        let browser = Arc::new(browser);
        let rate_limiter = Arc::new(rate_limiter);
        let cache = Arc::new(cache);
        let semaphore = Arc::new(Semaphore::new(4)); // Limit to 4 concurrent fetches

        #[derive(Debug)]
        enum FetchResult {
            Horse(String, HorseProfile),
            Jockey(String, JockeyProfile),
            Trainer(String, TrainerProfile),
            Error(String),
        }

        let results: Vec<FetchResult> = stream::iter(fetch_tasks)
            .map(|task| {
                let browser = Arc::clone(&browser);
                let rate_limiter = Arc::clone(&rate_limiter);
                let cache = Arc::clone(&cache);
                let semaphore = Arc::clone(&semaphore);
                let pb = pb.clone();

                async move {
                    // Acquire semaphore permit for concurrency control
                    let _permit = semaphore.acquire().await.unwrap();

                    // Rate limit before fetching
                    rate_limiter.acquire().await;

                    match task {
                        FetchTask::Horse(id) => {
                            pb.set_message(format!("horse:{}", &id[..id.len().min(8)]));
                            let url = crate::scraper::horse_url(&id);
                            match browser.fetch_page(&url).await {
                                Ok(html) => match HorseParser::parse(&html, &id) {
                                    Ok(profile) => {
                                        let _ = cache.set(CacheCategory::Horse, &id, &profile);
                                        pb.inc(1);
                                        FetchResult::Horse(id, profile)
                                    }
                                    Err(e) => {
                                        pb.inc(1);
                                        FetchResult::Error(format!("Horse parse error {}: {}", id, e))
                                    }
                                },
                                Err(e) => {
                                    pb.inc(1);
                                    FetchResult::Error(format!("Horse fetch error {}: {}", id, e))
                                }
                            }
                        }
                        FetchTask::Jockey(id) => {
                            pb.set_message(format!("jockey:{}", &id[..id.len().min(8)]));
                            let url = crate::scraper::jockey_url(&id);
                            match browser.fetch_page(&url).await {
                                Ok(html) => match JockeyParser::parse(&html, &id) {
                                    Ok(profile) => {
                                        let _ = cache.set(CacheCategory::Jockey, &id, &profile);
                                        pb.inc(1);
                                        FetchResult::Jockey(id, profile)
                                    }
                                    Err(e) => {
                                        pb.inc(1);
                                        FetchResult::Error(format!("Jockey parse error {}: {}", id, e))
                                    }
                                },
                                Err(e) => {
                                    pb.inc(1);
                                    FetchResult::Error(format!("Jockey fetch error {}: {}", id, e))
                                }
                            }
                        }
                        FetchTask::Trainer(id) => {
                            pb.set_message(format!("trainer:{}", &id[..id.len().min(8)]));
                            let url = crate::scraper::trainer_url(&id);
                            match browser.fetch_page(&url).await {
                                Ok(html) => match TrainerParser::parse(&html, &id) {
                                    Ok(profile) => {
                                        let _ = cache.set(CacheCategory::Trainer, &id, &profile);
                                        pb.inc(1);
                                        FetchResult::Trainer(id, profile)
                                    }
                                    Err(e) => {
                                        pb.inc(1);
                                        FetchResult::Error(format!("Trainer parse error {}: {}", id, e))
                                    }
                                },
                                Err(e) => {
                                    pb.inc(1);
                                    FetchResult::Error(format!("Trainer fetch error {}: {}", id, e))
                                }
                            }
                        }
                    }
                }
            })
            .buffer_unordered(4) // Process up to 4 fetches concurrently
            .collect()
            .await;

        pb.finish_with_message("Done");

        // Collect results
        let mut errors: Vec<String> = Vec::new();
        for result in results {
            match result {
                FetchResult::Horse(id, profile) => { horses.insert(id, profile); }
                FetchResult::Jockey(id, profile) => { jockeys.insert(id, profile); }
                FetchResult::Trainer(id, profile) => { trainers.insert(id, profile); }
                FetchResult::Error(msg) => {
                    if verbose { eprintln!("  Warning: {}", msg.yellow()); }
                    errors.push(msg);
                }
            }
        }

        if !errors.is_empty() && verbose {
            eprintln!("  {} fetch errors (continuing with available data)", errors.len());
        }

        // Close browser - need to unwrap Arc
        let browser = Arc::try_unwrap(browser)
            .map_err(|_| anyhow::anyhow!("Failed to unwrap browser Arc"))?;
        let _ = browser.close().await;
    } else {
        // All cached, just close browser
        let _ = browser.close().await;
    }
    eprintln!(
        "  Profiles loaded: {} horses, {} jockeys, {} trainers",
        format!("{}", horses.len()).cyan(),
        format!("{}", jockeys.len()).cyan(),
        format!("{}", trainers.len()).cyan()
    );

    // Step 3: Fetch odds
    eprintln!("{}", "Step 3: Fetching odds...".green());
    let odds_map = fetch_odds(&race_id, &bet_type).await?;
    eprintln!("  Loaded {} odds combinations", format!("{}", odds_map.len()).yellow());

    // Step 4: Build features
    eprintln!("{}", "Step 4: Building features...".green());
    let mut horse_features = Vec::new();
    let mut horse_ids = Vec::new();

    for entry in &entries {
        let horse = horses.get(&entry.horse_id);
        let jockey = jockeys.get(&entry.jockey_id);
        let trainer = trainers.get(&entry.trainer_id);

        let features = FeatureBuilder::build(&race_info, entry, horse, jockey, trainer);
        horse_features.push(features);
        horse_ids.push(entry.horse_id.clone());
    }

    // Step 5: Run model inference
    eprintln!("{}", "Step 5: Running model inference...".green());
    let config = AppConfig::load()?;
    let model = create_shared_model(&config.model.path)?;

    let n_horses = horse_features.len();
    let mut features_array = ndarray::Array2::<f32>::zeros((n_horses, crate::model::NUM_FEATURES));

    for (i, hf) in horse_features.iter().enumerate() {
        let arr = hf.to_array();
        for (j, &val) in arr.iter().enumerate() {
            features_array[[i, j]] = val;
        }
    }

    let position_probs = model.predict(features_array)?;
    let raw_win_probs = extract_win_probs(&position_probs, &horse_ids);

    // Load and apply calibration
    let calibrator = load_calibrator(&calibration_path);
    let win_probs = if calibrator.is_enabled() {
        if verbose {
            eprintln!("{}", "  Applying calibration...".dimmed());
        }
        calibrator.calibrate_map(&raw_win_probs)
    } else {
        raw_win_probs
    };

    // Step 6: Calculate probabilities and EV
    eprintln!("{}", "Step 6: Calculating probabilities and expected values...".green());
    let min_prob = config.betting.min_probability;
    let max_combos = config.betting.max_combinations;

    let results = match bet_type.as_str() {
        "trifecta" => {
            let probs = calculate_trifecta_probs(&win_probs, min_prob);
            let top = get_top_trifectas(&probs, max_combos);
            calculate_ev_trifecta(&top, &odds_map, ev_threshold)
        }
        _ => {
            let probs = calculate_exacta_probs(&win_probs, min_prob);
            let top = get_top_exactas(&probs, max_combos);
            calculate_ev_exacta(&top, &odds_map, ev_threshold)
        }
    };

    // Step 7: Output results
    eprintln!();
    eprintln!("{}", "=== Results ===".cyan().bold());
    println!();
    println!("{}: {} ({})", "Race".bold(), race_info.race_name.cyan(), race_id);
    println!("{}: {}", "Bet type".bold(), bet_type.yellow());
    println!("{}: ¥{:.0}", "Bankroll".bold(), config.betting.bankroll);
    println!();

    // Print win probabilities with edge detection
    println!("{}", "Win Probabilities (vs Implied Odds):".bold());
    let mut sorted_probs: Vec<_> = win_probs.iter().collect();
    sorted_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (id, prob) in sorted_probs.iter().take(5) {
        let entry = entries.iter().find(|e| &e.horse_id == *id);
        let name = entry.map(|e| e.horse_name.as_str()).unwrap_or("Unknown");

        // Calculate edge vs implied odds
        let win_odds = entry.and_then(|e| e.win_odds);
        let edge_str = if let Some(odds) = win_odds {
            let implied_prob = 100.0 / odds;
            let edge = (*prob * 100.0) - implied_prob;
            if edge > 0.0 {
                format!("[{:+.1}% edge]", edge).green().to_string()
            } else {
                format!("[{:+.1}%]", edge).dimmed().to_string()
            }
        } else {
            "".to_string()
        };

        println!("  {:>6}: {:>5.2}% - {} {}", id, *prob * 100.0, name.yellow(), edge_str);
    }
    println!();

    // Print recommended bets with Kelly sizing
    let bankroll = config.betting.bankroll;
    let kelly_fraction = config.betting.kelly_fraction;
    let bet_unit = config.betting.bet_unit;

    println!("{} (EV > {}):", "Recommended Bets".bold(), format!("{}", ev_threshold).yellow());
    if results.is_empty() {
        println!("  {}", "No bets meet the EV threshold".dimmed());
    } else {
        // Header
        println!("  {}", "─".repeat(75));
        println!(
            "  {:>3} {:>10} {:>8} {:>8} {:>8} {:>10} {:>12}",
            "#", "Combo", "Prob%", "Odds", "EV", "Kelly%", "Bet (¥)"
        );
        println!("  {}", "─".repeat(75));

        for (i, (combo, prob, odds, ev)) in results.iter().take(10).enumerate() {
            // Calculate Kelly bet sizing
            let decimal_odds = *odds / 100.0;
            let b = decimal_odds - 1.0;
            let q = 1.0 - *prob;
            let full_kelly = ((*prob * b - q) / b).max(0.0);
            let kelly_pct = full_kelly * kelly_fraction * 100.0;
            let bet_amount = (bankroll * full_kelly * kelly_fraction / bet_unit as f64).round() as u32 * bet_unit;
            let bet_amount = bet_amount.max(bet_unit);

            // Color coding
            let ev_str = if *ev >= 1.5 {
                format!("{:.3}", ev).green().bold().to_string()
            } else if *ev >= 1.2 {
                format!("{:.3}", ev).yellow().to_string()
            } else {
                format!("{:.3}", ev).normal().to_string()
            };

            let kelly_str = if kelly_pct >= 2.0 {
                format!("{:.2}", kelly_pct).green().to_string()
            } else if kelly_pct >= 1.0 {
                format!("{:.2}", kelly_pct).yellow().to_string()
            } else {
                format!("{:.2}", kelly_pct).dimmed().to_string()
            };

            println!(
                "  {:>3}. {:>10} {:>7.4} {:>8.1} {:>8} {:>9}% {:>12}",
                i + 1,
                combo.cyan(),
                prob * 100.0,
                odds,
                ev_str,
                kelly_str,
                format!("¥{:>6}", bet_amount),
            );
        }
        println!("  {}", "─".repeat(75));

        // Summary
        let total_bets: u32 = results.iter().take(10).map(|(_, prob, odds, _)| {
            let decimal_odds = *odds / 100.0;
            let b = decimal_odds - 1.0;
            let q = 1.0 - *prob;
            let full_kelly = ((*prob * b - q) / b).max(0.0);
            let bet_amount = (bankroll * full_kelly * kelly_fraction / bet_unit as f64).round() as u32 * bet_unit;
            bet_amount.max(bet_unit)
        }).sum();

        println!();
        println!(
            "  {} ¥{:>6} ({:.1}% of bankroll)",
            "Total suggested bets:".bold(),
            total_bets,
            (total_bets as f64 / bankroll) * 100.0
        );
    }

    // Save to file if requested
    if let Some(out_path) = output {
        let output_data = serde_json::json!({
            "race_id": race_id,
            "race_name": race_info.race_name,
            "bet_type": bet_type,
            "ev_threshold": ev_threshold,
            "bankroll": bankroll,
            "kelly_fraction": kelly_fraction,
            "win_probabilities": win_probs,
            "recommended_bets": results.iter().take(10).map(|(c, p, o, e)| {
                let decimal_odds = *o / 100.0;
                let b = decimal_odds - 1.0;
                let q = 1.0 - *p;
                let full_kelly = ((*p * b - q) / b).max(0.0);
                let kelly_pct = full_kelly * kelly_fraction;
                let bet_amount = (bankroll * full_kelly * kelly_fraction / bet_unit as f64).round() as u32 * bet_unit;
                serde_json::json!({
                    "combination": c,
                    "probability": p,
                    "odds": o,
                    "expected_value": e,
                    "kelly_fraction": kelly_pct,
                    "recommended_bet": bet_amount.max(bet_unit)
                })
            }).collect::<Vec<_>>()
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&output_data)?)?;
        eprintln!();
        eprintln!("{} {}", "Results saved to:".green(), out_path.display().to_string().cyan());
    }

    Ok(())
}

/// Fetch race card HTML using provided browser instance
async fn fetch_race_card_with_browser(
    race_id: &str,
    browser: &crate::scraper::Browser,
    rate_limiter: &crate::scraper::RateLimiter,
) -> anyhow::Result<String> {
    rate_limiter.acquire().await;
    let url = crate::scraper::race_card_url(race_id);
    browser.fetch_page(&url).await
}

/// Fetch odds from API
async fn fetch_odds(
    race_id: &str,
    bet_type: &str,
) -> anyhow::Result<std::collections::HashMap<String, f64>> {
    use crate::scraper::parsers::OddsParser;

    let client = reqwest::Client::new();
    let url = match bet_type {
        "trifecta" => crate::scraper::trifecta_odds_url(race_id),
        _ => crate::scraper::exacta_odds_url(race_id),
    };

    let response = client.get(&url).send().await?.text().await?;

    let mut odds_map = std::collections::HashMap::new();

    match bet_type {
        "trifecta" => {
            let parsed = OddsParser::parse_trifecta(&response)?;
            for ((first, second, third), odds) in parsed.odds {
                let key = format!("{:02}-{:02}-{:02}", first, second, third);
                odds_map.insert(key, odds);
            }
        }
        _ => {
            let parsed = OddsParser::parse_exacta(&response)?;
            for ((first, second), odds) in parsed.odds {
                let key = format!("{:02}-{:02}", first, second);
                odds_map.insert(key, odds);
            }
        }
    }

    Ok(odds_map)
}

/// Calculate EV for exacta bets
fn calculate_ev_exacta(
    top: &[((String, String), f64)],
    odds_map: &std::collections::HashMap<String, f64>,
    ev_threshold: f64,
) -> Vec<(String, f64, f64, f64)> {
    let mut results = Vec::new();

    for ((first, second), prob) in top {
        // Match horse_id to post_position (simplified - assumes ID contains position)
        // In real implementation, we'd need to map horse_id to post_position
        let key = format!("{}-{}", first, second);

        // Try to find matching odds
        if let Some(&odds) = odds_map.get(&key) {
            let ev = prob * odds;
            if ev >= ev_threshold {
                results.push((format!("{}-{}", first, second), *prob, odds, ev));
            }
        }
    }

    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    results
}

/// Calculate EV for trifecta bets
fn calculate_ev_trifecta(
    top: &[((String, String, String), f64)],
    odds_map: &std::collections::HashMap<String, f64>,
    ev_threshold: f64,
) -> Vec<(String, f64, f64, f64)> {
    let mut results = Vec::new();

    for ((first, second, third), prob) in top {
        let key = format!("{}-{}-{}", first, second, third);

        if let Some(&odds) = odds_map.get(&key) {
            let ev = prob * odds;
            if ev >= ev_threshold {
                results.push((
                    format!("{}-{}-{}", first, second, third),
                    *prob,
                    odds,
                    ev,
                ));
            }
        }
    }

    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    results
}

/// Run historical data scraping.
pub async fn run_scrape_historical(
    date: Option<String>,
    start: Option<String>,
    end: Option<String>,
    db_path: PathBuf,
    include_odds: bool,
    force: bool,
    verbose: bool,
) -> anyhow::Result<()> {
    use chrono::{Datelike, NaiveDate};
    use colored::Colorize;
    use indicatif::{ProgressBar, ProgressStyle};

    use crate::scraper::historical::{
        exacta_odds_history_url, quinella_odds_history_url, race_list_url, race_result_url,
        trifecta_odds_history_url, trio_odds_history_url, wide_odds_history_url,
        HistoricalOddsParser, RaceListParser, RaceResultParser,
    };
    use crate::scraper::Browser;
    use crate::storage::RaceRepository;

    // Determine date range
    let (start_date, end_date) = if let Some(d) = date {
        let d = NaiveDate::parse_from_str(&d, "%Y-%m-%d")
            .map_err(|e| anyhow::anyhow!("Invalid date format: {}. Use YYYY-MM-DD", e))?;
        (d, d)
    } else if let (Some(s), Some(e)) = (start.clone(), end.clone()) {
        let start_date = NaiveDate::parse_from_str(&s, "%Y-%m-%d")
            .map_err(|e| anyhow::anyhow!("Invalid start date: {}. Use YYYY-MM-DD", e))?;
        let end_date = NaiveDate::parse_from_str(&e, "%Y-%m-%d")
            .map_err(|e| anyhow::anyhow!("Invalid end date: {}. Use YYYY-MM-DD", e))?;
        (start_date, end_date)
    } else {
        return Err(anyhow::anyhow!(
            "Specify --date for single date or --start and --end for range"
        ));
    };

    println!(
        "{} Scraping historical data from {} to {}",
        "Historical Scraper".green().bold(),
        start_date,
        end_date
    );
    println!("Database: {}", db_path.display());
    println!("Include odds: {}", include_odds);
    println!();

    // Initialize repository
    let repo = RaceRepository::new(&db_path)?;

    // Count total days
    let total_days = (end_date - start_date).num_days() + 1;
    println!("Total days to scrape: {}", total_days);

    // Resume capability
    if !force {
        if let Some(last_date) = repo.get_last_race_date()? {
            println!(
                "{}",
                format!("Last scraped date: {}. Use --force to re-scrape.", last_date).yellow()
            );
        }
    }

    // Initialize browser
    println!("Launching browser...");
    let browser = Browser::launch().await?;

    let progress = ProgressBar::new(total_days as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut total_races = 0;
    let mut current_date = start_date;

    while current_date <= end_date {
        let year = current_date.year() as u32;
        let month = current_date.month();
        let day = current_date.day();

        // Fetch race list for the day
        let list_url = race_list_url(year, month, day);
        if verbose {
            println!("Fetching race list: {}", list_url);
        }

        match browser.fetch_page(&list_url).await {
            Ok(html) => {
                let race_ids = RaceListParser::parse(&html).unwrap_or_default();

                for race_id in &race_ids {
                    // Check if race already exists
                    if !force && repo.race_exists(race_id)? {
                        if verbose {
                            println!("  Skipping existing race: {}", race_id);
                        }
                        continue;
                    }

                    // Fetch race result
                    let result_url = race_result_url(race_id);
                    if verbose {
                        println!("  Fetching race: {}", race_id);
                    }

                    match browser.fetch_page(&result_url).await {
                        Ok(result_html) => {
                            if let Ok((race_info, entries)) =
                                RaceResultParser::parse(&result_html, race_id)
                            {
                                // Save race
                                repo.insert_race(&race_info)?;

                                // Save entries
                                for entry in &entries {
                                    repo.insert_entry(entry)?;
                                }

                                total_races += 1;

                                if verbose {
                                    println!(
                                        "    Saved race {} with {} entries",
                                        race_id,
                                        entries.len()
                                    );
                                }

                                // Fetch odds if include_odds is true
                                if include_odds {
                                    // Fetch each bet type's odds
                                    let odds_types: Vec<(&str, String, fn(&str) -> anyhow::Result<std::collections::HashMap<String, f64>>)> = vec![
                                        ("exacta", exacta_odds_history_url(&race_id), HistoricalOddsParser::parse_exacta),
                                        ("quinella", quinella_odds_history_url(&race_id), HistoricalOddsParser::parse_quinella),
                                        ("trifecta", trifecta_odds_history_url(&race_id), HistoricalOddsParser::parse_trifecta),
                                        ("trio", trio_odds_history_url(&race_id), HistoricalOddsParser::parse_trio),
                                        ("wide", wide_odds_history_url(&race_id), HistoricalOddsParser::parse_wide),
                                    ];

                                    for (bet_type, url, parser) in odds_types {
                                        match browser.fetch_page(&url).await {
                                            Ok(html) => {
                                                if let Ok(odds_map) = parser(&html) {
                                                    let mut odds_count = 0;
                                                    for (combo, odds) in odds_map {
                                                        if repo.insert_odds(&race_id, bet_type, &combo, odds).is_ok() {
                                                            odds_count += 1;
                                                        }
                                                    }
                                                    if verbose && odds_count > 0 {
                                                        println!("      {} {} odds saved", odds_count, bet_type);
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                if verbose {
                                                    eprintln!("      Failed to fetch {} odds: {}", bet_type, e);
                                                }
                                            }
                                        }
                                        // Rate limiting between odds fetches
                                        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            if verbose {
                                eprintln!("  Failed to fetch race {}: {}", race_id, e);
                            }
                        }
                    }

                    // Rate limiting
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                }
            }
            Err(e) => {
                if verbose {
                    eprintln!("Failed to fetch race list for {}: {}", current_date, e);
                }
            }
        }

        progress.inc(1);
        current_date = current_date.succ_opt().unwrap();
    }

    progress.finish_with_message("Done!");

    println!();
    println!("{}", "Summary:".green().bold());
    println!("  Total races scraped: {}", total_races);
    println!("  Total races in database: {}", repo.get_race_count()?);

    Ok(())
}

/// Run backtest using historical SQLite data.
pub async fn run_backtest_historical(
    db_path: PathBuf,
    start: String,
    end: String,
    bet_type_str: String,
    model_path: Option<PathBuf>,
    calibration_path: Option<PathBuf>,
    ev_threshold: f64,
    _format: String,
) -> anyhow::Result<()> {
    use chrono::NaiveDate;
    use colored::Colorize;
    use indicatif::{ProgressBar, ProgressStyle};
    use ndarray::Array2;

    use crate::backtest::BetType;
    use crate::betting::calculate_ev;
    use crate::calibration::Calibrator;
    use crate::exacta::calculate_exacta_probs;
    use crate::model::PositionModel;
    use crate::quinella::calculate_quinella_probs;
    use crate::storage::RaceRepository;
    use crate::trifecta::calculate_trifecta_probs;
    use crate::trio::calculate_trio_probs;
    use crate::wide::calculate_wide_probs;

    // Parse bet type
    let bet_type = BetType::from_str(&bet_type_str)
        .ok_or_else(|| anyhow::anyhow!("Invalid bet type: {}", bet_type_str))?;

    // Parse dates
    let start_date = NaiveDate::parse_from_str(&start, "%Y-%m-%d")
        .map_err(|e| anyhow::anyhow!("Invalid start date: {}. Use YYYY-MM-DD", e))?;
    let end_date = NaiveDate::parse_from_str(&end, "%Y-%m-%d")
        .map_err(|e| anyhow::anyhow!("Invalid end date: {}. Use YYYY-MM-DD", e))?;

    println!(
        "{} Running backtest from {} to {}",
        "Historical Backtest".green().bold(),
        start_date,
        end_date
    );
    println!("Database: {}", db_path.display());
    println!("Bet type: {}", bet_type.name());
    println!("EV threshold: {:.2}", ev_threshold);
    println!();

    // Check database exists
    if !db_path.exists() {
        return Err(anyhow::anyhow!(
            "Database not found: {}\nRun 'keiba-api scrape-historical' first to collect data.",
            db_path.display()
        ));
    }

    // Load model
    let model_file = model_path.unwrap_or_else(|| PathBuf::from("data/models/position_model.onnx"));
    println!("Loading model: {}", model_file.display());
    let model = PositionModel::load(&model_file)?;

    // Load calibrator
    let calibrator = if let Some(cal_path) = calibration_path {
        println!("Loading calibrator: {}", cal_path.display());
        match Calibrator::from_file(&cal_path) {
            Ok(cal) => cal,
            Err(e) => {
                println!("{}: Failed to load calibrator: {}", "Warning".yellow(), e);
                Calibrator::None
            }
        }
    } else {
        // Try default path
        let default_cal = PathBuf::from("data/models/calibration.json");
        if default_cal.exists() {
            println!("Loading calibrator: {}", default_cal.display());
            Calibrator::from_file(&default_cal).unwrap_or(Calibrator::None)
        } else {
            Calibrator::None
        }
    };

    // Load repository
    let repo = RaceRepository::new(&db_path)?;

    // Get race count
    let race_count = repo.get_race_count()?;
    println!("Total races in database: {}", race_count);

    // Get races in date range
    let races = repo.get_races_by_date_range(start_date, end_date)?;
    println!("Races in date range: {}", races.len());

    if races.is_empty() {
        return Err(anyhow::anyhow!(
            "No races found in date range. Run 'keiba-api scrape-historical' to collect more data."
        ));
    }

    println!();

    // Track results
    let mut total_bets = 0usize;
    let mut total_wins = 0usize;
    let mut total_wagered = 0.0f64;
    let mut total_returned = 0.0f64;
    let mut races_processed = 0usize;
    let mut races_skipped = 0usize;

    // Progress bar
    let pb = ProgressBar::new(races.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    for race in &races {
        pb.set_message(format!("{}", race.race_id));

        // Get entries for this race
        let entries = repo.get_race_entries(&race.race_id)?;

        // Skip races with too few entries
        if entries.len() < bet_type.min_horses() {
            races_skipped += 1;
            pb.inc(1);
            continue;
        }

        // Skip races without complete result data
        let valid_entries: Vec<_> = entries
            .iter()
            .filter(|e| e.finish_position.is_some() && e.win_odds.is_some())
            .collect();

        if valid_entries.len() < bet_type.min_horses() {
            races_skipped += 1;
            pb.inc(1);
            continue;
        }

        // Build feature array from entries
        // Using simplified features based on available data
        let n_horses = valid_entries.len();
        let mut features = Array2::<f32>::zeros((n_horses, 39));

        for (i, entry) in valid_entries.iter().enumerate() {
            // Build 39-feature vector with available data and defaults
            let odds_log = entry.win_odds.map(|o| o.ln() as f32).unwrap_or(2.0);

            // Feature indices (matching FEATURE_NAMES order):
            // 0: horse_age_num
            features[[i, 0]] = entry.horse_age.unwrap_or(4) as f32;
            // 1: horse_sex_encoded (0=牝, 1=牡, 2=セ)
            features[[i, 1]] = match entry.horse_sex.as_deref() {
                Some("牝") => 0.0,
                Some("牡") => 1.0,
                Some("セ") | Some("騸") => 2.0,
                _ => 1.0,
            };
            // 2: post_position_num
            features[[i, 2]] = entry.post_position as f32;
            // 3: weight_carried
            features[[i, 3]] = entry.weight_carried.unwrap_or(55.0) as f32;
            // 4: horse_weight
            features[[i, 4]] = entry.horse_weight.unwrap_or(480) as f32;
            // 5-9: jockey/trainer stats (defaults)
            features[[i, 5]] = 0.1;  // jockey_win_rate
            features[[i, 6]] = 0.3;  // jockey_place_rate
            features[[i, 7]] = 0.1;  // trainer_win_rate
            features[[i, 8]] = 100.0; // jockey_races
            features[[i, 9]] = 100.0; // trainer_races
            // 10-13: race conditions
            features[[i, 10]] = race.distance as f32;
            features[[i, 11]] = if race.surface == "turf" { 1.0 } else { 0.0 };
            features[[i, 12]] = if race.surface == "dirt" { 1.0 } else { 0.0 };
            features[[i, 13]] = match race.track_condition.as_deref() {
                Some("良") => 0.0,
                Some("稍重") => 1.0,
                Some("重") => 2.0,
                Some("不良") => 3.0,
                _ => 0.0,
            };
            // 14-21: past performance (defaults based on popularity proxy)
            let popularity_proxy = 1.0 / (1.0 + odds_log.exp());
            features[[i, 14]] = 5.0 - popularity_proxy * 3.0; // avg_position_last_3
            features[[i, 15]] = 5.0 - popularity_proxy * 3.0; // avg_position_last_5
            features[[i, 16]] = popularity_proxy * 0.2; // win_rate_last_3
            features[[i, 17]] = popularity_proxy * 0.2; // win_rate_last_5
            features[[i, 18]] = popularity_proxy * 0.4; // place_rate_last_3
            features[[i, 19]] = popularity_proxy * 0.4; // place_rate_last_5
            features[[i, 20]] = 5.0 - popularity_proxy * 3.0; // last_position
            features[[i, 21]] = 10.0; // career_races
            // 22: odds_log
            features[[i, 22]] = odds_log;
            // 23-25: running style (from corner positions if available)
            let early = entry.corner_1.or(entry.corner_2).unwrap_or(n_horses as u8 / 2) as f32;
            let late = entry.corner_4.or(entry.corner_3).unwrap_or(n_horses as u8 / 2) as f32;
            features[[i, 23]] = early / n_horses as f32; // early_position
            features[[i, 24]] = late / n_horses as f32;  // late_position
            features[[i, 25]] = late - early;            // position_change
            // 26-32: aptitude features (defaults)
            features[[i, 26]] = 0.5; // aptitude_sprint
            features[[i, 27]] = 0.5; // aptitude_mile
            features[[i, 28]] = 0.5; // aptitude_intermediate
            features[[i, 29]] = 0.5; // aptitude_long
            features[[i, 30]] = 0.5; // aptitude_turf
            features[[i, 31]] = 0.5; // aptitude_dirt
            features[[i, 32]] = 0.5; // aptitude_course
            // 33-35: pace features
            features[[i, 33]] = entry.last_3f.unwrap_or(35.0); // last_3f_avg
            features[[i, 34]] = entry.last_3f.unwrap_or(34.0); // last_3f_best
            features[[i, 35]] = entry.last_3f.unwrap_or(35.0); // last_3f_last
            // 36-38: race classification
            features[[i, 36]] = entry.weight_change.unwrap_or(0) as f32; // weight_change_kg
            features[[i, 37]] = if race.grade.is_some() { 1.0 } else { 0.0 }; // is_graded_race
            features[[i, 38]] = match race.grade.as_deref() {
                Some("G1") => 1.0,
                Some("G2") => 2.0,
                Some("G3") => 3.0,
                Some("L") => 4.0,
                Some("OP") => 5.0,
                _ => 6.0,
            }; // grade_level
        }

        // Run inference
        let probs = match model.predict(features) {
            Ok(p) => p,
            Err(_) => {
                races_skipped += 1;
                pb.inc(1);
                continue;
            }
        };

        // Extract win probabilities and apply calibration
        let mut win_probs_vec: Vec<f64> = probs.iter().map(|p| p[0]).collect();

        // Apply calibration
        win_probs_vec = win_probs_vec
            .iter()
            .map(|&p| calibrator.calibrate(p))
            .collect();

        // Normalize probabilities
        let sum: f64 = win_probs_vec.iter().sum();
        if sum > 0.0 {
            win_probs_vec.iter_mut().for_each(|p| *p /= sum);
        }

        // Build HashMap with horse indices as keys
        let win_probs: std::collections::HashMap<String, f64> = win_probs_vec
            .iter()
            .enumerate()
            .map(|(i, &p)| (i.to_string(), p))
            .collect();

        // Calculate combination probabilities based on bet type
        // These return HashMap<(String, String), f64> or similar
        let min_prob = 0.001;
        let combo_probs: Vec<(Vec<usize>, f64)> = match bet_type {
            BetType::Exacta => {
                let probs = calculate_exacta_probs(&win_probs, min_prob);
                probs
                    .into_iter()
                    .map(|((a, b), p)| (vec![a.parse().unwrap_or(0), b.parse().unwrap_or(0)], p))
                    .collect()
            }
            BetType::Trifecta => {
                let probs = calculate_trifecta_probs(&win_probs, min_prob);
                probs
                    .into_iter()
                    .map(|((a, b, c), p)| {
                        (
                            vec![
                                a.parse().unwrap_or(0),
                                b.parse().unwrap_or(0),
                                c.parse().unwrap_or(0),
                            ],
                            p,
                        )
                    })
                    .collect()
            }
            BetType::Quinella => {
                let probs = calculate_quinella_probs(&win_probs, min_prob);
                probs
                    .into_iter()
                    .map(|(set, p)| {
                        let mut combo: Vec<usize> = set
                            .into_iter()
                            .map(|s| s.parse().unwrap_or(0))
                            .collect();
                        combo.sort();
                        (combo, p)
                    })
                    .collect()
            }
            BetType::Trio => {
                let probs = calculate_trio_probs(&win_probs, min_prob);
                probs
                    .into_iter()
                    .map(|(set, p)| {
                        let mut combo: Vec<usize> = set
                            .into_iter()
                            .map(|s| s.parse().unwrap_or(0))
                            .collect();
                        combo.sort();
                        (combo, p)
                    })
                    .collect()
            }
            BetType::Wide => {
                let probs = calculate_wide_probs(&win_probs, min_prob);
                probs
                    .into_iter()
                    .map(|(set, p)| {
                        let mut combo: Vec<usize> = set
                            .into_iter()
                            .map(|s| s.parse().unwrap_or(0))
                            .collect();
                        combo.sort();
                        (combo, p)
                    })
                    .collect()
            }
        };

        // Get actual results (finish positions)
        let mut sorted_entries: Vec<_> = valid_entries.iter().enumerate().collect();
        sorted_entries.sort_by_key(|(_, e)| e.finish_position.unwrap_or(99));

        let actual_top3: Vec<usize> = sorted_entries
            .iter()
            .take(3)
            .map(|(idx, _)| *idx)
            .collect();

        // Determine winning combinations
        let winning_combo: Vec<usize> = match bet_type {
            BetType::Exacta => actual_top3.iter().take(2).cloned().collect(),
            BetType::Trifecta => actual_top3.clone(),
            BetType::Quinella => {
                let mut combo: Vec<usize> = actual_top3.iter().take(2).cloned().collect();
                combo.sort();
                combo
            }
            BetType::Trio => {
                let mut combo = actual_top3.clone();
                combo.sort();
                combo
            }
            BetType::Wide => {
                // Any pair from top 3 wins
                vec![] // We'll check differently for wide
            }
        };

        // Evaluate each high-EV combination
        for (combo, prob) in &combo_probs {
            // Use win_odds from entries to estimate combination odds
            // This is a rough approximation since we don't have actual combination odds
            // Returns decimal odds (e.g., 9.64x)
            let combo_odds_decimal = match bet_type {
                BetType::Exacta | BetType::Quinella => {
                    if combo.len() >= 2 {
                        let o1 = valid_entries[combo[0]].win_odds.unwrap_or(10.0);
                        let o2 = valid_entries[combo[1]].win_odds.unwrap_or(10.0);
                        (o1 * o2).sqrt() * 2.0 // Rough approximation
                    } else {
                        10.0
                    }
                }
                BetType::Trifecta | BetType::Trio => {
                    if combo.len() >= 3 {
                        let o1 = valid_entries[combo[0]].win_odds.unwrap_or(10.0);
                        let o2 = valid_entries[combo[1]].win_odds.unwrap_or(10.0);
                        let o3 = valid_entries[combo[2]].win_odds.unwrap_or(10.0);
                        (o1 * o2 * o3).powf(1.0 / 3.0) * 5.0
                    } else {
                        50.0
                    }
                }
                BetType::Wide => {
                    if combo.len() >= 2 {
                        let o1 = valid_entries[combo[0]].win_odds.unwrap_or(10.0);
                        let o2 = valid_entries[combo[1]].win_odds.unwrap_or(10.0);
                        (o1 * o2).sqrt() * 0.5
                    } else {
                        3.0
                    }
                }
            };

            // Convert to Japanese format (multiply by 100)
            let combo_odds = combo_odds_decimal * 100.0;

            let ev = calculate_ev(*prob, combo_odds);

            if ev >= ev_threshold {
                total_bets += 1;
                total_wagered += 100.0; // ¥100 per bet

                // Check if this combo won
                let won = match bet_type {
                    BetType::Exacta | BetType::Trifecta => *combo == winning_combo,
                    BetType::Quinella | BetType::Trio => {
                        let mut sorted_combo = combo.clone();
                        sorted_combo.sort();
                        sorted_combo == winning_combo
                    }
                    BetType::Wide => {
                        // Check if both horses in combo are in top 3
                        combo.len() >= 2
                            && actual_top3.contains(&combo[0])
                            && actual_top3.contains(&combo[1])
                    }
                };

                if won {
                    total_wins += 1;
                    total_returned += 100.0 * combo_odds_decimal;
                }
            }
        }

        races_processed += 1;
        pb.inc(1);
    }

    pb.finish_with_message("Done");
    println!();

    // Print results
    println!("{}", "═".repeat(50).cyan());
    println!("{}", "Backtest Results".green().bold());
    println!("{}", "═".repeat(50).cyan());
    println!();
    println!("Races processed: {}", races_processed);
    println!("Races skipped:   {}", races_skipped);
    println!();
    println!("{}", "Betting Statistics".yellow().bold());
    println!("{}", "─".repeat(30));
    println!("Total bets:      {}", total_bets);
    println!("Total wins:      {}", total_wins);
    println!(
        "Hit rate:        {:.2}%",
        if total_bets > 0 {
            (total_wins as f64 / total_bets as f64) * 100.0
        } else {
            0.0
        }
    );
    println!();
    println!("{}", "Financial Results".yellow().bold());
    println!("{}", "─".repeat(30));
    println!("Total wagered:   ¥{:.0}", total_wagered);
    println!("Total returned:  ¥{:.0}", total_returned);
    println!(
        "Profit/Loss:     ¥{:.0}",
        total_returned - total_wagered
    );
    println!(
        "ROI:             {:.2}%",
        if total_wagered > 0.0 {
            ((total_returned - total_wagered) / total_wagered) * 100.0
        } else {
            0.0
        }
    );
    println!();

    // Add warning about odds estimation
    println!("{}", "⚠ Note".yellow().bold());
    println!("Combination odds are estimated from win odds.");
    println!("For accurate results, scrape with --include-odds.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_race_id_valid() {
        assert!(validate_race_id("202506050811").is_ok());
        assert!(validate_race_id("202501010101").is_ok());
        assert!(validate_race_id("202010121212").is_ok());
    }

    #[test]
    fn test_validate_race_id_invalid_length() {
        assert!(validate_race_id("2025060508").is_err());
        assert!(validate_race_id("20250605081112").is_err());
        assert!(validate_race_id("").is_err());
    }

    #[test]
    fn test_validate_race_id_invalid_chars() {
        assert!(validate_race_id("20250605081a").is_err());
        assert!(validate_race_id("2025-06-05-08").is_err());
    }

    #[test]
    fn test_validate_race_id_invalid_year() {
        assert!(validate_race_id("201806050811").is_err()); // Too old
        assert!(validate_race_id("203106050811").is_err()); // Too future
    }

    #[test]
    fn test_validate_race_id_invalid_racecourse() {
        assert!(validate_race_id("202500050811").is_err()); // 00 invalid
        assert!(validate_race_id("202511050811").is_err()); // 11 invalid
    }

    #[test]
    fn test_validate_race_id_invalid_race_number() {
        assert!(validate_race_id("202506050800").is_err()); // Race 00
        assert!(validate_race_id("202506050813").is_err()); // Race 13
    }

    #[test]
    fn test_validate_bet_type_valid() {
        assert!(validate_bet_type("exacta").is_ok());
        assert!(validate_bet_type("trifecta").is_ok());
        assert!(validate_bet_type("quinella").is_ok());
        assert!(validate_bet_type("trio").is_ok());
        assert!(validate_bet_type("wide").is_ok());
        assert!(validate_bet_type("all").is_ok());
        assert!(validate_bet_type("EXACTA").is_ok()); // Case insensitive
    }

    #[test]
    fn test_validate_bet_type_invalid() {
        assert!(validate_bet_type("invalid").is_err());
        assert!(validate_bet_type("").is_err());
        assert!(validate_bet_type("win").is_err());
    }
}
