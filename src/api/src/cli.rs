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
