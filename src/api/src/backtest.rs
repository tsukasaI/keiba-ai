//! Backtesting module for walk-forward validation.
//!
//! Provides tools for backtesting betting strategies with historical data.

use chrono::NaiveDate;
use polars::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::betting::calculate_ev;
use crate::calibration::Calibrator;
use crate::config::{BettingConfig, FEATURE_NAMES};
use crate::exacta::{calculate_exacta_probs, extract_win_probs};
use crate::model::{SharedModel, NUM_FEATURES};
use crate::quinella::calculate_quinella_probs;
use crate::trifecta::calculate_trifecta_probs;
use crate::trio::calculate_trio_probs;
use crate::wide::calculate_wide_probs;

/// Bet type for backtesting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetType {
    Exacta,    // 馬単 - 1st and 2nd in order
    Trifecta,  // 三連単 - 1st, 2nd, 3rd in order
    Quinella,  // 馬連 - 1st and 2nd any order
    Trio,      // 三連複 - 1st, 2nd, 3rd any order
    Wide,      // ワイド - 2 horses in top 3
}

impl BetType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "exacta" | "umatan" => Some(BetType::Exacta),
            "trifecta" | "sanrentan" => Some(BetType::Trifecta),
            "quinella" | "umaren" => Some(BetType::Quinella),
            "trio" | "sanrenpuku" => Some(BetType::Trio),
            "wide" => Some(BetType::Wide),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            BetType::Exacta => "exacta",
            BetType::Trifecta => "trifecta",
            BetType::Quinella => "quinella",
            BetType::Trio => "trio",
            BetType::Wide => "wide",
        }
    }

    pub fn min_horses(&self) -> usize {
        match self {
            BetType::Exacta | BetType::Quinella | BetType::Wide => 2,
            BetType::Trifecta | BetType::Trio => 3,
        }
    }
}

/// A single bet result.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BetResult {
    pub race_id: String,
    pub race_date: NaiveDate,
    pub bet_type: BetType,
    pub combination: Vec<String>,     // Horse numbers in bet order
    pub actual_positions: Vec<String>, // Actual top finishers (1st, 2nd, 3rd)
    pub probability: f64,
    pub odds: f64,
    pub expected_value: f64,
    pub won: bool,
    pub profit: f64,
}

/// Results from a backtest period.
#[derive(Debug, Clone, Default)]
pub struct PeriodResult {
    pub period_name: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub num_bets: usize,
    pub num_wins: usize,
    pub total_bet: f64,
    pub total_return: f64,
}

impl PeriodResult {
    pub fn profit(&self) -> f64 {
        self.total_return - self.total_bet
    }

    pub fn roi(&self) -> f64 {
        if self.total_bet > 0.0 {
            (self.total_return - self.total_bet) / self.total_bet
        } else {
            0.0
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.num_bets > 0 {
            self.num_wins as f64 / self.num_bets as f64
        } else {
            0.0
        }
    }
}

/// Aggregate backtest results.
#[derive(Debug, Clone, Default)]
pub struct BacktestResults {
    pub bets: Vec<BetResult>,
    pub periods: Vec<PeriodResult>,
    pub total_bet: f64,
    pub total_return: f64,
    pub num_bets: usize,
    pub num_wins: usize,
}

impl BacktestResults {
    pub fn profit(&self) -> f64 {
        self.total_return - self.total_bet
    }

    pub fn roi(&self) -> f64 {
        if self.total_bet > 0.0 {
            (self.total_return - self.total_bet) / self.total_bet
        } else {
            0.0
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.num_bets > 0 {
            self.num_wins as f64 / self.num_bets as f64
        } else {
            0.0
        }
    }

    pub fn max_drawdown(&self) -> f64 {
        let mut peak = 0.0;
        let mut max_dd = 0.0;
        let mut cumulative = 0.0;

        for bet in &self.bets {
            cumulative += bet.profit;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = peak - cumulative;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }
}

/// Odds data for a single race.
#[derive(Debug, Clone, Default)]
pub struct RaceOdds {
    pub exacta_odds: HashMap<String, f64>,    // "1-2" -> 1520.0 (ordered)
    pub trifecta_odds: HashMap<String, f64>,  // "1-2-3" -> 15200.0 (ordered)
    pub quinella_odds: HashMap<String, f64>,  // "1-2" -> 760.0 (unordered, sorted)
    pub trio_odds: HashMap<String, f64>,      // "1-2-3" -> 5200.0 (unordered, sorted)
    pub wide_odds: HashMap<String, f64>,      // "1-2" -> 380.0 (unordered, sorted)
}

impl RaceOdds {
    #[allow(dead_code)]
    pub fn get_odds(&self, bet_type: BetType, key: &str) -> Option<f64> {
        match bet_type {
            BetType::Exacta => self.exacta_odds.get(key).copied(),
            BetType::Trifecta => self.trifecta_odds.get(key).copied(),
            BetType::Quinella => self.quinella_odds.get(key).copied(),
            BetType::Trio => self.trio_odds.get(key).copied(),
            BetType::Wide => self.wide_odds.get(key).copied(),
        }
    }
}

/// Lookup for odds data.
#[allow(dead_code)]
pub struct OddsLookup {
    data: HashMap<String, RaceOdds>, // race_id -> RaceOdds
    bet_type: BetType,
}

impl OddsLookup {
    /// Load odds from CSV file for a specific bet type.
    ///
    /// For 2-horse bets (exacta, quinella, wide):
    ///   Expected columns: race_id, first, second, odds
    /// For 3-horse bets (trifecta, trio):
    ///   Expected columns: race_id, first, second, third, odds
    pub fn from_csv<P: AsRef<Path>>(path: P, bet_type: BetType) -> anyhow::Result<Self> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;

        let mut data: HashMap<String, RaceOdds> = HashMap::new();

        // Handle race_id as either string or i64
        let race_id_col = df.column("race_id")?;
        let firsts = df.column("first")?.i64()?;
        let seconds = df.column("second")?.i64()?;
        let thirds = df.column("third").ok();
        let odds_col = df.column("odds")?.f64()?;

        for i in 0..df.height() {
            let race_id = match race_id_col.dtype() {
                DataType::String => race_id_col.str()?.get(i).unwrap_or("").to_string(),
                DataType::Int64 => race_id_col.i64()?.get(i).unwrap_or(0).to_string(),
                _ => continue,
            };
            let first = firsts.get(i).unwrap_or(0);
            let second = seconds.get(i).unwrap_or(0);
            let odds = odds_col.get(i).unwrap_or(0.0);

            let entry = data.entry(race_id.clone()).or_default();

            match bet_type {
                BetType::Exacta => {
                    let key = format!("{}-{}", first, second);
                    entry.exacta_odds.insert(key, odds);
                }
                BetType::Trifecta => {
                    if let Some(third_col) = &thirds {
                        let third = third_col.i64().ok().and_then(|c| c.get(i)).unwrap_or(0);
                        let key = format!("{}-{}-{}", first, second, third);
                        entry.trifecta_odds.insert(key, odds);
                    }
                }
                BetType::Quinella => {
                    // Unordered - use sorted key
                    let mut horses = [first, second];
                    horses.sort();
                    let key = format!("{}-{}", horses[0], horses[1]);
                    entry.quinella_odds.insert(key, odds);
                }
                BetType::Trio => {
                    if let Some(third_col) = &thirds {
                        let third = third_col.i64().ok().and_then(|c| c.get(i)).unwrap_or(0);
                        let mut horses = [first, second, third];
                        horses.sort();
                        let key = format!("{}-{}-{}", horses[0], horses[1], horses[2]);
                        entry.trio_odds.insert(key, odds);
                    }
                }
                BetType::Wide => {
                    // Unordered - use sorted key
                    let mut horses = [first, second];
                    horses.sort();
                    let key = format!("{}-{}", horses[0], horses[1]);
                    entry.wide_odds.insert(key, odds);
                }
            }
        }

        Ok(Self { data, bet_type })
    }

    /// Get odds for a race.
    pub fn get(&self, race_id: &str) -> Option<&RaceOdds> {
        self.data.get(race_id)
    }

    /// Get the bet type this lookup is for.
    #[allow(dead_code)]
    pub fn bet_type(&self) -> BetType {
        self.bet_type
    }
}

/// Race data for backtesting.
#[derive(Debug, Clone)]
pub struct RaceData {
    pub race_id: String,
    pub race_date: NaiveDate,
    pub horse_ids: Vec<String>,
    pub features: Vec<Vec<f32>>,      // [n_horses, n_features]
    pub actual_positions: Vec<usize>, // actual finishing position for each horse
}

/// Load race data from parquet.
pub fn load_race_data<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<RaceData>> {
    let df = LazyFrame::scan_parquet(path, Default::default())?
        .collect()?;

    // Group by race_id
    let race_ids = df.column("race_id")?.str()?;
    let race_dates = df.column("race_date")?.str()?;
    let horse_nums = df.column("horse_num")?.i64()?;
    let positions = df.column("position")?.i64()?;

    // Extract features
    let mut feature_cols: Vec<&Column> = Vec::new();
    for name in FEATURE_NAMES.iter() {
        if let Ok(col) = df.column(name) {
            feature_cols.push(col);
        }
    }

    // Build race data map
    let mut race_map: HashMap<String, RaceData> = HashMap::new();

    for i in 0..df.height() {
        let race_id = race_ids.get(i).unwrap_or("").to_string();
        let race_date_str = race_dates.get(i).unwrap_or("1970-01-01");
        let race_date =
            NaiveDate::parse_from_str(race_date_str, "%Y-%m-%d").unwrap_or(NaiveDate::MIN);
        let horse_num = horse_nums.get(i).unwrap_or(0) as usize;
        let position = positions.get(i).unwrap_or(0) as usize;

        // Extract features for this horse
        let mut features_row: Vec<f32> = Vec::with_capacity(NUM_FEATURES);
        for col in &feature_cols {
            let val = match col.dtype() {
                DataType::Float64 => col.f64().ok().and_then(|c| c.get(i)).unwrap_or(0.0) as f32,
                DataType::Float32 => col.f32().ok().and_then(|c| c.get(i)).unwrap_or(0.0),
                DataType::Int64 => col.i64().ok().and_then(|c| c.get(i)).unwrap_or(0) as f32,
                DataType::Int32 => col.i32().ok().and_then(|c| c.get(i)).unwrap_or(0) as f32,
                _ => 0.0,
            };
            features_row.push(val);
        }

        // Pad if needed
        while features_row.len() < NUM_FEATURES {
            features_row.push(0.0);
        }

        let entry = race_map.entry(race_id.clone()).or_insert_with(|| RaceData {
            race_id,
            race_date,
            horse_ids: Vec::new(),
            features: Vec::new(),
            actual_positions: Vec::new(),
        });

        entry.horse_ids.push(horse_num.to_string());
        entry.features.push(features_row);
        entry.actual_positions.push(position);
    }

    let mut races: Vec<RaceData> = race_map.into_values().collect();
    races.sort_by(|a, b| a.race_date.cmp(&b.race_date));

    Ok(races)
}

/// Backtester for betting strategies.
pub struct Backtester {
    model: SharedModel,
    calibrator: Calibrator,
    config: BettingConfig,
    bet_unit: f64,
    bet_type: BetType,
}

impl Backtester {
    pub fn new(
        model: SharedModel,
        calibrator: Calibrator,
        config: BettingConfig,
        bet_type: BetType,
    ) -> Self {
        let bet_unit = config.bet_unit as f64;
        Self {
            model,
            calibrator,
            config,
            bet_unit,
            bet_type,
        }
    }

    /// Run backtest on a list of races.
    pub fn run(&self, races: &[RaceData], odds_lookup: &OddsLookup) -> BacktestResults {
        let mut results = BacktestResults::default();
        let min_horses = self.bet_type.min_horses();

        for race in races {
            if race.horse_ids.len() < min_horses {
                continue;
            }

            // Build feature matrix
            let n_horses = race.horse_ids.len();
            let mut features = ndarray::Array2::<f32>::zeros((n_horses, NUM_FEATURES));
            for (i, feature_row) in race.features.iter().enumerate() {
                for (j, &val) in feature_row.iter().enumerate() {
                    if j < NUM_FEATURES {
                        features[[i, j]] = val;
                    }
                }
            }

            // Run inference
            let position_probs = match self.model.predict(features) {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Extract win probs
            let win_probs = extract_win_probs(&position_probs, &race.horse_ids);

            // Apply calibration
            let win_probs = if self.calibrator.is_enabled() {
                self.calibrator.calibrate_map(&win_probs)
            } else {
                win_probs
            };

            // Get odds for this race
            let race_odds = match odds_lookup.get(&race.race_id) {
                Some(o) => o,
                None => continue,
            };

            // Find actual top finishers
            let actual_top: Vec<String> = (1..=3)
                .filter_map(|pos| {
                    race.actual_positions
                        .iter()
                        .position(|&p| p == pos)
                        .map(|i| race.horse_ids[i].clone())
                })
                .collect();

            if actual_top.len() < min_horses {
                continue;
            }

            // Process bets based on bet type
            self.process_bets_for_type(
                &win_probs,
                race_odds,
                &actual_top,
                &race.race_id,
                race.race_date,
                &mut results,
            );
        }

        results
    }

    /// Process bets for the specific bet type.
    fn process_bets_for_type(
        &self,
        win_probs: &HashMap<String, f64>,
        race_odds: &RaceOdds,
        actual_top: &[String],
        race_id: &str,
        race_date: NaiveDate,
        results: &mut BacktestResults,
    ) {
        let min_prob = self.config.min_probability;

        match self.bet_type {
            BetType::Exacta => {
                let probs = calculate_exacta_probs(win_probs, min_prob);
                for ((first, second), &prob) in &probs {
                    let odds_key = format!("{}-{}", first, second);
                    if let Some(odds) = race_odds.exacta_odds.get(&odds_key) {
                        self.record_bet_if_value(
                            prob,
                            *odds,
                            vec![first.clone(), second.clone()],
                            actual_top,
                            race_id,
                            race_date,
                            results,
                            |combo, actual| combo[0] == actual[0] && combo[1] == actual[1],
                        );
                    }
                }
            }
            BetType::Trifecta => {
                let probs = calculate_trifecta_probs(win_probs, min_prob);
                for ((first, second, third), &prob) in &probs {
                    let odds_key = format!("{}-{}-{}", first, second, third);
                    if let Some(odds) = race_odds.trifecta_odds.get(&odds_key) {
                        self.record_bet_if_value(
                            prob,
                            *odds,
                            vec![first.clone(), second.clone(), third.clone()],
                            actual_top,
                            race_id,
                            race_date,
                            results,
                            |combo, actual| {
                                combo[0] == actual[0] && combo[1] == actual[1] && combo[2] == actual[2]
                            },
                        );
                    }
                }
            }
            BetType::Quinella => {
                let probs = calculate_quinella_probs(win_probs, min_prob);
                for (set, &prob) in &probs {
                    let horses: Vec<_> = set.iter().cloned().collect();
                    let mut sorted = horses.clone();
                    sorted.sort();
                    let odds_key = format!("{}-{}", sorted[0], sorted[1]);
                    if let Some(odds) = race_odds.quinella_odds.get(&odds_key) {
                        self.record_bet_if_value(
                            prob,
                            *odds,
                            horses,
                            actual_top,
                            race_id,
                            race_date,
                            results,
                            |combo, actual| {
                                let combo_set: BTreeSet<_> = combo.iter().collect();
                                let actual_set: BTreeSet<_> = actual[..2].iter().collect();
                                combo_set == actual_set
                            },
                        );
                    }
                }
            }
            BetType::Trio => {
                let probs = calculate_trio_probs(win_probs, min_prob);
                for (set, &prob) in &probs {
                    let horses: Vec<_> = set.iter().cloned().collect();
                    let mut sorted = horses.clone();
                    sorted.sort();
                    let odds_key = format!("{}-{}-{}", sorted[0], sorted[1], sorted[2]);
                    if let Some(odds) = race_odds.trio_odds.get(&odds_key) {
                        self.record_bet_if_value(
                            prob,
                            *odds,
                            horses,
                            actual_top,
                            race_id,
                            race_date,
                            results,
                            |combo, actual| {
                                let combo_set: BTreeSet<_> = combo.iter().collect();
                                let actual_set: BTreeSet<_> = actual.iter().collect();
                                combo_set == actual_set
                            },
                        );
                    }
                }
            }
            BetType::Wide => {
                let probs = calculate_wide_probs(win_probs, min_prob);
                for (set, &prob) in &probs {
                    let horses: Vec<_> = set.iter().cloned().collect();
                    let mut sorted = horses.clone();
                    sorted.sort();
                    let odds_key = format!("{}-{}", sorted[0], sorted[1]);
                    if let Some(odds) = race_odds.wide_odds.get(&odds_key) {
                        self.record_bet_if_value(
                            prob,
                            *odds,
                            horses,
                            actual_top,
                            race_id,
                            race_date,
                            results,
                            |combo, actual| {
                                // Wide wins if both horses finish in top 3
                                let actual_set: BTreeSet<_> = actual.iter().collect();
                                combo.iter().all(|h| actual_set.contains(h))
                            },
                        );
                    }
                }
            }
        }
    }

    /// Record a bet if it has positive expected value.
    #[allow(clippy::too_many_arguments)]
    fn record_bet_if_value<F>(
        &self,
        prob: f64,
        odds: f64,
        combination: Vec<String>,
        actual_top: &[String],
        race_id: &str,
        race_date: NaiveDate,
        results: &mut BacktestResults,
        win_check: F,
    ) where
        F: FnOnce(&[String], &[String]) -> bool,
    {
        let ev = calculate_ev(prob, odds);
        if ev > self.config.ev_threshold {
            let won = win_check(&combination, actual_top);
            let profit = if won {
                (odds / 100.0) * self.bet_unit - self.bet_unit
            } else {
                -self.bet_unit
            };

            let bet = BetResult {
                race_id: race_id.to_string(),
                race_date,
                bet_type: self.bet_type,
                combination,
                actual_positions: actual_top.to_vec(),
                probability: prob,
                odds,
                expected_value: ev,
                won,
                profit,
            };

            results.bets.push(bet);
            results.num_bets += 1;
            results.total_bet += self.bet_unit;

            if won {
                results.num_wins += 1;
                results.total_return += odds / 100.0 * self.bet_unit;
            }
        }
    }

    /// Run walk-forward backtest with period splits.
    pub fn run_walkforward(
        &self,
        races: &[RaceData],
        odds_lookup: &OddsLookup,
        n_periods: usize,
        train_months: usize,
        _test_months: usize,
    ) -> BacktestResults {
        if races.is_empty() {
            return BacktestResults::default();
        }

        let start_date = races.first().unwrap().race_date;
        let end_date = races.last().unwrap().race_date;

        // Calculate period boundaries
        let total_days = (end_date - start_date).num_days() as usize;
        let period_days = total_days / n_periods;

        let mut all_results = BacktestResults::default();

        for period in 0..n_periods {
            let period_start = start_date + chrono::Duration::days((period * period_days) as i64);
            let period_end = if period == n_periods - 1 {
                end_date
            } else {
                start_date + chrono::Duration::days(((period + 1) * period_days) as i64)
            };

            // Training window ends before test period (for reference, not used since model is pre-trained)
            let _train_start = period_start - chrono::Duration::days((train_months * 30) as i64);

            // Filter races for this test period
            let test_races: Vec<&RaceData> = races
                .iter()
                .filter(|r| r.race_date >= period_start && r.race_date < period_end)
                .collect();

            if test_races.is_empty() {
                continue;
            }

            // For now, run inference on test races (model is already trained externally)
            let test_races_owned: Vec<RaceData> = test_races.into_iter().cloned().collect();
            let period_results = self.run(&test_races_owned, odds_lookup);

            let period_roi = period_results.roi();
            let pr = PeriodResult {
                period_name: format!("Period {}", period + 1),
                start_date: period_start,
                end_date: period_end,
                num_bets: period_results.num_bets,
                num_wins: period_results.num_wins,
                total_bet: period_results.total_bet,
                total_return: period_results.total_return,
            };

            eprintln!(
                "Period {}: {} - {}, {} bets, ROI: {:.2}%",
                period + 1,
                period_start,
                period_end,
                period_results.num_bets,
                period_roi * 100.0
            );

            all_results.periods.push(pr);
            all_results.bets.extend(period_results.bets);
            all_results.num_bets += period_results.num_bets;
            all_results.num_wins += period_results.num_wins;
            all_results.total_bet += period_results.total_bet;
            all_results.total_return += period_results.total_return;
        }

        all_results
    }
}

/// Print backtest results in table format.
pub fn print_backtest_table(results: &BacktestResults) {
    println!("=== Backtest Results ===");
    println!();
    println!("Overall Statistics:");
    println!("  Total Bets:    {}", results.num_bets);
    println!("  Total Wins:    {}", results.num_wins);
    println!("  Hit Rate:      {:.2}%", results.hit_rate() * 100.0);
    println!("  Total Bet:     ¥{:.0}", results.total_bet);
    println!("  Total Return:  ¥{:.0}", results.total_return);
    println!("  Profit:        ¥{:.0}", results.profit());
    println!("  ROI:           {:.2}%", results.roi() * 100.0);
    println!("  Max Drawdown:  ¥{:.0}", results.max_drawdown());
    println!();

    if !results.periods.is_empty() {
        println!("Period Results:");
        println!(
            "  {:12} {:>10} {:>10} {:>10} {:>10}",
            "Period", "Bets", "Wins", "ROI", "Profit"
        );
        println!("  {}", "-".repeat(54));
        for pr in &results.periods {
            println!(
                "  {:12} {:>10} {:>10} {:>9.1}% {:>10.0}",
                pr.period_name,
                pr.num_bets,
                pr.num_wins,
                pr.roi() * 100.0,
                pr.profit()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_period_result_metrics() {
        let pr = PeriodResult {
            period_name: "Test".to_string(),
            start_date: NaiveDate::from_ymd_opt(2021, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2021, 3, 31).unwrap(),
            num_bets: 100,
            num_wins: 10,
            total_bet: 10000.0,
            total_return: 12000.0,
        };

        assert!((pr.profit() - 2000.0).abs() < 0.01);
        assert!((pr.roi() - 0.2).abs() < 0.01);
        assert!((pr.hit_rate() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_backtest_results_max_drawdown() {
        let mut results = BacktestResults::default();

        // Simulate some bets: +100, +100, -300, +200
        results.bets.push(BetResult {
            race_id: "R1".to_string(),
            race_date: NaiveDate::from_ymd_opt(2021, 1, 1).unwrap(),
            bet_type: BetType::Exacta,
            combination: vec!["1".to_string(), "2".to_string()],
            actual_positions: vec!["1".to_string(), "2".to_string()],
            probability: 0.1,
            odds: 1200.0,
            expected_value: 1.2,
            won: true,
            profit: 100.0,
        });
        results.bets.push(BetResult {
            race_id: "R2".to_string(),
            race_date: NaiveDate::from_ymd_opt(2021, 1, 2).unwrap(),
            bet_type: BetType::Exacta,
            combination: vec!["1".to_string(), "2".to_string()],
            actual_positions: vec!["1".to_string(), "2".to_string()],
            probability: 0.1,
            odds: 1200.0,
            expected_value: 1.2,
            won: true,
            profit: 100.0,
        });
        results.bets.push(BetResult {
            race_id: "R3".to_string(),
            race_date: NaiveDate::from_ymd_opt(2021, 1, 3).unwrap(),
            bet_type: BetType::Exacta,
            combination: vec!["1".to_string(), "2".to_string()],
            actual_positions: vec!["3".to_string(), "4".to_string()],
            probability: 0.1,
            odds: 1200.0,
            expected_value: 1.2,
            won: false,
            profit: -300.0,
        });
        results.bets.push(BetResult {
            race_id: "R4".to_string(),
            race_date: NaiveDate::from_ymd_opt(2021, 1, 4).unwrap(),
            bet_type: BetType::Exacta,
            combination: vec!["1".to_string(), "2".to_string()],
            actual_positions: vec!["1".to_string(), "2".to_string()],
            probability: 0.1,
            odds: 1300.0,
            expected_value: 1.3,
            won: true,
            profit: 200.0,
        });

        // Peak at 200, then drops to -100, then recovers to 100
        // Max drawdown = 200 - (-100) = 300
        let dd = results.max_drawdown();
        assert!((dd - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_bet_type_parsing() {
        assert_eq!(BetType::from_str("exacta"), Some(BetType::Exacta));
        assert_eq!(BetType::from_str("trifecta"), Some(BetType::Trifecta));
        assert_eq!(BetType::from_str("quinella"), Some(BetType::Quinella));
        assert_eq!(BetType::from_str("trio"), Some(BetType::Trio));
        assert_eq!(BetType::from_str("wide"), Some(BetType::Wide));
        assert_eq!(BetType::from_str("umatan"), Some(BetType::Exacta));
        assert_eq!(BetType::from_str("sanrentan"), Some(BetType::Trifecta));
        assert_eq!(BetType::from_str("invalid"), None);
    }

    #[test]
    fn test_bet_type_min_horses() {
        assert_eq!(BetType::Exacta.min_horses(), 2);
        assert_eq!(BetType::Quinella.min_horses(), 2);
        assert_eq!(BetType::Wide.min_horses(), 2);
        assert_eq!(BetType::Trifecta.min_horses(), 3);
        assert_eq!(BetType::Trio.min_horses(), 3);
    }
}
