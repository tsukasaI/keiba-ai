//! Betting logic: EV calculation, Kelly criterion, bet recommendations.

use crate::config::BettingConfig;
use crate::types::BettingSignal;
use std::collections::{BTreeSet, HashMap};

/// Calculate expected value.
///
/// # Arguments
/// * `probability` - Predicted probability of winning
/// * `odds` - Japanese format odds (e.g., 1520 = 15.2x return per 100 yen)
///
/// # Returns
/// Expected value (> 1.0 indicates positive edge)
pub fn calculate_ev(probability: f64, odds: f64) -> f64 {
    probability * (odds / 100.0)
}

/// Calculate Kelly criterion fraction.
///
/// Kelly fraction = (p * b - q) / b
/// where:
///   p = probability of winning
///   b = net odds (payout - 1)
///   q = 1 - p (probability of losing)
///
/// # Arguments
/// * `probability` - Predicted probability
/// * `odds` - Japanese format odds
///
/// # Returns
/// Optimal fraction of bankroll to bet (0 if negative EV)
pub fn calculate_kelly_fraction(probability: f64, odds: f64) -> f64 {
    if probability <= 0.0 || odds <= 100.0 {
        return 0.0;
    }

    let decimal_odds = odds / 100.0;
    let b = decimal_odds - 1.0; // Net odds
    let q = 1.0 - probability;

    let kelly = (probability * b - q) / b;
    kelly.max(0.0)
}

/// Calculate recommended bet size using fractional Kelly.
///
/// # Arguments
/// * `probability` - Predicted probability
/// * `odds` - Japanese format odds
/// * `bankroll` - Current bankroll
/// * `kelly_fraction` - Fraction of full Kelly to use (e.g., 0.25 for quarter Kelly)
/// * `bet_unit` - Minimum bet unit
///
/// # Returns
/// Recommended bet amount (rounded to bet unit)
pub fn calculate_bet_size(
    probability: f64,
    odds: f64,
    bankroll: f64,
    kelly_fraction: f64,
    bet_unit: u32,
) -> u32 {
    let full_kelly = calculate_kelly_fraction(probability, odds);
    let fraction = full_kelly * kelly_fraction;
    let bet_size = bankroll * fraction;

    // Round to bet unit
    let rounded = ((bet_size / bet_unit as f64).round() as u32) * bet_unit;
    rounded.max(bet_unit)
}

/// Find value bets where EV > threshold.
///
/// # Arguments
/// * `exacta_probs` - Map of (first, second) -> probability
/// * `exacta_odds` - Map of "first-second" -> odds
/// * `config` - Betting configuration
///
/// # Returns
/// List of betting signals sorted by EV
pub fn find_value_bets(
    exacta_probs: &HashMap<(String, String), f64>,
    exacta_odds: &HashMap<String, f64>,
    config: &BettingConfig,
) -> Vec<BettingSignal> {
    let mut signals = Vec::new();

    for ((first, second), &prob) in exacta_probs {
        let odds_key = format!("{}-{}", first, second);

        if let Some(&odds) = exacta_odds.get(&odds_key) {
            let ev = calculate_ev(prob, odds);

            if ev > config.ev_threshold {
                let kelly = calculate_kelly_fraction(prob, odds);

                signals.push(BettingSignal {
                    combination: (first.clone(), second.clone()),
                    bet_type: "exacta".to_string(),
                    probability: prob,
                    odds,
                    expected_value: ev,
                    kelly_fraction: kelly * config.kelly_fraction,
                    recommended_bet: config.bet_unit,
                });
            }
        }
    }

    // Sort by EV descending
    signals.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());

    signals
}

/// Find value bets for trifecta (3-horse ordered).
pub fn find_value_bets_trifecta(
    trifecta_probs: &HashMap<(String, String, String), f64>,
    trifecta_odds: &HashMap<String, f64>,
    config: &BettingConfig,
) -> Vec<BettingSignal> {
    let mut signals = Vec::new();

    for ((first, second, third), &prob) in trifecta_probs {
        let odds_key = format!("{}-{}-{}", first, second, third);

        if let Some(&odds) = trifecta_odds.get(&odds_key) {
            let ev = calculate_ev(prob, odds);

            if ev > config.ev_threshold {
                let kelly = calculate_kelly_fraction(prob, odds);

                signals.push(BettingSignal {
                    combination: (first.clone(), second.clone()),
                    bet_type: "trifecta".to_string(),
                    probability: prob,
                    odds,
                    expected_value: ev,
                    kelly_fraction: kelly * config.kelly_fraction,
                    recommended_bet: config.bet_unit,
                });
            }
        }
    }

    signals.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());
    signals
}

/// Find value bets for quinella (2-horse unordered).
pub fn find_value_bets_quinella(
    quinella_probs: &HashMap<BTreeSet<String>, f64>,
    quinella_odds: &HashMap<String, f64>,
    config: &BettingConfig,
) -> Vec<BettingSignal> {
    let mut signals = Vec::new();

    for (set, &prob) in quinella_probs {
        let horses: Vec<_> = set.iter().cloned().collect();
        if horses.len() != 2 {
            continue;
        }

        let odds_key = format!("{}-{}", horses[0], horses[1]);

        if let Some(&odds) = quinella_odds.get(&odds_key) {
            let ev = calculate_ev(prob, odds);

            if ev > config.ev_threshold {
                let kelly = calculate_kelly_fraction(prob, odds);

                signals.push(BettingSignal {
                    combination: (horses[0].clone(), horses[1].clone()),
                    bet_type: "quinella".to_string(),
                    probability: prob,
                    odds,
                    expected_value: ev,
                    kelly_fraction: kelly * config.kelly_fraction,
                    recommended_bet: config.bet_unit,
                });
            }
        }
    }

    signals.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());
    signals
}

/// Find value bets for trio (3-horse unordered).
pub fn find_value_bets_trio(
    trio_probs: &HashMap<BTreeSet<String>, f64>,
    trio_odds: &HashMap<String, f64>,
    config: &BettingConfig,
) -> Vec<BettingSignal> {
    let mut signals = Vec::new();

    for (set, &prob) in trio_probs {
        let horses: Vec<_> = set.iter().cloned().collect();
        if horses.len() != 3 {
            continue;
        }

        let odds_key = format!("{}-{}-{}", horses[0], horses[1], horses[2]);

        if let Some(&odds) = trio_odds.get(&odds_key) {
            let ev = calculate_ev(prob, odds);

            if ev > config.ev_threshold {
                let kelly = calculate_kelly_fraction(prob, odds);

                signals.push(BettingSignal {
                    combination: (horses[0].clone(), horses[1].clone()),
                    bet_type: "trio".to_string(),
                    probability: prob,
                    odds,
                    expected_value: ev,
                    kelly_fraction: kelly * config.kelly_fraction,
                    recommended_bet: config.bet_unit,
                });
            }
        }
    }

    signals.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());
    signals
}

/// Find value bets for wide (2-horse both in top 3).
pub fn find_value_bets_wide(
    wide_probs: &HashMap<BTreeSet<String>, f64>,
    wide_odds: &HashMap<String, f64>,
    config: &BettingConfig,
) -> Vec<BettingSignal> {
    let mut signals = Vec::new();

    for (set, &prob) in wide_probs {
        let horses: Vec<_> = set.iter().cloned().collect();
        if horses.len() != 2 {
            continue;
        }

        let odds_key = format!("{}-{}", horses[0], horses[1]);

        if let Some(&odds) = wide_odds.get(&odds_key) {
            let ev = calculate_ev(prob, odds);

            if ev > config.ev_threshold {
                let kelly = calculate_kelly_fraction(prob, odds);

                signals.push(BettingSignal {
                    combination: (horses[0].clone(), horses[1].clone()),
                    bet_type: "wide".to_string(),
                    probability: prob,
                    odds,
                    expected_value: ev,
                    kelly_fraction: kelly * config.kelly_fraction,
                    recommended_bet: config.bet_unit,
                });
            }
        }
    }

    signals.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());
    signals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_ev() {
        // 10% probability at 1500 odds (15x) = 1.5 EV
        let ev = calculate_ev(0.10, 1500.0);
        assert!((ev - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_calculate_ev_breakeven() {
        // 10% at 1000 odds (10x) = 1.0 EV (breakeven)
        let ev = calculate_ev(0.10, 1000.0);
        assert!((ev - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kelly_fraction() {
        // 50% at 300 odds (3x)
        // Kelly = (0.5 * 2 - 0.5) / 2 = 0.25
        let kelly = calculate_kelly_fraction(0.5, 300.0);
        assert!((kelly - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_kelly_negative_ev() {
        // 10% at 500 odds (5x) = 0.5 EV (negative)
        let kelly = calculate_kelly_fraction(0.10, 500.0);
        assert_eq!(kelly, 0.0);
    }

    #[test]
    fn test_find_value_bets() {
        let mut exacta_probs = HashMap::new();
        exacta_probs.insert(("A".to_string(), "B".to_string()), 0.10);
        exacta_probs.insert(("A".to_string(), "C".to_string()), 0.05);

        let mut exacta_odds = HashMap::new();
        exacta_odds.insert("A-B".to_string(), 1500.0); // EV = 1.5
        exacta_odds.insert("A-C".to_string(), 1500.0); // EV = 0.75

        let config = BettingConfig::default();
        let signals = find_value_bets(&exacta_probs, &exacta_odds, &config);

        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].combination.0, "A");
        assert_eq!(signals[0].combination.1, "B");
    }
}
