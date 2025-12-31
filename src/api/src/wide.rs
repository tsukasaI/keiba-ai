//! Wide (ワイド) probability calculation.
//! Wide = 2 horses both finish in top 3, order doesn't matter.

use std::collections::{BTreeSet, HashMap};

/// Calculate a single trifecta probability using Harville formula.
fn calculate_single_trifecta(
    win_probs: &HashMap<String, f64>,
    first: &str,
    second: &str,
    third: &str,
) -> f64 {
    let p_first = win_probs.get(first).copied().unwrap_or(0.0);
    let p_second = win_probs.get(second).copied().unwrap_or(0.0);
    let p_third = win_probs.get(third).copied().unwrap_or(0.0);

    let p_second_given_first = p_second / (1.0 - p_first + 1e-10);
    let remaining_prob = 1.0 - p_first - p_second + 1e-10;
    let p_third_given_first_second = p_third / remaining_prob;

    p_first * p_second_given_first * p_third_given_first_second
}

/// Calculate wide probabilities from win probabilities.
///
/// Wide(A,B) = Probability both A and B finish in top 3
///           = Sum of Trio(A,B,C) for all horses C != A,B
///           = Sum of all 6 trifecta permutations involving A and B
///
/// # Arguments
/// * `win_probs` - Map of horse_id to win probability
/// * `min_probability` - Minimum probability threshold
///
/// # Returns
/// Map of {horse_a, horse_b} (BTreeSet) -> probability
pub fn calculate_wide_probs(
    win_probs: &HashMap<String, f64>,
    min_probability: f64,
) -> HashMap<BTreeSet<String>, f64> {
    let mut wide_probs = HashMap::new();
    let horses: Vec<_> = win_probs.keys().cloned().collect();

    if horses.len() < 3 {
        return wide_probs;
    }

    // For each pair of horses (A, B)
    for i in 0..horses.len() {
        for j in (i + 1)..horses.len() {
            let a = &horses[i];
            let b = &horses[j];

            let mut wide_prob = 0.0;

            // Sum over all possible third horses
            for (k, c) in horses.iter().enumerate() {
                if k == i || k == j {
                    continue;
                }

                // Sum all 6 permutations of (A, B, C)
                wide_prob += calculate_single_trifecta(win_probs, a, b, c);
                wide_prob += calculate_single_trifecta(win_probs, a, c, b);
                wide_prob += calculate_single_trifecta(win_probs, b, a, c);
                wide_prob += calculate_single_trifecta(win_probs, b, c, a);
                wide_prob += calculate_single_trifecta(win_probs, c, a, b);
                wide_prob += calculate_single_trifecta(win_probs, c, b, a);
            }

            if wide_prob >= min_probability {
                let mut key = BTreeSet::new();
                key.insert(a.clone());
                key.insert(b.clone());
                wide_probs.insert(key, wide_prob);
            }
        }
    }

    wide_probs
}

/// Convert wide probs to tuple keys for JSON serialization.
/// Returns sorted tuple (smaller_id, larger_id) for consistency.
#[allow(dead_code)]
pub fn wide_probs_to_tuples(
    wide_probs: &HashMap<BTreeSet<String>, f64>,
) -> HashMap<(String, String), f64> {
    wide_probs
        .iter()
        .map(|(set, prob)| {
            let horses: Vec<_> = set.iter().cloned().collect();
            ((horses[0].clone(), horses[1].clone()), *prob)
        })
        .collect()
}

/// Get top N wide combinations sorted by probability.
pub fn get_top_wides(
    wide_probs: &HashMap<BTreeSet<String>, f64>,
    n: usize,
) -> Vec<(BTreeSet<String>, f64)> {
    let mut sorted: Vec<_> = wide_probs.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    sorted
        .into_iter()
        .take(n)
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_wide_probs() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.3);
        win_probs.insert("C".to_string(), 0.2);

        let wide_probs = calculate_wide_probs(&win_probs, 0.001);

        // Should have C(3,2) = 3 combinations
        assert_eq!(wide_probs.len(), 3);

        // Wide(A,B) = Trio(A,B,C) since there's only one third horse
        // Trio(A,B,C) should sum to 1.0
        let mut key = BTreeSet::new();
        key.insert("A".to_string());
        key.insert("B".to_string());
        let prob = wide_probs.get(&key).unwrap();
        assert!((*prob - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_wide_probs_4_horses() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.4);
        win_probs.insert("B".to_string(), 0.3);
        win_probs.insert("C".to_string(), 0.2);
        win_probs.insert("D".to_string(), 0.1);

        let wide_probs = calculate_wide_probs(&win_probs, 0.001);

        // Should have C(4,2) = 6 combinations
        assert_eq!(wide_probs.len(), 6);

        // Wide probs should be higher than quinella (includes more positions)
        let mut key = BTreeSet::new();
        key.insert("A".to_string());
        key.insert("B".to_string());
        let prob = wide_probs.get(&key).unwrap();
        assert!(*prob > 0.5); // Should be higher than quinella
    }

    #[test]
    fn test_get_top_wides() {
        let mut wide_probs = HashMap::new();

        let mut key1 = BTreeSet::new();
        key1.insert("A".to_string());
        key1.insert("B".to_string());
        wide_probs.insert(key1.clone(), 0.8);

        let mut key2 = BTreeSet::new();
        key2.insert("A".to_string());
        key2.insert("C".to_string());
        wide_probs.insert(key2, 0.5);

        let top = get_top_wides(&wide_probs, 1);

        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, key1);
    }

    #[test]
    fn test_insufficient_horses() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.5);

        let wide_probs = calculate_wide_probs(&win_probs, 0.001);
        assert!(wide_probs.is_empty());
    }
}
