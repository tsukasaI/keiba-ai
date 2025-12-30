//! Trio (三連複) probability calculation.
//! Trio = 3 horses in top 3, order doesn't matter.

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

/// Calculate trio probabilities from win probabilities.
///
/// Trio(A,B,C) = Sum of all 6 trifecta permutations:
///   P(A-B-C) + P(A-C-B) + P(B-A-C) + P(B-C-A) + P(C-A-B) + P(C-B-A)
///
/// # Arguments
/// * `win_probs` - Map of horse_id to win probability
/// * `min_probability` - Minimum probability threshold
///
/// # Returns
/// Map of {horse_a, horse_b, horse_c} (BTreeSet) -> probability
pub fn calculate_trio_probs(
    win_probs: &HashMap<String, f64>,
    min_probability: f64,
) -> HashMap<BTreeSet<String>, f64> {
    let mut trio_probs = HashMap::new();
    let horses: Vec<_> = win_probs.keys().cloned().collect();

    if horses.len() < 3 {
        return trio_probs;
    }

    // Generate all combinations of 3 horses
    for i in 0..horses.len() {
        for j in (i + 1)..horses.len() {
            for k in (j + 1)..horses.len() {
                let a = &horses[i];
                let b = &horses[j];
                let c = &horses[k];

                // Sum all 6 permutations
                let trio_prob = calculate_single_trifecta(win_probs, a, b, c)
                    + calculate_single_trifecta(win_probs, a, c, b)
                    + calculate_single_trifecta(win_probs, b, a, c)
                    + calculate_single_trifecta(win_probs, b, c, a)
                    + calculate_single_trifecta(win_probs, c, a, b)
                    + calculate_single_trifecta(win_probs, c, b, a);

                if trio_prob >= min_probability {
                    let mut key = BTreeSet::new();
                    key.insert(a.clone());
                    key.insert(b.clone());
                    key.insert(c.clone());
                    trio_probs.insert(key, trio_prob);
                }
            }
        }
    }

    trio_probs
}

/// Convert trio probs to tuple keys for JSON serialization.
/// Returns sorted tuple (id1, id2, id3) for consistency.
pub fn trio_probs_to_tuples(
    trio_probs: &HashMap<BTreeSet<String>, f64>,
) -> HashMap<(String, String, String), f64> {
    trio_probs
        .iter()
        .map(|(set, prob)| {
            let horses: Vec<_> = set.iter().cloned().collect();
            ((horses[0].clone(), horses[1].clone(), horses[2].clone()), *prob)
        })
        .collect()
}

/// Get top N trio combinations sorted by probability.
pub fn get_top_trios(
    trio_probs: &HashMap<BTreeSet<String>, f64>,
    n: usize,
) -> Vec<(BTreeSet<String>, f64)> {
    let mut sorted: Vec<_> = trio_probs.iter().collect();
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
    fn test_calculate_trio_probs() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.3);
        win_probs.insert("C".to_string(), 0.2);

        let trio_probs = calculate_trio_probs(&win_probs, 0.001);

        // Should have C(3,3) = 1 combination
        assert_eq!(trio_probs.len(), 1);

        // The trio probability should sum to 1.0 (all 6 permutations)
        let mut key = BTreeSet::new();
        key.insert("A".to_string());
        key.insert("B".to_string());
        key.insert("C".to_string());
        let prob = trio_probs.get(&key).unwrap();
        assert!((*prob - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_trio_probs_4_horses() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.4);
        win_probs.insert("B".to_string(), 0.3);
        win_probs.insert("C".to_string(), 0.2);
        win_probs.insert("D".to_string(), 0.1);

        let trio_probs = calculate_trio_probs(&win_probs, 0.001);

        // Should have C(4,3) = 4 combinations
        assert_eq!(trio_probs.len(), 4);
    }

    #[test]
    fn test_get_top_trios() {
        let mut trio_probs = HashMap::new();

        let mut key1 = BTreeSet::new();
        key1.insert("A".to_string());
        key1.insert("B".to_string());
        key1.insert("C".to_string());
        trio_probs.insert(key1.clone(), 0.8);

        let mut key2 = BTreeSet::new();
        key2.insert("A".to_string());
        key2.insert("B".to_string());
        key2.insert("D".to_string());
        trio_probs.insert(key2, 0.5);

        let top = get_top_trios(&trio_probs, 1);

        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, key1);
    }

    #[test]
    fn test_insufficient_horses() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.5);

        let trio_probs = calculate_trio_probs(&win_probs, 0.001);
        assert!(trio_probs.is_empty());
    }
}
