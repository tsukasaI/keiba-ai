//! Exacta probability calculation using Harville formula.

use std::collections::HashMap;

/// Calculate exacta probabilities from win probabilities using Harville formula.
///
/// Harville formula: P(A=1st, B=2nd) = P(A=1st) * P(B=1st) / (1 - P(A=1st))
///
/// # Arguments
/// * `win_probs` - Map of horse_id to win probability
/// * `min_probability` - Minimum probability threshold
///
/// # Returns
/// Map of (first, second) -> probability
pub fn calculate_exacta_probs(
    win_probs: &HashMap<String, f64>,
    min_probability: f64,
) -> HashMap<(String, String), f64> {
    let mut exacta_probs = HashMap::new();
    let horses: Vec<_> = win_probs.keys().collect();

    for first in &horses {
        let p_first = win_probs[*first];

        // Skip if probability too low
        if p_first < min_probability {
            continue;
        }

        for second in &horses {
            if first == second {
                continue;
            }

            let p_second = win_probs[*second];

            // Harville formula
            // P(B=2nd | A=1st) = P(B=1st) / (1 - P(A=1st))
            let p_second_given_first = p_second / (1.0 - p_first + 1e-10);
            let exacta_prob = p_first * p_second_given_first;

            if exacta_prob >= min_probability {
                exacta_probs.insert(((*first).clone(), (*second).clone()), exacta_prob);
            }
        }
    }

    exacta_probs
}

/// Get top N exacta combinations sorted by probability.
pub fn get_top_exactas(
    exacta_probs: &HashMap<(String, String), f64>,
    n: usize,
) -> Vec<((String, String), f64)> {
    let mut sorted: Vec<_> = exacta_probs.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    sorted
        .into_iter()
        .take(n)
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

/// Extract win probabilities from position probability matrix.
///
/// # Arguments
/// * `position_probs` - Matrix of shape (n_horses, n_positions)
/// * `horse_ids` - List of horse IDs
///
/// # Returns
/// Map of horse_id to win probability (position 0)
pub fn extract_win_probs(
    position_probs: &[Vec<f64>],
    horse_ids: &[String],
) -> HashMap<String, f64> {
    horse_ids
        .iter()
        .zip(position_probs.iter())
        .map(|(id, probs)| (id.clone(), probs.first().copied().unwrap_or(0.0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_exacta_probs() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.3);
        win_probs.insert("C".to_string(), 0.2);

        let exacta_probs = calculate_exacta_probs(&win_probs, 0.001);

        // A-B: 0.5 * 0.3 / (1 - 0.5) = 0.5 * 0.6 = 0.3
        let a_b = exacta_probs.get(&("A".to_string(), "B".to_string())).unwrap();
        assert!((a_b - 0.3).abs() < 0.01);

        // Should have 6 combinations
        assert_eq!(exacta_probs.len(), 6);
    }

    #[test]
    fn test_get_top_exactas() {
        let mut exacta_probs = HashMap::new();
        exacta_probs.insert(("A".to_string(), "B".to_string()), 0.3);
        exacta_probs.insert(("A".to_string(), "C".to_string()), 0.2);
        exacta_probs.insert(("B".to_string(), "A".to_string()), 0.15);

        let top = get_top_exactas(&exacta_probs, 2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, ("A".to_string(), "B".to_string()));
        assert_eq!(top[1].0, ("A".to_string(), "C".to_string()));
    }
}
