//! Trifecta (三連単) probability calculation using extended Harville formula.

use std::collections::HashMap;

/// Calculate trifecta probabilities from win probabilities using extended Harville formula.
///
/// Extended Harville: P(A=1st, B=2nd, C=3rd) = P(A) * P(B|A) * P(C|A,B)
///   where P(B|A) = P(B) / (1 - P(A))
///   and   P(C|A,B) = P(C) / (1 - P(A) - P(B))
///
/// # Arguments
/// * `win_probs` - Map of horse_id to win probability
/// * `min_probability` - Minimum probability threshold
///
/// # Returns
/// Map of (first, second, third) -> probability
pub fn calculate_trifecta_probs(
    win_probs: &HashMap<String, f64>,
    min_probability: f64,
) -> HashMap<(String, String, String), f64> {
    let mut trifecta_probs = HashMap::new();
    let horses: Vec<_> = win_probs.keys().collect();

    if horses.len() < 3 {
        return trifecta_probs;
    }

    for first in &horses {
        let p_first = win_probs[*first];

        if p_first < min_probability {
            continue;
        }

        for second in &horses {
            if first == second {
                continue;
            }

            let p_second = win_probs[*second];

            // P(B=2nd | A=1st) = P(B) / (1 - P(A))
            let p_second_given_first = p_second / (1.0 - p_first + 1e-10);
            let p_first_second = p_first * p_second_given_first;

            // Early pruning
            if p_first_second < min_probability {
                continue;
            }

            for third in &horses {
                if third == first || third == second {
                    continue;
                }

                let p_third = win_probs[*third];

                // P(C=3rd | A=1st, B=2nd) = P(C) / (1 - P(A) - P(B))
                let remaining_prob = 1.0 - p_first - p_second + 1e-10;
                let p_third_given_first_second = p_third / remaining_prob;

                let trifecta_prob = p_first_second * p_third_given_first_second;

                if trifecta_prob >= min_probability {
                    trifecta_probs.insert(
                        ((*first).clone(), (*second).clone(), (*third).clone()),
                        trifecta_prob,
                    );
                }
            }
        }
    }

    trifecta_probs
}

/// Get top N trifecta combinations sorted by probability.
pub fn get_top_trifectas(
    trifecta_probs: &HashMap<(String, String, String), f64>,
    n: usize,
) -> Vec<((String, String, String), f64)> {
    let mut sorted: Vec<_> = trifecta_probs.iter().collect();
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
    fn test_calculate_trifecta_probs() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.3);
        win_probs.insert("C".to_string(), 0.2);

        let trifecta_probs = calculate_trifecta_probs(&win_probs, 0.001);

        // Should have 6 combinations (3! = 6)
        assert_eq!(trifecta_probs.len(), 6);

        // Verify A-B-C calculation
        // P(A-B-C) = 0.5 * (0.3 / 0.5) * (0.2 / 0.2) = 0.5 * 0.6 * 1.0 = 0.3
        let a_b_c = trifecta_probs
            .get(&("A".to_string(), "B".to_string(), "C".to_string()))
            .unwrap();
        assert!((a_b_c - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_get_top_trifectas() {
        let mut trifecta_probs = HashMap::new();
        trifecta_probs.insert(
            ("A".to_string(), "B".to_string(), "C".to_string()),
            0.3,
        );
        trifecta_probs.insert(
            ("A".to_string(), "C".to_string(), "B".to_string()),
            0.2,
        );
        trifecta_probs.insert(
            ("B".to_string(), "A".to_string(), "C".to_string()),
            0.15,
        );

        let top = get_top_trifectas(&trifecta_probs, 2);

        assert_eq!(top.len(), 2);
        assert_eq!(
            top[0].0,
            ("A".to_string(), "B".to_string(), "C".to_string())
        );
    }

    #[test]
    fn test_insufficient_horses() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.5);

        let trifecta_probs = calculate_trifecta_probs(&win_probs, 0.001);
        assert!(trifecta_probs.is_empty());
    }
}
