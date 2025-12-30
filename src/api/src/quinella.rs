//! Quinella (馬連) probability calculation.
//! Quinella = 1st and 2nd place, order doesn't matter.

use std::collections::{BTreeSet, HashMap};

/// Calculate quinella probabilities from win probabilities.
///
/// Quinella(A,B) = Exacta(A,B) + Exacta(B,A)
///              = P(A)*P(B|A) + P(B)*P(A|B)
///
/// # Arguments
/// * `win_probs` - Map of horse_id to win probability
/// * `min_probability` - Minimum probability threshold
///
/// # Returns
/// Map of {horse_a, horse_b} (BTreeSet) -> probability
pub fn calculate_quinella_probs(
    win_probs: &HashMap<String, f64>,
    min_probability: f64,
) -> HashMap<BTreeSet<String>, f64> {
    let mut quinella_probs = HashMap::new();
    let horses: Vec<_> = win_probs.keys().cloned().collect();

    for i in 0..horses.len() {
        for j in (i + 1)..horses.len() {
            let horse_a = &horses[i];
            let horse_b = &horses[j];

            let p_a = win_probs[horse_a];
            let p_b = win_probs[horse_b];

            // Exacta(A,B) = P(A) * P(B) / (1 - P(A))
            let exacta_ab = p_a * p_b / (1.0 - p_a + 1e-10);

            // Exacta(B,A) = P(B) * P(A) / (1 - P(B))
            let exacta_ba = p_b * p_a / (1.0 - p_b + 1e-10);

            let quinella_prob = exacta_ab + exacta_ba;

            if quinella_prob >= min_probability {
                let mut key = BTreeSet::new();
                key.insert(horse_a.clone());
                key.insert(horse_b.clone());
                quinella_probs.insert(key, quinella_prob);
            }
        }
    }

    quinella_probs
}

/// Convert quinella probs to tuple keys for JSON serialization.
/// Returns sorted tuple (smaller_id, larger_id) for consistency.
#[allow(dead_code)]
pub fn quinella_probs_to_tuples(
    quinella_probs: &HashMap<BTreeSet<String>, f64>,
) -> HashMap<(String, String), f64> {
    quinella_probs
        .iter()
        .map(|(set, prob)| {
            let horses: Vec<_> = set.iter().cloned().collect();
            ((horses[0].clone(), horses[1].clone()), *prob)
        })
        .collect()
}

/// Get top N quinella combinations sorted by probability.
pub fn get_top_quinellas(
    quinella_probs: &HashMap<BTreeSet<String>, f64>,
    n: usize,
) -> Vec<(BTreeSet<String>, f64)> {
    let mut sorted: Vec<_> = quinella_probs.iter().collect();
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
    fn test_calculate_quinella_probs() {
        let mut win_probs = HashMap::new();
        win_probs.insert("A".to_string(), 0.5);
        win_probs.insert("B".to_string(), 0.3);
        win_probs.insert("C".to_string(), 0.2);

        let quinella_probs = calculate_quinella_probs(&win_probs, 0.001);

        // Should have C(3,2) = 3 combinations
        assert_eq!(quinella_probs.len(), 3);

        // Verify A-B calculation
        // Exacta(A,B) = 0.5 * 0.3 / 0.5 = 0.3
        // Exacta(B,A) = 0.3 * 0.5 / 0.7 ≈ 0.214
        // Quinella(A,B) ≈ 0.514
        let mut key = BTreeSet::new();
        key.insert("A".to_string());
        key.insert("B".to_string());
        let a_b = quinella_probs.get(&key).unwrap();
        assert!(*a_b > 0.5);
    }

    #[test]
    fn test_get_top_quinellas() {
        let mut quinella_probs = HashMap::new();

        let mut key1 = BTreeSet::new();
        key1.insert("A".to_string());
        key1.insert("B".to_string());
        quinella_probs.insert(key1, 0.5);

        let mut key2 = BTreeSet::new();
        key2.insert("A".to_string());
        key2.insert("C".to_string());
        quinella_probs.insert(key2, 0.3);

        let top = get_top_quinellas(&quinella_probs, 1);

        assert_eq!(top.len(), 1);
        assert!(top[0].0.contains("A"));
        assert!(top[0].0.contains("B"));
    }

    #[test]
    fn test_quinella_probs_to_tuples() {
        let mut quinella_probs = HashMap::new();

        let mut key = BTreeSet::new();
        key.insert("B".to_string());
        key.insert("A".to_string());
        quinella_probs.insert(key, 0.5);

        let tuples = quinella_probs_to_tuples(&quinella_probs);

        // BTreeSet orders alphabetically, so key should be ("A", "B")
        assert!(tuples.contains_key(&("A".to_string(), "B".to_string())));
    }
}
