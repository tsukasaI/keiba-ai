//! Sire statistics loader for blood features.
//!
//! Loads pre-computed sire performance statistics from JSON file.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;
use tracing::{info, warn};

/// Global sire stats cache
static SIRE_STATS: OnceLock<SireStatsData> = OnceLock::new();

/// Sire statistics data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SireStatsData {
    pub version: String,
    pub description: String,
    pub sire_count: u32,
    pub bms_count: u32,
    pub default_win_rate: f64,
    pub default_place_rate: f64,
    pub sires: HashMap<String, SireStats>,
}

/// Individual sire statistics
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SireStats {
    pub win_rate: f64,
    pub place_rate: f64,
    pub avg_earnings: f64,
    pub offspring_count: u32,
    pub total_races: u32,
}

impl Default for SireStatsData {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            description: "Default sire stats".to_string(),
            sire_count: 0,
            bms_count: 0,
            default_win_rate: 0.07,
            default_place_rate: 0.21,
            sires: HashMap::new(),
        }
    }
}

/// Load sire stats from JSON file (to be integrated with model)
#[allow(dead_code)]
pub fn load_sire_stats<P: AsRef<Path>>(path: P) -> Result<&'static SireStatsData> {
    let path = path.as_ref();

    // Try to get existing or initialize
    let stats = SIRE_STATS.get_or_init(|| {
        if !path.exists() {
            warn!("Sire stats file not found: {}, using defaults", path.display());
            return SireStatsData::default();
        }

        match std::fs::read_to_string(path) {
            Ok(content) => match serde_json::from_str::<SireStatsData>(&content) {
                Ok(data) => {
                    info!("Loaded sire stats: {} sires, {} BMS", data.sire_count, data.bms_count);
                    data
                }
                Err(e) => {
                    warn!("Failed to parse sire stats: {}, using defaults", e);
                    SireStatsData::default()
                }
            },
            Err(e) => {
                warn!("Failed to read sire stats file: {}, using defaults", e);
                SireStatsData::default()
            }
        }
    });

    Ok(stats)
}

/// Get sire stats (returns cached data or defaults)
pub fn get_sire_stats() -> &'static SireStatsData {
    SIRE_STATS.get_or_init(SireStatsData::default)
}

/// Look up sire statistics by name
pub fn lookup_sire(sire_name: &str) -> Option<&'static SireStats> {
    get_sire_stats().sires.get(sire_name)
}

/// Look up broodmare sire statistics by name
pub fn lookup_broodmare_sire(bms_name: &str) -> Option<&'static SireStats> {
    let key = format!("BMS:{}", bms_name);
    get_sire_stats().sires.get(&key)
}

/// Blood features computed from sire statistics
#[derive(Debug, Clone, Default)]
pub struct BloodFeatures {
    pub sire_win_rate: f32,
    pub sire_place_rate: f32,
    pub broodmare_sire_win_rate: f32,
    pub broodmare_sire_place_rate: f32,
}

impl BloodFeatures {
    /// Compute blood features from sire and broodmare sire names
    pub fn compute(sire: &str, broodmare_sire: &str) -> Self {
        let stats = get_sire_stats();
        let default_win = stats.default_win_rate as f32;
        let default_place = stats.default_place_rate as f32;

        let (sire_win, sire_place) = if let Some(s) = lookup_sire(sire) {
            (s.win_rate as f32, s.place_rate as f32)
        } else {
            (default_win, default_place)
        };

        let (bms_win, bms_place) = if let Some(s) = lookup_broodmare_sire(broodmare_sire) {
            (s.win_rate as f32, s.place_rate as f32)
        } else {
            (default_win, default_place)
        };

        Self {
            sire_win_rate: sire_win,
            sire_place_rate: sire_place,
            broodmare_sire_win_rate: bms_win,
            broodmare_sire_place_rate: bms_place,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_stats() {
        let stats = SireStatsData::default();
        assert_eq!(stats.default_win_rate, 0.07);
        assert_eq!(stats.default_place_rate, 0.21);
    }

    #[test]
    fn test_blood_features_unknown_sire() {
        let features = BloodFeatures::compute("UnknownSire", "UnknownBMS");
        // Should use default values
        assert!(features.sire_win_rate > 0.0);
        assert!(features.sire_place_rate > 0.0);
    }

    #[test]
    fn test_sire_stats_deserialize() {
        let json = r#"{
            "version": "1.0",
            "description": "Test",
            "sire_count": 1,
            "bms_count": 0,
            "default_win_rate": 0.07,
            "default_place_rate": 0.21,
            "sires": {
                "TestSire": {
                    "win_rate": 0.12,
                    "place_rate": 0.35,
                    "avg_earnings": 50000000,
                    "offspring_count": 100,
                    "total_races": 1000
                }
            }
        }"#;

        let data: SireStatsData = serde_json::from_str(json).unwrap();
        assert_eq!(data.sire_count, 1);
        assert!(data.sires.contains_key("TestSire"));
        assert_eq!(data.sires["TestSire"].win_rate, 0.12);
    }
}
