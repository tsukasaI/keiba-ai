//! Feature builder for ML model input.
//!
//! Generates 23 features from scraped data.

use crate::scraper::parsers::{HorseProfile, JockeyProfile, RaceEntry, RaceInfo, TrainerProfile};
use serde::{Deserialize, Serialize};

/// 23 features for the ML model
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HorseFeatures {
    // Basic (5)
    pub horse_age_num: f32,
    pub horse_sex_encoded: f32, // 牡:0, 牝:1, セ:2
    pub post_position_num: f32,
    pub weight_carried: f32,
    pub horse_weight: f32,
    // Jockey/Trainer (5)
    pub jockey_win_rate: f32,
    pub jockey_place_rate: f32,
    pub trainer_win_rate: f32,
    pub jockey_races: f32,
    pub trainer_races: f32,
    // Race conditions (4)
    pub distance_num: f32,
    pub is_turf: f32,
    pub is_dirt: f32,
    pub track_condition_num: f32, // 良:0, 稍重:1, 重:2, 不良:3
    // Past performance (8)
    pub avg_position_last_3: f32,
    pub avg_position_last_5: f32,
    pub win_rate_last_3: f32,
    pub win_rate_last_5: f32,
    pub place_rate_last_3: f32,
    pub place_rate_last_5: f32,
    pub last_position: f32,
    pub career_races: f32,
    // Odds (1)
    pub odds_log: f32,
}

impl HorseFeatures {
    /// Convert to array for model input
    pub fn to_array(&self) -> [f32; 23] {
        [
            self.horse_age_num,
            self.horse_sex_encoded,
            self.post_position_num,
            self.weight_carried,
            self.horse_weight,
            self.jockey_win_rate,
            self.jockey_place_rate,
            self.trainer_win_rate,
            self.jockey_races,
            self.trainer_races,
            self.distance_num,
            self.is_turf,
            self.is_dirt,
            self.track_condition_num,
            self.avg_position_last_3,
            self.avg_position_last_5,
            self.win_rate_last_3,
            self.win_rate_last_5,
            self.place_rate_last_3,
            self.place_rate_last_5,
            self.last_position,
            self.career_races,
            self.odds_log,
        ]
    }
}

/// Default values for missing data
struct Defaults;

impl Defaults {
    const HORSE_AGE: f32 = 4.0;
    const WEIGHT_CARRIED: f32 = 55.0;
    const HORSE_WEIGHT: f32 = 480.0;
    const JOCKEY_WIN_RATE: f32 = 0.07;
    const JOCKEY_PLACE_RATE: f32 = 0.21;
    const TRAINER_WIN_RATE: f32 = 0.07;
    const JOCKEY_RACES: f32 = 100.0;
    const TRAINER_RACES: f32 = 100.0;
    const AVG_POSITION: f32 = 10.0;
    const WIN_RATE: f32 = 0.0;
    const PLACE_RATE: f32 = 0.0;
    const LAST_POSITION: f32 = 10.0;
    const CAREER_RACES: f32 = 0.0;
    const ODDS_LOG: f32 = 2.303; // log(10)
}

/// Feature builder
pub struct FeatureBuilder;

impl FeatureBuilder {
    /// Build features for a horse
    #[allow(clippy::field_reassign_with_default)]
    pub fn build(
        race: &RaceInfo,
        entry: &RaceEntry,
        horse: Option<&HorseProfile>,
        jockey: Option<&JockeyProfile>,
        trainer: Option<&TrainerProfile>,
    ) -> HorseFeatures {
        let mut features = HorseFeatures::default();

        // Basic features from entry
        features.horse_age_num = if entry.horse_age > 0 {
            entry.horse_age as f32
        } else {
            horse
                .map(|h| h.horse_age as f32)
                .unwrap_or(Defaults::HORSE_AGE)
        };

        features.horse_sex_encoded = Self::encode_sex(&entry.horse_sex);

        features.post_position_num = entry.post_position as f32;

        features.weight_carried = if entry.weight_carried > 0.0 {
            entry.weight_carried as f32
        } else {
            Defaults::WEIGHT_CARRIED
        };

        features.horse_weight = entry
            .horse_weight
            .map(|w| w as f32)
            .unwrap_or(Defaults::HORSE_WEIGHT);

        // Jockey features
        if let Some(j) = jockey {
            features.jockey_win_rate = j.win_rate as f32;
            features.jockey_place_rate = j.place_rate as f32;
            features.jockey_races = j.total_races as f32;
        } else {
            features.jockey_win_rate = Defaults::JOCKEY_WIN_RATE;
            features.jockey_place_rate = Defaults::JOCKEY_PLACE_RATE;
            features.jockey_races = Defaults::JOCKEY_RACES;
        }

        // Trainer features
        if let Some(t) = trainer {
            features.trainer_win_rate = t.win_rate as f32;
            features.trainer_races = t.total_races as f32;
        } else {
            features.trainer_win_rate = Defaults::TRAINER_WIN_RATE;
            features.trainer_races = Defaults::TRAINER_RACES;
        }

        // Race condition features
        features.distance_num = race.distance as f32;
        features.is_turf = if race.surface == "turf" { 1.0 } else { 0.0 };
        features.is_dirt = if race.surface == "dirt" { 1.0 } else { 0.0 };
        features.track_condition_num = Self::encode_track_condition(&race.track_condition);

        // Past performance features from horse profile
        if let Some(h) = horse {
            features.avg_position_last_3 = h.avg_position_last_3 as f32;
            features.avg_position_last_5 = h.avg_position_last_5 as f32;
            features.win_rate_last_3 = h.win_rate_last_3 as f32;
            features.win_rate_last_5 = h.win_rate_last_5 as f32;
            features.place_rate_last_3 = h.place_rate_last_3 as f32;
            features.place_rate_last_5 = h.place_rate_last_5 as f32;
            features.last_position = h.last_position.map(|p| p as f32).unwrap_or(Defaults::LAST_POSITION);
            features.career_races = h.career_races as f32;
        } else {
            features.avg_position_last_3 = Defaults::AVG_POSITION;
            features.avg_position_last_5 = Defaults::AVG_POSITION;
            features.win_rate_last_3 = Defaults::WIN_RATE;
            features.win_rate_last_5 = Defaults::WIN_RATE;
            features.place_rate_last_3 = Defaults::PLACE_RATE;
            features.place_rate_last_5 = Defaults::PLACE_RATE;
            features.last_position = Defaults::LAST_POSITION;
            features.career_races = Defaults::CAREER_RACES;
        }

        // Odds feature
        features.odds_log = entry
            .win_odds
            .map(|o| (o as f32).ln())
            .unwrap_or(Defaults::ODDS_LOG);

        features
    }

    /// Encode sex to numeric value
    fn encode_sex(sex: &str) -> f32 {
        match sex {
            "牡" => 0.0,
            "牝" => 1.0,
            "セ" => 2.0,
            _ => 0.0,
        }
    }

    /// Encode track condition to numeric value
    fn encode_track_condition(condition: &str) -> f32 {
        match condition {
            "良" => 0.0,
            "稍重" => 1.0,
            "重" => 2.0,
            "不良" => 3.0,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_sex() {
        assert_eq!(FeatureBuilder::encode_sex("牡"), 0.0);
        assert_eq!(FeatureBuilder::encode_sex("牝"), 1.0);
        assert_eq!(FeatureBuilder::encode_sex("セ"), 2.0);
    }

    #[test]
    fn test_encode_track_condition() {
        assert_eq!(FeatureBuilder::encode_track_condition("良"), 0.0);
        assert_eq!(FeatureBuilder::encode_track_condition("稍重"), 1.0);
        assert_eq!(FeatureBuilder::encode_track_condition("重"), 2.0);
        assert_eq!(FeatureBuilder::encode_track_condition("不良"), 3.0);
    }

    #[test]
    fn test_to_array() {
        let features = HorseFeatures {
            horse_age_num: 4.0,
            horse_sex_encoded: 0.0,
            post_position_num: 5.0,
            ..Default::default()
        };

        let arr = features.to_array();
        assert_eq!(arr.len(), 23);
        assert_eq!(arr[0], 4.0);
        assert_eq!(arr[1], 0.0);
        assert_eq!(arr[2], 5.0);
    }
}
