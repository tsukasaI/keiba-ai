//! Feature builder for ML model input.
//!
//! Generates 39 features from scraped data (43 with blood features).

use crate::scraper::parsers::{HorseProfile, JockeyProfile, RaceEntry, RaceInfo, TrainerProfile};
use crate::scraper::sire_stats::BloodFeatures;
use serde::{Deserialize, Serialize};

/// 39 features for the ML model (43 with blood features)
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
    // Running style (3)
    pub early_position: f32,
    pub late_position: f32,
    pub position_change: f32,
    // Aptitude (7)
    pub aptitude_sprint: f32,
    pub aptitude_mile: f32,
    pub aptitude_intermediate: f32,
    pub aptitude_long: f32,
    pub aptitude_turf: f32,
    pub aptitude_dirt: f32,
    pub aptitude_course: f32,
    // Pace (3)
    pub last_3f_avg: f32,
    pub last_3f_best: f32,
    pub last_3f_last: f32,
    // Race classification (3)
    pub weight_change_kg: f32,
    pub is_graded_race: f32,
    pub grade_level: f32,
    // Blood features (4) - requires model retraining to use
    pub sire_win_rate: f32,
    pub sire_place_rate: f32,
    pub broodmare_sire_win_rate: f32,
    pub broodmare_sire_place_rate: f32,
}

impl HorseFeatures {
    /// Convert to array for model input (39 features - current model)
    pub fn to_array(&self) -> [f32; 39] {
        [
            // Basic (5)
            self.horse_age_num,
            self.horse_sex_encoded,
            self.post_position_num,
            self.weight_carried,
            self.horse_weight,
            // Jockey/Trainer (5)
            self.jockey_win_rate,
            self.jockey_place_rate,
            self.trainer_win_rate,
            self.jockey_races,
            self.trainer_races,
            // Race conditions (4)
            self.distance_num,
            self.is_turf,
            self.is_dirt,
            self.track_condition_num,
            // Past performance (8)
            self.avg_position_last_3,
            self.avg_position_last_5,
            self.win_rate_last_3,
            self.win_rate_last_5,
            self.place_rate_last_3,
            self.place_rate_last_5,
            self.last_position,
            self.career_races,
            // Odds (1)
            self.odds_log,
            // Running style (3)
            self.early_position,
            self.late_position,
            self.position_change,
            // Aptitude (7)
            self.aptitude_sprint,
            self.aptitude_mile,
            self.aptitude_intermediate,
            self.aptitude_long,
            self.aptitude_turf,
            self.aptitude_dirt,
            self.aptitude_course,
            // Pace (3)
            self.last_3f_avg,
            self.last_3f_best,
            self.last_3f_last,
            // Race classification (3)
            self.weight_change_kg,
            self.is_graded_race,
            self.grade_level,
        ]
    }

    /// Convert to array with blood features (43 features - requires retrained model)
    pub fn to_array_with_blood(&self) -> [f32; 43] {
        [
            // Basic (5)
            self.horse_age_num,
            self.horse_sex_encoded,
            self.post_position_num,
            self.weight_carried,
            self.horse_weight,
            // Jockey/Trainer (5)
            self.jockey_win_rate,
            self.jockey_place_rate,
            self.trainer_win_rate,
            self.jockey_races,
            self.trainer_races,
            // Race conditions (4)
            self.distance_num,
            self.is_turf,
            self.is_dirt,
            self.track_condition_num,
            // Past performance (8)
            self.avg_position_last_3,
            self.avg_position_last_5,
            self.win_rate_last_3,
            self.win_rate_last_5,
            self.place_rate_last_3,
            self.place_rate_last_5,
            self.last_position,
            self.career_races,
            // Odds (1)
            self.odds_log,
            // Running style (3)
            self.early_position,
            self.late_position,
            self.position_change,
            // Aptitude (7)
            self.aptitude_sprint,
            self.aptitude_mile,
            self.aptitude_intermediate,
            self.aptitude_long,
            self.aptitude_turf,
            self.aptitude_dirt,
            self.aptitude_course,
            // Pace (3)
            self.last_3f_avg,
            self.last_3f_best,
            self.last_3f_last,
            // Race classification (3)
            self.weight_change_kg,
            self.is_graded_race,
            self.grade_level,
            // Blood features (4)
            self.sire_win_rate,
            self.sire_place_rate,
            self.broodmare_sire_win_rate,
            self.broodmare_sire_place_rate,
        ]
    }
}

/// Default values for missing data
struct Defaults;

impl Defaults {
    // Basic
    const HORSE_AGE: f32 = 4.0;
    const WEIGHT_CARRIED: f32 = 55.0;
    const HORSE_WEIGHT: f32 = 480.0;
    // Jockey/Trainer
    const JOCKEY_WIN_RATE: f32 = 0.07;
    const JOCKEY_PLACE_RATE: f32 = 0.21;
    const TRAINER_WIN_RATE: f32 = 0.07;
    const JOCKEY_RACES: f32 = 100.0;
    const TRAINER_RACES: f32 = 100.0;
    // Past performance
    const AVG_POSITION: f32 = 10.0;
    const WIN_RATE: f32 = 0.0;
    const PLACE_RATE: f32 = 0.0;
    const LAST_POSITION: f32 = 10.0;
    const CAREER_RACES: f32 = 0.0;
    // Odds
    const ODDS_LOG: f32 = 2.303; // log(10)
    // Running style
    const EARLY_POSITION: f32 = 9.0;
    const LATE_POSITION: f32 = 9.0;
    const POSITION_CHANGE: f32 = 0.0;
    // Aptitude (default 0.0 = no data)
    const APTITUDE: f32 = 0.0;
    // Pace (上り3ハロン)
    const LAST_3F_AVG: f32 = 35.0;  // ~35 seconds for last 600m
    const LAST_3F_BEST: f32 = 34.0;
    const LAST_3F_LAST: f32 = 35.0;
    // Race classification
    const WEIGHT_CHANGE: f32 = 0.0;
    // Blood features
    const SIRE_WIN_RATE: f32 = 0.07;
    const SIRE_PLACE_RATE: f32 = 0.21;
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

        // Running style features from horse history
        let (early, late, change) = horse
            .map(|h| h.running_style_features())
            .unwrap_or((Defaults::EARLY_POSITION, Defaults::LATE_POSITION, Defaults::POSITION_CHANGE));
        features.early_position = early;
        features.late_position = late;
        features.position_change = change;

        // Aptitude features from horse history
        if let Some(h) = horse {
            let aptitude = h.aptitude_features(&race.racecourse);
            features.aptitude_sprint = aptitude.sprint;
            features.aptitude_mile = aptitude.mile;
            features.aptitude_intermediate = aptitude.intermediate;
            features.aptitude_long = aptitude.long;
            features.aptitude_turf = aptitude.turf;
            features.aptitude_dirt = aptitude.dirt;
            features.aptitude_course = aptitude.course;
        } else {
            features.aptitude_sprint = Defaults::APTITUDE;
            features.aptitude_mile = Defaults::APTITUDE;
            features.aptitude_intermediate = Defaults::APTITUDE;
            features.aptitude_long = Defaults::APTITUDE;
            features.aptitude_turf = Defaults::APTITUDE;
            features.aptitude_dirt = Defaults::APTITUDE;
            features.aptitude_course = Defaults::APTITUDE;
        }

        // Pace features from horse history (上り3ハロン)
        let (avg, best, last) = horse
            .map(|h| h.pace_features())
            .unwrap_or((Defaults::LAST_3F_AVG, Defaults::LAST_3F_BEST, Defaults::LAST_3F_LAST));
        features.last_3f_avg = avg;
        features.last_3f_best = best;
        features.last_3f_last = last;

        // Race classification features
        features.weight_change_kg = entry
            .weight_change
            .map(|w| w as f32)
            .unwrap_or(Defaults::WEIGHT_CHANGE);
        features.is_graded_race = Self::encode_grade(&race.grade);
        features.grade_level = Self::encode_grade_level(&race.grade);

        // Blood features from sire statistics
        if let Some(h) = horse {
            let blood = BloodFeatures::compute(&h.sire, &h.broodmare_sire);
            features.sire_win_rate = blood.sire_win_rate;
            features.sire_place_rate = blood.sire_place_rate;
            features.broodmare_sire_win_rate = blood.broodmare_sire_win_rate;
            features.broodmare_sire_place_rate = blood.broodmare_sire_place_rate;
        } else {
            features.sire_win_rate = Defaults::SIRE_WIN_RATE;
            features.sire_place_rate = Defaults::SIRE_PLACE_RATE;
            features.broodmare_sire_win_rate = Defaults::SIRE_WIN_RATE;
            features.broodmare_sire_place_rate = Defaults::SIRE_PLACE_RATE;
        }

        features
    }

    /// Encode race grade to binary (0 = non-graded, 1 = graded)
    fn encode_grade(grade: &str) -> f32 {
        match grade {
            "G1" | "G2" | "G3" => 1.0,
            _ => 0.0,
        }
    }

    /// Encode grade level (0 = non-graded, 1 = G3, 2 = G2, 3 = G1)
    fn encode_grade_level(grade: &str) -> f32 {
        match grade {
            "G1" => 3.0,
            "G2" => 2.0,
            "G3" => 1.0,
            _ => 0.0,
        }
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
        assert_eq!(arr.len(), 39);
        assert_eq!(arr[0], 4.0);
        assert_eq!(arr[1], 0.0);
        assert_eq!(arr[2], 5.0);
    }

    #[test]
    fn test_encode_grade() {
        assert_eq!(FeatureBuilder::encode_grade("G1"), 1.0);
        assert_eq!(FeatureBuilder::encode_grade("G2"), 1.0);
        assert_eq!(FeatureBuilder::encode_grade("G3"), 1.0);
        assert_eq!(FeatureBuilder::encode_grade("OP"), 0.0);
        assert_eq!(FeatureBuilder::encode_grade(""), 0.0);
    }

    #[test]
    fn test_encode_grade_level() {
        assert_eq!(FeatureBuilder::encode_grade_level("G1"), 3.0);
        assert_eq!(FeatureBuilder::encode_grade_level("G2"), 2.0);
        assert_eq!(FeatureBuilder::encode_grade_level("G3"), 1.0);
        assert_eq!(FeatureBuilder::encode_grade_level("OP"), 0.0);
    }

    #[test]
    fn test_to_array_with_blood() {
        let features = HorseFeatures {
            horse_age_num: 4.0,
            horse_sex_encoded: 0.0,
            post_position_num: 5.0,
            sire_win_rate: 0.12,
            sire_place_rate: 0.35,
            broodmare_sire_win_rate: 0.10,
            broodmare_sire_place_rate: 0.30,
            ..Default::default()
        };

        let arr = features.to_array_with_blood();
        assert_eq!(arr.len(), 43);
        // Check blood features at the end
        assert_eq!(arr[39], 0.12);  // sire_win_rate
        assert_eq!(arr[40], 0.35);  // sire_place_rate
        assert_eq!(arr[41], 0.10);  // broodmare_sire_win_rate
        assert_eq!(arr[42], 0.30);  // broodmare_sire_place_rate
    }

    #[test]
    fn test_blood_features_default() {
        let features = HorseFeatures::default();

        // Default blood features should be 0
        assert_eq!(features.sire_win_rate, 0.0);
        assert_eq!(features.sire_place_rate, 0.0);

        // to_array should still return 39 features
        assert_eq!(features.to_array().len(), 39);

        // to_array_with_blood should return 43 features
        assert_eq!(features.to_array_with_blood().len(), 43);
    }
}
