//! Horse profile parser for netkeiba.com.

use anyhow::Result;
use chrono::{Datelike, Utc};
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// Horse profile data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HorseProfile {
    pub horse_id: String,
    pub name: String,
    pub birth_date: Option<String>,
    pub horse_age: u8,
    pub sex: String,
    pub coat_color: String,
    // Bloodline
    pub sire: String,
    pub dam: String,
    pub broodmare_sire: String,
    // Career stats
    pub career_races: u32,
    pub wins: u32,
    pub seconds: u32,
    pub thirds: u32,
    pub earnings: u64,
    // Computed stats
    pub avg_position_last_3: f64,
    pub avg_position_last_5: f64,
    pub win_rate_last_3: f64,
    pub win_rate_last_5: f64,
    pub place_rate_last_3: f64,
    pub place_rate_last_5: f64,
    pub last_position: Option<u8>,
    // Past races
    pub past_races: Vec<PastRace>,
}

/// Past race record
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PastRace {
    pub date: String,
    pub racecourse: String,
    pub race_name: String,
    pub distance: u32,
    pub surface: String,
    pub track_condition: String,
    pub position: u8,
    pub field_size: u8,
    pub weight_carried: f64,
    pub jockey: String,
    pub odds: Option<f64>,
    pub popularity: Option<u8>,
    // Corner passing positions (1st-4th corner)
    pub corner_1: Option<u8>,
    pub corner_2: Option<u8>,
    pub corner_3: Option<u8>,
    pub corner_4: Option<u8>,
    // Last 600m time (上り) in seconds
    pub last_3f: Option<f32>,
}

/// Aptitude features computed from race history
#[derive(Debug, Clone, Default)]
pub struct AptitudeFeatures {
    pub sprint: f32,        // ≤1400m win rate
    pub mile: f32,          // 1400-1800m win rate
    pub intermediate: f32,  // 1800-2200m win rate
    pub long: f32,          // >2200m win rate
    pub turf: f32,          // turf place rate
    pub dirt: f32,          // dirt place rate
    pub course: f32,        // specific course place rate
}

impl HorseProfile {
    /// Calculate running style features from past races
    /// Returns (early_position, late_position, position_change)
    pub fn running_style_features(&self) -> (f32, f32, f32) {
        let mut early_sum = 0.0f32;
        let mut late_sum = 0.0f32;
        let mut count = 0u32;

        for race in &self.past_races {
            // Need at least 2 corner positions
            let corners: Vec<u8> = [race.corner_1, race.corner_2, race.corner_3, race.corner_4]
                .iter()
                .filter_map(|&c| c)
                .collect();

            if corners.len() >= 2 {
                // Early = first half, Late = second half
                let mid = corners.len() / 2;
                let early: f32 = corners[..mid].iter().map(|&c| c as f32).sum::<f32>() / mid as f32;
                let late: f32 = corners[mid..].iter().map(|&c| c as f32).sum::<f32>() / (corners.len() - mid) as f32;

                early_sum += early;
                late_sum += late;
                count += 1;
            }
        }

        if count > 0 {
            let early_avg = early_sum / count as f32;
            let late_avg = late_sum / count as f32;
            let change = late_avg - early_avg;
            (early_avg, late_avg, change)
        } else {
            (9.0, 9.0, 0.0) // defaults
        }
    }

    /// Calculate aptitude features from past races
    pub fn aptitude_features(&self, current_racecourse: &str) -> AptitudeFeatures {
        let mut features = AptitudeFeatures::default();

        // Count wins and races by distance category
        let mut sprint_wins = 0u32;
        let mut sprint_races = 0u32;
        let mut mile_wins = 0u32;
        let mut mile_races = 0u32;
        let mut intermediate_wins = 0u32;
        let mut intermediate_races = 0u32;
        let mut long_wins = 0u32;
        let mut long_races = 0u32;

        // Count places by surface
        let mut turf_places = 0u32;
        let mut turf_races = 0u32;
        let mut dirt_places = 0u32;
        let mut dirt_races = 0u32;

        // Count places at current course
        let mut course_places = 0u32;
        let mut course_races = 0u32;

        for race in &self.past_races {
            let is_win = race.position == 1;
            let is_place = race.position <= 3;

            // Distance categories
            match race.distance {
                0..=1400 => {
                    sprint_races += 1;
                    if is_win { sprint_wins += 1; }
                }
                1401..=1800 => {
                    mile_races += 1;
                    if is_win { mile_wins += 1; }
                }
                1801..=2200 => {
                    intermediate_races += 1;
                    if is_win { intermediate_wins += 1; }
                }
                _ => {
                    long_races += 1;
                    if is_win { long_wins += 1; }
                }
            }

            // Surface
            if race.surface == "turf" {
                turf_races += 1;
                if is_place { turf_places += 1; }
            } else {
                dirt_races += 1;
                if is_place { dirt_places += 1; }
            }

            // Course (use contains for flexible matching)
            if race.racecourse.contains(current_racecourse) || current_racecourse.contains(&race.racecourse) {
                course_races += 1;
                if is_place { course_places += 1; }
            }
        }

        // Calculate rates
        if sprint_races > 0 { features.sprint = sprint_wins as f32 / sprint_races as f32; }
        if mile_races > 0 { features.mile = mile_wins as f32 / mile_races as f32; }
        if intermediate_races > 0 { features.intermediate = intermediate_wins as f32 / intermediate_races as f32; }
        if long_races > 0 { features.long = long_wins as f32 / long_races as f32; }
        if turf_races > 0 { features.turf = turf_places as f32 / turf_races as f32; }
        if dirt_races > 0 { features.dirt = dirt_places as f32 / dirt_races as f32; }
        if course_races > 0 { features.course = course_places as f32 / course_races as f32; }

        features
    }

    /// Calculate pace features from past races
    /// Returns (last_3f_avg, last_3f_best, last_3f_last)
    pub fn pace_features(&self) -> (f32, f32, f32) {
        let times: Vec<f32> = self.past_races
            .iter()
            .filter_map(|r| r.last_3f)
            .collect();

        if times.is_empty() {
            return (35.0, 35.0, 35.0); // defaults
        }

        // Last = most recent (first in list)
        let last = times[0];

        // Best = minimum (fastest)
        let best = times.iter().cloned().fold(f32::MAX, f32::min);

        // Avg = mean of last 5 races
        let avg_count = times.len().min(5);
        let avg: f32 = times[..avg_count].iter().sum::<f32>() / avg_count as f32;

        (avg, best, last)
    }
}

/// Parser for horse profile pages
pub struct HorseParser;

impl HorseParser {
    /// Parse horse profile from HTML
    pub fn parse(html: &str, horse_id: &str) -> Result<HorseProfile> {
        let document = Html::parse_document(html);
        let mut profile = HorseProfile {
            horse_id: horse_id.to_string(),
            ..Default::default()
        };

        // Parse basic info
        Self::parse_basic_info(&document, &mut profile);

        // Parse bloodline
        Self::parse_bloodline(&document, &mut profile);

        // Parse career stats
        Self::parse_career_stats(&document, &mut profile);

        // Parse past races
        profile.past_races = Self::parse_past_races(&document);

        // Compute stats from past races
        Self::compute_stats(&mut profile);

        Ok(profile)
    }

    fn parse_basic_info(document: &Html, profile: &mut HorseProfile) {
        // Name
        let name_selectors = [".horse_title h1", ".db_head_name h1", "h1"];
        for sel_str in name_selectors {
            if let Ok(selector) = Selector::parse(sel_str) {
                if let Some(elem) = document.select(&selector).next() {
                    let text = elem.text().collect::<String>();
                    let cleaned = text.replace("競走成績", "").trim().to_string();
                    if !cleaned.is_empty() {
                        profile.name = cleaned;
                        break;
                    }
                }
            }
        }

        // Compile regex patterns once before the loop
        let date_re = Regex::new(r"(\d{4})年(\d{1,2})月(\d{1,2})日").unwrap();
        let sex_re = Regex::new(r"([牡牝セ])").unwrap();
        let color_re = Regex::new(r"[牡牝セ]\s*(.+)").unwrap();

        // Profile table
        if let Ok(selector) = Selector::parse(".db_prof_table td, .profile_table td") {
            let cells: Vec<_> = document.select(&selector).collect();
            for cell in cells.iter() {
                let text = cell.text().collect::<String>();

                // Birth date
                if let Some(caps) = date_re.captures(&text) {
                    let year: i32 = caps[1].parse().unwrap_or(2000);
                    let month: u32 = caps[2].parse().unwrap_or(1);
                    let day: u32 = caps[3].parse().unwrap_or(1);
                    profile.birth_date = Some(format!("{}-{:02}-{:02}", year, month, day));

                    // Calculate age
                    let now = Utc::now();
                    profile.horse_age = (now.year() - year) as u8;
                }

                // Sex
                if let Some(caps) = sex_re.captures(&text) {
                    profile.sex = caps[1].to_string();
                }

                // Coat color (毛色)
                if let Some(caps) = color_re.captures(&text) {
                    profile.coat_color = caps[1].trim().to_string();
                }
            }
        }
    }

    fn parse_bloodline(document: &Html, profile: &mut HorseProfile) {
        let table_selectors = [".blood_table", ".pedigree_table", "table.blood"];

        for sel_str in table_selectors {
            if let Ok(selector) = Selector::parse(sel_str) {
                if let Some(table) = document.select(&selector).next() {
                    let a_selector = Selector::parse("a").unwrap();
                    let links: Vec<_> = table.select(&a_selector).collect();

                    // Typically: sire, dam, broodmare_sire in specific positions
                    if !links.is_empty() {
                        profile.sire = links[0].text().collect::<String>().trim().to_string();
                    }
                    if links.len() >= 3 {
                        profile.dam = links[2].text().collect::<String>().trim().to_string();
                    }
                    if links.len() >= 4 {
                        profile.broodmare_sire = links[3].text().collect::<String>().trim().to_string();
                    }
                    break;
                }
            }
        }
    }

    fn parse_career_stats(document: &Html, profile: &mut HorseProfile) {
        // Career record: "3-2-1-4" format
        if let Ok(selector) = Selector::parse(".db_prof_area_02, .career_record") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>();
                let re = Regex::new(r"(\d+)-(\d+)-(\d+)-(\d+)").unwrap();
                if let Some(caps) = re.captures(&text) {
                    profile.wins = caps[1].parse().unwrap_or(0);
                    profile.seconds = caps[2].parse().unwrap_or(0);
                    profile.thirds = caps[3].parse().unwrap_or(0);
                    let fourths_plus: u32 = caps[4].parse().unwrap_or(0);
                    profile.career_races =
                        profile.wins + profile.seconds + profile.thirds + fourths_plus;
                }
            }
        }

        // Earnings
        if let Ok(selector) = Selector::parse(".prize, .earnings") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>();
                let re = Regex::new(r"([\d,]+)").unwrap();
                if let Some(caps) = re.captures(&text) {
                    let cleaned = caps[1].replace(',', "");
                    profile.earnings = cleaned.parse().unwrap_or(0);
                }
            }
        }
    }

    fn parse_past_races(document: &Html) -> Vec<PastRace> {
        let mut races = Vec::new();

        let table_selectors = [".db_h_race_results", "table.nk_tb_common", "table.race_table"];

        for sel_str in table_selectors {
            if let Ok(selector) = Selector::parse(sel_str) {
                if let Some(table) = document.select(&selector).next() {
                    let tr_selector = Selector::parse("tbody tr, tr").unwrap();

                    for (i, row) in table.select(&tr_selector).enumerate() {
                        // Skip header
                        if i == 0 {
                            continue;
                        }

                        if let Some(race) = Self::parse_past_race_row(&row) {
                            races.push(race);
                        }
                    }

                    if !races.is_empty() {
                        break;
                    }
                }
            }
        }

        races
    }

    fn parse_past_race_row(row: &scraper::ElementRef) -> Option<PastRace> {
        let td_selector = Selector::parse("td").unwrap();
        let cells: Vec<_> = row.select(&td_selector).collect();

        if cells.len() < 10 {
            return None;
        }

        // Compile regex patterns once
        let date_re = Regex::new(r"(\d{4})/(\d{2})/(\d{2})").unwrap();
        let surface_re = Regex::new(r"(芝|ダ)\s*(\d+)").unwrap();
        let position_re = Regex::new(r"(\d+)").unwrap();
        let field_size_re = Regex::new(r"/(\d+)").unwrap();
        let weight_re = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();
        // Corner passing pattern: "5-5-4-3" or "5-5-4" (2-4 corners)
        let corner_re = Regex::new(r"(\d+)-(\d+)(?:-(\d+))?(?:-(\d+))?").unwrap();
        // Last 3f time pattern: "33.5" or "34.0"
        let last_3f_re = Regex::new(r"(\d{2}\.\d)").unwrap();

        let mut race = PastRace::default();

        // Date (cell 0)
        let date_text = cells[0].text().collect::<String>();
        if let Some(caps) = date_re.captures(&date_text) {
            race.date = format!("{}-{}-{}", &caps[1], &caps[2], &caps[3]);
        }

        // Racecourse (cell 1)
        race.racecourse = cells[1].text().collect::<String>().trim().to_string();

        // Race name (cell 4)
        let a_selector = Selector::parse("a").unwrap();
        if let Some(link) = cells.get(4).and_then(|c| c.select(&a_selector).next()) {
            race.race_name = link.text().collect::<String>().trim().to_string();
        }

        // Distance and surface (cell 6 or 14)
        for &idx in &[6, 14] {
            if let Some(cell) = cells.get(idx) {
                let text = cell.text().collect::<String>();
                if let Some(caps) = surface_re.captures(&text) {
                    race.surface = if &caps[1] == "芝" {
                        "turf".to_string()
                    } else {
                        "dirt".to_string()
                    };
                    race.distance = caps[2].parse().unwrap_or(0);
                    break;
                }
            }
        }

        // Track condition (cell 7)
        if let Some(cell) = cells.get(7) {
            let text = cell.text().collect::<String>();
            for condition in ["良", "稍重", "重", "不良"] {
                if text.contains(condition) {
                    race.track_condition = condition.to_string();
                    break;
                }
            }
        }

        // Position (cell 11 or 5)
        for &idx in &[11, 5] {
            if let Some(cell) = cells.get(idx) {
                let text = cell.text().collect::<String>();
                if let Some(caps) = position_re.captures(&text) {
                    race.position = caps[1].parse().unwrap_or(0);
                    if race.position > 0 {
                        break;
                    }
                }
            }
        }

        // Field size from position cell (e.g., "1/18")
        if let Some(cell) = cells.get(6) {
            let text = cell.text().collect::<String>();
            if let Some(caps) = field_size_re.captures(&text) {
                race.field_size = caps[1].parse().unwrap_or(0);
            }
        }

        // Weight carried (cell 13)
        if let Some(cell) = cells.get(13) {
            let text = cell.text().collect::<String>();
            if let Some(caps) = weight_re.captures(&text) {
                race.weight_carried = caps[1].parse().unwrap_or(0.0);
            }
        }

        // Jockey (cell 12)
        if let Some(cell) = cells.get(12) {
            if let Some(link) = cell.select(&a_selector).next() {
                race.jockey = link.text().collect::<String>().trim().to_string();
            }
        }

        // Odds (cell 9)
        if let Some(cell) = cells.get(9) {
            let text = cell.text().collect::<String>();
            race.odds = text.trim().parse().ok();
        }

        // Popularity (cell 10)
        if let Some(cell) = cells.get(10) {
            let text = cell.text().collect::<String>();
            race.popularity = text.trim().parse().ok();
        }

        // Corner passing positions (通過) - typically cell 20 or search for pattern
        // Format: "5-5-4-3" (1st-2nd-3rd-4th corner positions)
        for idx in [20, 21, 22, 18, 19] {
            if let Some(cell) = cells.get(idx) {
                let text = cell.text().collect::<String>();
                if let Some(caps) = corner_re.captures(&text) {
                    race.corner_1 = caps.get(1).and_then(|m| m.as_str().parse().ok());
                    race.corner_2 = caps.get(2).and_then(|m| m.as_str().parse().ok());
                    race.corner_3 = caps.get(3).and_then(|m| m.as_str().parse().ok());
                    race.corner_4 = caps.get(4).and_then(|m| m.as_str().parse().ok());
                    break;
                }
            }
        }

        // Last 3f time (上り) - typically cell 22 or 23
        // Format: "33.5" seconds
        for idx in [22, 23, 24, 21] {
            if let Some(cell) = cells.get(idx) {
                let text = cell.text().collect::<String>();
                if let Some(caps) = last_3f_re.captures(&text) {
                    race.last_3f = caps.get(1).and_then(|m| m.as_str().parse().ok());
                    break;
                }
            }
        }

        // Validate
        if race.position == 0 {
            return None;
        }

        Some(race)
    }

    fn compute_stats(profile: &mut HorseProfile) {
        let races = &profile.past_races;

        // Last position
        if let Some(first) = races.first() {
            profile.last_position = Some(first.position);
        }

        // Last 3 races
        let last_3: Vec<_> = races.iter().take(3).collect();
        if !last_3.is_empty() {
            let sum: u32 = last_3.iter().map(|r| r.position as u32).sum();
            profile.avg_position_last_3 = sum as f64 / last_3.len() as f64;
            profile.win_rate_last_3 =
                last_3.iter().filter(|r| r.position == 1).count() as f64 / last_3.len() as f64;
            profile.place_rate_last_3 =
                last_3.iter().filter(|r| r.position <= 3).count() as f64 / last_3.len() as f64;
        } else {
            profile.avg_position_last_3 = 10.0;
        }

        // Last 5 races
        let last_5: Vec<_> = races.iter().take(5).collect();
        if !last_5.is_empty() {
            let sum: u32 = last_5.iter().map(|r| r.position as u32).sum();
            profile.avg_position_last_5 = sum as f64 / last_5.len() as f64;
            profile.win_rate_last_5 =
                last_5.iter().filter(|r| r.position == 1).count() as f64 / last_5.len() as f64;
            profile.place_rate_last_5 =
                last_5.iter().filter(|r| r.position <= 3).count() as f64 / last_5.len() as f64;
        } else {
            profile.avg_position_last_5 = 10.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_race(
        distance: u32,
        surface: &str,
        racecourse: &str,
        position: u8,
        corners: (Option<u8>, Option<u8>, Option<u8>, Option<u8>),
        last_3f: Option<f32>,
    ) -> PastRace {
        PastRace {
            date: "2024-01-01".to_string(),
            racecourse: racecourse.to_string(),
            race_name: "Test Race".to_string(),
            distance,
            surface: surface.to_string(),
            track_condition: "良".to_string(),
            position,
            field_size: 18,
            weight_carried: 55.0,
            jockey: "Test Jockey".to_string(),
            odds: Some(5.0),
            popularity: Some(3),
            corner_1: corners.0,
            corner_2: corners.1,
            corner_3: corners.2,
            corner_4: corners.3,
            last_3f,
        }
    }

    #[test]
    fn test_running_style_features() {
        let profile = HorseProfile {
            past_races: vec![
                create_test_race(1600, "turf", "東京", 1, (Some(5), Some(5), Some(4), Some(3)), Some(33.5)),
                create_test_race(1600, "turf", "東京", 2, (Some(6), Some(6), Some(5), Some(4)), Some(34.0)),
            ],
            ..Default::default()
        };

        let (early, late, change) = profile.running_style_features();

        // Early = avg of corner 1-2: (5+6)/2 = 5.5
        // Late = avg of corner 3-4: (4+5)/2 = 4.5
        // But we average across all races too
        assert!((early - 5.5).abs() < 0.1);
        assert!((late - 4.0).abs() < 0.1);
        assert!(change < 0.0); // Negative means horse moved up
    }

    #[test]
    fn test_running_style_features_empty() {
        let profile = HorseProfile::default();
        let (early, late, change) = profile.running_style_features();

        assert_eq!(early, 9.0);
        assert_eq!(late, 9.0);
        assert_eq!(change, 0.0);
    }

    #[test]
    fn test_aptitude_features_distance() {
        let profile = HorseProfile {
            past_races: vec![
                create_test_race(1200, "turf", "東京", 1, (None, None, None, None), None), // Sprint win
                create_test_race(1400, "turf", "東京", 5, (None, None, None, None), None), // Sprint loss
                create_test_race(1600, "turf", "中山", 1, (None, None, None, None), None), // Mile win
                create_test_race(2000, "turf", "東京", 2, (None, None, None, None), None), // Intermediate place
            ],
            ..Default::default()
        };

        let aptitude = profile.aptitude_features("東京");

        assert!((aptitude.sprint - 0.5).abs() < 0.01); // 1 win in 2 sprint races
        assert!((aptitude.mile - 1.0).abs() < 0.01);   // 1 win in 1 mile race
        assert!((aptitude.intermediate - 0.0).abs() < 0.01); // 0 wins in 1 intermediate race
    }

    #[test]
    fn test_aptitude_features_surface() {
        let profile = HorseProfile {
            past_races: vec![
                create_test_race(1600, "turf", "東京", 1, (None, None, None, None), None),
                create_test_race(1600, "turf", "東京", 3, (None, None, None, None), None),
                create_test_race(1400, "dirt", "東京", 5, (None, None, None, None), None),
            ],
            ..Default::default()
        };

        let aptitude = profile.aptitude_features("東京");

        assert!((aptitude.turf - 1.0).abs() < 0.01);   // 2 places in 2 turf races
        assert!((aptitude.dirt - 0.0).abs() < 0.01);   // 0 places in 1 dirt race
    }

    #[test]
    fn test_aptitude_features_course() {
        let profile = HorseProfile {
            past_races: vec![
                create_test_race(1600, "turf", "東京", 1, (None, None, None, None), None),
                create_test_race(1600, "turf", "東京", 2, (None, None, None, None), None),
                create_test_race(1600, "turf", "中山", 10, (None, None, None, None), None),
            ],
            ..Default::default()
        };

        let aptitude = profile.aptitude_features("東京");
        assert!((aptitude.course - 1.0).abs() < 0.01); // 2 places in 2 Tokyo races

        let aptitude2 = profile.aptitude_features("中山");
        assert!((aptitude2.course - 0.0).abs() < 0.01); // 0 places in 1 Nakayama race
    }

    #[test]
    fn test_pace_features() {
        let profile = HorseProfile {
            past_races: vec![
                create_test_race(1600, "turf", "東京", 1, (None, None, None, None), Some(33.0)), // Most recent
                create_test_race(1600, "turf", "東京", 2, (None, None, None, None), Some(34.0)),
                create_test_race(1600, "turf", "東京", 3, (None, None, None, None), Some(35.0)),
                create_test_race(1600, "turf", "東京", 4, (None, None, None, None), Some(34.5)),
                create_test_race(1600, "turf", "東京", 5, (None, None, None, None), Some(36.0)),
            ],
            ..Default::default()
        };

        let (avg, best, last) = profile.pace_features();

        assert_eq!(last, 33.0); // Most recent
        assert_eq!(best, 33.0); // Fastest
        // Avg of 5 races: (33 + 34 + 35 + 34.5 + 36) / 5 = 34.5
        assert!((avg - 34.5).abs() < 0.01);
    }

    #[test]
    fn test_pace_features_empty() {
        let profile = HorseProfile::default();
        let (avg, best, last) = profile.pace_features();

        assert_eq!(avg, 35.0);
        assert_eq!(best, 35.0);
        assert_eq!(last, 35.0);
    }

    #[test]
    fn test_corner_parsing_pattern() {
        // Test the regex pattern for corner positions
        let re = regex::Regex::new(r"(\d+)-(\d+)(?:-(\d+))?(?:-(\d+))?").unwrap();

        // 4 corners
        let caps = re.captures("5-5-4-3").unwrap();
        assert_eq!(caps.get(1).unwrap().as_str(), "5");
        assert_eq!(caps.get(2).unwrap().as_str(), "5");
        assert_eq!(caps.get(3).unwrap().as_str(), "4");
        assert_eq!(caps.get(4).unwrap().as_str(), "3");

        // 2 corners (short races)
        let caps2 = re.captures("3-2").unwrap();
        assert_eq!(caps2.get(1).unwrap().as_str(), "3");
        assert_eq!(caps2.get(2).unwrap().as_str(), "2");
        assert!(caps2.get(3).is_none());
        assert!(caps2.get(4).is_none());
    }

    #[test]
    fn test_last_3f_parsing_pattern() {
        let re = regex::Regex::new(r"(\d{2}\.\d)").unwrap();

        let caps = re.captures("33.5").unwrap();
        assert_eq!(caps.get(1).unwrap().as_str(), "33.5");

        let caps2 = re.captures("34.0").unwrap();
        assert_eq!(caps2.get(1).unwrap().as_str(), "34.0");
    }
}
