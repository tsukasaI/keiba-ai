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
