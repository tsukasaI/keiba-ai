//! Race card (shutuba) parser for netkeiba.com.

use anyhow::Result;
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// Race information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RaceInfo {
    pub race_id: String,
    pub race_name: String,
    pub race_number: u8,
    pub racecourse: String,
    pub date: String,
    pub distance: u32,
    pub surface: String,         // "turf" or "dirt"
    pub track_condition: String, // "良", "稍重", "重", "不良"
    pub grade: String,           // "G1", "G2", "G3", "OP", "" etc.
}

/// Entry in race card
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RaceEntry {
    pub post_position: u8,
    pub horse_id: String,
    pub horse_name: String,
    pub horse_age: u8,
    pub horse_sex: String, // "牡", "牝", "セ"
    pub weight_carried: f64,
    pub horse_weight: Option<u32>,
    pub weight_change: Option<i32>,
    pub jockey_id: String,
    pub jockey_name: String,
    pub trainer_id: String,
    pub trainer_name: String,
    pub win_odds: Option<f64>,
    pub popularity: Option<u8>,
}

/// Parser for race card pages
pub struct RaceCardParser;

impl RaceCardParser {
    /// Parse race card from HTML
    pub fn parse(html: &str, race_id: &str) -> Result<(RaceInfo, Vec<RaceEntry>)> {
        let document = Html::parse_document(html);

        let race_info = Self::parse_race_info(&document, race_id)?;
        let entries = Self::parse_entries(&document)?;

        Ok((race_info, entries))
    }

    fn parse_race_info(document: &Html, race_id: &str) -> Result<RaceInfo> {
        let mut info = RaceInfo {
            race_id: race_id.to_string(),
            ..Default::default()
        };

        // Race name
        if let Ok(selector) = Selector::parse(".RaceName") {
            if let Some(elem) = document.select(&selector).next() {
                info.race_name = elem.text().collect::<String>().trim().to_string();
            }
        }

        // Race number
        if let Ok(selector) = Selector::parse(".RaceNum") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>();
                let re = Regex::new(r"(\d+)R").unwrap();
                if let Some(caps) = re.captures(&text) {
                    info.race_number = caps[1].parse().unwrap_or(0);
                }
            }
        }

        // Parse RaceData01 for date, distance, surface
        if let Ok(selector) = Selector::parse(".RaceData01") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>();

                // Date: YYYY/MM/DD
                let date_re = Regex::new(r"(\d{4})/(\d{1,2})/(\d{1,2})").unwrap();
                if let Some(caps) = date_re.captures(&text) {
                    info.date = format!("{}-{:02}-{:02}", &caps[1],
                        caps[2].parse::<u32>().unwrap_or(1),
                        caps[3].parse::<u32>().unwrap_or(1));
                }

                // Distance and surface: 芝2500m or ダ1200m
                let dist_re = Regex::new(r"(芝|ダ)(\d+)m").unwrap();
                if let Some(caps) = dist_re.captures(&text) {
                    info.surface = if &caps[1] == "芝" {
                        "turf".to_string()
                    } else {
                        "dirt".to_string()
                    };
                    info.distance = caps[2].parse().unwrap_or(0);
                }
            }
        }

        // Racecourse from RaceData02
        if let Ok(selector) = Selector::parse(".RaceData02 span") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>();
                let re = Regex::new(r"回(.+?)(?:\d|$)").unwrap();
                if let Some(caps) = re.captures(&text) {
                    info.racecourse = caps[1].trim().to_string();
                }
            }
        }

        // Track condition from Item03
        if let Ok(selector) = Selector::parse(".Item03") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>();
                for condition in ["良", "稍重", "重", "不良"] {
                    if text.contains(condition) {
                        info.track_condition = condition.to_string();
                        break;
                    }
                }
            }
        }

        // Default track condition
        if info.track_condition.is_empty() {
            info.track_condition = "良".to_string();
        }

        // Race grade from race name or RaceData01/02
        info.grade = Self::extract_grade(&info.race_name);

        // Also check RaceData01 for grade icons/text
        if info.grade.is_empty() {
            if let Ok(selector) = Selector::parse(".RaceData01") {
                if let Some(elem) = document.select(&selector).next() {
                    let text = elem.text().collect::<String>();
                    info.grade = Self::extract_grade(&text);
                }
            }
        }

        Ok(info)
    }

    /// Extract race grade from text
    fn extract_grade(text: &str) -> String {
        // Check for GI, GII, GIII patterns (used in some displays)
        if text.contains("GI") && !text.contains("GII") && !text.contains("GIII") {
            return "G1".to_string();
        }
        if text.contains("GII") && !text.contains("GIII") {
            return "G2".to_string();
        }
        if text.contains("GIII") {
            return "G3".to_string();
        }
        // Check for G1, G2, G3 patterns
        if text.contains("(G1)") || text.contains("（G1）") || text.contains("G1") {
            return "G1".to_string();
        }
        if text.contains("(G2)") || text.contains("（G2）") || text.contains("G2") {
            return "G2".to_string();
        }
        if text.contains("(G3)") || text.contains("（G3）") || text.contains("G3") {
            return "G3".to_string();
        }
        // Check for Japanese grade notation
        if text.contains("（Ｇ１）") || text.contains("(Ｇ１)") {
            return "G1".to_string();
        }
        if text.contains("（Ｇ２）") || text.contains("(Ｇ２)") {
            return "G2".to_string();
        }
        if text.contains("（Ｇ３）") || text.contains("(Ｇ３)") {
            return "G3".to_string();
        }
        // Check for オープン (Open class)
        if text.contains("オープン") || text.contains("OP") {
            return "OP".to_string();
        }
        String::new()
    }

    fn parse_entries(document: &Html) -> Result<Vec<RaceEntry>> {
        let mut entries = Vec::new();

        // Try multiple table selectors
        let table_selectors = [".Shutuba_Table", ".HorseList", "table.ShutubaTable"];
        let mut table = None;

        for sel_str in table_selectors {
            if let Ok(selector) = Selector::parse(sel_str) {
                if let Some(t) = document.select(&selector).next() {
                    table = Some(t);
                    break;
                }
            }
        }

        let Some(table_elem) = table else {
            return Ok(entries);
        };

        // Row selectors
        let row_selector = Selector::parse("tr.HorseList, tr[class*='Horse'], tbody tr").unwrap();

        for row in table_elem.select(&row_selector) {
            if let Some(entry) = Self::parse_entry_row(&row) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    fn parse_entry_row(row: &scraper::ElementRef) -> Option<RaceEntry> {
        let mut entry = RaceEntry::default();

        let td_selector = Selector::parse("td").unwrap();
        let cells: Vec<_> = row.select(&td_selector).collect();

        if cells.len() < 4 {
            return None;
        }

        // Post position (枠番/馬番)
        if let Ok(sel) = Selector::parse("td.Umaban, td:nth-child(2)") {
            if let Some(elem) = row.select(&sel).next() {
                let text = elem.text().collect::<String>();
                entry.post_position = text.trim().parse().unwrap_or(0);
            }
        }

        // Horse ID and name
        if let Ok(sel) = Selector::parse("a[href*='/horse/']") {
            if let Some(elem) = row.select(&sel).next() {
                entry.horse_name = elem.text().collect::<String>().trim().to_string();
                if let Some(href) = elem.value().attr("href") {
                    let re = Regex::new(r"/horse/(\d+)").unwrap();
                    if let Some(caps) = re.captures(href) {
                        entry.horse_id = caps[1].to_string();
                    }
                }
            }
        }

        // Sex and age
        if let Ok(sel) = Selector::parse("td.Barei, .Age, td:nth-child(4)") {
            if let Some(elem) = row.select(&sel).next() {
                let text = elem.text().collect::<String>();
                let re = Regex::new(r"([牡牝セ])(\d+)").unwrap();
                if let Some(caps) = re.captures(&text) {
                    entry.horse_sex = caps[1].to_string();
                    entry.horse_age = caps[2].parse().unwrap_or(0);
                }
            }
        }

        // Weight carried (斤量)
        if let Ok(sel) = Selector::parse("td.Jockey + td, .Weight, td:nth-child(6)") {
            if let Some(elem) = row.select(&sel).next() {
                let text = elem.text().collect::<String>();
                let re = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();
                if let Some(caps) = re.captures(&text) {
                    entry.weight_carried = caps[1].parse().unwrap_or(0.0);
                }
            }
        }

        // Jockey
        if let Ok(sel) = Selector::parse("a[href*='/jockey/']") {
            if let Some(elem) = row.select(&sel).next() {
                entry.jockey_name = elem.text().collect::<String>().trim().to_string();
                if let Some(href) = elem.value().attr("href") {
                    let re = Regex::new(r"/jockey/(?:result/recent/)?(\d+)").unwrap();
                    if let Some(caps) = re.captures(href) {
                        entry.jockey_id = caps[1].to_string();
                    }
                }
            }
        }

        // Trainer
        if let Ok(sel) = Selector::parse("a[href*='/trainer/']") {
            if let Some(elem) = row.select(&sel).next() {
                entry.trainer_name = elem.text().collect::<String>().trim().to_string();
                if let Some(href) = elem.value().attr("href") {
                    let re = Regex::new(r"/trainer/(?:result/recent/)?(\d+)").unwrap();
                    if let Some(caps) = re.captures(href) {
                        entry.trainer_id = caps[1].to_string();
                    }
                }
            }
        }

        // Horse weight
        if let Ok(sel) = Selector::parse("td.Weight") {
            if let Some(elem) = row.select(&sel).next() {
                let text = elem.text().collect::<String>();
                let re = Regex::new(r"(\d+)(?:\(([+-]?\d+)\))?").unwrap();
                if let Some(caps) = re.captures(&text) {
                    entry.horse_weight = caps.get(1).and_then(|m| m.as_str().parse().ok());
                    entry.weight_change = caps.get(2).and_then(|m| m.as_str().parse().ok());
                }
            }
        }

        // Win odds
        if let Ok(sel) = Selector::parse("span[id^='odds-']") {
            if let Some(elem) = row.select(&sel).next() {
                let text = elem.text().collect::<String>();
                entry.win_odds = text.trim().parse().ok();
            }
        }

        // Popularity
        if let Ok(sel) = Selector::parse("span[id^='ninki-']") {
            if let Some(elem) = row.select(&sel).next() {
                let text = elem.text().collect::<String>();
                entry.popularity = text.trim().parse().ok();
            }
        }

        // Validate entry
        if entry.horse_id.is_empty() || entry.post_position == 0 {
            return None;
        }

        Some(entry)
    }
}
