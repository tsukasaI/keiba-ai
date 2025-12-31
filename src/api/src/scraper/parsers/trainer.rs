//! Trainer profile parser for netkeiba.com.

use anyhow::Result;
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// Trainer profile data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainerProfile {
    pub trainer_id: String,
    pub name: String,
    // Career stats
    pub total_races: u32,
    pub wins: u32,
    pub seconds: u32,
    pub thirds: u32,
    pub win_rate: f64,
    pub place_rate: f64,
    // Current year stats
    pub year_races: u32,
    pub year_wins: u32,
    pub year_win_rate: f64,
    pub year_place_rate: f64,
}

/// Parser for trainer profile pages
pub struct TrainerParser;

impl TrainerParser {
    /// Parse trainer profile from HTML
    pub fn parse(html: &str, trainer_id: &str) -> Result<TrainerProfile> {
        let document = Html::parse_document(html);
        let mut profile = TrainerProfile {
            trainer_id: trainer_id.to_string(),
            ..Default::default()
        };

        // Parse name
        profile.name = Self::parse_name(&document);

        // Parse stats table
        Self::parse_stats_table(&document, &mut profile);

        // Calculate rates
        if profile.total_races > 0 {
            profile.win_rate = profile.wins as f64 / profile.total_races as f64;
            profile.place_rate =
                (profile.wins + profile.seconds + profile.thirds) as f64 / profile.total_races as f64;
        }

        if profile.year_races > 0 {
            profile.year_win_rate = profile.year_wins as f64 / profile.year_races as f64;
        }

        Ok(profile)
    }

    fn parse_name(document: &Html) -> String {
        let selectors = [".Name_En", ".db_head_name h1", "h1"];

        for sel_str in selectors {
            if let Ok(selector) = Selector::parse(sel_str) {
                if let Some(element) = document.select(&selector).next() {
                    let text = element.text().collect::<String>();
                    let cleaned = Self::clean_name(&text);
                    if !cleaned.is_empty() {
                        return cleaned;
                    }
                }
            }
        }

        String::new()
    }

    fn clean_name(name: &str) -> String {
        let re = Regex::new(r"\([^)]+\)").unwrap();
        let cleaned = re.replace_all(name, "");

        let re2 = Regex::new(r"のプロフィール.*$").unwrap();
        let cleaned = re2.replace_all(&cleaned, "");

        cleaned.trim().to_string()
    }

    fn parse_stats_table(document: &Html, profile: &mut TrainerProfile) {
        let table_selector =
            Selector::parse("table.ResultsByYears").unwrap_or_else(|_| Selector::parse("table").unwrap());
        let tr_selector = Selector::parse("tbody tr, tr").unwrap();
        let td_selector = Selector::parse("td").unwrap();

        if let Some(table) = document.select(&table_selector).next() {
            for row in table.select(&tr_selector) {
                let cells: Vec<_> = row.select(&td_selector).collect();
                if cells.len() < 7 {
                    continue;
                }

                let year_text = cells[0].text().collect::<String>();

                let wins = Self::parse_number(&cells[2].text().collect::<String>());
                let seconds = Self::parse_number(&cells[3].text().collect::<String>());
                let thirds = Self::parse_number(&cells[4].text().collect::<String>());
                let total = Self::parse_number(&cells[6].text().collect::<String>());

                if year_text.contains("累計") || year_text.contains("通算") {
                    profile.wins = wins;
                    profile.seconds = seconds;
                    profile.thirds = thirds;
                    profile.total_races = total;
                } else if Self::is_current_year(&year_text) {
                    profile.year_wins = wins;
                    profile.year_races = total;
                    if total > 0 {
                        profile.year_place_rate = (wins + seconds + thirds) as f64 / total as f64;
                    }
                }
            }
        }
    }

    fn parse_number(text: &str) -> u32 {
        let cleaned = text.replace(',', "").trim().to_string();
        cleaned.parse().unwrap_or(0)
    }

    fn is_current_year(text: &str) -> bool {
        let current_year = chrono::Utc::now().format("%Y").to_string();
        text.contains(&current_year)
    }
}
