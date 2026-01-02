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

#[cfg(test)]
mod tests {
    use super::*;

    // HTML fixture matching netkeiba format: 年度, 順位, 1着, 2着, 3着, 着外, 出走数 (7 columns)
    const SAMPLE_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<div class="db_head_name"><h1>友道康夫</h1></div>
<div class="Name_En">Yasuo Tomomichi</div>
<table class="ResultsByYears">
  <thead>
    <tr><th>年度</th><th>順位</th><th>1着</th><th>2着</th><th>3着</th><th>着外</th><th>出走数</th></tr>
  </thead>
  <tbody>
    <tr><td>通算</td><td>-</td><td>850</td><td>720</td><td>650</td><td>2780</td><td>5000</td></tr>
    <tr><td>2025</td><td>5</td><td>42</td><td>35</td><td>28</td><td>95</td><td>200</td></tr>
  </tbody>
</table>
</body>
</html>"#;

    #[test]
    fn test_parse_trainer_profile() {
        let profile = TrainerParser::parse(SAMPLE_HTML, "01567").unwrap();

        assert_eq!(profile.trainer_id, "01567");
        // Name_En selector is checked first and returns English name
        assert!(profile.name.contains("Tomomichi") || profile.name.contains("友道"));
        assert_eq!(profile.total_races, 5000);
        assert_eq!(profile.wins, 850);
        assert_eq!(profile.seconds, 720);
        assert_eq!(profile.thirds, 650);
    }

    #[test]
    fn test_parse_trainer_rates() {
        let profile = TrainerParser::parse(SAMPLE_HTML, "01567").unwrap();

        // Win rate = 850 / 5000 = 0.17
        let expected_win_rate = 850.0 / 5000.0;
        assert!((profile.win_rate - expected_win_rate).abs() < 0.001);

        // Place rate = (850 + 720 + 650) / 5000 = 0.444
        let expected_place_rate = (850.0 + 720.0 + 650.0) / 5000.0;
        assert!((profile.place_rate - expected_place_rate).abs() < 0.001);
    }

    #[test]
    fn test_parse_empty_html() {
        let profile = TrainerParser::parse("<html></html>", "test").unwrap();
        assert_eq!(profile.trainer_id, "test");
        assert_eq!(profile.total_races, 0);
        assert_eq!(profile.win_rate, 0.0);
    }

    #[test]
    fn test_clean_name() {
        assert_eq!(
            TrainerParser::clean_name("友道康夫 (トモミチヤスオ)"),
            "友道康夫"
        );
    }
}
