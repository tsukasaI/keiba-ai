//! Jockey profile parser for netkeiba.com.

use anyhow::Result;
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

/// Jockey profile data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JockeyProfile {
    pub jockey_id: String,
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

/// Parser for jockey profile pages
pub struct JockeyParser;

impl JockeyParser {
    /// Parse jockey profile from HTML
    pub fn parse(html: &str, jockey_id: &str) -> Result<JockeyProfile> {
        let document = Html::parse_document(html);
        let mut profile = JockeyProfile {
            jockey_id: jockey_id.to_string(),
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
        // Try multiple selectors
        let selectors = [
            ".Name_En",
            ".db_head_name h1",
            "h1",
        ];

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

    fn parse_stats_table(document: &Html, profile: &mut JockeyProfile) {
        let table_selector = Selector::parse("table.ResultsByYears").unwrap_or_else(|_| {
            Selector::parse("table").unwrap()
        });
        let tr_selector = Selector::parse("tbody tr, tr").unwrap();
        let td_selector = Selector::parse("td").unwrap();

        if let Some(table) = document.select(&table_selector).next() {
            for row in table.select(&tr_selector) {
                let cells: Vec<_> = row.select(&td_selector).collect();
                if cells.len() < 7 {
                    continue;
                }

                let year_text = cells[0].text().collect::<String>();

                // Parse numbers
                let wins = Self::parse_number(&cells[2].text().collect::<String>());
                let seconds = Self::parse_number(&cells[3].text().collect::<String>());
                let thirds = Self::parse_number(&cells[4].text().collect::<String>());
                let _fourths_plus = Self::parse_number(&cells[5].text().collect::<String>());
                let total = Self::parse_number(&cells[6].text().collect::<String>());

                if year_text.contains("累計") || year_text.contains("通算") {
                    // Career totals
                    profile.wins = wins;
                    profile.seconds = seconds;
                    profile.thirds = thirds;
                    profile.total_races = total;
                } else if Self::is_current_year(&year_text) {
                    // Current year
                    profile.year_wins = wins;
                    profile.year_races = total;
                    if total > 0 {
                        profile.year_place_rate =
                            (wins + seconds + thirds) as f64 / total as f64;
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

    // HTML fixture matching netkeiba format: 年度, 順位, 1着, 2着, 3着, 着外, 騎乗数 (7 columns)
    const SAMPLE_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<div class="db_head_name"><h1>武豊</h1></div>
<div class="Name_En">Yutaka Take</div>
<table class="ResultsByYears">
  <thead>
    <tr><th>年度</th><th>順位</th><th>1着</th><th>2着</th><th>3着</th><th>着外</th><th>騎乗数</th></tr>
  </thead>
  <tbody>
    <tr><td>通算</td><td>-</td><td>4500</td><td>3200</td><td>2800</td><td>10500</td><td>21000</td></tr>
    <tr><td>2025</td><td>1</td><td>85</td><td>70</td><td>55</td><td>190</td><td>400</td></tr>
  </tbody>
</table>
</body>
</html>"#;

    #[test]
    fn test_clean_name() {
        assert_eq!(
            JockeyParser::clean_name("山田太郎 (やまだたろう)"),
            "山田太郎"
        );
        assert_eq!(
            JockeyParser::clean_name("山田太郎のプロフィール - netkeiba"),
            "山田太郎"
        );
    }

    #[test]
    fn test_parse_jockey_profile() {
        let profile = JockeyParser::parse(SAMPLE_HTML, "01234").unwrap();

        assert_eq!(profile.jockey_id, "01234");
        // Name_En selector is checked first and returns English name
        assert!(profile.name.contains("Take") || profile.name.contains("武豊"));
        assert_eq!(profile.total_races, 21000);
        assert_eq!(profile.wins, 4500);
        assert_eq!(profile.seconds, 3200);
        assert_eq!(profile.thirds, 2800);
    }

    #[test]
    fn test_parse_jockey_rates() {
        let profile = JockeyParser::parse(SAMPLE_HTML, "01234").unwrap();

        // Win rate = 4500 / 21000 = 0.214...
        // Note: rates are calculated after parsing
        let expected_win_rate = 4500.0 / 21000.0;
        assert!((profile.win_rate - expected_win_rate).abs() < 0.001);

        // Place rate = (4500 + 3200 + 2800) / 21000 = 0.5
        let expected_place_rate = (4500.0 + 3200.0 + 2800.0) / 21000.0;
        assert!((profile.place_rate - expected_place_rate).abs() < 0.001);
    }

    #[test]
    fn test_parse_empty_html() {
        let profile = JockeyParser::parse("<html></html>", "test").unwrap();
        assert_eq!(profile.jockey_id, "test");
        assert_eq!(profile.total_races, 0);
        assert_eq!(profile.win_rate, 0.0);
    }
}
