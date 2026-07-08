//! Race list parser for db.netkeiba.com
//!
//! Parses race schedule pages to extract race IDs for a given date.
//! URL: https://db.netkeiba.com/race/list/YYYYMMDD/

use anyhow::Result;
use regex::Regex;
use scraper::{Html, Selector};

/// Parser for race list pages
pub struct RaceListParser;

impl RaceListParser {
    /// True iff the race ID belongs to a JRA racecourse.
    ///
    /// db.netkeiba race-list pages mix JRA and NAR (regional) meetings on the
    /// same date. Chars 4..6 of the 12-digit race ID are the racecourse code:
    /// 01-10 for JRA (Sapporo..Kokura), 30+ for NAR tracks. This project only
    /// models JRA; ingesting NAR silently poisons the dataset (nearly all
    /// dirt, different class system, thin betting pools).
    fn is_jra(race_id: &str) -> bool {
        race_id
            .get(4..6)
            .and_then(|s| s.parse::<u32>().ok())
            .is_some_and(|code| (1..=10).contains(&code))
    }

    /// Parse race list HTML and extract JRA race IDs
    ///
    /// Returns a list of race IDs (format: YYYYRRCCNNDD)
    pub fn parse(html: &str) -> Result<Vec<String>> {
        let document = Html::parse_document(html);
        let mut race_ids = Vec::new();

        // Race links are typically in the format /race/RACEID/
        let link_selector = Selector::parse("a[href*='/race/']").unwrap();
        let race_id_re = Regex::new(r"/race/(\d{12})/").unwrap();

        for elem in document.select(&link_selector) {
            if let Some(href) = elem.value().attr("href") {
                if let Some(caps) = race_id_re.captures(href) {
                    let race_id = caps[1].to_string();
                    // Avoid duplicates
                    if Self::is_jra(&race_id) && !race_ids.contains(&race_id) {
                        race_ids.push(race_id);
                    }
                }
            }
        }

        // Sort by race ID (which includes racecourse and race number)
        race_ids.sort();

        Ok(race_ids)
    }

    /// Parse and return race IDs grouped by racecourse (for advanced filtering)
    #[allow(dead_code)]
    pub fn parse_grouped(html: &str) -> Result<Vec<(String, Vec<String>)>> {
        let document = Html::parse_document(html);
        let mut grouped: Vec<(String, Vec<String>)> = Vec::new();

        // Look for racecourse sections
        let section_selector =
            Selector::parse(".race_kaisai_info, .RaceList_Box, div[class*='Race']").unwrap();
        let link_selector = Selector::parse("a[href*='/race/']").unwrap();
        let race_id_re = Regex::new(r"/race/(\d{12})/").unwrap();

        // Try to find racecourse name in headers
        let racecourse_re =
            Regex::new(r"(札幌|函館|福島|新潟|中山|東京|中京|京都|阪神|小倉)").unwrap();

        for section in document.select(&section_selector) {
            let section_text = section.text().collect::<String>();
            let racecourse = racecourse_re
                .find(&section_text)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            let mut race_ids = Vec::new();
            for elem in section.select(&link_selector) {
                if let Some(href) = elem.value().attr("href") {
                    if let Some(caps) = race_id_re.captures(href) {
                        let race_id = caps[1].to_string();
                        if Self::is_jra(&race_id) && !race_ids.contains(&race_id) {
                            race_ids.push(race_id);
                        }
                    }
                }
            }

            if !race_ids.is_empty() {
                race_ids.sort();
                grouped.push((racecourse, race_ids));
            }
        }

        // If no sections found, fall back to flat parsing
        if grouped.is_empty() {
            let race_ids = Self::parse(html)?;
            if !race_ids.is_empty() {
                grouped.push(("unknown".to_string(), race_ids));
            }
        }

        Ok(grouped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<div class="race_kaisai_info">
    <h2>中山競馬</h2>
    <ul>
        <li><a href="/race/202406010101/">1R 未勝利</a></li>
        <li><a href="/race/202406010102/">2R 未勝利</a></li>
        <li><a href="/race/202406010111/">11R 皐月賞(G1)</a></li>
        <li><a href="/race/202406010112/">12R オープン</a></li>
    </ul>
</div>
<div class="race_kaisai_info">
    <h2>阪神競馬</h2>
    <ul>
        <li><a href="/race/202409010101/">1R 未勝利</a></li>
        <li><a href="/race/202409010111/">11R 桜花賞(G1)</a></li>
    </ul>
</div>
</body>
</html>"#;

    #[test]
    fn test_parse_race_list() {
        let race_ids = RaceListParser::parse(SAMPLE_HTML).unwrap();

        assert_eq!(race_ids.len(), 6);
        assert!(race_ids.contains(&"202406010101".to_string()));
        assert!(race_ids.contains(&"202406010111".to_string()));
        assert!(race_ids.contains(&"202409010101".to_string()));
    }

    #[test]
    fn test_parse_race_list_sorted() {
        let race_ids = RaceListParser::parse(SAMPLE_HTML).unwrap();

        // Should be sorted
        for i in 1..race_ids.len() {
            assert!(race_ids[i - 1] <= race_ids[i]);
        }
    }

    #[test]
    fn test_parse_grouped() {
        let grouped = RaceListParser::parse_grouped(SAMPLE_HTML).unwrap();

        assert!(!grouped.is_empty());
        // Should have races from both racecourses
        let total_races: usize = grouped.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(total_races, 6);
    }

    #[test]
    fn test_empty_html() {
        let result = RaceListParser::parse("<html></html>");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_no_duplicates() {
        let html = r#"
        <a href="/race/202406010101/">Race 1</a>
        <a href="/race/202406010101/">Race 1 Again</a>
        <a href="/race/202406010102/">Race 2</a>
        "#;
        let race_ids = RaceListParser::parse(html).unwrap();
        assert_eq!(race_ids.len(), 2);
    }

    #[test]
    fn test_nar_races_filtered_out() {
        // Course codes 30+ are NAR (regional) tracks; db.netkeiba lists them
        // on the same date pages as JRA meetings.
        let html = r#"
        <a href="/race/202406010101/">JRA Nakayama 1R</a>
        <a href="/race/202354012402/">NAR Himeji C3</a>
        <a href="/race/202344010509/">NAR Oi</a>
        <a href="/race/202230011201/">NAR Monbetsu</a>
        "#;
        let race_ids = RaceListParser::parse(html).unwrap();
        assert_eq!(race_ids, vec!["202406010101".to_string()]);
    }

    #[test]
    fn test_is_jra_bounds() {
        assert!(RaceListParser::is_jra("202401010101")); // 01 Sapporo
        assert!(RaceListParser::is_jra("202410010101")); // 10 Kokura
        assert!(!RaceListParser::is_jra("202400010101")); // 00 invalid
        assert!(!RaceListParser::is_jra("202411010101")); // 11 not JRA
        assert!(!RaceListParser::is_jra("202454012402")); // 54 NAR
        assert!(!RaceListParser::is_jra("short")); // malformed
    }
}
