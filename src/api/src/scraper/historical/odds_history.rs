//! Historical odds parser for db.netkeiba.com
//!
//! Parses historical odds pages for all bet types.
//! URLs:
//! - Exacta: https://db.netkeiba.com/odds/RACEID/umatan/
//! - Trifecta: https://db.netkeiba.com/odds/RACEID/sanrentan/
//! - Quinella: https://db.netkeiba.com/odds/RACEID/umaren/
//! - Trio: https://db.netkeiba.com/odds/RACEID/sanrenpuku/
//! - Wide: https://db.netkeiba.com/odds/RACEID/wide/

use anyhow::Result;
use regex::Regex;
use scraper::{Html, Selector};
use std::collections::HashMap;

/// Parser for historical odds pages (used when --include-odds is specified)
#[allow(dead_code)]
pub struct HistoricalOddsParser;

#[allow(dead_code)]
impl HistoricalOddsParser {
    /// Parse exacta (馬単) odds
    ///
    /// Returns HashMap of (first-second) combination to odds
    /// Key format: "01-02" (2-digit padded horse numbers)
    pub fn parse_exacta(html: &str) -> Result<HashMap<String, f64>> {
        Self::parse_pair_odds(html)
    }

    /// Parse quinella (馬連) odds
    ///
    /// Returns HashMap of (horse1-horse2) combination to odds
    /// Key format: "01-02" (always smaller number first)
    pub fn parse_quinella(html: &str) -> Result<HashMap<String, f64>> {
        let mut odds = Self::parse_pair_odds(html)?;

        // Quinella is unordered, so normalize keys (smaller first)
        let normalized: HashMap<String, f64> = odds
            .drain()
            .map(|(k, v)| {
                let parts: Vec<&str> = k.split('-').collect();
                if parts.len() == 2 {
                    let a = parts[0];
                    let b = parts[1];
                    if a < b {
                        (format!("{}-{}", a, b), v)
                    } else {
                        (format!("{}-{}", b, a), v)
                    }
                } else {
                    (k, v)
                }
            })
            .collect();

        Ok(normalized)
    }

    /// Parse trifecta (三連単) odds
    ///
    /// Returns HashMap of (first-second-third) combination to odds
    /// Key format: "01-02-03" (2-digit padded horse numbers)
    pub fn parse_trifecta(html: &str) -> Result<HashMap<String, f64>> {
        Self::parse_triple_odds(html, true)
    }

    /// Parse trio (三連複) odds
    ///
    /// Returns HashMap of (horse1-horse2-horse3) combination to odds
    /// Key format: "01-02-03" (sorted, smaller numbers first)
    pub fn parse_trio(html: &str) -> Result<HashMap<String, f64>> {
        let mut odds = Self::parse_triple_odds(html, false)?;

        // Trio is unordered, so normalize keys (sorted)
        let normalized: HashMap<String, f64> = odds
            .drain()
            .map(|(k, v)| {
                let mut parts: Vec<&str> = k.split('-').collect();
                if parts.len() == 3 {
                    parts.sort();
                    (parts.join("-"), v)
                } else {
                    (k, v)
                }
            })
            .collect();

        Ok(normalized)
    }

    /// Parse wide (ワイド) odds
    ///
    /// Wide bets have a range (min-max odds) because payout depends on whether
    /// the combination finishes 1-2, 1-3, or 2-3.
    /// Returns HashMap with average of min-max odds.
    /// Key format: "01-02" (sorted, smaller number first)
    pub fn parse_wide(html: &str) -> Result<HashMap<String, f64>> {
        let document = Html::parse_document(html);
        let mut odds_map = HashMap::new();

        // Wide odds often show as "1.5-2.0" (range)
        let odds_re = Regex::new(r"(\d+(?:\.\d+)?)\s*[-〜~]\s*(\d+(?:\.\d+)?)").unwrap();
        let single_odds_re = Regex::new(r"^(\d+(?:\.\d+)?)$").unwrap();

        // Try table parsing
        let table_selector = Selector::parse("table").unwrap();
        let row_selector = Selector::parse("tr").unwrap();
        let cell_selector = Selector::parse("td, th").unwrap();

        for table in document.select(&table_selector) {
            for row in table.select(&row_selector) {
                let cells: Vec<_> = row.select(&cell_selector).collect();

                if cells.len() >= 2 {
                    let combo = Self::extract_combination(&cells[0]);
                    if let Some((h1, h2)) = combo {
                        // Get odds from remaining cells
                        for cell in cells.iter().skip(1) {
                            let text = cell.text().collect::<String>();

                            // Try range format first
                            if let Some(caps) = odds_re.captures(&text) {
                                let min: f64 = caps[1].parse().unwrap_or(0.0);
                                let max: f64 = caps[2].parse().unwrap_or(0.0);
                                if min > 0.0 && max > 0.0 {
                                    // Use average of range
                                    let avg = (min + max) / 2.0;
                                    let key = Self::make_sorted_key(h1, h2);
                                    odds_map.insert(key, avg);
                                    break;
                                }
                            }

                            // Try single value
                            let trimmed = text.trim();
                            if let Some(caps) = single_odds_re.captures(trimmed) {
                                if let Ok(odds) = caps[1].parse::<f64>() {
                                    if odds > 0.0 {
                                        let key = Self::make_sorted_key(h1, h2);
                                        odds_map.insert(key, odds);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(odds_map)
    }

    /// Parse pair (2-horse) odds from HTML
    fn parse_pair_odds(html: &str) -> Result<HashMap<String, f64>> {
        let document = Html::parse_document(html);
        let mut odds_map = HashMap::new();

        let table_selector = Selector::parse("table").unwrap();
        let row_selector = Selector::parse("tr").unwrap();
        let cell_selector = Selector::parse("td, th").unwrap();

        // Odds value regex
        let odds_re = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();

        for table in document.select(&table_selector) {
            // Check if this looks like an odds table
            let table_text = table.text().collect::<String>();
            if !table_text.contains("オッズ") && !table_text.contains("組合") {
                // Might still be odds table, continue checking
            }

            for row in table.select(&row_selector) {
                let cells: Vec<_> = row.select(&cell_selector).collect();

                if cells.len() >= 2 {
                    // Look for combination in first cell
                    let combo = Self::extract_combination(&cells[0]);

                    if let Some((h1, h2)) = combo {
                        // Look for odds in subsequent cells
                        for cell in cells.iter().skip(1) {
                            let text = cell.text().collect::<String>().trim().to_string();

                            // Skip if it looks like another combination
                            if text.contains("-") || text.contains("→") {
                                continue;
                            }

                            if let Some(caps) = odds_re.captures(&text) {
                                if let Ok(odds) = caps[1].parse::<f64>() {
                                    if odds > 1.0 && odds < 100000.0 {
                                        let key = format!("{:02}-{:02}", h1, h2);
                                        odds_map.insert(key, odds);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Also try parsing from div/span elements (alternative layout)
        if odds_map.is_empty() {
            Self::parse_odds_from_divs(&document, &mut odds_map)?;
        }

        Ok(odds_map)
    }

    /// Parse triple (3-horse) odds from HTML
    fn parse_triple_odds(html: &str, ordered: bool) -> Result<HashMap<String, f64>> {
        let document = Html::parse_document(html);
        let mut odds_map = HashMap::new();

        let table_selector = Selector::parse("table").unwrap();
        let row_selector = Selector::parse("tr").unwrap();
        let cell_selector = Selector::parse("td, th").unwrap();

        let odds_re = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();
        let triple_re = Regex::new(r"(\d+)\s*[-→]\s*(\d+)\s*[-→]\s*(\d+)").unwrap();

        for table in document.select(&table_selector) {
            for row in table.select(&row_selector) {
                let cells: Vec<_> = row.select(&cell_selector).collect();

                if cells.len() >= 2 {
                    let first_text = cells[0].text().collect::<String>();

                    // Try to extract triple combination
                    if let Some(caps) = triple_re.captures(&first_text) {
                        let h1: u8 = caps[1].parse().unwrap_or(0);
                        let h2: u8 = caps[2].parse().unwrap_or(0);
                        let h3: u8 = caps[3].parse().unwrap_or(0);

                        if h1 > 0 && h2 > 0 && h3 > 0 {
                            // Look for odds in subsequent cells
                            for cell in cells.iter().skip(1) {
                                let text = cell.text().collect::<String>().trim().to_string();

                                if let Some(caps) = odds_re.captures(&text) {
                                    if let Ok(odds) = caps[1].parse::<f64>() {
                                        if odds > 1.0 && odds < 1000000.0 {
                                            let key = if ordered {
                                                format!("{:02}-{:02}-{:02}", h1, h2, h3)
                                            } else {
                                                let mut nums = vec![h1, h2, h3];
                                                nums.sort();
                                                format!("{:02}-{:02}-{:02}", nums[0], nums[1], nums[2])
                                            };
                                            odds_map.insert(key, odds);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(odds_map)
    }

    /// Parse odds from div/span elements (alternative layout)
    fn parse_odds_from_divs(document: &Html, odds_map: &mut HashMap<String, f64>) -> Result<()> {
        let odds_re = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();

        // Look for elements with class containing "odds"
        if let Ok(selector) = Selector::parse("[class*='odds'], [class*='Odds']") {
            for elem in document.select(&selector) {
                let text = elem.text().collect::<String>();

                // Check if element contains a combination and odds
                if let Some((h1, h2)) = Self::extract_combination_from_text(&text) {
                    if let Some(caps) = odds_re.captures(&text) {
                        if let Ok(odds) = caps[1].parse::<f64>() {
                            if odds > 1.0 && odds < 100000.0 {
                                let key = format!("{:02}-{:02}", h1, h2);
                                odds_map.entry(key).or_insert(odds);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract combination from cell element
    fn extract_combination(cell: &scraper::ElementRef) -> Option<(u8, u8)> {
        let text = cell.text().collect::<String>();
        Self::extract_combination_from_text(&text)
    }

    /// Extract combination from text
    fn extract_combination_from_text(text: &str) -> Option<(u8, u8)> {
        // Patterns: "1-2", "1→2", "01-02", "1 - 2"
        let re = Regex::new(r"(\d+)\s*[-→]\s*(\d+)").unwrap();
        if let Some(caps) = re.captures(text) {
            let h1: u8 = caps[1].parse().ok()?;
            let h2: u8 = caps[2].parse().ok()?;
            if h1 >= 1 && h1 <= 18 && h2 >= 1 && h2 <= 18 {
                return Some((h1, h2));
            }
        }
        None
    }

    /// Create sorted key for unordered bets
    fn make_sorted_key(h1: u8, h2: u8) -> String {
        if h1 < h2 {
            format!("{:02}-{:02}", h1, h2)
        } else {
            format!("{:02}-{:02}", h2, h1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXACTA_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<table class="odds_table">
  <tr><th>組合せ</th><th>オッズ</th></tr>
  <tr><td>1→2</td><td>15.5</td></tr>
  <tr><td>2→1</td><td>22.0</td></tr>
  <tr><td>1→3</td><td>45.0</td></tr>
  <tr><td>3→1</td><td>55.5</td></tr>
</table>
</body>
</html>"#;

    const QUINELLA_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<table>
  <tr><th>組合せ</th><th>オッズ</th></tr>
  <tr><td>1-2</td><td>8.5</td></tr>
  <tr><td>1-3</td><td>25.0</td></tr>
  <tr><td>2-3</td><td>12.5</td></tr>
</table>
</body>
</html>"#;

    const TRIFECTA_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<table>
  <tr><th>組合せ</th><th>オッズ</th></tr>
  <tr><td>1→2→3</td><td>150.5</td></tr>
  <tr><td>1→3→2</td><td>185.0</td></tr>
  <tr><td>2→1→3</td><td>220.0</td></tr>
</table>
</body>
</html>"#;

    const WIDE_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<table>
  <tr><th>組合せ</th><th>オッズ</th></tr>
  <tr><td>1-2</td><td>1.5-2.0</td></tr>
  <tr><td>1-3</td><td>3.0-4.5</td></tr>
  <tr><td>2-3</td><td>2.5~3.5</td></tr>
</table>
</body>
</html>"#;

    #[test]
    fn test_parse_exacta() {
        let odds = HistoricalOddsParser::parse_exacta(EXACTA_HTML).unwrap();

        assert_eq!(odds.len(), 4);
        assert_eq!(odds.get("01-02"), Some(&15.5));
        assert_eq!(odds.get("02-01"), Some(&22.0));
        assert_eq!(odds.get("01-03"), Some(&45.0));
    }

    #[test]
    fn test_parse_quinella() {
        let odds = HistoricalOddsParser::parse_quinella(QUINELLA_HTML).unwrap();

        assert_eq!(odds.len(), 3);
        assert_eq!(odds.get("01-02"), Some(&8.5));
        assert_eq!(odds.get("01-03"), Some(&25.0));
        assert_eq!(odds.get("02-03"), Some(&12.5));
    }

    #[test]
    fn test_parse_trifecta() {
        let odds = HistoricalOddsParser::parse_trifecta(TRIFECTA_HTML).unwrap();

        assert_eq!(odds.len(), 3);
        assert_eq!(odds.get("01-02-03"), Some(&150.5));
        assert_eq!(odds.get("01-03-02"), Some(&185.0));
        assert_eq!(odds.get("02-01-03"), Some(&220.0));
    }

    #[test]
    fn test_parse_wide() {
        let odds = HistoricalOddsParser::parse_wide(WIDE_HTML).unwrap();

        assert_eq!(odds.len(), 3);

        // Wide uses average of range
        assert!((odds.get("01-02").unwrap() - 1.75).abs() < 0.01);
        assert!((odds.get("01-03").unwrap() - 3.75).abs() < 0.01);
        assert!((odds.get("02-03").unwrap() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_quinella_normalization() {
        let html = r#"<table>
          <tr><td>2-1</td><td>10.0</td></tr>
        </table>"#;
        let odds = HistoricalOddsParser::parse_quinella(html).unwrap();

        // Should be normalized to 01-02, not 02-01
        assert!(odds.contains_key("01-02"));
        assert!(!odds.contains_key("02-01"));
    }

    #[test]
    fn test_empty_html() {
        let odds = HistoricalOddsParser::parse_exacta("<html></html>").unwrap();
        assert!(odds.is_empty());
    }

    #[test]
    fn test_make_sorted_key() {
        assert_eq!(HistoricalOddsParser::make_sorted_key(1, 2), "01-02");
        assert_eq!(HistoricalOddsParser::make_sorted_key(5, 3), "03-05");
        assert_eq!(HistoricalOddsParser::make_sorted_key(10, 10), "10-10");
    }
}
