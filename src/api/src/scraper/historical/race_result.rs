//! Race result parser for db.netkeiba.com
//!
//! Parses historical race result pages to extract race info and results.
//! URL: https://db.netkeiba.com/race/RACEID/

use anyhow::Result;
use chrono::{Datelike, NaiveDate};
use regex::Regex;
use scraper::{Html, Selector};

use crate::storage::repository::{HistoricalRaceEntry, HistoricalRaceInfo};

/// Parser for race result pages
pub struct RaceResultParser;

impl RaceResultParser {
    /// Parse race result HTML
    ///
    /// Returns race info and list of entries with results
    pub fn parse(html: &str, race_id: &str) -> Result<(HistoricalRaceInfo, Vec<HistoricalRaceEntry>)> {
        let document = Html::parse_document(html);

        let race_info = Self::parse_race_info(&document, race_id)?;
        let entries = Self::parse_entries(&document, race_id)?;

        Ok((race_info, entries))
    }

    fn parse_race_info(document: &Html, race_id: &str) -> Result<HistoricalRaceInfo> {
        let mut info = HistoricalRaceInfo {
            race_id: race_id.to_string(),
            race_date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
            racecourse: String::new(),
            race_number: 0,
            race_name: None,
            distance: 0,
            surface: String::new(),
            track_condition: None,
            grade: None,
            field_size: None,
            weather: None,
        };

        // Race name from title or heading
        // db.netkeiba.com structure: <dl class="racedata"><dd><h1>Race Name(Grade)</h1></dd></dl>
        for selector_str in ["dl.racedata h1", ".racedata h1", "dd h1", ".RaceName", "h1"] {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(elem) = document.select(&selector).next() {
                    let text = elem.text().collect::<String>().trim().to_string();
                    if !text.is_empty() && text.len() > 2 {
                        info.race_name = Some(text.clone());
                        info.grade = Self::extract_grade(&text);
                        break;
                    }
                }
            }
        }

        // Race data containing date, distance, surface, etc.
        // db.netkeiba.com uses multiple nested elements
        // First, get all text from the main race data section
        let mut full_text = String::new();

        // Try multiple selectors to get race info text
        for selector_str in [
            ".data_intro",
            ".mainrace_data",
            "dl.racedata",
            ".racedata",
            ".RaceData01",
        ] {
            if let Ok(selector) = Selector::parse(selector_str) {
                for elem in document.select(&selector) {
                    full_text.push_str(&elem.text().collect::<String>());
                    full_text.push(' ');
                }
            }
        }

        // Also check smalltxt for date/racecourse info
        if let Ok(selector) = Selector::parse(".smalltxt, p.smalltxt") {
            for elem in document.select(&selector) {
                full_text.push_str(&elem.text().collect::<String>());
                full_text.push(' ');
            }
        }

        // Date: YYYY年MM月DD日 or YYYY/MM/DD
        let date_re = Regex::new(r"(\d{4})年(\d{1,2})月(\d{1,2})日").unwrap();
        if let Some(caps) = date_re.captures(&full_text) {
            let year: i32 = caps[1].parse().unwrap_or(2000);
            let month: u32 = caps[2].parse().unwrap_or(1);
            let day: u32 = caps[3].parse().unwrap_or(1);
            if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
                info.race_date = d;
            }
        }

        // Alternative date format
        if info.race_date.year() == 2000 {
            let date_re2 = Regex::new(r"(\d{4})/(\d{1,2})/(\d{1,2})").unwrap();
            if let Some(caps) = date_re2.captures(&full_text) {
                let year: i32 = caps[1].parse().unwrap_or(2000);
                let month: u32 = caps[2].parse().unwrap_or(1);
                let day: u32 = caps[3].parse().unwrap_or(1);
                if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
                    info.race_date = d;
                }
            }
        }

        // Distance and surface: 芝右2500m or 芝2500m or ダ1200m or ダート1800m
        // Handle direction markers: 右(right), 左(left), 直(straight)
        let dist_re = Regex::new(r"(芝|ダ(?:ート)?)[右左直]?(\d+)m?").unwrap();
        if let Some(caps) = dist_re.captures(&full_text) {
            info.surface = if caps[1].starts_with("芝") {
                "turf".to_string()
            } else {
                "dirt".to_string()
            };
            info.distance = caps[2].parse().unwrap_or(0);
        }

        // Race number: XXR or XX R
        let race_num_re = Regex::new(r"(\d+)\s*R").unwrap();
        if let Some(caps) = race_num_re.captures(&full_text) {
            info.race_number = caps[1].parse().unwrap_or(0);
        }

        // Racecourse - look for pattern like "5回中山8日目"
        let racecourse_re = Regex::new(r"(札幌|函館|福島|新潟|中山|東京|中京|京都|阪神|小倉)").unwrap();
        if let Some(caps) = racecourse_re.captures(&full_text) {
            info.racecourse = caps[1].to_string();
        }

        // Track condition - look for pattern like "芝 : 良" or "ダート : 稍重"
        let condition_re = Regex::new(r"(?:芝|ダート?)\s*[:：]\s*(良|稍重|重|不良)").unwrap();
        if let Some(caps) = condition_re.captures(&full_text) {
            info.track_condition = Some(caps[1].to_string());
        } else {
            // Fallback: look for condition without prefix
            for condition in ["良", "稍重", "重", "不良"] {
                if full_text.contains(condition) {
                    info.track_condition = Some(condition.to_string());
                    break;
                }
            }
        }

        // Weather - look for pattern like "天候 : 晴"
        let weather_re = Regex::new(r"天候\s*[:：]\s*(晴|曇|雨|小雨|雪|小雪)").unwrap();
        if let Some(caps) = weather_re.captures(&full_text) {
            info.weather = Some(caps[1].to_string());
        }

        // Parse race_id for date if not found in HTML
        if info.race_date.year() == 2000 && race_id.len() >= 8 {
            if let Ok(year) = race_id[0..4].parse::<i32>() {
                // Race ID format: YYYYRRCCNNDD
                // Try to extract date from the race calendar
                info.race_date = NaiveDate::from_ymd_opt(year, 1, 1).unwrap_or(info.race_date);
            }
        }

        // Extract field size from entries table later
        Ok(info)
    }

    fn extract_grade(text: &str) -> Option<String> {
        // Check for various grade patterns
        // Order matters: check G3/GIII before G2 before G1 to avoid false matches

        // G3/GIII patterns (check first)
        if text.contains("GIII")
            || text.contains("GⅢ")
            || text.contains("G3")
            || text.contains("(G3)")
            || text.contains("（G3）")
        {
            return Some("G3".to_string());
        }

        // G2/GII patterns (check before G1)
        if text.contains("GII") && !text.contains("GIII") {
            return Some("G2".to_string());
        }
        if text.contains("GⅡ")
            || text.contains("G2")
            || text.contains("(G2)")
            || text.contains("（G2）")
        {
            return Some("G2".to_string());
        }

        // G1/GI patterns
        if text.contains("GI") && !text.contains("GII") && !text.contains("GIII") {
            return Some("G1".to_string());
        }
        if text.contains("GⅠ")
            || text.contains("G1")
            || text.contains("(G1)")
            || text.contains("（G1）")
        {
            return Some("G1".to_string());
        }

        // Listed
        if text.contains("(L)") || text.contains("（L）") || text.contains("(Ｌ)") {
            return Some("L".to_string());
        }

        // Open class
        if text.contains("オープン") || text.contains("OP") {
            return Some("OP".to_string());
        }

        None
    }

    fn parse_entries(document: &Html, race_id: &str) -> Result<Vec<HistoricalRaceEntry>> {
        let mut entries = Vec::new();

        // Result table selectors
        let table_selectors = [
            ".race_table_01",
            ".result_table",
            "table.nk_tb_common",
            "table[summary*='レース結果']",
            "table",
        ];

        let mut table = None;
        for sel_str in table_selectors {
            if let Ok(selector) = Selector::parse(sel_str) {
                if let Some(t) = document.select(&selector).next() {
                    // Check if it looks like a result table
                    let text = t.text().collect::<String>();
                    if text.contains("着順") || text.contains("馬番") || text.contains("馬名") {
                        table = Some(t);
                        break;
                    }
                }
            }
        }

        let Some(table_elem) = table else {
            return Ok(entries);
        };

        // Row selector - skip header rows with th elements
        let row_selector = Selector::parse("tr").unwrap();
        let th_selector = Selector::parse("th").unwrap();

        for row in table_elem.select(&row_selector) {
            // Skip header rows
            if row.select(&th_selector).next().is_some() {
                continue;
            }

            if let Some(entry) = Self::parse_entry_row(&row, race_id) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    fn parse_entry_row(row: &scraper::ElementRef, race_id: &str) -> Option<HistoricalRaceEntry> {
        let td_selector = Selector::parse("td").unwrap();
        let cells: Vec<_> = row.select(&td_selector).collect();

        // Need at least a few cells for a valid entry
        if cells.len() < 4 {
            return None;
        }

        let mut entry = HistoricalRaceEntry {
            race_id: race_id.to_string(),
            post_position: 0,
            horse_id: String::new(),
            horse_name: String::new(),
            horse_age: None,
            horse_sex: None,
            weight_carried: None,
            horse_weight: None,
            weight_change: None,
            jockey_id: None,
            jockey_name: None,
            trainer_id: None,
            trainer_name: None,
            finish_position: None,
            finish_time: None,
            margin: None,
            last_3f: None,
            corner_1: None,
            corner_2: None,
            corner_3: None,
            corner_4: None,
            win_odds: None,
            popularity: None,
        };

        // Finish position (着順) - usually first column
        let text0 = cells[0].text().collect::<String>().trim().to_string();
        if let Ok(pos) = text0.parse::<u8>() {
            entry.finish_position = Some(pos);
        } else if text0.contains("取消") || text0.contains("除外") || text0.contains("中止") {
            // Scratched/withdrawn
            entry.finish_position = None;
        }

        // Post position (馬番) - typically in column 3 (after 着順 and 枠番)
        // Check column 3 first, then fall back to column 2
        // Using cells vector directly instead of selectors
        if cells.len() >= 3 {
            // Try column 3 (index 2) first - this is typically umaban
            let text = cells[2].text().collect::<String>().trim().to_string();
            if let Ok(pos) = text.parse::<u8>() {
                if pos >= 1 && pos <= 18 {
                    entry.post_position = pos;
                }
            }
        }
        // Fall back to column 2 (index 1) if no valid post position found
        if entry.post_position == 0 && cells.len() >= 2 {
            let text = cells[1].text().collect::<String>().trim().to_string();
            if let Ok(pos) = text.parse::<u8>() {
                if pos >= 1 && pos <= 18 {
                    entry.post_position = pos;
                }
            }
        }

        // Horse ID and name
        if let Ok(selector) = Selector::parse("a[href*='/horse/']") {
            if let Some(elem) = row.select(&selector).next() {
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
        let age_re = Regex::new(r"([牡牝セ騸])(\d+)").unwrap();
        for cell in &cells {
            let text = cell.text().collect::<String>();
            if let Some(caps) = age_re.captures(&text) {
                entry.horse_sex = Some(caps[1].to_string());
                entry.horse_age = caps[2].parse().ok();
                break;
            }
        }

        // Jockey
        if let Ok(selector) = Selector::parse("a[href*='/jockey/']") {
            if let Some(elem) = row.select(&selector).next() {
                entry.jockey_name = Some(elem.text().collect::<String>().trim().to_string());
                if let Some(href) = elem.value().attr("href") {
                    let re = Regex::new(r"/jockey/(?:result/)?(\d+)").unwrap();
                    if let Some(caps) = re.captures(href) {
                        entry.jockey_id = Some(caps[1].to_string());
                    }
                }
            }
        }

        // Trainer
        if let Ok(selector) = Selector::parse("a[href*='/trainer/']") {
            if let Some(elem) = row.select(&selector).next() {
                entry.trainer_name = Some(elem.text().collect::<String>().trim().to_string());
                if let Some(href) = elem.value().attr("href") {
                    let re = Regex::new(r"/trainer/(?:result/)?(\d+)").unwrap();
                    if let Some(caps) = re.captures(href) {
                        entry.trainer_id = Some(caps[1].to_string());
                    }
                }
            }
        }

        // Weight carried (斤量)
        let weight_re = Regex::new(r"(\d+(?:\.\d+)?)\s*(?:kg)?").unwrap();
        for cell in &cells {
            let class = cell.value().attr("class").unwrap_or("");
            if class.contains("weight") || class.contains("kinryo") {
                let text = cell.text().collect::<String>();
                if let Some(caps) = weight_re.captures(&text) {
                    if let Ok(w) = caps[1].parse::<f64>() {
                        if w >= 48.0 && w <= 65.0 {
                            entry.weight_carried = Some(w);
                            break;
                        }
                    }
                }
            }
        }

        // Horse weight: 480(+4) or 480(-2)
        let horse_weight_re = Regex::new(r"(\d{3,4})\s*(?:\(([+-]?\d+)\))?").unwrap();
        for cell in &cells {
            let text = cell.text().collect::<String>();
            if let Some(caps) = horse_weight_re.captures(&text) {
                if let Ok(w) = caps[1].parse::<u32>() {
                    if w >= 350 && w <= 600 {
                        entry.horse_weight = Some(w);
                        entry.weight_change = caps.get(2).and_then(|m| m.as_str().parse().ok());
                        break;
                    }
                }
            }
        }

        // Finish time: 1:35.4 or 2:01.5
        let time_re = Regex::new(r"(\d):(\d{2})\.(\d)").unwrap();
        for cell in &cells {
            let text = cell.text().collect::<String>();
            if let Some(caps) = time_re.captures(&text) {
                let min: f64 = caps[1].parse().unwrap_or(0.0);
                let sec: f64 = caps[2].parse().unwrap_or(0.0);
                let dec: f64 = caps[3].parse().unwrap_or(0.0) / 10.0;
                let total = min * 60.0 + sec + dec;
                if total > 60.0 && total < 300.0 {
                    entry.finish_time = Some(total);
                    break;
                }
            }
        }

        // Last 3f (上り)
        let last_3f_re = Regex::new(r"(\d{2})\.(\d)").unwrap();
        for cell in &cells {
            let class = cell.value().attr("class").unwrap_or("");
            if class.contains("agari") || class.contains("last3f") {
                let text = cell.text().collect::<String>();
                if let Some(caps) = last_3f_re.captures(&text) {
                    let sec: f32 = caps[1].parse().unwrap_or(0.0);
                    let dec: f32 = caps[2].parse().unwrap_or(0.0) / 10.0;
                    let total = sec + dec;
                    if total >= 30.0 && total <= 45.0 {
                        entry.last_3f = Some(total);
                        break;
                    }
                }
            }
        }

        // Corner positions (通過)
        if let Ok(selector) = Selector::parse("td:nth-child(11), td.corner") {
            if let Some(elem) = row.select(&selector).next() {
                let text = elem.text().collect::<String>();
                let corner_re = Regex::new(r"(\d+)-(\d+)-(\d+)-(\d+)").unwrap();
                if let Some(caps) = corner_re.captures(&text) {
                    entry.corner_1 = caps[1].parse().ok();
                    entry.corner_2 = caps[2].parse().ok();
                    entry.corner_3 = caps[3].parse().ok();
                    entry.corner_4 = caps[4].parse().ok();
                }
            }
        }

        // Win odds (単勝) and Popularity (人気)
        // First try: class-based identification (works for test HTML and some real pages)
        for cell in &cells {
            let class = cell.value().attr("class").unwrap_or("");
            if class.contains("odds") || class.contains("tanshow") {
                let text = cell.text().collect::<String>();
                let simple_odds_re = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();
                if let Some(caps) = simple_odds_re.captures(&text) {
                    entry.win_odds = caps[1].parse().ok();
                    break;
                }
            }
        }

        for cell in &cells {
            let class = cell.value().attr("class").unwrap_or("");
            if class.contains("ninki") || class.contains("popularity") {
                let text = cell.text().collect::<String>().trim().to_string();
                entry.popularity = text.parse().ok();
                break;
            }
        }

        // Fallback for db.netkeiba.com which doesn't use special classes
        // Column order: 着順, 枠番, 馬番, 馬名, 性齢, 斤量, 騎手, タイム, 着差, [タイム指数], [通過], [上り], 単勝, 人気, 馬体重, ...
        // Look for odds pattern: decimal number that's NOT in last 3f range (30-45)
        if entry.win_odds.is_none() && cells.len() >= 12 {
            let odds_re = Regex::new(r"^(\d+\.\d)$").unwrap();
            for (idx, cell) in cells.iter().enumerate() {
                if idx < 9 {
                    continue; // Skip early columns (up to 着差)
                }
                let text = cell.text().collect::<String>().trim().to_string();
                if let Some(caps) = odds_re.captures(&text) {
                    if let Ok(odds) = caps[1].parse::<f64>() {
                        // Skip values in last 3f range (30-50 seconds)
                        // Odds are typically 1.0-30.0 for favorites, or >50 for longshots
                        if (odds >= 1.0 && odds < 30.0) || odds > 50.0 {
                            entry.win_odds = Some(odds);
                            // Next cell might be popularity
                            if idx + 1 < cells.len() {
                                let pop_text = cells[idx + 1].text().collect::<String>().trim().to_string();
                                if let Ok(pop) = pop_text.parse::<u8>() {
                                    if pop >= 1 && pop <= 18 {
                                        entry.popularity = Some(pop);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Validate entry
        if entry.horse_id.is_empty() || entry.post_position == 0 {
            return None;
        }

        Some(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_HTML: &str = r#"<!DOCTYPE html>
<html>
<body>
<div class="racedata">
    <h1>有馬記念(G1)</h1>
    <p>2024年12月22日 中山 11R 芝2500m 良</p>
</div>
<table class="race_table_01">
  <tr>
    <th>着順</th>
    <th>枠</th>
    <th>馬番</th>
    <th>馬名</th>
    <th>性齢</th>
    <th>斤量</th>
    <th>騎手</th>
    <th>タイム</th>
    <th>着差</th>
    <th>上り</th>
    <th>通過</th>
    <th>単勝</th>
    <th>人気</th>
    <th>馬体重</th>
    <th>調教師</th>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td><a href="/horse/2019104567/">ドウデュース</a></td>
    <td>牡5</td>
    <td class="weight">57.0</td>
    <td><a href="/jockey/01234/">武豊</a></td>
    <td>2:32.5</td>
    <td></td>
    <td class="agari">35.2</td>
    <td class="corner">5-5-3-2</td>
    <td class="odds">2.1</td>
    <td class="ninki">1</td>
    <td>486(+2)</td>
    <td><a href="/trainer/01567/">友道康夫</a></td>
  </tr>
  <tr>
    <td>2</td>
    <td>2</td>
    <td>3</td>
    <td><a href="/horse/2020105678/">ジャスティンパレス</a></td>
    <td>牡4</td>
    <td class="weight">57.0</td>
    <td><a href="/jockey/01235/">C.ルメール</a></td>
    <td>2:32.7</td>
    <td>1 1/4</td>
    <td class="agari">35.5</td>
    <td class="corner">8-8-6-4</td>
    <td class="odds">3.5</td>
    <td class="ninki">2</td>
    <td>492(-4)</td>
    <td><a href="/trainer/01568/">杉山晴紀</a></td>
  </tr>
</table>
</body>
</html>"#;

    #[test]
    fn test_parse_race_info() {
        let (info, _) = RaceResultParser::parse(SAMPLE_HTML, "202406050811").unwrap();

        assert_eq!(info.race_id, "202406050811");
        assert_eq!(info.race_name.as_deref(), Some("有馬記念(G1)"));
        assert_eq!(info.racecourse, "中山");
        assert_eq!(info.race_number, 11);
        assert_eq!(info.distance, 2500);
        assert_eq!(info.surface, "turf");
        assert_eq!(info.track_condition.as_deref(), Some("良"));
        assert_eq!(info.grade.as_deref(), Some("G1"));
        assert_eq!(info.race_date, NaiveDate::from_ymd_opt(2024, 12, 22).unwrap());
    }

    #[test]
    fn test_parse_entries() {
        let (_, entries) = RaceResultParser::parse(SAMPLE_HTML, "202406050811").unwrap();

        assert_eq!(entries.len(), 2);

        let entry1 = &entries[0];
        assert_eq!(entry1.post_position, 1);
        assert_eq!(entry1.horse_id, "2019104567");
        assert_eq!(entry1.horse_name, "ドウデュース");
        assert_eq!(entry1.horse_sex.as_deref(), Some("牡"));
        assert_eq!(entry1.horse_age, Some(5));
        assert_eq!(entry1.finish_position, Some(1));
        assert_eq!(entry1.jockey_id.as_deref(), Some("01234"));
        assert_eq!(entry1.jockey_name.as_deref(), Some("武豊"));
        assert_eq!(entry1.trainer_id.as_deref(), Some("01567"));
        assert_eq!(entry1.win_odds, Some(2.1));
        assert_eq!(entry1.popularity, Some(1));

        let entry2 = &entries[1];
        assert_eq!(entry2.post_position, 3);
        assert_eq!(entry2.finish_position, Some(2));
        assert_eq!(entry2.horse_weight, Some(492));
        assert_eq!(entry2.weight_change, Some(-4));
    }

    #[test]
    fn test_extract_grade() {
        assert_eq!(RaceResultParser::extract_grade("有馬記念(G1)"), Some("G1".to_string()));
        assert_eq!(RaceResultParser::extract_grade("日経賞（G2）"), Some("G2".to_string()));
        assert_eq!(RaceResultParser::extract_grade("中山牝馬S(GIII)"), Some("G3".to_string()));
        assert_eq!(RaceResultParser::extract_grade("オープン"), Some("OP".to_string()));
        assert_eq!(RaceResultParser::extract_grade("未勝利"), None);
    }

    #[test]
    fn test_empty_html() {
        let result = RaceResultParser::parse("<html></html>", "test123");
        assert!(result.is_ok());
        let (info, entries) = result.unwrap();
        assert_eq!(info.race_id, "test123");
        assert!(entries.is_empty());
    }
}
