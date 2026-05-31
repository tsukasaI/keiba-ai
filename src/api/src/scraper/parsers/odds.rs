//! Odds parser for netkeiba.com JSON API.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exacta odds (馬単)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactaOdds {
    /// (first_post, second_post) -> odds
    pub odds: HashMap<(u8, u8), f64>,
    /// Official datetime
    pub official_datetime: Option<String>,
}

/// Trifecta odds (三連単)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrifectaOdds {
    /// (first_post, second_post, third_post) -> odds
    pub odds: HashMap<(u8, u8, u8), f64>,
    /// Official datetime
    pub official_datetime: Option<String>,
}

/// Win odds (単勝)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinOdds {
    /// post_position (馬番) -> win odds
    pub odds: HashMap<u8, f64>,
    /// Official datetime
    pub official_datetime: Option<String>,
}

/// Parser for odds JSON API
pub struct OddsParser;

impl OddsParser {
    /// Parse exacta odds from JSON response
    pub fn parse_exacta(json: &str) -> Result<ExactaOdds> {
        let response: OddsResponse = serde_json::from_str(json)?;

        // Live (pre-race) odds report status "middle"; only finalized odds report
        // "result". Both are valid to read, so we don't reject on status — a payload
        // with no odds yet arrives as an empty `data` field (handled as None below).
        let mut odds = HashMap::new();

        if let Some(data) = response.data {
            let official_datetime = data.official_datetime;

            // type=6 is exacta
            if let Some(odds_map) = data.odds.get("6") {
                for (combo_str, values) in odds_map {
                    if values.is_empty() {
                        continue;
                    }

                    // Parse combo: "0102" -> (1, 2)
                    if combo_str.len() == 4 {
                        if let (Ok(first), Ok(second)) =
                            (combo_str[0..2].parse::<u8>(), combo_str[2..4].parse::<u8>())
                        {
                            // Parse odds value (remove commas)
                            if let Ok(odds_val) = values[0].replace(',', "").parse::<f64>() {
                                odds.insert((first, second), odds_val);
                            }
                        }
                    }
                }
            }

            return Ok(ExactaOdds {
                odds,
                official_datetime,
            });
        }

        Ok(ExactaOdds {
            odds,
            official_datetime: None,
        })
    }

    /// Parse trifecta odds from JSON response
    pub fn parse_trifecta(json: &str) -> Result<TrifectaOdds> {
        let response: OddsResponse = serde_json::from_str(json)?;

        // Live (pre-race) odds report status "middle"; only finalized odds report
        // "result". Both are valid to read, so we don't reject on status — a payload
        // with no odds yet arrives as an empty `data` field (handled as None below).
        let mut odds = HashMap::new();

        if let Some(data) = response.data {
            let official_datetime = data.official_datetime;

            // type=8 is trifecta
            if let Some(odds_map) = data.odds.get("8") {
                for (combo_str, values) in odds_map {
                    if values.is_empty() {
                        continue;
                    }

                    // Parse combo: "010203" -> (1, 2, 3)
                    if combo_str.len() == 6 {
                        if let (Ok(first), Ok(second), Ok(third)) = (
                            combo_str[0..2].parse::<u8>(),
                            combo_str[2..4].parse::<u8>(),
                            combo_str[4..6].parse::<u8>(),
                        ) {
                            // Parse odds value (remove commas)
                            if let Ok(odds_val) = values[0].replace(',', "").parse::<f64>() {
                                odds.insert((first, second, third), odds_val);
                            }
                        }
                    }
                }
            }

            return Ok(TrifectaOdds {
                odds,
                official_datetime,
            });
        }

        Ok(TrifectaOdds {
            odds,
            official_datetime: None,
        })
    }

    /// Parse win (単勝) odds from a type=1 JSON response. Reads `data.odds["1"]`,
    /// whose keys are zero-padded post positions ("01".."18") and whose value's
    /// first element is the decimal win odds. Returns empty odds when none are
    /// published yet (`data:""` -> None).
    pub fn parse_win(json: &str) -> Result<WinOdds> {
        let response: OddsResponse = serde_json::from_str(json)?;

        let mut odds = HashMap::new();
        let Some(data) = response.data else {
            return Ok(WinOdds {
                odds,
                official_datetime: None,
            });
        };

        // type=1 is win (単勝); type=2 (place) shares the payload and is ignored.
        if let Some(win_map) = data.odds.get("1") {
            for (post_str, values) in win_map {
                if let (Ok(post), Some(first)) = (post_str.parse::<u8>(), values.first()) {
                    if let Ok(odds_val) = first.replace(',', "").parse::<f64>() {
                        odds.insert(post, odds_val);
                    }
                }
            }
        }

        Ok(WinOdds {
            odds,
            official_datetime: data.official_datetime,
        })
    }
}

/// Internal: API response structure
#[derive(Deserialize)]
struct OddsResponse {
    #[serde(default, deserialize_with = "data_opt")]
    data: Option<OddsData>,
}

/// netkeiba returns `"data": ""` (empty string) when no odds are published yet for
/// the requested type, and `"data": { ... }` once they are. Treat anything that
/// isn't a JSON object as `None` so an empty/absent payload degrades to "no odds"
/// instead of a hard parse error.
fn data_opt<'de, D>(deserializer: D) -> Result<Option<OddsData>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Object(_) => serde_json::from_value(value)
            .map(Some)
            .map_err(serde::de::Error::custom),
        _ => Ok(None),
    }
}

#[derive(Deserialize)]
struct OddsData {
    official_datetime: Option<String>,
    odds: HashMap<String, HashMap<String, Vec<String>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_exacta() {
        let json = r#"{
            "status": "result",
            "data": {
                "official_datetime": "2025-01-01 12:00:00",
                "odds": {
                    "6": {
                        "0102": ["12.5", "0", "2"],
                        "0201": ["15.3", "0", "5"]
                    }
                }
            }
        }"#;

        let result = OddsParser::parse_exacta(json).unwrap();
        assert_eq!(result.odds.get(&(1, 2)), Some(&12.5));
        assert_eq!(result.odds.get(&(2, 1)), Some(&15.3));
    }

    #[test]
    fn test_parse_trifecta() {
        let json = r#"{
            "status": "result",
            "data": {
                "official_datetime": "2025-01-01 12:00:00",
                "odds": {
                    "8": {
                        "010203": ["45.2", "0", "1"],
                        "010302": ["52.3", "0", "3"]
                    }
                }
            }
        }"#;

        let result = OddsParser::parse_trifecta(json).unwrap();
        assert_eq!(result.odds.get(&(1, 2, 3)), Some(&45.2));
        assert_eq!(result.odds.get(&(1, 3, 2)), Some(&52.3));
    }

    /// netkeiba returns `"data": ""` (empty string) before odds are published.
    /// This must degrade to empty odds, not a hard parse error (regression: the
    /// live `paper-record` path crashed here with "invalid type: string ...").
    #[test]
    fn test_parse_empty_data_string() {
        let json =
            r#"{"status":"middle","data":"","update_count":"0","reason":"result odds empty"}"#;

        let exacta = OddsParser::parse_exacta(json).unwrap();
        assert!(exacta.odds.is_empty());
        assert_eq!(exacta.official_datetime, None);

        let trifecta = OddsParser::parse_trifecta(json).unwrap();
        assert!(trifecta.odds.is_empty());
    }

    /// Live (pre-race) odds carry status "middle", not "result". They must still
    /// parse — the parser must not reject on status.
    #[test]
    fn test_parse_exacta_middle_status() {
        let json = r#"{
            "status": "middle",
            "data": {
                "official_datetime": "2026-05-31 09:50:12",
                "odds": {
                    "6": {
                        "0102": ["472.5", "", "75"]
                    }
                }
            }
        }"#;

        let result = OddsParser::parse_exacta(json).unwrap();
        assert_eq!(result.odds.get(&(1, 2)), Some(&472.5));
        assert_eq!(
            result.official_datetime.as_deref(),
            Some("2026-05-31 09:50:12")
        );
    }

    #[test]
    fn test_parse_win() {
        let json = r#"{
            "status": "middle",
            "data": {
                "official_datetime": "2026-05-31 10:17:07",
                "odds": {
                    "1": {
                        "05": ["2.3", "0.0", "1"],
                        "13": ["5.2", "0.0", "3"],
                        "14": ["642.3", "0.0", "16"]
                    },
                    "2": { "05": ["1.1", "0.0", "1"] }
                }
            }
        }"#;

        let result = OddsParser::parse_win(json).unwrap();
        // Only type "1" (win) is read; type "2" (place) is ignored.
        assert_eq!(result.odds.len(), 3);
        assert_eq!(result.odds.get(&5), Some(&2.3));
        assert_eq!(result.odds.get(&14), Some(&642.3));
        assert_eq!(
            result.official_datetime.as_deref(),
            Some("2026-05-31 10:17:07")
        );
    }

    #[test]
    fn test_parse_win_empty_data() {
        let json = r#"{"status":"middle","data":"","reason":"result odds empty"}"#;
        let result = OddsParser::parse_win(json).unwrap();
        assert!(result.odds.is_empty());
    }
}
