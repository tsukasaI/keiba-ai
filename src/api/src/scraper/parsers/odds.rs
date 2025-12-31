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

/// Parser for odds JSON API
pub struct OddsParser;

impl OddsParser {
    /// Parse exacta odds from JSON response
    pub fn parse_exacta(json: &str) -> Result<ExactaOdds> {
        let response: OddsResponse = serde_json::from_str(json)?;

        if response.status != "result" {
            anyhow::bail!("Invalid response status: {}", response.status);
        }

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
                        if let (Ok(first), Ok(second)) = (
                            combo_str[0..2].parse::<u8>(),
                            combo_str[2..4].parse::<u8>(),
                        ) {
                            // Parse odds value (remove commas)
                            if let Ok(odds_val) =
                                values[0].replace(',', "").parse::<f64>()
                            {
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

        if response.status != "result" {
            anyhow::bail!("Invalid response status: {}", response.status);
        }

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
                            if let Ok(odds_val) =
                                values[0].replace(',', "").parse::<f64>()
                            {
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
}

/// Internal: API response structure
#[derive(Deserialize)]
struct OddsResponse {
    status: String,
    data: Option<OddsData>,
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
}
