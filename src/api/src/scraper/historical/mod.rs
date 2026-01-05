//! Historical data scraper for db.netkeiba.com
//!
//! Provides parsers for historical race data:
//! - Race list (schedule) pages
//! - Race result pages
//! - Historical odds pages

pub mod odds_history;
pub mod race_list;
pub mod race_result;

pub use odds_history::HistoricalOddsParser;
pub use race_list::RaceListParser;
pub use race_result::RaceResultParser;

use super::DB_URL;

/// Build race list URL for a date
/// URL: https://db.netkeiba.com/race/list/YYYYMMDD/
pub fn race_list_url(year: u32, month: u32, day: u32) -> String {
    format!("{}/race/list/{:04}{:02}{:02}/", DB_URL, year, month, day)
}

/// Build race result URL
/// URL: https://db.netkeiba.com/race/RACEID/
pub fn race_result_url(race_id: &str) -> String {
    format!("{}/race/{}/", DB_URL, race_id)
}

/// Build historical exacta odds URL
/// URL: https://db.netkeiba.com/odds/RACEID/umatan/
pub fn exacta_odds_history_url(race_id: &str) -> String {
    format!("{}/odds/{}/umatan/", DB_URL, race_id)
}

/// Build historical trifecta odds URL
/// URL: https://db.netkeiba.com/odds/RACEID/sanrentan/
pub fn trifecta_odds_history_url(race_id: &str) -> String {
    format!("{}/odds/{}/sanrentan/", DB_URL, race_id)
}

/// Build historical quinella odds URL
/// URL: https://db.netkeiba.com/odds/RACEID/umaren/
pub fn quinella_odds_history_url(race_id: &str) -> String {
    format!("{}/odds/{}/umaren/", DB_URL, race_id)
}

/// Build historical trio odds URL
/// URL: https://db.netkeiba.com/odds/RACEID/sanrenpuku/
pub fn trio_odds_history_url(race_id: &str) -> String {
    format!("{}/odds/{}/sanrenpuku/", DB_URL, race_id)
}

/// Build historical wide odds URL
/// URL: https://db.netkeiba.com/odds/RACEID/wide/
pub fn wide_odds_history_url(race_id: &str) -> String {
    format!("{}/odds/{}/wide/", DB_URL, race_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_race_list_url() {
        let url = race_list_url(2024, 1, 6);
        assert_eq!(url, "https://db.netkeiba.com/race/list/20240106/");
    }

    #[test]
    fn test_race_result_url() {
        let url = race_result_url("202406050811");
        assert_eq!(url, "https://db.netkeiba.com/race/202406050811/");
    }

    #[test]
    fn test_odds_urls() {
        let race_id = "202406050811";
        assert_eq!(
            exacta_odds_history_url(race_id),
            "https://db.netkeiba.com/odds/202406050811/umatan/"
        );
        assert_eq!(
            trifecta_odds_history_url(race_id),
            "https://db.netkeiba.com/odds/202406050811/sanrentan/"
        );
    }
}
