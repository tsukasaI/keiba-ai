//! Historical data scraper for db.netkeiba.com
//!
//! Provides parsers for historical race data:
//! - Race list (schedule) pages
//! - Race result pages
//! - Historical odds pages

pub mod race_list;
pub mod race_result;

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
}
