//! Web scraper module for netkeiba.com
//!
//! Provides browser automation, HTML parsing, and feature extraction.

pub mod browser;
pub mod cache;
pub mod feature_builder;
pub mod historical;
pub mod parsers;
pub mod rate_limiter;
pub mod sire_stats;

// Blood features (infrastructure ready, to be integrated with model)
#[allow(unused_imports)]
pub use sire_stats::{load_sire_stats, BloodFeatures};

pub use browser::Browser;
pub use rate_limiter::RateLimiter;

/// Base URLs for netkeiba.com
pub const BASE_URL: &str = "https://race.netkeiba.com";
pub const DB_URL: &str = "https://db.netkeiba.com";

/// CSS selectors for DOM readiness detection (used by PageLoadConfig::with_selector)
pub mod selectors {
    /// Race card page - wait for horse list table
    pub const RACE_CARD: &str = ".HorseList";
    /// Horse profile page - wait for profile area
    pub const HORSE_PROFILE: &str = ".db_prof_area_02";
    /// Jockey profile page - wait for profile area
    pub const JOCKEY_PROFILE: &str = ".db_prof_area_02";
    /// Trainer profile page - wait for profile area
    pub const TRAINER_PROFILE: &str = ".db_prof_area_02";
}

/// Build race card URL
pub fn race_card_url(race_id: &str) -> String {
    format!("{}/race/shutuba.html?race_id={}", BASE_URL, race_id)
}

/// Build the live result page URL (race.netkeiba.com). This publishes results
/// (finish order + 払戻) immediately, whereas db.netkeiba lags same-day races.
pub fn result_url_live(race_id: &str) -> String {
    format!("{}/race/result.html?race_id={}", BASE_URL, race_id)
}

/// Build horse profile URL
pub fn horse_url(horse_id: &str) -> String {
    format!("{}/horse/{}/", DB_URL, horse_id)
}

/// Build jockey profile URL
pub fn jockey_url(jockey_id: &str) -> String {
    format!("{}/jockey/{}/", DB_URL, jockey_id)
}

/// Build trainer profile URL
pub fn trainer_url(trainer_id: &str) -> String {
    format!("{}/trainer/{}/", DB_URL, trainer_id)
}

/// Build win (単勝) odds API URL
pub fn win_odds_url(race_id: &str) -> String {
    format!(
        "{}/api/api_get_jra_odds.html?race_id={}&type=1&action=update",
        BASE_URL, race_id
    )
}

/// Build exacta odds API URL
pub fn exacta_odds_url(race_id: &str) -> String {
    format!(
        "{}/api/api_get_jra_odds.html?race_id={}&type=6&action=update",
        BASE_URL, race_id
    )
}

/// Build trifecta odds API URL
pub fn trifecta_odds_url(race_id: &str) -> String {
    format!(
        "{}/api/api_get_jra_odds.html?race_id={}&type=8&action=update",
        BASE_URL, race_id
    )
}
