//! Web scraper module for netkeiba.com
//!
//! Provides browser automation, HTML parsing, and feature extraction.

pub mod browser;
pub mod cache;
pub mod feature_builder;
pub mod parsers;
pub mod rate_limiter;
pub mod sire_stats;

pub use sire_stats::{BloodFeatures, load_sire_stats};

pub use browser::Browser;
pub use rate_limiter::RateLimiter;

/// Base URLs for netkeiba.com
pub const BASE_URL: &str = "https://race.netkeiba.com";
pub const DB_URL: &str = "https://db.netkeiba.com";

/// Build race card URL
pub fn race_card_url(race_id: &str) -> String {
    format!("{}/race/shutuba.html?race_id={}", BASE_URL, race_id)
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

/// Build exacta odds API URL
pub fn exacta_odds_url(race_id: &str) -> String {
    format!(
        "{}/api/api_get_jra_odds.html?race_id={}&type=6",
        BASE_URL, race_id
    )
}

/// Build trifecta odds API URL
pub fn trifecta_odds_url(race_id: &str) -> String {
    format!(
        "{}/api/api_get_jra_odds.html?race_id={}&type=8",
        BASE_URL, race_id
    )
}
