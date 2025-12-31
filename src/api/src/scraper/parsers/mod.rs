//! HTML and JSON parsers for netkeiba.com data.

pub mod horse;
pub mod jockey;
pub mod odds;
pub mod race_card;
pub mod trainer;

pub use horse::{HorseParser, HorseProfile};
pub use jockey::{JockeyParser, JockeyProfile};
pub use odds::OddsParser;
pub use race_card::{RaceCardParser, RaceEntry, RaceInfo};
pub use trainer::{TrainerParser, TrainerProfile};
