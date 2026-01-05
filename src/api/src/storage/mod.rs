//! SQLite storage module for historical race data
//!
//! Provides persistent storage for scraped historical race data,
//! including race results, entries, and odds for all bet types.

pub mod repository;
pub mod schema;

pub use repository::RaceRepository;
pub use schema::create_tables;
