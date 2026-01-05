//! SQLite repository for CRUD operations on historical race data

use anyhow::{Context, Result};
use chrono::NaiveDate;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;

use super::schema::create_tables;

/// Historical race information
#[derive(Debug, Clone)]
pub struct HistoricalRaceInfo {
    pub race_id: String,
    pub race_date: NaiveDate,
    pub racecourse: String,
    pub race_number: u8,
    pub race_name: Option<String>,
    pub distance: u32,
    pub surface: String,
    pub track_condition: Option<String>,
    pub grade: Option<String>,
    pub field_size: Option<u8>,
    pub weather: Option<String>,
}

/// Historical race entry with results
#[derive(Debug, Clone)]
pub struct HistoricalRaceEntry {
    pub race_id: String,
    pub post_position: u8,
    pub horse_id: String,
    pub horse_name: String,
    pub horse_age: Option<u8>,
    pub horse_sex: Option<String>,
    pub weight_carried: Option<f64>,
    pub horse_weight: Option<u32>,
    pub weight_change: Option<i32>,
    pub jockey_id: Option<String>,
    pub jockey_name: Option<String>,
    pub trainer_id: Option<String>,
    pub trainer_name: Option<String>,
    pub finish_position: Option<u8>,
    pub finish_time: Option<f64>,
    pub margin: Option<String>,
    pub last_3f: Option<f32>,
    pub corner_1: Option<u8>,
    pub corner_2: Option<u8>,
    pub corner_3: Option<u8>,
    pub corner_4: Option<u8>,
    pub win_odds: Option<f64>,
    pub popularity: Option<u8>,
}

/// Horse statistics snapshot at race time
#[derive(Debug, Clone, Default)]
pub struct HorseStatsSnapshot {
    pub career_races: Option<i32>,
    pub wins: Option<i32>,
    pub seconds: Option<i32>,
    pub thirds: Option<i32>,
    pub earnings: Option<i64>,
    pub avg_position_last_3: Option<f64>,
    pub avg_position_last_5: Option<f64>,
    pub win_rate_last_3: Option<f64>,
    pub win_rate_last_5: Option<f64>,
    pub place_rate_last_3: Option<f64>,
    pub place_rate_last_5: Option<f64>,
    pub last_position: Option<i32>,
    pub early_position_avg: Option<f64>,
    pub late_position_avg: Option<f64>,
    pub sire: Option<String>,
    pub broodmare_sire: Option<String>,
}

/// Jockey statistics snapshot at race time
#[derive(Debug, Clone, Default)]
pub struct JockeyStatsSnapshot {
    pub total_races: Option<i32>,
    pub wins: Option<i32>,
    pub win_rate: Option<f64>,
    pub place_rate: Option<f64>,
}

/// Trainer statistics snapshot at race time
#[derive(Debug, Clone, Default)]
pub struct TrainerStatsSnapshot {
    pub total_races: Option<i32>,
    pub wins: Option<i32>,
    pub win_rate: Option<f64>,
}

/// Repository for historical race data
pub struct RaceRepository {
    conn: Connection,
}

impl RaceRepository {
    /// Create a new repository, initializing the database if needed
    pub fn new(db_path: &Path) -> Result<Self> {
        // Create parent directories if needed
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create database directory")?;
        }

        let conn = Connection::open(db_path)
            .context("Failed to open database")?;

        // Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON", [])?;

        // Create tables if they don't exist
        create_tables(&conn)?;

        Ok(Self { conn })
    }

    /// Create an in-memory repository (for testing)
    #[cfg(test)]
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        create_tables(&conn)?;
        Ok(Self { conn })
    }

    // ==================== Insert Operations ====================

    /// Insert a race (upsert)
    pub fn insert_race(&self, race: &HistoricalRaceInfo) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO races
            (race_id, race_date, racecourse, race_number, race_name, distance,
             surface, track_condition, grade, field_size, weather)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            "#,
            params![
                race.race_id,
                race.race_date.to_string(),
                race.racecourse,
                race.race_number,
                race.race_name,
                race.distance,
                race.surface,
                race.track_condition,
                race.grade,
                race.field_size,
                race.weather,
            ],
        )?;
        Ok(())
    }

    /// Insert a race entry (upsert)
    pub fn insert_entry(&self, entry: &HistoricalRaceEntry) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO race_entries
            (race_id, post_position, horse_id, horse_name, horse_age, horse_sex,
             weight_carried, horse_weight, weight_change, jockey_id, jockey_name,
             trainer_id, trainer_name, finish_position, finish_time, margin,
             last_3f, corner_1, corner_2, corner_3, corner_4, win_odds, popularity)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)
            "#,
            params![
                entry.race_id,
                entry.post_position,
                entry.horse_id,
                entry.horse_name,
                entry.horse_age,
                entry.horse_sex,
                entry.weight_carried,
                entry.horse_weight,
                entry.weight_change,
                entry.jockey_id,
                entry.jockey_name,
                entry.trainer_id,
                entry.trainer_name,
                entry.finish_position,
                entry.finish_time,
                entry.margin,
                entry.last_3f,
                entry.corner_1,
                entry.corner_2,
                entry.corner_3,
                entry.corner_4,
                entry.win_odds,
                entry.popularity,
            ],
        )?;
        Ok(())
    }

    /// Insert odds for a combination (upsert)
    pub fn insert_odds(
        &self,
        race_id: &str,
        bet_type: &str,
        combination: &str,
        odds: f64,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO odds_snapshots
            (race_id, bet_type, combination, odds)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            params![race_id, bet_type, combination, odds],
        )?;
        Ok(())
    }

    /// Insert horse statistics snapshot
    pub fn insert_horse_stats(
        &self,
        race_id: &str,
        horse_id: &str,
        stats: &HorseStatsSnapshot,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO horse_stats_snapshot
            (race_id, horse_id, career_races, wins, seconds, thirds, earnings,
             avg_position_last_3, avg_position_last_5, win_rate_last_3, win_rate_last_5,
             place_rate_last_3, place_rate_last_5, last_position, early_position_avg,
             late_position_avg, sire, broodmare_sire)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
            "#,
            params![
                race_id,
                horse_id,
                stats.career_races,
                stats.wins,
                stats.seconds,
                stats.thirds,
                stats.earnings,
                stats.avg_position_last_3,
                stats.avg_position_last_5,
                stats.win_rate_last_3,
                stats.win_rate_last_5,
                stats.place_rate_last_3,
                stats.place_rate_last_5,
                stats.last_position,
                stats.early_position_avg,
                stats.late_position_avg,
                stats.sire,
                stats.broodmare_sire,
            ],
        )?;
        Ok(())
    }

    /// Insert jockey statistics snapshot
    pub fn insert_jockey_stats(
        &self,
        race_id: &str,
        jockey_id: &str,
        stats: &JockeyStatsSnapshot,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO jockey_stats_snapshot
            (race_id, jockey_id, total_races, wins, win_rate, place_rate)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                race_id,
                jockey_id,
                stats.total_races,
                stats.wins,
                stats.win_rate,
                stats.place_rate,
            ],
        )?;
        Ok(())
    }

    /// Insert trainer statistics snapshot
    pub fn insert_trainer_stats(
        &self,
        race_id: &str,
        trainer_id: &str,
        stats: &TrainerStatsSnapshot,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO trainer_stats_snapshot
            (race_id, trainer_id, total_races, wins, win_rate)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            params![
                race_id,
                trainer_id,
                stats.total_races,
                stats.wins,
                stats.win_rate,
            ],
        )?;
        Ok(())
    }

    // ==================== Query Operations ====================

    /// Check if a race exists
    pub fn race_exists(&self, race_id: &str) -> Result<bool> {
        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM races WHERE race_id = ?1",
            [race_id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get races in a date range
    pub fn get_races_by_date_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<HistoricalRaceInfo>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT race_id, race_date, racecourse, race_number, race_name,
                   distance, surface, track_condition, grade, field_size, weather
            FROM races
            WHERE race_date BETWEEN ?1 AND ?2
            ORDER BY race_date, race_id
            "#,
        )?;

        let races = stmt
            .query_map([start.to_string(), end.to_string()], |row| {
                let date_str: String = row.get(1)?;
                let race_date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                    .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());

                Ok(HistoricalRaceInfo {
                    race_id: row.get(0)?,
                    race_date,
                    racecourse: row.get(2)?,
                    race_number: row.get(3)?,
                    race_name: row.get(4)?,
                    distance: row.get(5)?,
                    surface: row.get(6)?,
                    track_condition: row.get(7)?,
                    grade: row.get(8)?,
                    field_size: row.get(9)?,
                    weather: row.get(10)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(races)
    }

    /// Get entries for a race
    pub fn get_race_entries(&self, race_id: &str) -> Result<Vec<HistoricalRaceEntry>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT race_id, post_position, horse_id, horse_name, horse_age, horse_sex,
                   weight_carried, horse_weight, weight_change, jockey_id, jockey_name,
                   trainer_id, trainer_name, finish_position, finish_time, margin,
                   last_3f, corner_1, corner_2, corner_3, corner_4, win_odds, popularity
            FROM race_entries
            WHERE race_id = ?1
            ORDER BY post_position
            "#,
        )?;

        let entries = stmt
            .query_map([race_id], |row| {
                Ok(HistoricalRaceEntry {
                    race_id: row.get(0)?,
                    post_position: row.get(1)?,
                    horse_id: row.get(2)?,
                    horse_name: row.get(3)?,
                    horse_age: row.get(4)?,
                    horse_sex: row.get(5)?,
                    weight_carried: row.get(6)?,
                    horse_weight: row.get(7)?,
                    weight_change: row.get(8)?,
                    jockey_id: row.get(9)?,
                    jockey_name: row.get(10)?,
                    trainer_id: row.get(11)?,
                    trainer_name: row.get(12)?,
                    finish_position: row.get(13)?,
                    finish_time: row.get(14)?,
                    margin: row.get(15)?,
                    last_3f: row.get(16)?,
                    corner_1: row.get(17)?,
                    corner_2: row.get(18)?,
                    corner_3: row.get(19)?,
                    corner_4: row.get(20)?,
                    win_odds: row.get(21)?,
                    popularity: row.get(22)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(entries)
    }

    /// Get odds for a race and bet type
    pub fn get_odds(&self, race_id: &str, bet_type: &str) -> Result<HashMap<String, f64>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT combination, odds
            FROM odds_snapshots
            WHERE race_id = ?1 AND bet_type = ?2
            "#,
        )?;

        let odds = stmt
            .query_map([race_id, bet_type], |row| {
                let combo: String = row.get(0)?;
                let odds: f64 = row.get(1)?;
                Ok((combo, odds))
            })?
            .collect::<std::result::Result<HashMap<_, _>, _>>()?;

        Ok(odds)
    }

    /// Get the last scraped date (for resume capability)
    pub fn get_last_race_date(&self) -> Result<Option<NaiveDate>> {
        let result: Option<String> = self.conn.query_row(
            "SELECT MAX(race_date) FROM races",
            [],
            |row| row.get(0),
        )?;

        Ok(result.and_then(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok()))
    }

    /// Get race count
    pub fn get_race_count(&self) -> Result<i32> {
        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM races",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Get total odds count
    pub fn get_odds_count(&self) -> Result<i32> {
        let count: i32 = self.conn.query_row(
            "SELECT COUNT(*) FROM odds_snapshots",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_race() -> HistoricalRaceInfo {
        HistoricalRaceInfo {
            race_id: "202401010101".to_string(),
            race_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            racecourse: "中山".to_string(),
            race_number: 1,
            race_name: Some("新春賞".to_string()),
            distance: 1600,
            surface: "turf".to_string(),
            track_condition: Some("良".to_string()),
            grade: Some("OP".to_string()),
            field_size: Some(16),
            weather: Some("晴".to_string()),
        }
    }

    fn create_test_entry(race_id: &str, post: u8) -> HistoricalRaceEntry {
        HistoricalRaceEntry {
            race_id: race_id.to_string(),
            post_position: post,
            horse_id: format!("horse_{}", post),
            horse_name: format!("Horse {}", post),
            horse_age: Some(4),
            horse_sex: Some("牡".to_string()),
            weight_carried: Some(57.0),
            horse_weight: Some(480),
            weight_change: Some(0),
            jockey_id: Some(format!("jockey_{}", post)),
            jockey_name: Some(format!("Jockey {}", post)),
            trainer_id: Some(format!("trainer_{}", post)),
            trainer_name: Some(format!("Trainer {}", post)),
            finish_position: Some(post),
            finish_time: Some(96.5),
            margin: None,
            last_3f: Some(34.5),
            corner_1: Some(post),
            corner_2: Some(post),
            corner_3: Some(post),
            corner_4: Some(post),
            win_odds: Some(5.0 * post as f64),
            popularity: Some(post),
        }
    }

    #[test]
    fn test_insert_and_get_race() {
        let repo = RaceRepository::in_memory().unwrap();
        let race = create_test_race();

        repo.insert_race(&race).unwrap();
        assert!(repo.race_exists(&race.race_id).unwrap());

        let races = repo
            .get_races_by_date_range(
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 31).unwrap(),
            )
            .unwrap();
        assert_eq!(races.len(), 1);
        assert_eq!(races[0].race_id, race.race_id);
    }

    #[test]
    fn test_insert_and_get_entries() {
        let repo = RaceRepository::in_memory().unwrap();
        let race = create_test_race();
        repo.insert_race(&race).unwrap();

        // Insert 5 entries
        for i in 1..=5 {
            let entry = create_test_entry(&race.race_id, i);
            repo.insert_entry(&entry).unwrap();
        }

        let entries = repo.get_race_entries(&race.race_id).unwrap();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].post_position, 1);
        assert_eq!(entries[4].post_position, 5);
    }

    #[test]
    fn test_insert_and_get_odds() {
        let repo = RaceRepository::in_memory().unwrap();
        let race = create_test_race();
        repo.insert_race(&race).unwrap();

        repo.insert_odds(&race.race_id, "exacta", "01-02", 15.5).unwrap();
        repo.insert_odds(&race.race_id, "exacta", "02-01", 22.0).unwrap();
        repo.insert_odds(&race.race_id, "trifecta", "01-02-03", 150.0).unwrap();

        let exacta_odds = repo.get_odds(&race.race_id, "exacta").unwrap();
        assert_eq!(exacta_odds.len(), 2);
        assert_eq!(exacta_odds.get("01-02"), Some(&15.5));

        let trifecta_odds = repo.get_odds(&race.race_id, "trifecta").unwrap();
        assert_eq!(trifecta_odds.len(), 1);
    }

    #[test]
    fn test_upsert_race() {
        let repo = RaceRepository::in_memory().unwrap();
        let mut race = create_test_race();

        repo.insert_race(&race).unwrap();
        assert_eq!(repo.get_race_count().unwrap(), 1);

        // Update race name
        race.race_name = Some("Updated Name".to_string());
        repo.insert_race(&race).unwrap();

        // Should still be 1 race (upsert)
        assert_eq!(repo.get_race_count().unwrap(), 1);
    }

    #[test]
    fn test_get_last_race_date() {
        let repo = RaceRepository::in_memory().unwrap();

        // No races yet
        assert!(repo.get_last_race_date().unwrap().is_none());

        // Add races
        let mut race = create_test_race();
        repo.insert_race(&race).unwrap();

        race.race_id = "202401150101".to_string();
        race.race_date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        repo.insert_race(&race).unwrap();

        let last_date = repo.get_last_race_date().unwrap().unwrap();
        assert_eq!(last_date, NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }
}
