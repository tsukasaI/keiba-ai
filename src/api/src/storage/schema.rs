//! SQLite schema definitions for historical race data
//!
//! Tables:
//! - races: Core race information
//! - race_entries: Horse entries with results
//! - odds_snapshots: Pre-race odds for all bet types
//! - horse_stats_snapshot: Historical horse statistics at race time
//! - jockey_stats_snapshot: Historical jockey statistics at race time
//! - trainer_stats_snapshot: Historical trainer statistics at race time

use rusqlite::{Connection, Result};

/// Create all tables in the database
pub fn create_tables(conn: &Connection) -> Result<()> {
    // Core race information
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS races (
            race_id TEXT PRIMARY KEY,
            race_date TEXT NOT NULL,
            racecourse TEXT NOT NULL,
            race_number INTEGER NOT NULL,
            race_name TEXT,
            distance INTEGER NOT NULL,
            surface TEXT NOT NULL,
            track_condition TEXT,
            grade TEXT,
            field_size INTEGER,
            weather TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        "#,
        [],
    )?;

    // Race entries and results
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS race_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL REFERENCES races(race_id),
            post_position INTEGER NOT NULL,
            horse_id TEXT NOT NULL,
            horse_name TEXT NOT NULL,
            horse_age INTEGER,
            horse_sex TEXT,
            weight_carried REAL,
            horse_weight INTEGER,
            weight_change INTEGER,
            jockey_id TEXT,
            jockey_name TEXT,
            trainer_id TEXT,
            trainer_name TEXT,
            finish_position INTEGER,
            finish_time REAL,
            margin TEXT,
            last_3f REAL,
            corner_1 INTEGER,
            corner_2 INTEGER,
            corner_3 INTEGER,
            corner_4 INTEGER,
            win_odds REAL,
            popularity INTEGER,
            UNIQUE(race_id, post_position)
        )
        "#,
        [],
    )?;

    // Pre-race combination odds
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL REFERENCES races(race_id),
            bet_type TEXT NOT NULL,
            combination TEXT NOT NULL,
            odds REAL NOT NULL,
            snapshot_time TEXT,
            UNIQUE(race_id, bet_type, combination)
        )
        "#,
        [],
    )?;

    // Historical horse statistics at race time
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS horse_stats_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL REFERENCES races(race_id),
            horse_id TEXT NOT NULL,
            career_races INTEGER,
            wins INTEGER,
            seconds INTEGER,
            thirds INTEGER,
            earnings INTEGER,
            avg_position_last_3 REAL,
            avg_position_last_5 REAL,
            win_rate_last_3 REAL,
            win_rate_last_5 REAL,
            place_rate_last_3 REAL,
            place_rate_last_5 REAL,
            last_position INTEGER,
            early_position_avg REAL,
            late_position_avg REAL,
            sire TEXT,
            broodmare_sire TEXT,
            UNIQUE(race_id, horse_id)
        )
        "#,
        [],
    )?;

    // Historical jockey statistics
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS jockey_stats_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL REFERENCES races(race_id),
            jockey_id TEXT NOT NULL,
            total_races INTEGER,
            wins INTEGER,
            win_rate REAL,
            place_rate REAL,
            UNIQUE(race_id, jockey_id)
        )
        "#,
        [],
    )?;

    // Historical trainer statistics
    conn.execute(
        r#"
        CREATE TABLE IF NOT EXISTS trainer_stats_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL REFERENCES races(race_id),
            trainer_id TEXT NOT NULL,
            total_races INTEGER,
            wins INTEGER,
            win_rate REAL,
            UNIQUE(race_id, trainer_id)
        )
        "#,
        [],
    )?;

    // Create indexes for common queries
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_race_entries_race ON race_entries(race_id)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_race_entries_horse ON race_entries(horse_id)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_race ON odds_snapshots(race_id)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_odds_race_type ON odds_snapshots(race_id, bet_type)",
        [],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_create_tables() {
        let conn = Connection::open_in_memory().unwrap();
        create_tables(&conn).unwrap();

        // Verify tables exist
        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN
                 ('races', 'race_entries', 'odds_snapshots', 'horse_stats_snapshot',
                  'jockey_stats_snapshot', 'trainer_stats_snapshot')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 6);
    }

    #[test]
    fn test_create_tables_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        create_tables(&conn).unwrap();
        // Should not fail on second call
        create_tables(&conn).unwrap();
    }
}
