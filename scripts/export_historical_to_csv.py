#!/usr/bin/env python3
"""
Export historical SQLite data to CSV format for model retraining.

Maps SQLite schema to the format expected by feature_engineering.py
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DATA_DIR

DB_PATH = Path(__file__).parent.parent / "data" / "historical" / "keiba.db"
OUTPUT_PATH = PROCESSED_DATA_DIR / "historical_2024.csv"


def export_to_csv():
    """Export SQLite data to CSV format."""
    print(f"Reading from: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Join races and entries
    query = """
    SELECT
        r.race_id as "レースID",
        r.race_id || '-' || e.post_position as "レース馬番ID",
        r.race_date as "レース日付",
        r.racecourse as "競馬場コード",
        r.racecourse as "競馬場名",
        r.race_number as "レース番号",
        r.race_name as "レース名",
        r.distance as "距離(m)",
        CASE r.surface
            WHEN 'turf' THEN '芝'
            WHEN 'dirt' THEN 'ダート'
            ELSE r.surface
        END as "芝・ダート区分",
        r.track_condition as "馬場状態1",
        r.grade as "グレード",
        r.field_size as "頭数",
        e.post_position as "馬番",
        e.post_position as "枠番",
        e.horse_id as "馬ID",
        e.horse_name as "馬名",
        e.horse_age as "馬齢",
        e.horse_sex as "性別",
        e.weight_carried as "斤量",
        e.horse_weight as "馬体重",
        e.weight_change as "馬体重増減",
        e.jockey_id as "騎手ID",
        e.jockey_name as "騎手",
        e.trainer_id as "調教師ID",
        e.trainer_name as "調教師",
        e.finish_position as "着順",
        e.finish_time as "タイム",
        e.margin as "着差",
        e.last_3f as "上がり3ハロンタイム",
        e.corner_1 as "コーナー通過順位1",
        e.corner_2 as "コーナー通過順位2",
        e.corner_3 as "コーナー通過順位3",
        e.corner_4 as "コーナー通過順位4",
        e.win_odds as "単勝",
        e.popularity as "人気順"
    FROM races r
    JOIN race_entries e ON r.race_id = e.race_id
    WHERE e.finish_position IS NOT NULL
    ORDER BY r.race_date, r.race_id, e.finish_position
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df)} entries from {df['レースID'].nunique()} races")
    print(f"Date range: {df['レース日付'].min()} to {df['レース日付'].max()}")

    # Create output directory if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")

    # Also save race data separately for exacta odds generation
    races_df = df.groupby('レースID').first().reset_index()[['レースID', 'レース日付']]
    races_path = PROCESSED_DATA_DIR / "historical_races_2024.csv"
    races_df.to_csv(races_path, index=False)
    print(f"Saved races to: {races_path}")

    return df


if __name__ == "__main__":
    export_to_csv()
