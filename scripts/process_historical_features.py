#!/usr/bin/env python3
"""
Process historical 2024 data into features for model retraining.

Creates features.parquet from historical_2024.csv with same 39 features
as the original Kaggle-based training.
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DATA_DIR
from src.models.config import FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_PATH = PROCESSED_DATA_DIR / "historical_2024.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "features.parquet"
BACKTEST_PATH = PROCESSED_DATA_DIR / "backtest_features.parquet"


def compute_jockey_trainer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling jockey and trainer statistics.

    Calculates:
    - jockey_win_rate: Rolling win rate (last 50 races)
    - jockey_place_rate: Rolling place rate (top 3)
    - trainer_win_rate: Rolling win rate (last 50 races)
    - jockey_races: Total races ridden
    - trainer_races: Total races trained
    """
    logger.info("Computing jockey/trainer statistics...")

    # Sort by date for proper time-series calculation
    df = df.sort_values(['レース日付', 'レースID', '馬番']).copy()
    df = df.reset_index(drop=True)

    # Create win/place flags
    df['_win'] = (df['着順'] == 1).astype(float)
    df['_place'] = (df['着順'] <= 3).astype(float)

    # Rolling window size
    window = 50

    # Jockey stats - use expanding window with shift to avoid leakage
    logger.info("  Computing jockey stats...")

    # Sort and compute within groups
    df['_jockey_cumwins'] = df.groupby('騎手')['_win'].transform(
        lambda x: x.expanding().sum().shift(1)
    )
    df['_jockey_cumplaces'] = df.groupby('騎手')['_place'].transform(
        lambda x: x.expanding().sum().shift(1)
    )
    df['_jockey_cumraces'] = df.groupby('騎手').cumcount()  # 0-indexed count before current

    # Win rate: cumulative wins / cumulative races
    df['jockey_win_rate'] = (
        df['_jockey_cumwins'] / df['_jockey_cumraces'].replace(0, np.nan)
    ).fillna(0.10).clip(0, 1)

    # Place rate
    df['jockey_place_rate'] = (
        df['_jockey_cumplaces'] / df['_jockey_cumraces'].replace(0, np.nan)
    ).fillna(0.30).clip(0, 1)

    # Race count (add 1 since cumcount is 0-indexed)
    df['jockey_races'] = (df['_jockey_cumraces'] + 1).clip(lower=1)

    # Trainer stats
    logger.info("  Computing trainer stats...")

    df['_trainer_cumwins'] = df.groupby('調教師')['_win'].transform(
        lambda x: x.expanding().sum().shift(1)
    )
    df['_trainer_cumraces'] = df.groupby('調教師').cumcount()

    df['trainer_win_rate'] = (
        df['_trainer_cumwins'] / df['_trainer_cumraces'].replace(0, np.nan)
    ).fillna(0.10).clip(0, 1)

    df['trainer_races'] = (df['_trainer_cumraces'] + 1).clip(lower=1)

    # Clean up temporary columns
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')])

    logger.info(f"  Jockey win rate range: {df['jockey_win_rate'].min():.3f} - {df['jockey_win_rate'].max():.3f}")
    logger.info(f"  Trainer win rate range: {df['trainer_win_rate'].min():.3f} - {df['trainer_win_rate'].max():.3f}")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 39 features from historical data."""
    logger.info("Creating features...")

    # Sort by date
    df = df.sort_values(['レース日付', 'レースID', '着順']).copy()

    # Basic features (use Japanese column names as expected by config)
    result = pd.DataFrame()
    result['レースID'] = df['レースID']
    result['レース日付'] = pd.to_datetime(df['レース日付'])
    result['着順'] = df['着順']

    # 0: horse_age_num
    result['horse_age_num'] = df['馬齢'].fillna(4).astype(int)

    # 1: horse_sex_encoded (0=牝, 1=牡, 2=セ)
    sex_map = {'牝': 0, '牡': 1, 'セ': 2, '騸': 2}
    result['horse_sex_encoded'] = df['性別'].map(sex_map).fillna(1)

    # 2: post_position_num
    result['post_position_num'] = df['馬番'].fillna(8).astype(int)

    # 3: weight_carried (斤量)
    result['斤量'] = df['斤量'].fillna(55.0)

    # 4: horse_weight (馬体重)
    result['馬体重'] = df['馬体重'].fillna(480)

    # 5-9: jockey/trainer stats (from pre-computed rolling stats)
    result['jockey_win_rate'] = df['jockey_win_rate'].fillna(0.10)
    result['jockey_place_rate'] = df['jockey_place_rate'].fillna(0.30)
    result['trainer_win_rate'] = df['trainer_win_rate'].fillna(0.10)
    result['jockey_races'] = df['jockey_races'].fillna(1.0)
    result['trainer_races'] = df['trainer_races'].fillna(1.0)

    # 10-13: race conditions
    result['distance_num'] = df['距離(m)'].fillna(1600)
    result['is_turf'] = (df['芝・ダート区分'] == '芝').astype(float)
    result['is_dirt'] = (df['芝・ダート区分'] == 'ダート').astype(float)

    condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    result['track_condition_num'] = df['馬場状態1'].map(condition_map).fillna(0)

    # 14-21: past performance features (use win_odds as proxy for quality)
    odds = df['単勝'].fillna(10.0)
    odds_log = np.log(odds.clip(lower=1.1))
    popularity_proxy = 1.0 / (1.0 + np.exp(odds_log - 2.0))  # Sigmoid transform

    result['avg_position_last_3'] = 5.0 - popularity_proxy * 3.0
    result['avg_position_last_5'] = 5.0 - popularity_proxy * 3.0
    result['win_rate_last_3'] = popularity_proxy * 0.2
    result['win_rate_last_5'] = popularity_proxy * 0.2
    result['place_rate_last_3'] = popularity_proxy * 0.4
    result['place_rate_last_5'] = popularity_proxy * 0.4
    result['last_position'] = 5.0 - popularity_proxy * 3.0
    result['career_races'] = 10.0

    # 22: odds_log
    result['odds_log'] = odds_log

    # 23-25: running style (from corner positions)
    corner_1 = df['コーナー通過順位1'].fillna(df['頭数'].fillna(10) / 2)
    corner_4 = df['コーナー通過順位4'].fillna(df['頭数'].fillna(10) / 2)
    field_size = df['頭数'].fillna(10)

    result['early_position'] = corner_1 / field_size.clip(lower=1)
    result['late_position'] = corner_4 / field_size.clip(lower=1)
    result['position_change'] = corner_4 - corner_1

    # 26-32: aptitude features (defaults)
    result['aptitude_sprint'] = 0.5
    result['aptitude_mile'] = 0.5
    result['aptitude_intermediate'] = 0.5
    result['aptitude_long'] = 0.5
    result['aptitude_turf'] = 0.5
    result['aptitude_dirt'] = 0.5
    result['aptitude_course'] = 0.5

    # 33-35: pace features
    last_3f = df['上がり3ハロンタイム'].fillna(35.0)
    result['last_3f_avg'] = last_3f
    result['last_3f_best'] = last_3f - 0.5
    result['last_3f_last'] = last_3f

    # 36-38: race classification
    result['weight_change_kg'] = df['馬体重増減'].fillna(0)
    result['is_graded_race'] = df['グレード'].notna().astype(float)

    grade_map = {'G1': 1, 'G2': 2, 'G3': 3, 'L': 4, 'OP': 5}
    result['grade_level'] = df['グレード'].map(grade_map).fillna(6)

    # Verify we have 39 features
    feature_cols = [c for c in result.columns if c not in ['レースID', 'レース日付', '着順']]
    logger.info(f"Created {len(feature_cols)} features")

    # Check feature names match expected
    missing = set(FEATURES) - set(feature_cols)
    extra = set(feature_cols) - set(FEATURES)

    if missing:
        logger.warning(f"Missing features: {missing}")
    if extra:
        logger.warning(f"Extra features: {extra}")

    return result


def main():
    logger.info(f"Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded {len(df)} entries from {df['レースID'].nunique()} races")

    # Filter to complete entries (with finishing position)
    df = df[df['着順'].notna() & (df['着順'] > 0)]
    logger.info(f"After filtering: {len(df)} entries")

    # Compute jockey/trainer stats first
    df = compute_jockey_trainer_stats(df)

    # Create features
    features_df = create_features(df)

    # Save to parquet
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved features to: {OUTPUT_PATH}")

    # Also save backtest features (same format for now)
    features_df.to_parquet(BACKTEST_PATH, index=False)
    logger.info(f"Saved backtest features to: {BACKTEST_PATH}")

    # Print summary
    logger.info(f"\nFeature summary:")
    logger.info(f"  Total entries: {len(features_df):,}")
    logger.info(f"  Date range: {features_df['レース日付'].min()} to {features_df['レース日付'].max()}")
    logger.info(f"  Races: {features_df['レースID'].nunique():,}")

    return features_df


if __name__ == "__main__":
    main()
