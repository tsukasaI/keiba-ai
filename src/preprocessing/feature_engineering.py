"""
Keiba AI Prediction System - Feature Engineering

This module creates features for horse racing prediction:
1. Basic features (horse, race conditions)
2. Jockey/Trainer statistics (rolling win rates)
3. Running style (from corner passing positions)
4. Aptitude features (distance, surface, track)
5. Past performance features (with time decay)
6. Blood features (sire statistics)
"""

import sys
from pathlib import Path
import logging
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATA_CONFIG,
    MODEL_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for horse racing data."""

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = data_dir
        self.dfs = {}
        self.main_df = None

        # Configuration
        self.start_year = DATA_CONFIG["start_year"]
        self.end_year = DATA_CONFIG["end_year"]
        self.decay_half_life = MODEL_CONFIG["decay_half_life_days"]
        self.past_races = MODEL_CONFIG["past_races_to_consider"]

        # Column mappings (will be detected or set)
        self.col_map = {}

    def load_data(self) -> bool:
        """Load all CSV files from data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            logger.error(f"No CSV files found in {self.data_dir}")
            logger.info("Please download the dataset first:")
            logger.info("  1. Go to: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset")
            logger.info("  2. Download and extract to: data/raw/")
            return False

        logger.info(f"Found {len(csv_files)} CSV files")

        for f in csv_files:
            name = f.stem
            logger.info(f"Loading {name}...")
            self.dfs[name] = pd.read_csv(f)
            logger.info(f"  Shape: {self.dfs[name].shape}")

        return True

    def detect_columns(self) -> None:
        """Auto-detect column names based on patterns."""
        if self.main_df is None:
            return

        columns = self.main_df.columns.tolist()

        # Exact column mappings for this Kaggle dataset
        # These take priority over pattern matching
        exact_mappings = {
            'race_id': 'レースID',
            'race_horse_id': 'レース馬番ID',
            'horse_name': '馬名',
            'jockey': '騎手',
            'trainer': '調教師',
            'date': 'レース日付',
            'position': '着順',
            'odds': '単勝',
            'distance': '距離(m)',
            'surface': '芝・ダート区分',
            'track_condition': '馬場状態1',
            'racecourse': '競馬場コード',
            'racecourse_name': '競馬場名',
            'horse_weight': '馬体重',
            'weight_carried': '斤量',
            'gate_number': '枠番',
            'post_position': '馬番',
            'corner_1': '1コーナー',
            'corner_2': '2コーナー',
            'corner_3': '3コーナー',
            'corner_4': '4コーナー',
            'time': 'タイム',
            'horse_age': '馬齢',
            'horse_sex': '性別',
            'popularity': '人気',
            'last_3f': '上り',
        }

        # Apply exact mappings
        for key, col_name in exact_mappings.items():
            if col_name in columns:
                self.col_map[key] = col_name

        # Fallback pattern matching for any missing columns
        patterns = {
            'race_id': ['race_id', 'raceid', 'レースid'],
            'horse_id': ['horse_id', 'horseid', '馬id'],
            'horse_name': ['馬名'],
            'jockey': ['騎手'],
            'trainer': ['調教師'],
            'date': ['日付', 'race_date'],
            'position': ['着順'],
            'odds': ['単勝'],
            'distance': ['距離'],
            'surface': ['芝・ダート'],
            'track_condition': ['馬場状態'],
            'racecourse': ['競馬場'],
            'horse_weight': ['馬体重'],
            'weight_carried': ['斤量'],
            'post_position': ['馬番'],
            'corner_1': ['1コーナー'],
            'corner_2': ['2コーナー'],
            'corner_3': ['3コーナー'],
            'corner_4': ['4コーナー'],
        }

        for key, pats in patterns.items():
            if key not in self.col_map:
                for col in columns:
                    if any(col == p or col.startswith(p) for p in pats):
                        self.col_map[key] = col
                        break

        logger.info(f"Detected columns: {self.col_map}")

    def identify_main_dataframe(self) -> None:
        """Identify the main results dataframe."""
        # Try to find by name
        for name in self.dfs.keys():
            if any(x in name.lower() for x in ['result', 'race', 'data']):
                self.main_df = self.dfs[name].copy()
                logger.info(f"Using '{name}' as main dataframe")
                return

        # Use largest dataframe
        if self.dfs:
            main_name = max(self.dfs.keys(), key=lambda x: len(self.dfs[x]))
            self.main_df = self.dfs[main_name].copy()
            logger.info(f"Using largest dataframe '{main_name}' as main")

    def filter_by_year(self) -> None:
        """Filter data to target years."""
        if self.main_df is None:
            return

        date_col = self.col_map.get('date')
        if date_col is None:
            # Try to find year from race_id or similar
            logger.warning("No date column found, skipping year filter")
            return

        try:
            if self.main_df[date_col].dtype == 'object':
                self.main_df['_date'] = pd.to_datetime(self.main_df[date_col], errors='coerce')
                self.main_df['_year'] = self.main_df['_date'].dt.year
            else:
                self.main_df['_year'] = self.main_df[date_col]

            before_count = len(self.main_df)
            self.main_df = self.main_df[
                (self.main_df['_year'] >= self.start_year) &
                (self.main_df['_year'] <= self.end_year)
            ].copy()

            logger.info(f"Filtered to {self.start_year}-{self.end_year}: {before_count} -> {len(self.main_df)} rows")
        except Exception as e:
            logger.warning(f"Year filter failed: {e}")

    def create_basic_features(self) -> None:
        """Create basic features from raw data."""
        if self.main_df is None:
            return

        logger.info("Creating basic features...")

        # Horse age
        age_col = self.col_map.get('horse_age')
        if age_col and age_col in self.main_df.columns:
            self.main_df['horse_age_num'] = pd.to_numeric(self.main_df[age_col], errors='coerce')
        else:
            # Fallback: try to find age column
            age_cols = [c for c in self.main_df.columns if c == '馬齢']
            if age_cols:
                self.main_df['horse_age_num'] = pd.to_numeric(self.main_df[age_cols[0]], errors='coerce')

        # Horse sex encoding
        sex_col = self.col_map.get('horse_sex')
        if sex_col and sex_col in self.main_df.columns:
            sex_map = {'牡': 1, '牝': 2, 'セ': 3, 'せん': 3}
            self.main_df['horse_sex_encoded'] = self.main_df[sex_col].map(sex_map).fillna(0).astype(int)
        else:
            sex_cols = [c for c in self.main_df.columns if c == '性別']
            if sex_cols:
                sex_map = {'牡': 1, '牝': 2, 'セ': 3, 'せん': 3}
                self.main_df['horse_sex_encoded'] = self.main_df[sex_cols[0]].map(sex_map).fillna(0).astype(int)

        # Post position (gate number)
        post_col = self.col_map.get('post_position')
        if post_col:
            self.main_df['post_position_num'] = pd.to_numeric(self.main_df[post_col], errors='coerce')

        # Distance category
        dist_col = self.col_map.get('distance')
        if dist_col:
            self.main_df['distance_num'] = pd.to_numeric(self.main_df[dist_col], errors='coerce')
            self.main_df['distance_category'] = pd.cut(
                self.main_df['distance_num'],
                bins=[0, 1400, 1800, 2200, 3000, 4000],
                labels=['sprint', 'mile', 'intermediate', 'long', 'extended']
            )

        # Surface encoding
        surface_col = self.col_map.get('surface')
        if surface_col:
            self.main_df['is_turf'] = self.main_df[surface_col].astype(str).str.contains('芝|turf', case=False).astype(int)
            self.main_df['is_dirt'] = self.main_df[surface_col].astype(str).str.contains('ダート|dirt', case=False).astype(int)

        # Track condition encoding
        condition_col = self.col_map.get('track_condition')
        if condition_col and condition_col in self.main_df.columns:
            condition_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
            self.main_df['track_condition_num'] = self.main_df[condition_col].map(condition_map).fillna(0).astype(int)

        # Odds log transform (odds are often log-normal)
        odds_col = self.col_map.get('odds')
        if odds_col:
            self.main_df['odds_num'] = pd.to_numeric(self.main_df[odds_col], errors='coerce')
            self.main_df['odds_log'] = np.log1p(self.main_df['odds_num'])

    def create_running_style_features(self) -> None:
        """Calculate running style from corner positions."""
        if self.main_df is None:
            return

        logger.info("Creating running style features...")

        corner_cols = []
        for i in range(1, 5):
            col = self.col_map.get(f'corner_{i}')
            if col:
                corner_cols.append(col)
                self.main_df[f'corner_{i}_pos'] = pd.to_numeric(self.main_df[col], errors='coerce')

        if len(corner_cols) >= 2:
            # Average early position (first 2 corners)
            early_cols = [f'corner_{i}_pos' for i in range(1, min(3, len(corner_cols)+1))]
            self.main_df['early_position'] = self.main_df[early_cols].mean(axis=1)

            # Average late position (last 2 corners)
            if len(corner_cols) >= 4:
                late_cols = ['corner_3_pos', 'corner_4_pos']
                self.main_df['late_position'] = self.main_df[late_cols].mean(axis=1)

            # Position change (negative = moving up)
            if 'late_position' in self.main_df.columns:
                self.main_df['position_change'] = self.main_df['late_position'] - self.main_df['early_position']

            # Running style classification
            if 'early_position' in self.main_df.columns:
                conditions = [
                    (self.main_df['early_position'] <= 3),
                    (self.main_df['early_position'] <= 6),
                    (self.main_df['early_position'] <= 10),
                ]
                choices = ['front_runner', 'stalker', 'mid_pack']
                self.main_df['running_style'] = np.select(conditions, choices, default='closer')

    def create_jockey_trainer_features(self) -> None:
        """Calculate jockey and trainer statistics."""
        if self.main_df is None:
            return

        logger.info("Creating jockey/trainer features...")

        position_col = self.col_map.get('position')
        if position_col is None:
            logger.warning("No position column found, skipping jockey/trainer features")
            return

        self.main_df['_position'] = pd.to_numeric(self.main_df[position_col], errors='coerce')
        self.main_df['_is_win'] = (self.main_df['_position'] == 1).astype(int)
        self.main_df['_is_place'] = (self.main_df['_position'] <= 3).astype(int)

        # Sort by date for rolling calculations
        date_col = self.col_map.get('date')
        if date_col and '_date' in self.main_df.columns:
            self.main_df = self.main_df.sort_values('_date')

        # Jockey stats (use transform to keep per-group operations)
        jockey_col = self.col_map.get('jockey')
        if jockey_col:
            # Count races per jockey (before current race)
            self.main_df['jockey_races'] = self.main_df.groupby(jockey_col).cumcount()

            # Cumulative wins shifted by 1 (wins BEFORE current race)
            self.main_df['jockey_wins'] = (
                self.main_df.groupby(jockey_col)['_is_win']
                .transform(lambda x: x.cumsum().shift(1).fillna(0))
            )
            self.main_df['jockey_win_rate'] = (
                self.main_df['jockey_wins'] / self.main_df['jockey_races'].clip(lower=1)
            )

            # Place rate
            self.main_df['jockey_places'] = (
                self.main_df.groupby(jockey_col)['_is_place']
                .transform(lambda x: x.cumsum().shift(1).fillna(0))
            )
            self.main_df['jockey_place_rate'] = (
                self.main_df['jockey_places'] / self.main_df['jockey_races'].clip(lower=1)
            )

        # Trainer stats (use transform to keep per-group operations)
        trainer_col = self.col_map.get('trainer')
        if trainer_col:
            self.main_df['trainer_races'] = self.main_df.groupby(trainer_col).cumcount()
            self.main_df['trainer_wins'] = (
                self.main_df.groupby(trainer_col)['_is_win']
                .transform(lambda x: x.cumsum().shift(1).fillna(0))
            )
            self.main_df['trainer_win_rate'] = (
                self.main_df['trainer_wins'] / self.main_df['trainer_races'].clip(lower=1)
            )

    def create_past_performance_features(self) -> None:
        """Calculate past performance features for each horse."""
        if self.main_df is None:
            return

        logger.info("Creating past performance features...")

        horse_col = self.col_map.get('horse_id') or self.col_map.get('horse_name')
        position_col = self.col_map.get('position')

        if horse_col is None or position_col is None:
            logger.warning("Required columns not found, skipping past performance features")
            return

        self.main_df['_position'] = pd.to_numeric(self.main_df[position_col], errors='coerce')

        # Sort by horse and date
        sort_cols = [horse_col]
        if '_date' in self.main_df.columns:
            sort_cols.append('_date')
        self.main_df = self.main_df.sort_values(sort_cols)

        # Calculate rolling stats per horse
        for window in [3, 5]:
            # Average finish position
            self.main_df[f'avg_position_last_{window}'] = (
                self.main_df.groupby(horse_col)['_position']
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

            # Win rate
            self.main_df[f'win_rate_last_{window}'] = (
                self.main_df.groupby(horse_col)['_is_win']
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

            # Place rate
            self.main_df[f'place_rate_last_{window}'] = (
                self.main_df.groupby(horse_col)['_is_place']
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

        # Last race position
        self.main_df['last_position'] = self.main_df.groupby(horse_col)['_position'].shift(1)

        # Total career races
        self.main_df['career_races'] = self.main_df.groupby(horse_col).cumcount()

    def create_aptitude_features(self) -> None:
        """Calculate aptitude features (distance, surface, track)."""
        if self.main_df is None:
            return

        logger.info("Creating aptitude features...")

        horse_col = self.col_map.get('horse_id') or self.col_map.get('horse_name')
        position_col = self.col_map.get('position')

        if horse_col is None:
            return

        self.main_df['_position'] = pd.to_numeric(self.main_df[position_col], errors='coerce')

        # Distance aptitude
        if 'distance_category' in self.main_df.columns:
            for cat in ['sprint', 'mile', 'intermediate', 'long']:
                mask = self.main_df['distance_category'] == cat
                self.main_df.loc[mask, f'aptitude_{cat}'] = (
                    self.main_df[mask].groupby(horse_col)['_is_win']
                    .transform(lambda x: x.shift(1).expanding().mean())
                )

        # Surface aptitude
        if 'is_turf' in self.main_df.columns:
            turf_mask = self.main_df['is_turf'] == 1
            self.main_df.loc[turf_mask, 'aptitude_turf'] = (
                self.main_df[turf_mask].groupby(horse_col)['_is_place']
                .transform(lambda x: x.shift(1).expanding().mean())
            )

            dirt_mask = self.main_df['is_dirt'] == 1
            self.main_df.loc[dirt_mask, 'aptitude_dirt'] = (
                self.main_df[dirt_mask].groupby(horse_col)['_is_place']
                .transform(lambda x: x.shift(1).expanding().mean())
            )

        # Racecourse aptitude
        course_col = self.col_map.get('racecourse')
        if course_col:
            self.main_df['aptitude_course'] = (
                self.main_df.groupby([horse_col, course_col])['_is_place']
                .transform(lambda x: x.shift(1).expanding().mean())
            )

    def create_blood_features(self) -> None:
        """Calculate sire/bloodline features.

        Note: The Kaggle dataset does not include sire/dam information.
        This method is kept for future use with JRA-VAN data.
        """
        if self.main_df is None:
            return

        logger.info("Checking for blood features...")

        # Check if we have actual sire data (not just race symbols)
        sire_col = self.col_map.get('sire')

        # Skip if no sire column or if the column is actually race symbols
        if sire_col is None:
            logger.info("No sire data available in this dataset, skipping blood features")
            return

        # Check if this is real sire data or just race symbols
        if 'レース記号' in sire_col:
            logger.info("No sire data available in this dataset, skipping blood features")
            return

        logger.info(f"Creating blood features from column: {sire_col}")

        position_col = self.col_map.get('position')
        if position_col is None:
            return

        self.main_df['_position'] = pd.to_numeric(self.main_df[position_col], errors='coerce')
        self.main_df['_is_win'] = (self.main_df['_position'] == 1).astype(int)

        # Sire win rate
        self.main_df['sire_races'] = self.main_df.groupby(sire_col).cumcount()
        self.main_df['sire_wins'] = (
            self.main_df.groupby(sire_col)['_is_win']
            .cumsum().shift(1).fillna(0)
        )
        self.main_df['sire_win_rate'] = (
            self.main_df['sire_wins'] / self.main_df['sire_races'].clip(lower=1)
        )

    def clean_features(self) -> None:
        """Clean up temporary columns and handle missing values."""
        if self.main_df is None:
            return

        logger.info("Cleaning features...")

        # Remove temporary columns
        temp_cols = [c for c in self.main_df.columns if c.startswith('_')]
        self.main_df = self.main_df.drop(columns=temp_cols, errors='ignore')

        # Fill NaN in engineered features with reasonable defaults
        fill_values = {
            'avg_position_last_3': 10,  # Middle of pack
            'avg_position_last_5': 10,
            'win_rate_last_3': 0,
            'win_rate_last_5': 0,
            'place_rate_last_3': 0,
            'place_rate_last_5': 0,
            'horse_age_num': 3,  # Most common age
        }

        for col, default in fill_values.items():
            if col in self.main_df.columns:
                self.main_df[col] = self.main_df[col].fillna(default)

    def save_features(self, output_path: Optional[Path] = None) -> Path:
        """Save engineered features to parquet file."""
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "features.parquet"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.main_df.to_parquet(output_path, index=False)
        logger.info(f"Features saved to: {output_path}")
        logger.info(f"Shape: {self.main_df.shape}")

        # Also save as CSV for inspection
        csv_path = output_path.with_suffix('.csv')
        self.main_df.head(1000).to_csv(csv_path, index=False)
        logger.info(f"Sample CSV saved to: {csv_path}")

        return output_path

    def run(self) -> Optional[Path]:
        """Run the full feature engineering pipeline."""
        logger.info("="*60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("="*60)

        # Load data
        if not self.load_data():
            return None

        # Identify main dataframe
        self.identify_main_dataframe()
        if self.main_df is None:
            logger.error("Could not identify main dataframe")
            return None

        # Detect columns
        self.detect_columns()

        # Filter by year
        self.filter_by_year()

        # Create features
        self.create_basic_features()
        self.create_running_style_features()
        self.create_jockey_trainer_features()
        self.create_past_performance_features()
        self.create_aptitude_features()
        self.create_blood_features()

        # Clean up
        self.clean_features()

        # Save
        output_path = self.save_features()

        logger.info("="*60)
        logger.info("Feature Engineering Complete!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Total features: {len(self.main_df.columns)}")
        logger.info("="*60)

        return output_path


def main():
    """Run feature engineering from command line."""
    engineer = FeatureEngineer()
    result = engineer.run()

    if result is None:
        logger.error("Feature engineering failed")
        sys.exit(1)

    # Print feature summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)

    df = pd.read_parquet(result)
    print(f"\nTotal samples: {len(df):,}")
    print(f"Total features: {len(df.columns)}")

    print("\nFeature columns:")
    for i, col in enumerate(sorted(df.columns)):
        dtype = df[col].dtype
        null_pct = df[col].isnull().mean() * 100
        print(f"  {i+1:3}. {col:<40} {str(dtype):<15} ({null_pct:.1f}% null)")


if __name__ == "__main__":
    main()
