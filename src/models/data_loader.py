"""
Keiba AI Prediction System - Data Loader

Load and prepare training data for position prediction model.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import (
    FEATURES,
    TARGET_COL,
    RACE_ID_COL,
    DATE_COL,
    MISSING_DEFAULTS,
    TRAINING_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RaceDataLoader:
    """Load and prepare data for training position prediction model."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            # Default path relative to project root
            self.data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.parquet"
        else:
            self.data_path = Path(data_path)

        self.df = None
        self.feature_cols = FEATURES.copy()

    def load_features(self) -> pd.DataFrame:
        """Load processed features from parquet file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading features from: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")

        return self.df

    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert finishing position to 0-indexed class labels.

        Position 1 -> class 0
        Position 2 -> class 1
        ...
        Position 18 -> class 17
        """
        target = df[TARGET_COL].copy()

        # Convert to numeric, coerce errors
        target = pd.to_numeric(target, errors="coerce")

        # Clip to valid range [1, 18]
        target = target.clip(lower=1, upper=18)

        # Convert to 0-indexed
        target = (target - 1).astype(int)

        return target

    def filter_valid_races(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid races.

        Filters:
        - Races with fewer than 5 horses
        - Rows with missing target values
        - Rows with missing critical features
        """
        initial_count = len(df)

        # Remove rows with missing target
        df = df[df[TARGET_COL].notna()].copy()
        logger.info(f"After removing missing targets: {len(df):,} rows")

        # Remove races with fewer than 5 horses
        race_sizes = df.groupby(RACE_ID_COL).size()
        valid_races = race_sizes[race_sizes >= 5].index
        df = df[df[RACE_ID_COL].isin(valid_races)].copy()
        logger.info(f"After removing small races (<5 horses): {len(df):,} rows")

        # Check for critical features
        missing_features = [f for f in self.feature_cols if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            self.feature_cols = [f for f in self.feature_cols if f in df.columns]

        logger.info(f"Filtered {initial_count - len(df):,} rows ({(initial_count - len(df)) / initial_count * 100:.1f}%)")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with defaults."""
        df = df.copy()

        for col, default in MISSING_DEFAULTS.items():
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    df[col] = df[col].fillna(default)
                    logger.info(f"Filled {null_count:,} nulls in '{col}' with {default}")

        return df

    def create_time_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits based on time.

        Uses last N months for test, remaining for train/validation.
        """
        # Parse dates
        df = df.copy()
        df["_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")

        # Sort by date
        df = df.sort_values("_date")

        # Get date range
        min_date = df["_date"].min()
        max_date = df["_date"].max()
        logger.info(f"Date range: {min_date.date()} to {max_date.date()}")

        # Split: last N months for test
        test_months = TRAINING_CONFIG["test_size_months"]
        test_start = max_date - pd.DateOffset(months=test_months)

        train_val_df = df[df["_date"] < test_start].copy()
        test_df = df[df["_date"] >= test_start].copy()

        # Further split train_val into train and validation (last 20% by time)
        train_val_dates = train_val_df["_date"].unique()
        split_idx = int(len(train_val_dates) * 0.8)
        val_start_date = sorted(train_val_dates)[split_idx]

        train_df = train_val_df[train_val_df["_date"] < val_start_date].copy()
        val_df = train_val_df[train_val_df["_date"] >= val_start_date].copy()

        # Drop temp column
        for d in [train_df, val_df, test_df]:
            d.drop(columns=["_date"], inplace=True)

        logger.info(f"Train: {len(train_df):,} rows")
        logger.info(f"Validation: {len(val_df):,} rows")
        logger.info(f"Test: {len(test_df):,} rows")

        return train_df, val_df, test_df

    def get_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and target vector."""
        X = df[self.feature_cols].copy()
        y = self.prepare_target(df)

        return X, y

    def prepare_training_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline to prepare training data.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Load data
        df = self.load_features()

        # Filter invalid races
        df = self.filter_valid_races(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Time-based splits
        train_df, val_df, test_df = self.create_time_splits(df)

        # Extract features and targets
        X_train, y_train = self.get_features_and_target(train_df)
        X_val, y_val = self.get_features_and_target(val_df)
        X_test, y_test = self.get_features_and_target(test_df)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_race_data(self, df: pd.DataFrame) -> dict:
        """
        Get race-level data for evaluation.

        Returns dict with race_id -> DataFrame mapping.
        """
        races = {}
        for race_id, group in df.groupby(RACE_ID_COL):
            races[race_id] = group.copy()
        return races


def main():
    """Test data loading."""
    loader = RaceDataLoader()

    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data()

    print("\n" + "=" * 60)
    print("DATA LOADING SUMMARY")
    print("=" * 60)
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nFeatures used: {len(loader.feature_cols)}")
    print(f"Feature list: {loader.feature_cols}")
    print("\nTarget distribution (train):")
    print(y_train.value_counts().sort_index().head(10))


if __name__ == "__main__":
    main()
