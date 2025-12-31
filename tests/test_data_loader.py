"""Tests for the RaceDataLoader class."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.models.data_loader import RaceDataLoader
from src.models.config import (
    FEATURES,
    TARGET_COL,
    RACE_ID_COL,
    DATE_COL,
    MISSING_DEFAULTS,
)


class TestRaceDataLoaderInit:
    """Test RaceDataLoader initialization."""

    def test_default_initialization(self):
        """Test initialization with default path."""
        loader = RaceDataLoader()

        assert loader.data_path is not None
        assert "features.parquet" in str(loader.data_path)
        assert loader.df is None
        assert loader.feature_cols == FEATURES

    def test_custom_path_initialization(self):
        """Test initialization with custom path."""
        custom_path = Path("/custom/path/data.parquet")
        loader = RaceDataLoader(data_path=custom_path)

        assert loader.data_path == custom_path


class TestRaceDataLoaderPrepareTarget:
    """Test prepare_target method."""

    @pytest.fixture
    def loader(self):
        return RaceDataLoader()

    def test_convert_positions_to_zero_indexed(self, loader):
        """Positions should be converted to 0-indexed classes."""
        df = pd.DataFrame({TARGET_COL: [1, 2, 3, 10, 18]})
        result = loader.prepare_target(df)

        assert list(result) == [0, 1, 2, 9, 17]

    def test_clip_positions_out_of_range(self, loader):
        """Positions outside 1-18 should be clipped."""
        df = pd.DataFrame({TARGET_COL: [0, 1, 18, 19, 25]})
        result = loader.prepare_target(df)

        assert list(result) == [0, 0, 17, 17, 17]

    def test_handle_string_values(self, loader):
        """String values should be converted correctly."""
        df = pd.DataFrame({TARGET_COL: ["1", "5", "10"]})
        result = loader.prepare_target(df)

        assert list(result) == [0, 4, 9]


class TestRaceDataLoaderFilterValidRaces:
    """Test filter_valid_races method."""

    @pytest.fixture
    def loader(self):
        loader = RaceDataLoader()
        # Use minimal feature set for testing
        loader.feature_cols = ["feature_1", "feature_2"]
        return loader

    def test_remove_missing_targets(self, loader):
        """Rows with missing targets should be removed."""
        df = pd.DataFrame({
            RACE_ID_COL: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            TARGET_COL: [1, 2, 3, 4, 5, 1, 2, None, 4, 5, 6],
            "feature_1": range(11),
            "feature_2": range(11),
        })

        result = loader.filter_valid_races(df)

        # Race 1: 5 horses, Race 2: 5 valid horses (1 null removed)
        assert len(result) == 10  # One row with None removed
        assert result[TARGET_COL].isna().sum() == 0

    def test_remove_small_races(self, loader):
        """Races with fewer than 5 horses should be removed."""
        df = pd.DataFrame({
            RACE_ID_COL: [1, 1, 1, 1, 1, 2, 2, 2],  # Race 2 has only 3 horses
            TARGET_COL: [1, 2, 3, 4, 5, 1, 2, 3],
            "feature_1": range(8),
            "feature_2": range(8),
        })

        result = loader.filter_valid_races(df)

        assert len(result) == 5  # Only race 1 remains
        assert all(result[RACE_ID_COL] == 1)

    def test_update_feature_cols_for_missing(self, loader):
        """Feature columns should be updated if some are missing."""
        loader.feature_cols = ["feature_1", "feature_2", "missing_feature"]

        df = pd.DataFrame({
            RACE_ID_COL: [1, 1, 1, 1, 1],
            TARGET_COL: [1, 2, 3, 4, 5],
            "feature_1": range(5),
            "feature_2": range(5),
        })

        loader.filter_valid_races(df)

        assert "missing_feature" not in loader.feature_cols
        assert "feature_1" in loader.feature_cols
        assert "feature_2" in loader.feature_cols


class TestRaceDataLoaderHandleMissingValues:
    """Test handle_missing_values method."""

    @pytest.fixture
    def loader(self):
        return RaceDataLoader()

    def test_fill_missing_with_defaults(self, loader):
        """Missing values should be filled with defaults."""
        # Get a column that has a default value
        col_with_default = list(MISSING_DEFAULTS.keys())[0]
        default_value = MISSING_DEFAULTS[col_with_default]

        df = pd.DataFrame({
            col_with_default: [1.0, None, 3.0, None, 5.0],
        })

        result = loader.handle_missing_values(df)

        assert result[col_with_default].isna().sum() == 0
        assert result[col_with_default].iloc[1] == default_value
        assert result[col_with_default].iloc[3] == default_value

    def test_no_mutation_of_original(self, loader):
        """Original DataFrame should not be mutated."""
        col_with_default = list(MISSING_DEFAULTS.keys())[0]

        df = pd.DataFrame({
            col_with_default: [1.0, None, 3.0],
        })

        original_null_count = df[col_with_default].isna().sum()
        loader.handle_missing_values(df)

        assert df[col_with_default].isna().sum() == original_null_count


class TestRaceDataLoaderCreateTimeSplits:
    """Test create_time_splits method."""

    @pytest.fixture
    def loader(self):
        return RaceDataLoader()

    @pytest.fixture
    def sample_data(self):
        """Create sample data spanning 12 months."""
        from datetime import datetime, timedelta

        base_date = datetime(2020, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(365)]

        df = pd.DataFrame({
            DATE_COL: dates,
            RACE_ID_COL: range(365),
            TARGET_COL: np.random.randint(1, 19, 365),
            "feature_1": np.random.randn(365),
        })

        return df

    def test_returns_three_dataframes(self, loader, sample_data):
        """Should return train, validation, and test DataFrames."""
        train_df, val_df, test_df = loader.create_time_splits(sample_data)

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_no_overlap_between_splits(self, loader, sample_data):
        """Train, val, and test should have no overlapping dates."""
        train_df, val_df, test_df = loader.create_time_splits(sample_data)

        # Re-parse dates for comparison (they were dropped in the function)
        sample_data["_date"] = pd.to_datetime(sample_data[DATE_COL])
        train_dates = set(sample_data.loc[train_df.index, "_date"])
        val_dates = set(sample_data.loc[val_df.index, "_date"])
        test_dates = set(sample_data.loc[test_df.index, "_date"])

        assert len(train_dates & val_dates) == 0
        assert len(train_dates & test_dates) == 0
        assert len(val_dates & test_dates) == 0

    def test_test_data_is_most_recent(self, loader, sample_data):
        """Test data should be from the most recent period."""
        train_df, val_df, test_df = loader.create_time_splits(sample_data)

        # Test should contain the latest dates
        all_indices = set(sample_data.index)
        test_indices = set(test_df.index)

        # The last few indices should be in test
        max_idx = max(all_indices)
        assert max_idx in test_indices


class TestRaceDataLoaderGetFeaturesAndTarget:
    """Test get_features_and_target method."""

    @pytest.fixture
    def loader(self):
        loader = RaceDataLoader()
        loader.feature_cols = ["feature_1", "feature_2"]
        return loader

    def test_returns_x_and_y(self, loader):
        """Should return feature matrix and target vector."""
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
            TARGET_COL: [1, 2, 3],
        })

        X, y = loader.get_features_and_target(df)

        assert X.shape == (3, 2)
        assert len(y) == 3
        assert list(X.columns) == ["feature_1", "feature_2"]

    def test_target_is_zero_indexed(self, loader):
        """Target should be 0-indexed."""
        df = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
            TARGET_COL: [1, 5, 10],
        })

        _, y = loader.get_features_and_target(df)

        assert list(y) == [0, 4, 9]


class TestRaceDataLoaderGetRaceData:
    """Test get_race_data method."""

    @pytest.fixture
    def loader(self):
        return RaceDataLoader()

    def test_returns_dict_of_dataframes(self, loader):
        """Should return dict mapping race_id to DataFrame."""
        df = pd.DataFrame({
            RACE_ID_COL: [1, 1, 2, 2, 2],
            "horse": ["A", "B", "C", "D", "E"],
        })

        result = loader.get_race_data(df)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 2
        assert len(result[2]) == 3


class TestRaceDataLoaderLoadFeatures:
    """Test load_features method."""

    def test_load_from_parquet(self):
        """Should load data from parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test parquet file
            df = pd.DataFrame({
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
            })
            path = Path(tmpdir) / "test.parquet"
            df.to_parquet(path)

            loader = RaceDataLoader(data_path=path)
            result = loader.load_features()

            assert len(result) == 3
            assert list(result.columns) == ["col1", "col2"]

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError if file doesn't exist."""
        loader = RaceDataLoader(data_path=Path("/nonexistent/path.parquet"))

        with pytest.raises(FileNotFoundError):
            loader.load_features()
