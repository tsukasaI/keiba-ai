"""Tests for the OddsLoader class."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.models.odds_loader import OddsLoader


class TestOddsLoaderInit:
    """Test OddsLoader initialization."""

    def test_default_initialization(self):
        """Test initialization with default path."""
        loader = OddsLoader()

        assert loader.odds_path is not None
        assert "odds.csv" in str(loader.odds_path)
        assert loader.odds_df is None
        assert loader.exacta_results is None
        assert loader.trifecta_results is None

    def test_custom_path_initialization(self):
        """Test initialization with custom path."""
        custom_path = Path("/custom/path/odds.csv")
        loader = OddsLoader(odds_path=custom_path)

        assert loader.odds_path == custom_path


class TestOddsLoaderLoadOdds:
    """Test load_odds method."""

    @pytest.fixture
    def sample_odds_csv(self):
        """Create sample odds CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "レースID": [202001010101, 202001010102, 202101010101, 201901010101],
                "馬単1_組合せ1": [1, 2, 3, 4],
                "馬単1_組合せ2": [2, 3, 4, 5],
                "馬単1_オッズ": [10.5, 25.3, 15.0, 8.5],
                "三連単1_組合せ1": [1, 2, 3, 4],
                "三連単1_組合せ2": [2, 3, 4, 5],
                "三連単1_組合せ3": [3, 4, 5, 6],
                "三連単1_オッズ": [100.0, 250.0, 150.0, 85.0],
            })

            path = Path(tmpdir) / "odds.csv"
            df.to_csv(path, index=False)

            yield path

    def test_load_filters_by_year(self, sample_odds_csv):
        """Should filter data by year range."""
        loader = OddsLoader(odds_path=sample_odds_csv)
        result = loader.load_odds(start_year=2020, end_year=2020)

        # Only 2020 races should be loaded
        assert len(result) == 2
        assert all(result["year"] == 2020)

    def test_load_stores_in_odds_df(self, sample_odds_csv):
        """Loaded data should be stored in odds_df."""
        loader = OddsLoader(odds_path=sample_odds_csv)
        loader.load_odds(start_year=2019, end_year=2021)

        assert loader.odds_df is not None
        assert len(loader.odds_df) == 4


class TestOddsLoaderGetExactaResults:
    """Test get_exacta_results method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with pre-loaded data."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002, 1003],
            "馬単1_組合せ1": [1, 2, 3],
            "馬単1_組合せ2": [2, 3, 4],
            "馬単1_オッズ": [15.5, 25.0, 8.0],
        })
        return loader

    def test_returns_dataframe(self, loader_with_data):
        """Should return DataFrame with exacta results."""
        result = loader_with_data.get_exacta_results()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_correct_columns(self, loader_with_data):
        """Result should have expected columns."""
        result = loader_with_data.get_exacta_results()

        assert "レースID" in result.columns
        assert "exacta_1st" in result.columns
        assert "exacta_2nd" in result.columns
        assert "exacta_odds" in result.columns

    def test_horse_numbers_are_int(self, loader_with_data):
        """Horse numbers should be converted to int."""
        result = loader_with_data.get_exacta_results()

        assert result["exacta_1st"].dtype == int
        assert result["exacta_2nd"].dtype == int

    def test_removes_missing_data(self):
        """Rows with missing data should be removed."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002, 1003],
            "馬単1_組合せ1": [1, None, 3],
            "馬単1_組合せ2": [2, 3, 4],
            "馬単1_オッズ": [15.5, 25.0, 8.0],
        })

        result = loader.get_exacta_results()

        assert len(result) == 2  # One row removed


class TestOddsLoaderGetTrifectaResults:
    """Test get_trifecta_results method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with pre-loaded data."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002],
            "三連単1_組合せ1": [1, 2],
            "三連単1_組合せ2": [2, 3],
            "三連単1_組合せ3": [3, 4],
            "三連単1_オッズ": [150.0, 250.0],
        })
        return loader

    def test_returns_dataframe(self, loader_with_data):
        """Should return DataFrame with trifecta results."""
        result = loader_with_data.get_trifecta_results()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_correct_columns(self, loader_with_data):
        """Result should have expected columns."""
        result = loader_with_data.get_trifecta_results()

        assert "trifecta_1st" in result.columns
        assert "trifecta_2nd" in result.columns
        assert "trifecta_3rd" in result.columns
        assert "trifecta_odds" in result.columns


class TestOddsLoaderGetQuinellaResults:
    """Test get_quinella_results method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with pre-loaded data."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002],
            "馬連1_組合せ1": [1, 2],
            "馬連1_組合せ2": [2, 3],
            "馬連1_オッズ": [8.5, 12.0],
        })
        return loader

    def test_returns_dataframe(self, loader_with_data):
        """Should return DataFrame with quinella results."""
        result = loader_with_data.get_quinella_results()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "quinella_horse1" in result.columns
        assert "quinella_horse2" in result.columns
        assert "quinella_odds" in result.columns


class TestOddsLoaderGetTrioResults:
    """Test get_trio_results method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with pre-loaded data."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002],
            "三連複1_組合せ1": [1, 2],
            "三連複1_組合せ2": [2, 3],
            "三連複1_組合せ3": [3, 4],
            "三連複1_オッズ": [50.0, 80.0],
        })
        return loader

    def test_returns_dataframe(self, loader_with_data):
        """Should return DataFrame with trio results."""
        result = loader_with_data.get_trio_results()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "trio_horse1" in result.columns
        assert "trio_horse2" in result.columns
        assert "trio_horse3" in result.columns
        assert "trio_odds" in result.columns


class TestOddsLoaderGetWideResults:
    """Test get_wide_results method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with pre-loaded data."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002],
            "ワイド1_組合せ1": [1, 2],
            "ワイド1_組合せ2": [2, 3],
            "ワイド1_オッズ": [2.5, 3.0],
            "ワイド2_組合せ1": [1, 2],
            "ワイド2_組合せ2": [3, 4],
            "ワイド2_オッズ": [3.5, 4.0],
        })
        return loader

    def test_returns_dataframe(self, loader_with_data):
        """Should return DataFrame with wide results."""
        result = loader_with_data.get_wide_results()

        assert isinstance(result, pd.DataFrame)
        assert "wide_horse1" in result.columns
        assert "wide_horse2" in result.columns
        assert "wide_odds" in result.columns
        assert "wide_rank" in result.columns

    def test_combines_multiple_wide_results(self, loader_with_data):
        """Should combine results from ワイド1 through ワイド7."""
        result = loader_with_data.get_wide_results()

        # 2 races * 2 wide results each = 4 rows
        assert len(result) == 4


class TestOddsLoaderGetWinOdds:
    """Test get_win_odds method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with pre-loaded data."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002],
            "単勝1_馬番": [1, 5],
            "単勝1_オッズ": [3.5, 8.0],
        })
        return loader

    def test_returns_dataframe(self, loader_with_data):
        """Should return DataFrame with win odds."""
        result = loader_with_data.get_win_odds()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "win_1st_horse" in result.columns
        assert "win_1st_odds" in result.columns


class TestOddsLoaderCreateRaceOddsLookup:
    """Test create_race_odds_lookup method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with pre-loaded data."""
        loader = OddsLoader()
        loader.odds_df = pd.DataFrame({
            "レースID": [1001, 1002],
            "馬単1_組合せ1": [1, 2],
            "馬単1_組合せ2": [2, 3],
            "馬単1_オッズ": [15.5, 25.0],
            "三連単1_組合せ1": [1, 2],
            "三連単1_組合せ2": [2, 3],
            "三連単1_組合せ3": [3, 4],
            "三連単1_オッズ": [150.0, 250.0],
            "単勝1_馬番": [1, 2],
            "単勝1_オッズ": [3.5, 5.0],
        })
        return loader

    def test_returns_dict(self, loader_with_data):
        """Should return dictionary."""
        result = loader_with_data.create_race_odds_lookup()

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 1001 in result
        assert 1002 in result

    def test_contains_exacta_data(self, loader_with_data):
        """Lookup should contain exacta data."""
        result = loader_with_data.create_race_odds_lookup()

        assert "exacta" in result[1001]
        exacta = result[1001]["exacta"]
        assert exacta == (1, 2, 15.5)

    def test_contains_trifecta_data(self, loader_with_data):
        """Lookup should contain trifecta data."""
        result = loader_with_data.create_race_odds_lookup()

        assert "trifecta" in result[1001]
        trifecta = result[1001]["trifecta"]
        assert trifecta == (1, 2, 3, 150.0)

    def test_contains_win_data(self, loader_with_data):
        """Lookup should contain win odds data."""
        result = loader_with_data.create_race_odds_lookup()

        assert "win" in result[1001]
        win = result[1001]["win"]
        assert win == (1, 3.5)


class TestOddsLoaderMergeWithFeatures:
    """Test merge_with_features method."""

    @pytest.fixture
    def loader_with_data(self):
        """Create loader with exacta results."""
        loader = OddsLoader()
        loader.exacta_results = pd.DataFrame({
            "レースID": [1001, 1002],
            "exacta_1st": [1, 2],
            "exacta_2nd": [2, 3],
            "exacta_odds": [15.5, 25.0],
        })
        return loader

    def test_merges_on_race_id(self, loader_with_data):
        """Should merge features with exacta results on race ID."""
        features_df = pd.DataFrame({
            "レースID": [1001, 1001, 1002, 1002, 1003],
            "horse": ["A", "B", "C", "D", "E"],
        })

        result = loader_with_data.merge_with_features(features_df)

        assert len(result) == 5
        assert "exacta_odds" in result.columns

    def test_left_join_preserves_all_features(self, loader_with_data):
        """Left join should preserve all feature rows."""
        features_df = pd.DataFrame({
            "レースID": [1001, 9999],  # 9999 has no odds
            "horse": ["A", "B"],
        })

        result = loader_with_data.merge_with_features(features_df)

        assert len(result) == 2
        assert pd.isna(result.loc[result["レースID"] == 9999, "exacta_odds"].iloc[0])
