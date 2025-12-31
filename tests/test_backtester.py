"""Tests for the Backtester class."""

import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.models.backtester import Backtester
from src.models.types import BacktestResults, BetResult
from src.models.config import RACE_ID_COL, DATE_COL, FEATURES


class TestBacktesterInit:
    """Test Backtester initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        backtester = Backtester()

        assert backtester.ev_threshold == 1.0
        assert backtester.bet_unit == 100
        assert backtester.max_bets_per_race == 3
        assert backtester.calibration_method is None
        assert backtester.filter_segments is False
        assert backtester.calibrator is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_filters = {"surface": ["turf"], "distance_category": ["sprint"]}

        backtester = Backtester(
            ev_threshold=1.2,
            bet_unit=200,
            max_bets_per_race=5,
            calibration_method="temperature",
            filter_segments=True,
            custom_filters=custom_filters,
        )

        assert backtester.ev_threshold == 1.2
        assert backtester.bet_unit == 200
        assert backtester.max_bets_per_race == 5
        assert backtester.calibration_method == "temperature"
        assert backtester.filter_segments is True
        assert backtester.segment_filters == custom_filters


class TestGetRaceMetadata:
    """Test _get_race_metadata method."""

    @pytest.fixture
    def backtester(self):
        return Backtester()

    def test_turf_race_metadata(self, backtester):
        """Test metadata extraction for turf race."""
        race_df = pd.DataFrame([{
            "is_turf": 1,
            "distance_num": 1600,
            "競争条件": "G1 天皇賞",
            "track_condition_num": 0,
            "競馬場名": "東京",
            DATE_COL: "2021-04-25",
        }])

        meta = backtester._get_race_metadata(race_df)

        assert meta["surface"] == "turf"
        assert meta["distance_category"] == "mile"
        assert meta["race_grade"] == "graded"
        assert meta["track_condition"] == "good"
        assert meta["racecourse"] == "東京"
        assert meta["race_date"] == "2021-04-25"

    def test_dirt_race_metadata(self, backtester):
        """Test metadata extraction for dirt race."""
        race_df = pd.DataFrame([{
            "is_turf": 0,
            "distance_num": 1200,
            "競争条件": "OP オープン特別",
            "track_condition_num": 2,
            "競馬場名": "中山",
            DATE_COL: "2021-05-01",
        }])

        meta = backtester._get_race_metadata(race_df)

        assert meta["surface"] == "dirt"
        assert meta["distance_category"] == "sprint"
        assert meta["race_grade"] == "open"
        assert meta["track_condition"] == "soft"
        assert meta["racecourse"] == "中山"

    def test_distance_categories(self, backtester):
        """Test different distance categories."""
        test_cases = [
            (1200, "sprint"),   # <= 1400
            (1400, "sprint"),   # <= 1400
            (1600, "mile"),     # <= 1800
            (1800, "mile"),     # <= 1800
            (2000, "intermediate"),  # <= 2200
            (2200, "intermediate"),  # <= 2200
            (2400, "long"),     # > 2200
            (3200, "long"),     # > 2200
        ]

        for distance, expected_cat in test_cases:
            race_df = pd.DataFrame([{
                "is_turf": 1,
                "distance_num": distance,
                "競争条件": "条件戦",
                "track_condition_num": 0,
                "競馬場名": "東京",
                DATE_COL: "2021-01-01",
            }])

            meta = backtester._get_race_metadata(race_df)
            assert meta["distance_category"] == expected_cat, f"Expected {expected_cat} for {distance}m"

    def test_race_grades(self, backtester):
        """Test different race grade classifications."""
        test_cases = [
            ("G1 日本ダービー", "graded"),
            ("G2 阪神大賞典", "graded"),
            ("G3 京王杯", "graded"),
            ("オープン特別", "open"),
            ("OP レース", "open"),
            ("新馬戦", "maiden"),
            ("未勝利戦", "maiden"),
            ("2勝クラス", "conditions"),
            ("条件戦", "conditions"),
        ]

        for condition, expected_grade in test_cases:
            race_df = pd.DataFrame([{
                "is_turf": 1,
                "distance_num": 1600,
                "競争条件": condition,
                "track_condition_num": 0,
                "競馬場名": "東京",
                DATE_COL: "2021-01-01",
            }])

            meta = backtester._get_race_metadata(race_df)
            assert meta["race_grade"] == expected_grade, f"Expected {expected_grade} for {condition}"

    def test_track_conditions(self, backtester):
        """Test different track condition mappings."""
        test_cases = [
            (0, "good"),
            (1, "yielding"),
            (2, "soft"),
            (3, "heavy"),
            (99, "good"),  # Unknown maps to good
        ]

        for condition_num, expected in test_cases:
            race_df = pd.DataFrame([{
                "is_turf": 1,
                "distance_num": 1600,
                "競争条件": "条件戦",
                "track_condition_num": condition_num,
                "競馬場名": "東京",
                DATE_COL: "2021-01-01",
            }])

            meta = backtester._get_race_metadata(race_df)
            assert meta["track_condition"] == expected


class TestMatchesSegmentFilter:
    """Test _matches_segment_filter method."""

    def test_filter_disabled_returns_true(self):
        """When filter_segments is False, should always return True."""
        backtester = Backtester(filter_segments=False)
        race_meta = {"surface": "anything", "distance_category": "anything"}

        assert backtester._matches_segment_filter(race_meta) is True

    def test_filter_enabled_matches_surface(self):
        """When filter matches surface, should return True."""
        backtester = Backtester(
            filter_segments=True,
            custom_filters={"surface": ["turf"]},
        )

        race_meta = {"surface": "turf", "distance_category": "mile"}
        assert backtester._matches_segment_filter(race_meta) is True

        race_meta = {"surface": "dirt", "distance_category": "mile"}
        assert backtester._matches_segment_filter(race_meta) is False

    def test_filter_enabled_matches_distance(self):
        """When filter matches distance_category, should return True."""
        backtester = Backtester(
            filter_segments=True,
            custom_filters={"distance_category": ["sprint", "mile"]},
        )

        race_meta = {"surface": "dirt", "distance_category": "sprint"}
        assert backtester._matches_segment_filter(race_meta) is True

        race_meta = {"surface": "turf", "distance_category": "long"}
        assert backtester._matches_segment_filter(race_meta) is False

    def test_filter_enabled_multiple_criteria(self):
        """Filter matches if ANY criterion is satisfied."""
        backtester = Backtester(
            filter_segments=True,
            custom_filters={
                "surface": ["turf"],
                "race_grade": ["graded"],
            },
        )

        # Matches surface
        race_meta = {"surface": "turf", "race_grade": "conditions"}
        assert backtester._matches_segment_filter(race_meta) is True

        # Matches grade
        race_meta = {"surface": "dirt", "race_grade": "graded"}
        assert backtester._matches_segment_filter(race_meta) is True

        # Matches neither
        race_meta = {"surface": "dirt", "race_grade": "conditions"}
        assert backtester._matches_segment_filter(race_meta) is False


class TestPrepareTarget:
    """Test _prepare_target method."""

    @pytest.fixture
    def backtester(self):
        return Backtester()

    def test_convert_positions_to_zero_indexed(self, backtester):
        """Positions should be converted to 0-indexed classes."""
        target = pd.Series([1, 2, 3, 10, 18])
        result = backtester._prepare_target(target)

        expected = pd.Series([0, 1, 2, 9, 17])
        pd.testing.assert_series_equal(result, expected)

    def test_clip_positions_out_of_range(self, backtester):
        """Positions outside 1-18 should be clipped."""
        target = pd.Series([0, 1, 18, 19, 25])
        result = backtester._prepare_target(target)

        # 0 -> 1 (clipped) -> 0; 19 -> 18 (clipped) -> 17
        expected = pd.Series([0, 0, 17, 17, 17])
        pd.testing.assert_series_equal(result, expected)

    def test_handle_string_numeric_values(self, backtester):
        """String numeric values should be converted correctly."""
        target = pd.Series(["1", "2", "5", "10"])
        result = backtester._prepare_target(target)

        assert len(result) == 4
        assert result.iloc[0] == 0
        assert result.iloc[1] == 1
        assert result.iloc[2] == 4
        assert result.iloc[3] == 9


class TestRunBacktest:
    """Test run_backtest method."""

    @pytest.fixture
    def backtester(self):
        return Backtester(max_bets_per_race=2)

    @pytest.fixture
    def sample_test_df(self):
        """Create sample test dataframe."""
        # Create data for 2 races, 3 horses each
        data = []
        for race_id in [1001, 1002]:
            for horse_num in range(1, 4):
                row = {
                    RACE_ID_COL: race_id,
                    "馬番": horse_num,
                    "is_turf": 1,
                    "distance_num": 1600,
                    "競争条件": "条件戦",
                    "track_condition_num": 0,
                    "競馬場名": "東京",
                    DATE_COL: "2021-01-01",
                }
                # Add feature columns
                for feat in FEATURES:
                    row[feat] = np.random.random()
                data.append(row)

        return pd.DataFrame(data)

    @pytest.fixture
    def mock_model(self):
        """Create mock position probability model."""
        model = Mock()
        # Return consistent position probabilities
        def predict_race(X, horse_names):
            return {
                name: {str(pos): 1.0/len(horse_names) for pos in range(1, len(horse_names)+1)}
                for name in horse_names
            }
        model.predict_race = Mock(side_effect=predict_race)
        return model

    def test_backtest_with_no_odds_skips_race(self, backtester, sample_test_df, mock_model):
        """Races without odds data should be skipped."""
        odds_lookup = {}  # Empty odds

        results = backtester.run_backtest(sample_test_df, mock_model, odds_lookup)

        assert results.num_bets == 0
        assert results.total_bet == 0

    def test_backtest_with_odds_places_bets(self, backtester, sample_test_df, mock_model):
        """Races with odds data should have bets placed."""
        # Mock exacta calculator to return high probability predictions
        with patch.object(backtester.exacta_calc, 'calculate_exacta_probs') as mock_calc, \
             patch.object(backtester.exacta_calc, 'get_top_exactas') as mock_top:

            mock_calc.return_value = {("1", "2"): 0.1, ("2", "1"): 0.08}
            mock_top.return_value = [(("1", "2"), 0.1), (("2", "1"), 0.08)]

            odds_lookup = {
                1001: {"exacta": (1, 2, 3560)},  # actual_1st=1, actual_2nd=2, odds=3560
                1002: {"exacta": (2, 3, 2400)},
            }

            results = backtester.run_backtest(sample_test_df, mock_model, odds_lookup)

            # Should have bets for races with odds
            assert results.num_bets > 0

    def test_backtest_winning_bet_calculates_payout(self, backtester, sample_test_df, mock_model):
        """Winning bets should calculate payout correctly."""
        with patch.object(backtester.exacta_calc, 'calculate_exacta_probs') as mock_calc, \
             patch.object(backtester.exacta_calc, 'get_top_exactas') as mock_top:

            # Predict (1, 2) with high probability
            mock_calc.return_value = {("1", "2"): 0.15}
            mock_top.return_value = [(("1", "2"), 0.15)]

            odds_lookup = {
                1001: {"exacta": (1, 2, 3560)},  # Winning combination
            }

            # Filter to single race
            single_race_df = sample_test_df[sample_test_df[RACE_ID_COL] == 1001]

            results = backtester.run_backtest(single_race_df, mock_model, odds_lookup)

            # Should have 1 winning bet
            assert results.num_wins == 1
            assert results.total_return == 3560  # Payout = odds for winning bet


class TestGetStratifiedResults:
    """Test get_stratified_results method."""

    @pytest.fixture
    def backtester(self):
        return Backtester()

    @pytest.fixture
    def sample_results(self):
        """Create sample backtest results with varied stratification fields."""
        results = BacktestResults()

        # Add bets with different surfaces
        results.bets.append(BetResult(
            race_id=1001, race_date="2021-01-01",
            predicted_1st="1", predicted_2nd="2",
            actual_1st=1, actual_2nd=2,
            predicted_prob=0.1, actual_odds=3560,
            expected_value=1.5, bet_amount=100,
            payout=3560, profit=3460, won=True,
            surface="turf", distance_category="mile",
            race_grade="graded", track_condition="good",
            racecourse="東京",
        ))

        results.bets.append(BetResult(
            race_id=1002, race_date="2021-01-02",
            predicted_1st="2", predicted_2nd="3",
            actual_1st=1, actual_2nd=3,
            predicted_prob=0.08, actual_odds=0,
            expected_value=0, bet_amount=100,
            payout=0, profit=-100, won=False,
            surface="dirt", distance_category="sprint",
            race_grade="maiden", track_condition="soft",
            racecourse="中山",
        ))

        results.bets.append(BetResult(
            race_id=1003, race_date="2021-01-03",
            predicted_1st="3", predicted_2nd="1",
            actual_1st=3, actual_2nd=1,
            predicted_prob=0.05, actual_odds=8900,
            expected_value=1.2, bet_amount=100,
            payout=8900, profit=8800, won=True,
            surface="turf", distance_category="long",
            race_grade="graded", track_condition="good",
            racecourse="東京",
        ))

        results.total_bet = 300
        results.total_return = 3560 + 0 + 8900
        results.num_bets = 3
        results.num_wins = 2

        return results

    def test_stratify_by_surface(self, backtester, sample_results):
        """Test stratification by surface."""
        stratified = backtester.get_stratified_results(sample_results)

        assert "surface" in stratified
        assert "turf" in stratified["surface"]
        assert "dirt" in stratified["surface"]

        turf_results = stratified["surface"]["turf"]
        assert turf_results.num_bets == 2
        assert turf_results.num_wins == 2

        dirt_results = stratified["surface"]["dirt"]
        assert dirt_results.num_bets == 1
        assert dirt_results.num_wins == 0

    def test_stratify_by_distance(self, backtester, sample_results):
        """Test stratification by distance category."""
        stratified = backtester.get_stratified_results(sample_results)

        assert "distance_category" in stratified
        assert stratified["distance_category"]["mile"].num_bets == 1
        assert stratified["distance_category"]["sprint"].num_bets == 1
        assert stratified["distance_category"]["long"].num_bets == 1

    def test_stratify_by_grade(self, backtester, sample_results):
        """Test stratification by race grade."""
        stratified = backtester.get_stratified_results(sample_results)

        assert "race_grade" in stratified
        assert stratified["race_grade"]["graded"].num_bets == 2
        assert stratified["race_grade"]["maiden"].num_bets == 1

    def test_stratify_by_track_condition(self, backtester, sample_results):
        """Test stratification by track condition."""
        stratified = backtester.get_stratified_results(sample_results)

        assert "track_condition" in stratified
        assert stratified["track_condition"]["good"].num_bets == 2
        assert stratified["track_condition"]["soft"].num_bets == 1

    def test_stratify_by_racecourse(self, backtester, sample_results):
        """Test stratification by racecourse."""
        stratified = backtester.get_stratified_results(sample_results)

        assert "racecourse" in stratified
        assert stratified["racecourse"]["東京"].num_bets == 2
        assert stratified["racecourse"]["中山"].num_bets == 1

    def test_stratified_totals_match_aggregate(self, backtester, sample_results):
        """Stratified totals should match aggregate results."""
        stratified = backtester.get_stratified_results(sample_results)

        # Sum up all surfaces
        total_bets = sum(r.num_bets for r in stratified["surface"].values())
        total_wins = sum(r.num_wins for r in stratified["surface"].values())
        total_bet = sum(r.total_bet for r in stratified["surface"].values())

        assert total_bets == sample_results.num_bets
        assert total_wins == sample_results.num_wins
        assert total_bet == sample_results.total_bet


class TestBacktestResultsCalculations:
    """Test BacktestResults computed properties."""

    def test_empty_results(self):
        """Empty results should return zeros."""
        results = BacktestResults()

        assert results.profit == 0
        assert results.roi == 0
        assert results.hit_rate == 0
        assert results.avg_odds_hit == 0
        assert results.get_max_drawdown() == 0
        assert results.get_sharpe_ratio() == 0

    def test_roi_calculation(self):
        """ROI should be calculated correctly."""
        results = BacktestResults(
            total_bet=1000,
            total_return=1200,
            num_bets=10,
            num_wins=2,
        )

        assert results.profit == 200
        assert results.roi == 20.0  # (1200-1000)/1000 * 100

    def test_hit_rate_calculation(self):
        """Hit rate should be calculated correctly."""
        results = BacktestResults(
            total_bet=1000,
            total_return=1200,
            num_bets=10,
            num_wins=3,
        )

        assert results.hit_rate == 30.0  # 3/10 * 100

    def test_avg_odds_hit(self):
        """Average odds of winning bets should be calculated correctly."""
        results = BacktestResults()
        results.bets = [
            BetResult(
                race_id=1, race_date="2021-01-01",
                predicted_1st="1", predicted_2nd="2",
                actual_1st=1, actual_2nd=2,
                predicted_prob=0.1, actual_odds=3000,  # 30x
                expected_value=1.5, bet_amount=100,
                payout=3000, profit=2900, won=True,
            ),
            BetResult(
                race_id=2, race_date="2021-01-02",
                predicted_1st="1", predicted_2nd="2",
                actual_1st=2, actual_2nd=1,
                predicted_prob=0.1, actual_odds=0,
                expected_value=0, bet_amount=100,
                payout=0, profit=-100, won=False,
            ),
            BetResult(
                race_id=3, race_date="2021-01-03",
                predicted_1st="1", predicted_2nd="2",
                actual_1st=1, actual_2nd=2,
                predicted_prob=0.1, actual_odds=5000,  # 50x
                expected_value=1.5, bet_amount=100,
                payout=5000, profit=4900, won=True,
            ),
        ]

        # Average of (3000/100, 5000/100) = (30 + 50) / 2 = 40
        assert results.avg_odds_hit == 40.0
