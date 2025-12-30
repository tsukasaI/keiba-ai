"""Tests for feature builder functionality."""

import pytest
import math

from src.scraper.pipeline.feature_builder import FeatureBuilder, HorseFeatures
from src.scraper.parsers.race_card import RaceCardData, HorseEntryData
from src.scraper.parsers.horse import HorseData, PastRaceData
from src.scraper.parsers.jockey import JockeyData
from src.scraper.parsers.trainer import TrainerData


class TestHorseFeatures:
    """Tests for HorseFeatures class."""

    def test_to_array_length(self):
        """Test that to_array returns 23 features."""
        features = HorseFeatures(
            horse_age_num=3.0,
            horse_sex_encoded=0.0,
            post_position_num=1.0,
            weight_carried=55.0,
            horse_weight=480.0,
            jockey_win_rate=0.1,
            jockey_place_rate=0.3,
            trainer_win_rate=0.1,
            jockey_races=100.0,
            trainer_races=100.0,
            distance_num=2000.0,
            is_turf=1.0,
            is_dirt=0.0,
            track_condition_num=0.0,
            avg_position_last_3=5.0,
            avg_position_last_5=6.0,
            win_rate_last_3=0.33,
            win_rate_last_5=0.2,
            place_rate_last_3=0.67,
            place_rate_last_5=0.6,
            last_position=2.0,
            career_races=10.0,
            odds_log=2.3,
        )
        
        arr = features.to_array()
        assert len(arr) == 23

    def test_to_dict_keys(self):
        """Test that to_dict has all required keys."""
        features = HorseFeatures(
            horse_age_num=3.0,
            horse_sex_encoded=0.0,
            post_position_num=1.0,
            weight_carried=55.0,
            horse_weight=480.0,
            jockey_win_rate=0.1,
            jockey_place_rate=0.3,
            trainer_win_rate=0.1,
            jockey_races=100.0,
            trainer_races=100.0,
            distance_num=2000.0,
            is_turf=1.0,
            is_dirt=0.0,
            track_condition_num=0.0,
            avg_position_last_3=5.0,
            avg_position_last_5=6.0,
            win_rate_last_3=0.33,
            win_rate_last_5=0.2,
            place_rate_last_3=0.67,
            place_rate_last_5=0.6,
            last_position=2.0,
            career_races=10.0,
            odds_log=2.3,
        )
        
        d = features.to_dict()
        required_keys = [
            "horse_age_num", "horse_sex_encoded", "post_position_num",
            "weight_carried", "horse_weight", "jockey_win_rate",
            "jockey_place_rate", "trainer_win_rate", "jockey_races",
            "trainer_races", "distance_num", "is_turf", "is_dirt",
            "track_condition_num", "avg_position_last_3", "avg_position_last_5",
            "win_rate_last_3", "win_rate_last_5", "place_rate_last_3",
            "place_rate_last_5", "last_position", "career_races", "odds_log",
        ]
        
        for key in required_keys:
            assert key in d, f"Missing key: {key}"


class TestFeatureBuilder:
    """Tests for FeatureBuilder class."""

    def _create_race_data(self) -> RaceCardData:
        """Create sample race data."""
        return RaceCardData(
            race_id="202506050811",
            race_name="有馬記念",
            race_number=11,
            racecourse="中山",
            racecourse_code="06",
            distance=2500,
            surface="turf",
            track_condition="良",
            race_date="2025-12-28",
        )

    def _create_entry_data(self) -> HorseEntryData:
        """Create sample horse entry."""
        return HorseEntryData(
            horse_id="2019104975",
            horse_name="テスト馬",
            post_position=1,
            horse_age=5,
            horse_sex="牡",
            horse_weight=480.0,
            weight_carried=57.0,
            jockey_id="01170",
            trainer_id="01106",
            win_odds=10.0,
        )

    def test_build_basic(self):
        """Test basic feature building."""
        builder = FeatureBuilder()
        race = self._create_race_data()
        entry = self._create_entry_data()
        
        features = builder.build(race, entry)
        
        assert features.horse_age_num == 5.0
        assert features.horse_sex_encoded == 0.0  # 牡 = 0
        assert features.post_position_num == 1.0
        assert features.weight_carried == 57.0
        assert features.horse_weight == 480.0
        assert features.distance_num == 2500.0
        assert features.is_turf == 1.0
        assert features.is_dirt == 0.0

    def test_build_with_horse_data(self):
        """Test feature building with horse profile."""
        builder = FeatureBuilder()
        race = self._create_race_data()
        entry = self._create_entry_data()
        
        horse = HorseData(
            horse_id="2019104975",
            horse_name="テスト馬",
            birth_date="2019-03-15",
            sex="牡",
            career_races=20,
            career_wins=5,
            past_races=[
                PastRaceData(
                    race_date="2025-11-01",
                    racecourse="東京",
                    race_name="天皇賞秋",
                    distance=2000,
                    surface="turf",
                    track_condition="良",
                    position=1,
                    field_size=18,
                ),
                PastRaceData(
                    race_date="2025-10-01",
                    racecourse="中山",
                    race_name="オールカマー",
                    distance=2200,
                    surface="turf",
                    track_condition="良",
                    position=2,
                    field_size=16,
                ),
                PastRaceData(
                    race_date="2025-06-01",
                    racecourse="東京",
                    race_name="安田記念",
                    distance=1600,
                    surface="turf",
                    track_condition="良",
                    position=3,
                    field_size=18,
                ),
            ],
        )
        
        features = builder.build(race, entry, horse=horse)
        
        assert features.career_races == 20.0
        assert features.avg_position_last_3 == 2.0  # (1+2+3)/3
        assert features.win_rate_last_3 == pytest.approx(1/3)
        assert features.last_position == 1.0

    def test_build_with_jockey_data(self):
        """Test feature building with jockey profile."""
        builder = FeatureBuilder()
        race = self._create_race_data()
        entry = self._create_entry_data()
        
        jockey = JockeyData(
            jockey_id="01170",
            jockey_name="テスト騎手",
            total_races=1000,
            wins=150,
            seconds=100,
            thirds=80,
        )
        
        features = builder.build(race, entry, jockey=jockey)
        
        assert features.jockey_races == 1000.0
        assert features.jockey_win_rate == pytest.approx(0.15)
        assert features.jockey_place_rate == pytest.approx(0.33)

    def test_build_odds_log(self):
        """Test odds log transformation."""
        builder = FeatureBuilder()
        race = self._create_race_data()
        entry = self._create_entry_data()
        entry.win_odds = 10.0
        
        features = builder.build(race, entry)
        
        assert features.odds_log == pytest.approx(math.log(10.0))

    def test_build_default_values(self):
        """Test that missing data uses defaults."""
        builder = FeatureBuilder()
        race = self._create_race_data()
        entry = HorseEntryData(
            horse_id="test",
            horse_name="Test",
            post_position=1,
            horse_age=3,
            horse_sex="牡",
            horse_weight=None,  # Missing
            win_odds=None,  # Missing
        )
        
        features = builder.build(race, entry)
        
        # Should use defaults
        assert features.horse_weight == 480.0
        assert features.odds_log == pytest.approx(math.log(10.0))
