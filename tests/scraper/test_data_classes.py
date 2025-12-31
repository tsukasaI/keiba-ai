"""Tests for scraper data classes and their computed properties."""

import pytest
from datetime import datetime

from src.scraper.parsers.horse import HorseData, PastRaceData
from src.scraper.parsers.jockey import JockeyData
from src.scraper.parsers.trainer import TrainerData
from src.scraper.parsers.race_card import HorseEntryData, RaceCardData


class TestPastRaceData:
    """Test PastRaceData properties."""

    def test_is_win_first_place(self):
        """Position 1 should be a win."""
        race = PastRaceData(
            race_date="2024-01-01",
            racecourse="東京",
            race_name="新春特別",
            distance=2000,
            surface="turf",
            track_condition="良",
            position=1,
            field_size=16,
        )
        assert race.is_win is True
        assert race.is_place is True

    def test_is_win_not_first(self):
        """Position > 1 should not be a win."""
        race = PastRaceData(
            race_date="2024-01-01",
            racecourse="東京",
            race_name="新春特別",
            distance=2000,
            surface="turf",
            track_condition="良",
            position=2,
            field_size=16,
        )
        assert race.is_win is False

    def test_is_place_top_three(self):
        """Positions 1-3 should be places."""
        for pos in [1, 2, 3]:
            race = PastRaceData(
                race_date="2024-01-01",
                racecourse="東京",
                race_name="テスト",
                distance=1600,
                surface="turf",
                track_condition="良",
                position=pos,
                field_size=16,
            )
            assert race.is_place is True, f"Position {pos} should be a place"

    def test_is_place_not_top_three(self):
        """Positions > 3 should not be places."""
        race = PastRaceData(
            race_date="2024-01-01",
            racecourse="東京",
            race_name="テスト",
            distance=1600,
            surface="turf",
            track_condition="良",
            position=4,
            field_size=16,
        )
        assert race.is_place is False


class TestHorseData:
    """Test HorseData properties."""

    @pytest.fixture
    def sample_past_races(self):
        """Create sample past race data."""
        return [
            PastRaceData(
                race_date="2024-12-01",
                racecourse="中山",
                race_name="レース1",
                distance=2000,
                surface="turf",
                track_condition="良",
                position=1,  # Win
                field_size=16,
            ),
            PastRaceData(
                race_date="2024-11-01",
                racecourse="東京",
                race_name="レース2",
                distance=2000,
                surface="turf",
                track_condition="良",
                position=2,  # Place
                field_size=16,
            ),
            PastRaceData(
                race_date="2024-10-01",
                racecourse="中山",
                race_name="レース3",
                distance=2000,
                surface="turf",
                track_condition="良",
                position=5,  # Out of places
                field_size=16,
            ),
            PastRaceData(
                race_date="2024-09-01",
                racecourse="中山",
                race_name="レース4",
                distance=2000,
                surface="turf",
                track_condition="良",
                position=3,  # Place
                field_size=16,
            ),
            PastRaceData(
                race_date="2024-08-01",
                racecourse="中山",
                race_name="レース5",
                distance=2000,
                surface="turf",
                track_condition="良",
                position=1,  # Win
                field_size=16,
            ),
        ]

    def test_horse_age_from_birth_date(self):
        """Age should be calculated from birth date."""
        current_year = datetime.now().year
        horse = HorseData(
            horse_id="2019104975",
            horse_name="テスト馬",
            birth_date=f"{current_year - 4}-04-15",
        )
        assert horse.horse_age == 4

    def test_horse_age_default_when_no_birth_date(self):
        """Default age should be 3 when no birth date."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            birth_date=None,
        )
        assert horse.horse_age == 3

    def test_horse_age_invalid_birth_date(self):
        """Default age should be 3 for invalid birth date format."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            birth_date="invalid-date",
        )
        assert horse.horse_age == 3

    def test_avg_position_last_3(self, sample_past_races):
        """Average position from last 3 races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=sample_past_races,
        )
        # Last 3: positions 1, 2, 5 → avg = 8/3 ≈ 2.67
        assert horse.avg_position_last_3 == pytest.approx(8 / 3, abs=0.01)

    def test_avg_position_last_5(self, sample_past_races):
        """Average position from last 5 races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=sample_past_races,
        )
        # Last 5: positions 1, 2, 5, 3, 1 → avg = 12/5 = 2.4
        assert horse.avg_position_last_5 == pytest.approx(2.4, abs=0.01)

    def test_avg_position_empty_races(self):
        """Default average position when no races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=[],
        )
        assert horse.avg_position_last_3 == 10.0
        assert horse.avg_position_last_5 == 10.0

    def test_win_rate_last_3(self, sample_past_races):
        """Win rate from last 3 races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=sample_past_races,
        )
        # Last 3: 1 win out of 3 → 1/3
        assert horse.win_rate_last_3 == pytest.approx(1 / 3, abs=0.01)

    def test_win_rate_last_5(self, sample_past_races):
        """Win rate from last 5 races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=sample_past_races,
        )
        # Last 5: 2 wins out of 5 → 2/5 = 0.4
        assert horse.win_rate_last_5 == pytest.approx(0.4, abs=0.01)

    def test_win_rate_empty_races(self):
        """Default win rate when no races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=[],
        )
        assert horse.win_rate_last_3 == 0.0
        assert horse.win_rate_last_5 == 0.0

    def test_place_rate_last_3(self, sample_past_races):
        """Place rate from last 3 races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=sample_past_races,
        )
        # Last 3: 2 places (pos 1, 2) out of 3 → 2/3
        assert horse.place_rate_last_3 == pytest.approx(2 / 3, abs=0.01)

    def test_place_rate_last_5(self, sample_past_races):
        """Place rate from last 5 races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=sample_past_races,
        )
        # Last 5: 4 places (pos 1, 2, 3, 1) out of 5 → 4/5 = 0.8
        assert horse.place_rate_last_5 == pytest.approx(0.8, abs=0.01)

    def test_place_rate_empty_races(self):
        """Default place rate when no races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=[],
        )
        assert horse.place_rate_last_3 == 0.0
        assert horse.place_rate_last_5 == 0.0

    def test_last_position(self, sample_past_races):
        """Last position should be from most recent race."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=sample_past_races,
        )
        assert horse.last_position == 1.0

    def test_last_position_empty_races(self):
        """Default last position when no races."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            past_races=[],
        )
        assert horse.last_position == 10.0


class TestJockeyData:
    """Test JockeyData properties."""

    def test_win_rate_with_races(self):
        """Win rate calculation with races."""
        jockey = JockeyData(
            jockey_id="05339",
            jockey_name="ルメール",
            total_races=1000,
            wins=200,
            seconds=150,
            thirds=100,
        )
        assert jockey.win_rate == pytest.approx(0.20, abs=0.01)

    def test_win_rate_no_races(self):
        """Default win rate when no races."""
        jockey = JockeyData(
            jockey_id="test",
            jockey_name="テスト騎手",
            total_races=0,
        )
        assert jockey.win_rate == 0.07  # Default

    def test_place_rate_with_races(self):
        """Place rate calculation with races."""
        jockey = JockeyData(
            jockey_id="05339",
            jockey_name="ルメール",
            total_races=1000,
            wins=200,
            seconds=150,
            thirds=100,
        )
        # (200 + 150 + 100) / 1000 = 0.45
        assert jockey.place_rate == pytest.approx(0.45, abs=0.01)

    def test_place_rate_no_races(self):
        """Default place rate when no races."""
        jockey = JockeyData(
            jockey_id="test",
            jockey_name="テスト騎手",
            total_races=0,
        )
        assert jockey.place_rate == 0.21  # Default

    def test_year_win_rate_with_year_races(self):
        """Year win rate when year races available."""
        jockey = JockeyData(
            jockey_id="05339",
            jockey_name="ルメール",
            total_races=1000,
            wins=200,
            year_races=100,
            year_wins=25,
        )
        assert jockey.year_win_rate == pytest.approx(0.25, abs=0.01)

    def test_year_win_rate_fallback_to_career(self):
        """Year win rate falls back to career when no year races."""
        jockey = JockeyData(
            jockey_id="05339",
            jockey_name="ルメール",
            total_races=1000,
            wins=200,
            year_races=0,
        )
        assert jockey.year_win_rate == pytest.approx(0.20, abs=0.01)

    def test_year_place_rate_with_year_races(self):
        """Year place rate when year races available."""
        jockey = JockeyData(
            jockey_id="05339",
            jockey_name="ルメール",
            total_races=1000,
            wins=200,
            seconds=150,
            thirds=100,
            year_races=100,
            year_wins=20,
            year_seconds=15,
            year_thirds=10,
        )
        # (20 + 15 + 10) / 100 = 0.45
        assert jockey.year_place_rate == pytest.approx(0.45, abs=0.01)

    def test_year_place_rate_fallback_to_career(self):
        """Year place rate falls back to career when no year races."""
        jockey = JockeyData(
            jockey_id="05339",
            jockey_name="ルメール",
            total_races=1000,
            wins=200,
            seconds=150,
            thirds=100,
            year_races=0,
        )
        assert jockey.year_place_rate == pytest.approx(0.45, abs=0.01)


class TestTrainerData:
    """Test TrainerData properties."""

    def test_win_rate_with_races(self):
        """Win rate calculation with races."""
        trainer = TrainerData(
            trainer_id="01061",
            trainer_name="矢作芳人",
            total_races=500,
            wins=75,
            seconds=60,
            thirds=50,
        )
        assert trainer.win_rate == pytest.approx(0.15, abs=0.01)

    def test_win_rate_no_races(self):
        """Default win rate when no races."""
        trainer = TrainerData(
            trainer_id="test",
            trainer_name="テスト調教師",
            total_races=0,
        )
        assert trainer.win_rate == 0.07  # Default

    def test_place_rate_with_races(self):
        """Place rate calculation with races."""
        trainer = TrainerData(
            trainer_id="01061",
            trainer_name="矢作芳人",
            total_races=500,
            wins=75,
            seconds=60,
            thirds=50,
        )
        # (75 + 60 + 50) / 500 = 0.37
        assert trainer.place_rate == pytest.approx(0.37, abs=0.01)

    def test_place_rate_no_races(self):
        """Default place rate when no races."""
        trainer = TrainerData(
            trainer_id="test",
            trainer_name="テスト調教師",
            total_races=0,
        )
        assert trainer.place_rate == 0.21  # Default

    def test_year_win_rate_with_year_races(self):
        """Year win rate when year races available."""
        trainer = TrainerData(
            trainer_id="01061",
            trainer_name="矢作芳人",
            total_races=500,
            wins=75,
            year_races=50,
            year_wins=10,
        )
        assert trainer.year_win_rate == pytest.approx(0.20, abs=0.01)

    def test_year_place_rate_with_year_races(self):
        """Year place rate when year races available."""
        trainer = TrainerData(
            trainer_id="01061",
            trainer_name="矢作芳人",
            total_races=500,
            wins=75,
            seconds=60,
            thirds=50,
            year_races=50,
            year_wins=10,
            year_seconds=8,
            year_thirds=7,
        )
        # (10 + 8 + 7) / 50 = 0.50
        assert trainer.year_place_rate == pytest.approx(0.50, abs=0.01)


class TestHorseEntryData:
    """Test HorseEntryData properties."""

    def test_horse_sex_encoded_male(self):
        """Male horse should encode to 0."""
        entry = HorseEntryData(
            horse_id="test",
            horse_name="テスト馬",
            post_position=1,
            horse_age=3,
            horse_sex="牡",
        )
        assert entry.horse_sex_encoded == 0

    def test_horse_sex_encoded_female(self):
        """Female horse should encode to 1."""
        entry = HorseEntryData(
            horse_id="test",
            horse_name="テスト馬",
            post_position=1,
            horse_age=3,
            horse_sex="牝",
        )
        assert entry.horse_sex_encoded == 1

    def test_horse_sex_encoded_gelding(self):
        """Gelding should encode to 2."""
        entry = HorseEntryData(
            horse_id="test",
            horse_name="テスト馬",
            post_position=1,
            horse_age=3,
            horse_sex="セ",
        )
        assert entry.horse_sex_encoded == 2

    def test_horse_sex_encoded_unknown(self):
        """Unknown sex should encode to 0."""
        entry = HorseEntryData(
            horse_id="test",
            horse_name="テスト馬",
            post_position=1,
            horse_age=3,
            horse_sex="不明",
        )
        assert entry.horse_sex_encoded == 0


class TestRaceCardData:
    """Test RaceCardData properties."""

    def test_is_turf_turf_race(self):
        """Turf race should return is_turf=1."""
        race = RaceCardData(
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
        assert race.is_turf == 1
        assert race.is_dirt == 0

    def test_is_dirt_dirt_race(self):
        """Dirt race should return is_dirt=1."""
        race = RaceCardData(
            race_id="202506050811",
            race_name="フェブラリーS",
            race_number=11,
            racecourse="東京",
            racecourse_code="05",
            distance=1600,
            surface="dirt",
            track_condition="良",
            race_date="2025-02-23",
        )
        assert race.is_turf == 0
        assert race.is_dirt == 1

    def test_track_condition_num_good(self):
        """Good condition should encode to 0."""
        race = RaceCardData(
            race_id="test",
            race_name="テスト",
            race_number=1,
            racecourse="東京",
            racecourse_code="05",
            distance=1600,
            surface="turf",
            track_condition="良",
            race_date="2025-01-01",
        )
        assert race.track_condition_num == 0

    def test_track_condition_num_yielding(self):
        """Yielding condition should encode to 1."""
        race = RaceCardData(
            race_id="test",
            race_name="テスト",
            race_number=1,
            racecourse="東京",
            racecourse_code="05",
            distance=1600,
            surface="turf",
            track_condition="稍重",
            race_date="2025-01-01",
        )
        assert race.track_condition_num == 1

    def test_track_condition_num_soft(self):
        """Soft condition should encode to 2."""
        race = RaceCardData(
            race_id="test",
            race_name="テスト",
            race_number=1,
            racecourse="東京",
            racecourse_code="05",
            distance=1600,
            surface="turf",
            track_condition="重",
            race_date="2025-01-01",
        )
        assert race.track_condition_num == 2

    def test_track_condition_num_heavy(self):
        """Heavy condition should encode to 3."""
        race = RaceCardData(
            race_id="test",
            race_name="テスト",
            race_number=1,
            racecourse="東京",
            racecourse_code="05",
            distance=1600,
            surface="turf",
            track_condition="不良",
            race_date="2025-01-01",
        )
        assert race.track_condition_num == 3

    def test_track_condition_num_unknown(self):
        """Unknown condition should encode to 0."""
        race = RaceCardData(
            race_id="test",
            race_name="テスト",
            race_number=1,
            racecourse="東京",
            racecourse_code="05",
            distance=1600,
            surface="turf",
            track_condition="不明",
            race_date="2025-01-01",
        )
        assert race.track_condition_num == 0
