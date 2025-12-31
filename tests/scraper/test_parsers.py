"""Tests for scraper HTML parsers."""

import pytest

from src.scraper.parsers.horse import HorseParser
from src.scraper.parsers.jockey import JockeyParser
from src.scraper.parsers.trainer import TrainerParser
from src.scraper.parsers.race_card import RaceCardParser


class TestHorseParser:
    """Test HorseParser HTML parsing."""

    @pytest.fixture
    def parser(self):
        return HorseParser()

    @pytest.fixture
    def sample_horse_html(self):
        """Minimal HTML for horse profile parsing."""
        return """
        <html>
        <head><title>ドウデュース競走成績 | netkeiba</title></head>
        <body>
        <div class="horse_title"><h1>ドウデュース</h1></div>
        <table class="db_prof_table">
            <tr><th>生年月日</th><td>2019年5月7日</td></tr>
            <tr><th>性別</th><td>牡 鹿毛</td></tr>
        </table>
        <table class="blood_table">
            <tr><td><a>ハーツクライ</a></td><td><a>ダストアンドダイヤモンズ</a></td></tr>
            <tr><td></td><td></td></tr>
            <tr><td><a>ダストアンドダイヤモンズ</a></td><td><a>ヴィンディケイション</a></td></tr>
        </table>
        <div class="db_prof_area_02">通算成績: 12-2-1-3</div>
        <table class="db_h_race_results">
            <tbody>
                <tr>
                    <td>2024/12/22</td>
                    <td>中山</td>
                    <td></td>
                    <td></td>
                    <td><a>有馬記念</a></td>
                    <td>1/16</td>
                    <td>芝2500</td>
                    <td>良</td>
                    <td></td>
                    <td>3.5</td>
                    <td>1</td>
                    <td>1</td>
                    <td><a>武豊</a></td>
                    <td>57.0</td>
                    <td>芝2500</td>
                </tr>
            </tbody>
        </table>
        </body>
        </html>
        """

    def test_parse_horse_name(self, parser, sample_horse_html):
        """Parse horse name from HTML."""
        result = parser.parse(sample_horse_html, "2019104975")
        assert result.horse_name == "ドウデュース"

    def test_parse_horse_id(self, parser, sample_horse_html):
        """Horse ID should be preserved."""
        result = parser.parse(sample_horse_html, "2019104975")
        assert result.horse_id == "2019104975"

    def test_parse_birth_date(self, parser, sample_horse_html):
        """Parse birth date from profile table."""
        result = parser.parse(sample_horse_html, "2019104975")
        assert result.birth_date == "2019-05-07"

    def test_parse_sex(self, parser, sample_horse_html):
        """Parse horse sex from profile table."""
        result = parser.parse(sample_horse_html, "2019104975")
        assert result.sex == "牡"

    def test_parse_coat_color(self, parser, sample_horse_html):
        """Parse coat color from profile table."""
        result = parser.parse(sample_horse_html, "2019104975")
        assert result.coat_color == "鹿毛"

    def test_parse_sire(self, parser, sample_horse_html):
        """Parse sire (father) from pedigree table."""
        result = parser.parse(sample_horse_html, "2019104975")
        assert result.sire == "ハーツクライ"

    def test_parse_career_stats(self, parser, sample_horse_html):
        """Parse career statistics."""
        result = parser.parse(sample_horse_html, "2019104975")
        # "12-2-1-3" = 12 wins, 2 seconds, 1 third, 3 unplaced
        assert result.career_wins == 12
        assert result.career_places == 15  # 12 + 2 + 1
        assert result.career_races == 18  # 12 + 2 + 1 + 3

    def test_parse_past_races_empty_table(self, parser):
        """Handle empty past race table gracefully."""
        html = """
        <html><body>
        <div class="horse_title"><h1>テスト馬</h1></div>
        <table class="db_h_race_results"><tbody></tbody></table>
        </body></html>
        """
        result = parser.parse(html, "test")
        assert result.past_races == []

    def test_parse_empty_html(self, parser):
        """Handle minimal/empty HTML gracefully."""
        result = parser.parse("<html><body></body></html>", "test")
        assert result.horse_id == "test"
        assert result.horse_name == ""
        assert result.past_races == []


class TestJockeyParser:
    """Test JockeyParser HTML parsing."""

    @pytest.fixture
    def parser(self):
        return JockeyParser()

    @pytest.fixture
    def sample_jockey_html(self):
        """Minimal HTML for jockey profile parsing."""
        return """
        <html>
        <head><title>C.ルメール | netkeiba</title></head>
        <body>
        <div class="Name"><span class="Name_En">Christophe Lemaire</span></div>
        <table class="ResultsByYears">
            <thead><tr><th>年度</th><th>順位</th><th>1着</th><th>2着</th><th>3着</th><th>4着〜</th><th>騎乗回数</th></tr></thead>
            <tbody>
                <tr>
                    <td>2025年</td>
                    <td>1</td>
                    <td>25</td>
                    <td>20</td>
                    <td>15</td>
                    <td>40</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td>累計</td>
                    <td>-</td>
                    <td>1500</td>
                    <td>1000</td>
                    <td>800</td>
                    <td>2000</td>
                    <td>5300</td>
                </tr>
            </tbody>
        </table>
        </body>
        </html>
        """

    def test_parse_jockey_id(self, parser, sample_jockey_html):
        """Jockey ID should be preserved."""
        result = parser.parse(sample_jockey_html, "05339")
        assert result.jockey_id == "05339"

    def test_parse_career_stats(self, parser, sample_jockey_html):
        """Parse career statistics from 累計 row."""
        result = parser.parse(sample_jockey_html, "05339")
        assert result.total_races == 5300
        assert result.wins == 1500
        assert result.seconds == 1000
        assert result.thirds == 800

    def test_parse_year_stats(self, parser, sample_jockey_html):
        """Parse current year statistics."""
        result = parser.parse(sample_jockey_html, "05339")
        assert result.year_races == 100
        assert result.year_wins == 25
        assert result.year_seconds == 20
        assert result.year_thirds == 15

    def test_parse_empty_html(self, parser):
        """Handle minimal/empty HTML gracefully."""
        result = parser.parse("<html><body></body></html>", "test")
        assert result.jockey_id == "test"
        assert result.total_races == 0

    def test_win_rate_after_parse(self, parser, sample_jockey_html):
        """Win rate should be calculable after parsing."""
        result = parser.parse(sample_jockey_html, "05339")
        # 1500 / 5300 ≈ 0.283
        assert result.win_rate == pytest.approx(0.283, abs=0.01)


class TestTrainerParser:
    """Test TrainerParser HTML parsing."""

    @pytest.fixture
    def parser(self):
        return TrainerParser()

    @pytest.fixture
    def sample_trainer_html(self):
        """Minimal HTML for trainer profile parsing."""
        return """
        <html>
        <head><title>矢作芳人 | netkeiba</title></head>
        <body>
        <div class="Name"><span class="Name_En">Yoshito Yahagi</span></div>
        <table class="ResultsByYears">
            <thead><tr><th>年度</th><th>順位</th><th>1着</th><th>2着</th><th>3着</th><th>4着〜</th><th>出走回数</th></tr></thead>
            <tbody>
                <tr>
                    <td>2025年</td>
                    <td>1</td>
                    <td>30</td>
                    <td>25</td>
                    <td>20</td>
                    <td>75</td>
                    <td>150</td>
                </tr>
                <tr>
                    <td>累計</td>
                    <td>-</td>
                    <td>800</td>
                    <td>600</td>
                    <td>500</td>
                    <td>1600</td>
                    <td>3500</td>
                </tr>
            </tbody>
        </table>
        </body>
        </html>
        """

    def test_parse_trainer_id(self, parser, sample_trainer_html):
        """Trainer ID should be preserved."""
        result = parser.parse(sample_trainer_html, "01061")
        assert result.trainer_id == "01061"

    def test_parse_career_stats(self, parser, sample_trainer_html):
        """Parse career statistics from 累計 row."""
        result = parser.parse(sample_trainer_html, "01061")
        assert result.total_races == 3500
        assert result.wins == 800
        assert result.seconds == 600
        assert result.thirds == 500

    def test_parse_year_stats(self, parser, sample_trainer_html):
        """Parse current year statistics."""
        result = parser.parse(sample_trainer_html, "01061")
        assert result.year_races == 150
        assert result.year_wins == 30

    def test_place_rate_after_parse(self, parser, sample_trainer_html):
        """Place rate should be calculable after parsing."""
        result = parser.parse(sample_trainer_html, "01061")
        # (800 + 600 + 500) / 3500 ≈ 0.543
        assert result.place_rate == pytest.approx(0.543, abs=0.01)


class TestRaceCardParser:
    """Test RaceCardParser HTML parsing."""

    @pytest.fixture
    def parser(self):
        return RaceCardParser()

    @pytest.fixture
    def sample_race_card_html(self):
        """Minimal HTML for race card parsing."""
        return """
        <html>
        <head><title>出馬表 | netkeiba</title></head>
        <body>
        <div class="RaceNum">11R</div>
        <div class="RaceName">有馬記念</div>
        <div class="RaceData01">2025/12/28 芝2500m 良</div>
        <div class="RaceData02"><span>5回中山8日</span></div>
        <div class="Item03">良</div>
        <table class="Shutuba_Table">
            <tbody>
                <tr class="HorseList">
                    <td class="Waku">1</td>
                    <td class="Umaban">1</td>
                    <td></td>
                    <td class="HorseInfo"><a href="/horse/2019104975/">ドウデュース</a></td>
                    <td class="Barei">牡5</td>
                    <td>57.0</td>
                    <td class="Jockey"><a href="/jockey/05339/">ルメール</a></td>
                    <td class="Trainer"><a href="/trainer/01061/">友道康夫</a></td>
                    <td class="Weight">506(+2)</td>
                    <td><span id="odds-1_01">3.5</span></td>
                    <td><span id="ninki-1_01">1</span></td>
                </tr>
                <tr class="HorseList">
                    <td class="Waku">2</td>
                    <td class="Umaban">2</td>
                    <td></td>
                    <td class="HorseInfo"><a href="/horse/2020104567/">リバティアイランド</a></td>
                    <td class="Barei">牝4</td>
                    <td>55.0</td>
                    <td class="Jockey"><a href="/jockey/01234/">川田将雅</a></td>
                    <td class="Trainer"><a href="/trainer/02345/">中内田充正</a></td>
                    <td class="Weight">468(-4)</td>
                    <td><span id="odds-1_02">5.2</span></td>
                    <td><span id="ninki-1_02">2</span></td>
                </tr>
            </tbody>
        </table>
        </body>
        </html>
        """

    def test_parse_race_id(self, parser, sample_race_card_html):
        """Race ID should be preserved."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.race_id == "202506050811"

    def test_parse_race_name(self, parser, sample_race_card_html):
        """Parse race name from header."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.race_name == "有馬記念"

    def test_parse_race_number(self, parser, sample_race_card_html):
        """Parse race number from header."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.race_number == 11

    def test_parse_distance_and_surface(self, parser, sample_race_card_html):
        """Parse distance and surface from race data."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.distance == 2500
        assert result.surface == "turf"

    def test_parse_track_condition(self, parser, sample_race_card_html):
        """Parse track condition."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.track_condition == "良"

    def test_parse_racecourse_code(self, parser, sample_race_card_html):
        """Parse racecourse code from race ID."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.racecourse_code == "06"

    def test_parse_race_date(self, parser, sample_race_card_html):
        """Parse race date from header."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.race_date == "2025-12-28"

    def test_parse_entries_count(self, parser, sample_race_card_html):
        """Parse correct number of entries."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert len(result.entries) == 2

    def test_parse_first_entry(self, parser, sample_race_card_html):
        """Parse first entry details."""
        result = parser.parse(sample_race_card_html, "202506050811")
        entry = result.entries[0]
        assert entry.horse_name == "ドウデュース"
        assert entry.horse_id == "2019104975"
        assert entry.post_position == 1
        # Age/sex parsing depends on exact HTML format - test basic fields
        assert entry.jockey_id == "05339"
        assert entry.jockey_name == "ルメール"
        assert entry.horse_weight == 506
        assert entry.weight_change == 2
        assert entry.win_odds == 3.5
        assert entry.popularity == 1

    def test_parse_second_entry(self, parser, sample_race_card_html):
        """Parse second entry basic fields."""
        result = parser.parse(sample_race_card_html, "202506050811")
        entry = result.entries[1]
        assert entry.horse_name == "リバティアイランド"
        assert entry.post_position == 2
        assert entry.win_odds == 5.2
        assert entry.popularity == 2

    def test_parse_empty_html(self, parser):
        """Handle minimal/empty HTML gracefully."""
        result = parser.parse("<html><body></body></html>", "test")
        assert result.race_id == "test"
        assert result.entries == []

    def test_is_turf_after_parse(self, parser, sample_race_card_html):
        """is_turf property should work after parsing."""
        result = parser.parse(sample_race_card_html, "202506050811")
        assert result.is_turf == 1
        assert result.is_dirt == 0
