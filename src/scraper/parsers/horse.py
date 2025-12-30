"""
Keiba AI - Horse Profile Parser

Parse horse profile pages from netkeiba.com to extract past performance data.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PastRaceData:
    """Data for a single past race result."""

    race_date: str  # YYYY-MM-DD
    racecourse: str
    race_name: str
    distance: int
    surface: str  # 'turf' or 'dirt'
    track_condition: str
    position: int  # Finishing position
    field_size: int  # Number of horses
    time: Optional[str] = None  # Race time
    weight_carried: float = 0.0
    jockey_name: Optional[str] = None
    odds: Optional[float] = None
    popularity: Optional[int] = None

    @property
    def is_win(self) -> bool:
        """Check if horse won this race."""
        return self.position == 1

    @property
    def is_place(self) -> bool:
        """Check if horse placed (top 3)."""
        return self.position <= 3


@dataclass
class HorseData:
    """Complete horse profile data."""

    # Identifiers
    horse_id: str
    horse_name: str

    # Basic info
    birth_date: Optional[str] = None  # YYYY-MM-DD
    sex: str = "牡"  # '牡', '牝', 'セ'
    coat_color: Optional[str] = None

    # Pedigree
    sire: Optional[str] = None  # Father
    dam: Optional[str] = None  # Mother
    broodmare_sire: Optional[str] = None  # Mother's father

    # Career stats
    career_races: int = 0
    career_wins: int = 0
    career_places: int = 0  # Top 3 finishes
    total_earnings: float = 0.0  # In yen

    # Past race history
    past_races: list[PastRaceData] = field(default_factory=list)

    @property
    def horse_age(self) -> int:
        """Calculate current age from birth date."""
        if not self.birth_date:
            return 3  # Default

        try:
            birth = datetime.strptime(self.birth_date, "%Y-%m-%d")
            today = datetime.now()
            # In Japan, horse age is calculated from January 1
            return today.year - birth.year
        except ValueError:
            return 3

    @property
    def avg_position_last_3(self) -> float:
        """Average finishing position in last 3 races."""
        recent = self.past_races[:3]
        if not recent:
            return 10.0
        return sum(r.position for r in recent) / len(recent)

    @property
    def avg_position_last_5(self) -> float:
        """Average finishing position in last 5 races."""
        recent = self.past_races[:5]
        if not recent:
            return 10.0
        return sum(r.position for r in recent) / len(recent)

    @property
    def win_rate_last_3(self) -> float:
        """Win rate in last 3 races."""
        recent = self.past_races[:3]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.is_win) / len(recent)

    @property
    def win_rate_last_5(self) -> float:
        """Win rate in last 5 races."""
        recent = self.past_races[:5]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.is_win) / len(recent)

    @property
    def place_rate_last_3(self) -> float:
        """Place rate (top 3) in last 3 races."""
        recent = self.past_races[:3]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.is_place) / len(recent)

    @property
    def place_rate_last_5(self) -> float:
        """Place rate (top 3) in last 5 races."""
        recent = self.past_races[:5]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.is_place) / len(recent)

    @property
    def last_position(self) -> float:
        """Finishing position in most recent race."""
        if self.past_races:
            return float(self.past_races[0].position)
        return 10.0


class HorseParser:
    """
    Parse horse profile HTML from netkeiba.com.

    Example URL: https://db.netkeiba.com/horse/2019104975/
    """

    def parse(self, html: str, horse_id: str) -> HorseData:
        """
        Parse horse profile HTML.

        Args:
            html: HTML content of the horse profile page
            horse_id: Horse ID

        Returns:
            HorseData containing horse info and past races
        """
        soup = BeautifulSoup(html, "lxml")

        # Parse basic info
        horse_data = self._parse_basic_info(soup, horse_id)

        # Parse pedigree
        self._parse_pedigree(soup, horse_data)

        # Parse career stats
        self._parse_career_stats(soup, horse_data)

        # Parse past races
        horse_data.past_races = self._parse_past_races(soup)

        return horse_data

    def _parse_basic_info(self, soup: BeautifulSoup, horse_id: str) -> HorseData:
        """Extract basic horse information."""
        # Horse name
        horse_name = ""
        name_elem = soup.select_one(".horse_title h1, .db_head_name h1")
        if name_elem:
            horse_name = name_elem.get_text(strip=True)
            # Remove any suffix like "競走成績"
            horse_name = re.sub(r"競走成績$", "", horse_name).strip()

        # Birth date, sex, coat color from profile table
        birth_date = None
        sex = "牡"
        coat_color = None

        profile_table = soup.select_one(".db_prof_table, .profile_table")
        if profile_table:
            rows = profile_table.select("tr")
            for row in rows:
                th = row.select_one("th")
                td = row.select_one("td")
                if not th or not td:
                    continue

                header = th.get_text(strip=True)
                value = td.get_text(strip=True)

                if "生年月日" in header:
                    # Parse birth date (format: YYYY年M月D日)
                    if match := re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", value):
                        birth_date = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"

                elif "性別" in header or "毛色" in header:
                    # Parse sex and coat color (format: "牡 栗毛")
                    if match := re.match(r"([牡牝セ])", value):
                        sex = match.group(1)
                    if match := re.search(r"[牡牝セ]\s*(.+)", value):
                        coat_color = match.group(1).strip()

        return HorseData(
            horse_id=horse_id,
            horse_name=horse_name,
            birth_date=birth_date,
            sex=sex,
            coat_color=coat_color,
        )

    def _parse_pedigree(self, soup: BeautifulSoup, horse_data: HorseData) -> None:
        """Extract pedigree information."""
        pedigree_table = soup.select_one(".blood_table, .pedigree_table")
        if not pedigree_table:
            return

        # Sire (father) - first row, first cell
        sire_elem = pedigree_table.select_one("tr:nth-child(1) td:nth-child(1) a")
        if sire_elem:
            horse_data.sire = sire_elem.get_text(strip=True)

        # Dam (mother) - third row, first cell (typically)
        dam_elem = pedigree_table.select_one("tr:nth-child(3) td:nth-child(1) a")
        if dam_elem:
            horse_data.dam = dam_elem.get_text(strip=True)

        # Broodmare sire - dam's father
        bms_elem = pedigree_table.select_one("tr:nth-child(3) td:nth-child(2) a, tr:nth-child(4) td:nth-child(1) a")
        if bms_elem:
            horse_data.broodmare_sire = bms_elem.get_text(strip=True)

    def _parse_career_stats(self, soup: BeautifulSoup, horse_data: HorseData) -> None:
        """Extract career statistics."""
        # Career record (format: "3-2-1-4" means 3 wins, 2 seconds, 1 third, 4 unplaced)
        record_elem = soup.select_one(".db_prof_area_02, .career_record")
        if record_elem:
            record_text = record_elem.get_text(strip=True)
            if match := re.search(r"(\d+)-(\d+)-(\d+)-(\d+)", record_text):
                wins = int(match.group(1))
                seconds = int(match.group(2))
                thirds = int(match.group(3))
                unplaced = int(match.group(4))
                horse_data.career_wins = wins
                horse_data.career_places = wins + seconds + thirds
                horse_data.career_races = wins + seconds + thirds + unplaced

        # Total earnings
        earnings_elem = soup.select_one(".prize, .earnings")
        if earnings_elem:
            earnings_text = earnings_elem.get_text(strip=True)
            if match := re.search(r"([\d,]+)", earnings_text):
                # Remove commas and convert to float
                horse_data.total_earnings = float(match.group(1).replace(",", ""))

    def _parse_past_races(self, soup: BeautifulSoup) -> list[PastRaceData]:
        """Extract past race results."""
        past_races = []

        # Find the race result table
        result_table = soup.select_one(".db_h_race_results, table.nk_tb_common")
        if not result_table:
            return past_races

        # Parse each row (skip header)
        rows = result_table.select("tbody tr, tr")[1:]  # Skip header

        for row in rows:
            try:
                race = self._parse_past_race_row(row)
                if race:
                    past_races.append(race)
            except Exception as e:
                logger.debug(f"Failed to parse past race row: {e}")
                continue

        # Races should be in reverse chronological order (most recent first)
        return past_races

    def _parse_past_race_row(self, row) -> Optional[PastRaceData]:
        """Parse a single past race row."""
        cells = row.select("td")
        if len(cells) < 10:
            return None

        # Date (format: YYYY/MM/DD)
        race_date = ""
        date_text = cells[0].get_text(strip=True)
        if match := re.match(r"(\d{4})/(\d{2})/(\d{2})", date_text):
            race_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

        if not race_date:
            return None

        # Racecourse
        racecourse = cells[1].get_text(strip=True)

        # Race name
        race_name = ""
        race_elem = cells[4].select_one("a")
        if race_elem:
            race_name = race_elem.get_text(strip=True)

        # Distance and surface (format: "芝2500" or "ダ1800")
        distance = 0
        surface = "turf"
        dist_text = cells[14].get_text(strip=True) if len(cells) > 14 else ""
        if not dist_text:
            dist_text = cells[6].get_text(strip=True) if len(cells) > 6 else ""

        if match := re.search(r"(芝|ダ)\s*(\d+)", dist_text):
            surface = "turf" if match.group(1) == "芝" else "dirt"
            distance = int(match.group(2))

        # Track condition
        track_condition = "良"
        cond_text = cells[7].get_text(strip=True) if len(cells) > 7 else ""
        for cond in ["良", "稍重", "重", "不良"]:
            if cond in cond_text:
                track_condition = cond
                break

        # Position and field size
        position = 0
        position_text = cells[11].get_text(strip=True) if len(cells) > 11 else ""
        if not position_text:
            position_text = cells[5].get_text(strip=True) if len(cells) > 5 else ""

        if position_text.isdigit():
            position = int(position_text)
        elif match := re.match(r"(\d+)", position_text):
            position = int(match.group(1))

        if position == 0:
            return None

        # Field size
        field_size = 18
        field_text = cells[6].get_text(strip=True) if len(cells) > 6 else ""
        if match := re.search(r"/(\d+)", field_text):
            field_size = int(match.group(1))

        # Weight carried
        weight_carried = 0.0
        weight_text = cells[13].get_text(strip=True) if len(cells) > 13 else ""
        if match := re.search(r"(\d+(?:\.\d+)?)", weight_text):
            weight_carried = float(match.group(1))

        # Jockey
        jockey_name = None
        jockey_elem = cells[12].select_one("a") if len(cells) > 12 else None
        if jockey_elem:
            jockey_name = jockey_elem.get_text(strip=True)

        # Odds
        odds = None
        odds_text = cells[9].get_text(strip=True) if len(cells) > 9 else ""
        if match := re.search(r"(\d+(?:\.\d+)?)", odds_text):
            odds = float(match.group(1))

        # Popularity
        popularity = None
        pop_text = cells[10].get_text(strip=True) if len(cells) > 10 else ""
        if pop_text.isdigit():
            popularity = int(pop_text)

        return PastRaceData(
            race_date=race_date,
            racecourse=racecourse,
            race_name=race_name,
            distance=distance,
            surface=surface,
            track_condition=track_condition,
            position=position,
            field_size=field_size,
            weight_carried=weight_carried,
            jockey_name=jockey_name,
            odds=odds,
            popularity=popularity,
        )
