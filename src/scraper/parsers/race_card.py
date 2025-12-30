"""
Keiba AI - Race Card Parser

Parse race card (shutuba.html) pages from netkeiba.com.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup

from ..config import TRACK_CONDITION_MAP, HORSE_SEX_MAP

logger = logging.getLogger(__name__)


@dataclass
class HorseEntryData:
    """Data for a single horse entry in a race."""

    # Identifiers
    horse_id: str
    horse_name: str
    post_position: int  # Gate number (1-18)

    # Horse info
    horse_age: int
    horse_sex: str  # '牡', '牝', 'セ'
    horse_weight: Optional[float] = None  # Body weight in kg
    weight_change: Optional[float] = None  # Weight change from last race

    # Jockey/Trainer IDs
    jockey_id: Optional[str] = None
    jockey_name: Optional[str] = None
    trainer_id: Optional[str] = None
    trainer_name: Optional[str] = None

    # Race conditions
    weight_carried: float = 0.0  # Handicap weight in kg

    # Odds
    win_odds: Optional[float] = None
    popularity: Optional[int] = None  # Betting rank

    @property
    def horse_sex_encoded(self) -> int:
        """Encode horse sex for model input."""
        return HORSE_SEX_MAP.get(self.horse_sex, 0)


@dataclass
class RaceCardData:
    """Data for an entire race card."""

    # Race identifiers
    race_id: str
    race_name: str
    race_number: int  # 1-12

    # Racecourse info
    racecourse: str  # e.g., '東京', '中山'
    racecourse_code: str  # e.g., '05', '06'

    # Race conditions
    distance: int  # Distance in meters
    surface: str  # 'turf' or 'dirt'
    track_condition: str  # '良', '稍重', '重', '不良'
    race_date: str  # YYYY-MM-DD

    # Horse entries
    entries: list[HorseEntryData] = field(default_factory=list)

    @property
    def is_turf(self) -> int:
        """Binary flag for turf race."""
        return 1 if self.surface == "turf" else 0

    @property
    def is_dirt(self) -> int:
        """Binary flag for dirt race."""
        return 1 if self.surface == "dirt" else 0

    @property
    def track_condition_num(self) -> int:
        """Numeric encoding of track condition."""
        return TRACK_CONDITION_MAP.get(self.track_condition, 0)


class RaceCardParser:
    """
    Parse race card HTML from netkeiba.com.

    Example URL: https://race.netkeiba.com/race/shutuba.html?race_id=202506050811
    """

    def parse(self, html: str, race_id: str) -> RaceCardData:
        """
        Parse race card HTML.

        Args:
            html: HTML content of the race card page
            race_id: Race ID (e.g., "202506050811")

        Returns:
            RaceCardData containing race info and horse entries
        """
        soup = BeautifulSoup(html, "lxml")

        # Parse race info
        race_data = self._parse_race_info(soup, race_id)

        # Parse horse entries
        race_data.entries = self._parse_entries(soup)

        return race_data

    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> RaceCardData:
        """Extract race information from the page header."""
        # Race name from title
        race_name = ""
        title_elem = soup.select_one(".RaceName")
        if title_elem:
            race_name = title_elem.get_text(strip=True)

        # Race number from header
        race_number = 1
        race_num_elem = soup.select_one(".RaceNum")
        if race_num_elem:
            race_num_text = race_num_elem.get_text(strip=True)
            if match := re.search(r"(\d+)R", race_num_text):
                race_number = int(match.group(1))

        # Racecourse from header
        racecourse = ""
        racecourse_elem = soup.select_one(".RaceData02 span")
        if racecourse_elem:
            racecourse = racecourse_elem.get_text(strip=True)
            # Remove day info like "1回東京1日"
            if match := re.search(r"回(.+?)(?:\d|$)", racecourse):
                racecourse = match.group(1)

        # Extract racecourse code from race_id
        # Format: YYYYRRCCNN where RR is racecourse code
        racecourse_code = race_id[4:6] if len(race_id) >= 6 else ""

        # Race date from race_id
        race_date = ""
        if len(race_id) >= 8:
            year = race_id[0:4]
            # Note: netkeiba race_id format may vary, need to handle
            # Format seems to be: YYYY + racecourse + kai + day + race_num
            # We'll need to get date from the page itself

        # Date from header
        date_elem = soup.select_one(".RaceData01")
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            if match := re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_text):
                race_date = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"

        # Distance and surface from RaceData01
        distance = 0
        surface = "turf"
        if date_elem:
            race_info = date_elem.get_text(strip=True)
            # Parse distance (e.g., "芝2500m" or "ダ1800m")
            if match := re.search(r"(芝|ダ)(\d+)m", race_info):
                surface = "turf" if match.group(1) == "芝" else "dirt"
                distance = int(match.group(2))

        # Track condition (may not be available before race)
        track_condition = "良"  # Default to good
        condition_elem = soup.select_one(".Item03")
        if condition_elem:
            cond_text = condition_elem.get_text(strip=True)
            for cond in TRACK_CONDITION_MAP:
                if cond in cond_text:
                    track_condition = cond
                    break

        return RaceCardData(
            race_id=race_id,
            race_name=race_name,
            race_number=race_number,
            racecourse=racecourse,
            racecourse_code=racecourse_code,
            distance=distance,
            surface=surface,
            track_condition=track_condition,
            race_date=race_date,
        )

    def _parse_entries(self, soup: BeautifulSoup) -> list[HorseEntryData]:
        """Parse horse entries from the entry table."""
        entries = []

        # Find the shutuba table (different selectors for different page versions)
        table = soup.select_one(".Shutuba_Table, .HorseList, table.ShutubaTable")
        if not table:
            logger.warning("Could not find entry table")
            return entries

        # Parse each row
        rows = table.select("tr.HorseList, tr[class*='Horse']")
        if not rows:
            # Alternative: try all rows except header
            rows = table.select("tbody tr")

        for row in rows:
            try:
                entry = self._parse_entry_row(row)
                if entry:
                    entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to parse entry row: {e}")
                continue

        return entries

    def _parse_entry_row(self, row) -> Optional[HorseEntryData]:
        """Parse a single horse entry row."""
        # Post position (gate number)
        waku_elem = row.select_one("td.Waku, td:nth-child(1)")
        umaban_elem = row.select_one("td.Umaban, td:nth-child(2)")

        post_position = 0
        if umaban_elem:
            text = umaban_elem.get_text(strip=True)
            if text.isdigit():
                post_position = int(text)

        if post_position == 0:
            return None  # Skip invalid rows

        # Horse name and ID
        horse_elem = row.select_one("td.HorseInfo a, .HorseName a, a[href*='/horse/']")
        if not horse_elem:
            return None

        horse_name = horse_elem.get_text(strip=True)
        horse_id = ""
        href = horse_elem.get("href", "")
        if match := re.search(r"/horse/(\d+)", href):
            horse_id = match.group(1)

        # Horse sex and age
        horse_sex = "牡"
        horse_age = 3
        info_elem = row.select_one("td.Barei, .Age, td:nth-child(4)")
        if info_elem:
            info_text = info_elem.get_text(strip=True)
            # Format: "牡3" or "セ5"
            if match := re.match(r"([牡牝セ])(\d+)", info_text):
                horse_sex = match.group(1)
                horse_age = int(match.group(2))

        # Weight carried
        weight_carried = 0.0
        weight_elem = row.select_one("td.Jockey + td, .Weight, td:nth-child(6)")
        if weight_elem:
            weight_text = weight_elem.get_text(strip=True)
            if match := re.search(r"(\d+(?:\.\d+)?)", weight_text):
                weight_carried = float(match.group(1))

        # Jockey
        jockey_id = None
        jockey_name = None
        jockey_elem = row.select_one("td.Jockey a, a[href*='/jockey/']")
        if jockey_elem:
            jockey_name = jockey_elem.get_text(strip=True)
            href = jockey_elem.get("href", "")
            if match := re.search(r"/jockey/(?:result/recent/)?(\d+)", href):
                jockey_id = match.group(1)

        # Trainer
        trainer_id = None
        trainer_name = None
        trainer_elem = row.select_one("td.Trainer a, a[href*='/trainer/']")
        if trainer_elem:
            trainer_name = trainer_elem.get_text(strip=True)
            href = trainer_elem.get("href", "")
            if match := re.search(r"/trainer/(?:result/recent/)?(\d+)", href):
                trainer_id = match.group(1)

        # Horse weight (body weight)
        horse_weight = None
        weight_change = None
        weight_cell = row.select_one("td.Weight")
        if weight_cell:
            # Get full text including small tag
            weight_text = weight_cell.get_text(strip=True)
            # Format: "480(+2)" or "502(-4)" or just "480"
            if match := re.match(r"(\d+)(?:\(([+-]?\d+)\))?", weight_text):
                horse_weight = float(match.group(1))
                if match.group(2):
                    weight_change = float(match.group(2))

        # Odds - look for span with id like "odds-1_01"
        win_odds = None
        odds_elem = row.select_one("span[id^='odds-']")
        if odds_elem:
            odds_text = odds_elem.get_text(strip=True)
            if match := re.search(r"(\d+(?:\.\d+)?)", odds_text):
                win_odds = float(match.group(1))

        # Popularity - look for span with id like "ninki-1_01"
        popularity = None
        pop_elem = row.select_one("span[id^='ninki-']")
        if pop_elem:
            pop_text = pop_elem.get_text(strip=True)
            if pop_text.isdigit():
                popularity = int(pop_text)

        return HorseEntryData(
            horse_id=horse_id,
            horse_name=horse_name,
            post_position=post_position,
            horse_age=horse_age,
            horse_sex=horse_sex,
            horse_weight=horse_weight,
            weight_change=weight_change,
            jockey_id=jockey_id,
            jockey_name=jockey_name,
            trainer_id=trainer_id,
            trainer_name=trainer_name,
            weight_carried=weight_carried,
            win_odds=win_odds,
            popularity=popularity,
        )
