"""
Keiba AI - Jockey Profile Parser

Parse jockey profile pages from netkeiba.com.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class JockeyData:
    """Jockey profile data."""

    # Identifiers
    jockey_id: str
    jockey_name: str

    # Career stats (current year or lifetime)
    total_races: int = 0
    wins: int = 0
    seconds: int = 0
    thirds: int = 0

    # Current year stats
    year_races: int = 0
    year_wins: int = 0
    year_seconds: int = 0
    year_thirds: int = 0

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        if self.total_races == 0:
            return 0.07  # Default
        return self.wins / self.total_races

    @property
    def place_rate(self) -> float:
        """Place rate (top 3 finishes)."""
        if self.total_races == 0:
            return 0.21  # Default
        return (self.wins + self.seconds + self.thirds) / self.total_races

    @property
    def year_win_rate(self) -> float:
        """Current year win rate."""
        if self.year_races == 0:
            return self.win_rate
        return self.year_wins / self.year_races

    @property
    def year_place_rate(self) -> float:
        """Current year place rate."""
        if self.year_races == 0:
            return self.place_rate
        return (self.year_wins + self.year_seconds + self.year_thirds) / self.year_races


class JockeyParser:
    """
    Parse jockey profile HTML from netkeiba.com.

    Example URL: https://db.netkeiba.com/jockey/05339/
    """

    def parse(self, html: str, jockey_id: str) -> JockeyData:
        """
        Parse jockey profile HTML.

        Args:
            html: HTML content of the jockey profile page
            jockey_id: Jockey ID

        Returns:
            JockeyData containing jockey stats
        """
        soup = BeautifulSoup(html, "lxml")

        # Get jockey name from page title or header
        jockey_name = ""
        name_elem = soup.select_one(".Name_En")
        if name_elem:
            # Get the previous sibling which contains the Japanese name
            parent = name_elem.parent
            if parent:
                jockey_name = parent.get_text(strip=True)
                # Remove English name and clean up
                jockey_name = re.sub(r"\([^)]+\)", "", jockey_name).strip()
                jockey_name = re.sub(r"\s+", " ", jockey_name).strip()

        if not jockey_name:
            # Fallback: try title
            title = soup.select_one("title")
            if title:
                title_text = title.get_text(strip=True)
                if match := re.match(r"(.+?)[\s|]", title_text):
                    jockey_name = match.group(1)

        # Clean up name (remove suffixes like "のプロフィール")
        jockey_name = re.sub(r"のプロフィール.*$", "", jockey_name).strip()

        data = JockeyData(
            jockey_id=jockey_id,
            jockey_name=jockey_name,
        )

        # Parse stats from ResultsByYears table
        self._parse_results_table(soup, data)

        return data

    def _parse_results_table(self, soup: BeautifulSoup, data: JockeyData) -> None:
        """Extract statistics from ResultsByYears table."""
        # Find the ResultsByYears table
        stats_table = soup.select_one("table.ResultsByYears")
        if not stats_table:
            logger.debug("ResultsByYears table not found")
            return

        current_year = str(__import__("datetime").datetime.now().year)

        rows = stats_table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 7:
                continue

            # First cell contains year label (累計, 2025, 2024, etc.)
            year_cell = cells[0].get_text(strip=True)

            if "累計" in year_cell:
                # Lifetime stats
                self._parse_stats_row(cells, data, is_year=False)
            elif current_year in year_cell:
                # Current year stats
                self._parse_stats_row(cells, data, is_year=True)

    def _parse_stats_row(self, cells, data: JockeyData, is_year: bool) -> None:
        """
        Parse a statistics row from ResultsByYears table.

        Column order:
        0: 年度 (Year)
        1: 順位 (Rank)
        2: 1着 (Wins)
        3: 2着 (2nd)
        4: 3着 (3rd)
        5: 4着〜 (4th+)
        6: 騎乗回数 (Rides)
        7-12: Other stats
        """
        if len(cells) < 7:
            return

        try:
            wins = self._parse_int(cells[2].get_text(strip=True))
            seconds = self._parse_int(cells[3].get_text(strip=True))
            thirds = self._parse_int(cells[4].get_text(strip=True))
            fourths_plus = self._parse_int(cells[5].get_text(strip=True))
            races = self._parse_int(cells[6].get_text(strip=True))

            # Verify by summing (races should equal sum of placements)
            if races == 0:
                races = wins + seconds + thirds + fourths_plus

            if is_year:
                data.year_races = races
                data.year_wins = wins
                data.year_seconds = seconds
                data.year_thirds = thirds
            else:
                data.total_races = races
                data.wins = wins
                data.seconds = seconds
                data.thirds = thirds

            logger.debug(
                f"Parsed {'year' if is_year else 'career'} stats: "
                f"{wins}w/{seconds}s/{thirds}t from {races} races"
            )

        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse stats row: {e}")

    def _parse_int(self, text: str) -> int:
        """Parse integer from text, handling commas."""
        text = text.replace(",", "").strip()
        if match := re.search(r"(\d+)", text):
            return int(match.group(1))
        return 0
