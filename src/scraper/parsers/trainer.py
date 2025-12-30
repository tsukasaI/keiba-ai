"""
Keiba AI - Trainer Profile Parser

Parse trainer profile pages from netkeiba.com.
"""

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class TrainerData:
    """Trainer profile data."""

    # Identifiers
    trainer_id: str
    trainer_name: str

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


class TrainerParser:
    """
    Parse trainer profile HTML from netkeiba.com.

    Example URL: https://db.netkeiba.com/trainer/01061/
    """

    def parse(self, html: str, trainer_id: str) -> TrainerData:
        """
        Parse trainer profile HTML.

        Args:
            html: HTML content of the trainer profile page
            trainer_id: Trainer ID

        Returns:
            TrainerData containing trainer stats
        """
        soup = BeautifulSoup(html, "lxml")

        # Get trainer name from page title or header
        trainer_name = ""
        name_elem = soup.select_one(".Name_En")
        if name_elem:
            parent = name_elem.parent
            if parent:
                trainer_name = parent.get_text(strip=True)
                # Remove English name and clean up
                trainer_name = re.sub(r"\([^)]+\)", "", trainer_name).strip()
                trainer_name = re.sub(r"\s+", " ", trainer_name).strip()

        if not trainer_name:
            # Fallback: try title
            title = soup.select_one("title")
            if title:
                title_text = title.get_text(strip=True)
                if match := re.match(r"(.+?)[\s|]", title_text):
                    trainer_name = match.group(1)

        # Clean up name (remove suffixes)
        trainer_name = re.sub(r"のプロフィール.*$", "", trainer_name).strip()

        data = TrainerData(
            trainer_id=trainer_id,
            trainer_name=trainer_name,
        )

        # Parse stats from ResultsByYears table
        self._parse_results_table(soup, data)

        return data

    def _parse_results_table(self, soup: BeautifulSoup, data: TrainerData) -> None:
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

    def _parse_stats_row(self, cells, data: TrainerData, is_year: bool) -> None:
        """
        Parse a statistics row from ResultsByYears table.

        Column order:
        0: 年度 (Year)
        1: 順位 (Rank)
        2: 1着 (Wins)
        3: 2着 (2nd)
        4: 3着 (3rd)
        5: 4着〜 (4th+)
        6: 出走回数 (Starts)
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
