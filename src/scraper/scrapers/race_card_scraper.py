"""
Keiba AI - Race Card Scraper

Scrape race card (shutuba) pages from netkeiba.com.
"""

from dataclasses import asdict
from typing import Optional

from ..browser import BrowserManager
from ..cache import FileCache
from ..config import ScraperConfig
from ..parsers.race_card import RaceCardData, RaceCardParser, HorseEntryData
from ..rate_limiter import RateLimiter
from .base import BaseScraper


class RaceCardScraper(BaseScraper[RaceCardData]):
    """
    Scrape race card pages from netkeiba.com.

    Example:
        async with BrowserManager() as browser:
            scraper = RaceCardScraper(browser)
            race_data = await scraper.scrape("202506050811")
    """

    def __init__(
        self,
        browser: BrowserManager,
        config: Optional[ScraperConfig] = None,
        cache: Optional[FileCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(browser, config, cache, rate_limiter)
        self._parser = RaceCardParser()

    @property
    def cache_category(self) -> str:
        return "race_card"

    def get_url(self, race_id: str) -> str:
        return self.config.get_race_card_url(race_id)

    def parse(self, html: str, race_id: str) -> RaceCardData:
        return self._parser.parse(html, race_id)

    def to_cache_data(self, data: RaceCardData) -> dict:
        """Convert RaceCardData to cache-friendly dict."""
        result = {
            "race_id": data.race_id,
            "race_name": data.race_name,
            "race_number": data.race_number,
            "racecourse": data.racecourse,
            "racecourse_code": data.racecourse_code,
            "distance": data.distance,
            "surface": data.surface,
            "track_condition": data.track_condition,
            "race_date": data.race_date,
            "entries": [asdict(e) for e in data.entries],
        }
        return result

    def from_cache_data(self, data: dict) -> RaceCardData:
        """Reconstruct RaceCardData from cached dict."""
        entries = [HorseEntryData(**e) for e in data.get("entries", [])]
        return RaceCardData(
            race_id=data["race_id"],
            race_name=data["race_name"],
            race_number=data["race_number"],
            racecourse=data["racecourse"],
            racecourse_code=data["racecourse_code"],
            distance=data["distance"],
            surface=data["surface"],
            track_condition=data["track_condition"],
            race_date=data["race_date"],
            entries=entries,
        )
