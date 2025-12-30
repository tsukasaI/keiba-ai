"""
Keiba AI - Jockey Profile Scraper

Scrape jockey profile pages from netkeiba.com.
"""

from typing import Optional

from ..browser import BrowserManager
from ..cache import FileCache
from ..config import ScraperConfig
from ..parsers.jockey import JockeyData, JockeyParser
from ..rate_limiter import RateLimiter
from .base import BaseScraper


class JockeyScraper(BaseScraper[JockeyData]):
    """
    Scrape jockey profile pages from netkeiba.com.

    Example:
        async with BrowserManager() as browser:
            scraper = JockeyScraper(browser)
            jockey_data = await scraper.scrape("01170")
    """

    def __init__(
        self,
        browser: BrowserManager,
        config: Optional[ScraperConfig] = None,
        cache: Optional[FileCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(browser, config, cache, rate_limiter)
        self._parser = JockeyParser()

    @property
    def cache_category(self) -> str:
        return "jockey"

    def get_url(self, jockey_id: str) -> str:
        return self.config.get_jockey_url(jockey_id)

    def parse(self, html: str, jockey_id: str) -> JockeyData:
        return self._parser.parse(html, jockey_id)

    def to_cache_data(self, data: JockeyData) -> dict:
        """Convert JockeyData to cache-friendly dict."""
        return {
            "jockey_id": data.jockey_id,
            "jockey_name": data.jockey_name,
            "total_races": data.total_races,
            "wins": data.wins,
            "seconds": data.seconds,
            "thirds": data.thirds,
            "year_races": data.year_races,
            "year_wins": data.year_wins,
            "year_seconds": data.year_seconds,
            "year_thirds": data.year_thirds,
        }

    def from_cache_data(self, data: dict) -> JockeyData:
        """Reconstruct JockeyData from cached dict."""
        return JockeyData(
            jockey_id=data["jockey_id"],
            jockey_name=data["jockey_name"],
            total_races=data.get("total_races", 0),
            wins=data.get("wins", 0),
            seconds=data.get("seconds", 0),
            thirds=data.get("thirds", 0),
            year_races=data.get("year_races", 0),
            year_wins=data.get("year_wins", 0),
            year_seconds=data.get("year_seconds", 0),
            year_thirds=data.get("year_thirds", 0),
        )
