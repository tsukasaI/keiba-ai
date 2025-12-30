"""
Keiba AI - Trainer Profile Scraper

Scrape trainer profile pages from netkeiba.com.
"""

from typing import Optional

from ..browser import BrowserManager
from ..cache import FileCache
from ..config import ScraperConfig
from ..parsers.trainer import TrainerData, TrainerParser
from ..rate_limiter import RateLimiter
from .base import BaseScraper


class TrainerScraper(BaseScraper[TrainerData]):
    """
    Scrape trainer profile pages from netkeiba.com.

    Example:
        async with BrowserManager() as browser:
            scraper = TrainerScraper(browser)
            trainer_data = await scraper.scrape("01106")
    """

    def __init__(
        self,
        browser: BrowserManager,
        config: Optional[ScraperConfig] = None,
        cache: Optional[FileCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(browser, config, cache, rate_limiter)
        self._parser = TrainerParser()

    @property
    def cache_category(self) -> str:
        return "trainer"

    def get_url(self, trainer_id: str) -> str:
        return self.config.get_trainer_url(trainer_id)

    def parse(self, html: str, trainer_id: str) -> TrainerData:
        return self._parser.parse(html, trainer_id)

    def to_cache_data(self, data: TrainerData) -> dict:
        """Convert TrainerData to cache-friendly dict."""
        return {
            "trainer_id": data.trainer_id,
            "trainer_name": data.trainer_name,
            "total_races": data.total_races,
            "wins": data.wins,
            "seconds": data.seconds,
            "thirds": data.thirds,
            "year_races": data.year_races,
            "year_wins": data.year_wins,
            "year_seconds": data.year_seconds,
            "year_thirds": data.year_thirds,
        }

    def from_cache_data(self, data: dict) -> TrainerData:
        """Reconstruct TrainerData from cached dict."""
        return TrainerData(
            trainer_id=data["trainer_id"],
            trainer_name=data["trainer_name"],
            total_races=data.get("total_races", 0),
            wins=data.get("wins", 0),
            seconds=data.get("seconds", 0),
            thirds=data.get("thirds", 0),
            year_races=data.get("year_races", 0),
            year_wins=data.get("year_wins", 0),
            year_seconds=data.get("year_seconds", 0),
            year_thirds=data.get("year_thirds", 0),
        )
