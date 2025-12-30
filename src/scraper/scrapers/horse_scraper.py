"""
Keiba AI - Horse Profile Scraper

Scrape horse profile pages from netkeiba.com.
"""

from dataclasses import asdict
from typing import Optional

from ..browser import BrowserManager
from ..cache import FileCache
from ..config import ScraperConfig
from ..parsers.horse import HorseData, HorseParser, PastRaceData
from ..rate_limiter import RateLimiter
from .base import BaseScraper


class HorseScraper(BaseScraper[HorseData]):
    """
    Scrape horse profile pages from netkeiba.com.

    Example:
        async with BrowserManager() as browser:
            scraper = HorseScraper(browser)
            horse_data = await scraper.scrape("2019104975")
    """

    def __init__(
        self,
        browser: BrowserManager,
        config: Optional[ScraperConfig] = None,
        cache: Optional[FileCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(browser, config, cache, rate_limiter)
        self._parser = HorseParser()

    @property
    def cache_category(self) -> str:
        return "horse"

    def get_url(self, horse_id: str) -> str:
        return self.config.get_horse_url(horse_id)

    def parse(self, html: str, horse_id: str) -> HorseData:
        return self._parser.parse(html, horse_id)

    def to_cache_data(self, data: HorseData) -> dict:
        """Convert HorseData to cache-friendly dict."""
        result = {
            "horse_id": data.horse_id,
            "horse_name": data.horse_name,
            "birth_date": data.birth_date,
            "sex": data.sex,
            "coat_color": data.coat_color,
            "sire": data.sire,
            "dam": data.dam,
            "broodmare_sire": data.broodmare_sire,
            "career_races": data.career_races,
            "career_wins": data.career_wins,
            "career_places": data.career_places,
            "total_earnings": data.total_earnings,
            "past_races": [asdict(r) for r in data.past_races],
        }
        return result

    def from_cache_data(self, data: dict) -> HorseData:
        """Reconstruct HorseData from cached dict."""
        past_races = [PastRaceData(**r) for r in data.get("past_races", [])]
        return HorseData(
            horse_id=data["horse_id"],
            horse_name=data["horse_name"],
            birth_date=data.get("birth_date"),
            sex=data.get("sex", "牡"),
            coat_color=data.get("coat_color"),
            sire=data.get("sire"),
            dam=data.get("dam"),
            broodmare_sire=data.get("broodmare_sire"),
            career_races=data.get("career_races", 0),
            career_wins=data.get("career_wins", 0),
            career_places=data.get("career_places", 0),
            total_earnings=data.get("total_earnings", 0.0),
            past_races=past_races,
        )
