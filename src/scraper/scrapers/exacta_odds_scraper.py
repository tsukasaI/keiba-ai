"""
Keiba AI - Exacta Odds Scraper

Scrape exacta (umatan/馬単) odds from netkeiba.com API.
"""

import asyncio
import json
import logging
from typing import Optional

import aiohttp

from ..cache import FileCache
from ..config import ScraperConfig
from ..rate_limiter import RateLimiter
from ..parsers.exacta_odds import ExactaOddsParser, ExactaOddsData

logger = logging.getLogger(__name__)


class ExactaOddsScraper:
    """
    Scrape exacta odds from netkeiba.com API.

    API URL: https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=6
    """

    API_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        cache: Optional[FileCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.config = config or ScraperConfig()
        self.cache = cache or FileCache.from_config(self.config)
        self.rate_limiter = rate_limiter or RateLimiter.from_config(self.config)
        self.parser = ExactaOddsParser()

    @property
    def cache_category(self) -> str:
        return "exacta_odds"

    async def scrape(self, race_id: str, force_refresh: bool = False) -> ExactaOddsData:
        """
        Scrape exacta odds for a race.

        Args:
            race_id: Race ID (e.g., "202506050811")
            force_refresh: If True, bypass cache

        Returns:
            ExactaOddsData containing all exacta combinations
        """
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(self.cache_category, race_id)
            if cached is not None:
                logger.debug(f"Cache hit for {self.cache_category}/{race_id}")
                return self._from_cache_data(cached)

        # Fetch from API
        json_data = await self._fetch_with_retry(race_id)

        # Parse
        data = self.parser.parse(json_data, race_id)

        # Cache result
        cache_data = self._to_cache_data(data)
        self.cache.set(self.cache_category, race_id, cache_data)

        return data

    async def _fetch_with_retry(self, race_id: str) -> dict:
        """Fetch API with rate limiting and retry logic."""
        url = f"{self.API_URL}?race_id={race_id}&type=6"
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                async with self.rate_limiter:
                    logger.debug(f"Fetching {url} (attempt {attempt + 1})")
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            if response.status != 200:
                                raise RuntimeError(f"API returned status {response.status}")
                            return await response.json()

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Fetch failed for {url} (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (2 ** attempt)
                    logger.debug(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

        raise RuntimeError(f"Failed to fetch {url} after {self.config.max_retries} attempts: {last_error}")

    def _to_cache_data(self, data: ExactaOddsData) -> dict:
        """Convert to cache-friendly dict."""
        return {
            "race_id": data.race_id,
            "official_datetime": data.official_datetime,
            "odds": {f"{k[0]:02d}{k[1]:02d}": v for k, v in data.odds.items()},
            "popularity": {f"{k[0]:02d}{k[1]:02d}": v for k, v in data.popularity.items()},
        }

    def _from_cache_data(self, data: dict) -> ExactaOddsData:
        """Reconstruct from cached dict."""
        result = ExactaOddsData(
            race_id=data["race_id"],
            official_datetime=data.get("official_datetime", ""),
        )

        for k, v in data.get("odds", {}).items():
            first = int(k[:2])
            second = int(k[2:])
            result.odds[(first, second)] = v

        for k, v in data.get("popularity", {}).items():
            first = int(k[:2])
            second = int(k[2:])
            result.popularity[(first, second)] = v

        return result
