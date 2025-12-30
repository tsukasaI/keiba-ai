"""
Keiba AI - Base Scraper

Abstract base class for all scrapers with retry logic and caching.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from ..browser import BrowserManager
from ..cache import FileCache
from ..config import ScraperConfig
from ..rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseScraper(ABC, Generic[T]):
    """
    Abstract base class for scrapers.

    Provides:
    - Rate limiting
    - Caching
    - Retry logic with exponential backoff
    - Browser management

    Subclasses must implement:
    - cache_category: str property
    - get_url(id: str) -> str
    - parse(html: str, id: str) -> T
    - to_cache_data(data: T) -> dict
    - from_cache_data(data: dict) -> T
    """

    def __init__(
        self,
        browser: BrowserManager,
        config: Optional[ScraperConfig] = None,
        cache: Optional[FileCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.browser = browser
        self.config = config or ScraperConfig()
        self.cache = cache or FileCache.from_config(self.config)
        self.rate_limiter = rate_limiter or RateLimiter.from_config(self.config)

    @property
    @abstractmethod
    def cache_category(self) -> str:
        """Cache category for this scraper (e.g., 'race_card', 'horse')."""
        pass

    @abstractmethod
    def get_url(self, id: str) -> str:
        """Get URL for the given ID."""
        pass

    @abstractmethod
    def parse(self, html: str, id: str) -> T:
        """Parse HTML content and return data object."""
        pass

    @abstractmethod
    def to_cache_data(self, data: T) -> dict:
        """Convert data object to cache-friendly dict."""
        pass

    @abstractmethod
    def from_cache_data(self, data: dict) -> T:
        """Reconstruct data object from cached dict."""
        pass

    async def scrape(self, id: str, force_refresh: bool = False) -> T:
        """
        Scrape data for the given ID.

        Uses cache if available and not expired.
        Applies rate limiting and retry logic.

        Args:
            id: Entity ID (race_id, horse_id, etc.)
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            Parsed data object
        """
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(self.cache_category, id)
            if cached is not None:
                logger.debug(f"Cache hit for {self.cache_category}/{id}")
                return self.from_cache_data(cached)

        # Fetch with retry logic
        html = await self._fetch_with_retry(id)

        # Parse
        data = self.parse(html, id)

        # Cache result
        cache_data = self.to_cache_data(data)
        self.cache.set(self.cache_category, id, cache_data)

        return data

    async def _fetch_with_retry(self, id: str) -> str:
        """Fetch URL with rate limiting and retry logic."""
        url = self.get_url(id)
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Rate limit
                async with self.rate_limiter:
                    logger.debug(f"Fetching {url} (attempt {attempt + 1})")
                    html = await self.browser.fetch_page(url)
                    return html

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Fetch failed for {url} (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay_seconds * (2**attempt)
                    logger.debug(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

        raise RuntimeError(f"Failed to fetch {url} after {self.config.max_retries} attempts: {last_error}")

    async def scrape_many(
        self,
        ids: list[str],
        force_refresh: bool = False,
        concurrency: int = 1,
    ) -> list[T]:
        """
        Scrape multiple entities.

        Args:
            ids: List of entity IDs
            force_refresh: If True, bypass cache
            concurrency: Number of concurrent requests (use 1 for safety)

        Returns:
            List of parsed data objects
        """
        if concurrency == 1:
            # Sequential scraping (safest)
            results = []
            for id in ids:
                try:
                    data = await self.scrape(id, force_refresh)
                    results.append(data)
                except Exception as e:
                    logger.error(f"Failed to scrape {self.cache_category}/{id}: {e}")
            return results

        # Concurrent scraping with semaphore
        semaphore = asyncio.Semaphore(concurrency)

        async def scrape_with_semaphore(id: str) -> Optional[T]:
            async with semaphore:
                try:
                    return await self.scrape(id, force_refresh)
                except Exception as e:
                    logger.error(f"Failed to scrape {self.cache_category}/{id}: {e}")
                    return None

        tasks = [scrape_with_semaphore(id) for id in ids]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]
