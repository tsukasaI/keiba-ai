"""
Keiba AI - Browser Management

Async Playwright browser management with stealth mode for scraping netkeiba.com.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright_stealth import Stealth

from .config import ScraperConfig

# Configure stealth with Japanese locale
_stealth = Stealth(
    navigator_languages_override=("ja-JP", "ja", "en-US", "en"),
    navigator_platform_override="MacIntel",
)

logger = logging.getLogger(__name__)


class BrowserManager:
    """
    Manages Playwright browser lifecycle with stealth mode.

    Usage:
        async with BrowserManager(config) as manager:
            async with manager.new_page() as page:
                await page.goto(url)
                content = await page.content()
    """

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "BrowserManager":
        """Start browser on context enter."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close browser on context exit."""
        await self.close()

    async def start(self) -> None:
        """Start Playwright and launch browser."""
        if self._browser is not None:
            return

        logger.info("Starting Playwright browser...")
        self._playwright = await async_playwright().start()

        # Launch Chromium with stealth-friendly options
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )

        # Create browser context with realistic viewport
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=self.config.user_agent,
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
        )

        # Set default timeout
        self._context.set_default_timeout(self.config.timeout_ms)

        logger.info("Browser started successfully")

    async def close(self) -> None:
        """Close browser and cleanup resources."""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Browser closed")

    @asynccontextmanager
    async def new_page(self) -> AsyncGenerator[Page, None]:
        """
        Create a new page with stealth mode applied.

        Yields:
            Page: Playwright page with stealth mode
        """
        if self._context is None:
            raise RuntimeError("Browser not started. Call start() first or use async context manager.")

        page = await self._context.new_page()

        # Apply stealth mode to avoid detection
        await _stealth.apply_stealth_async(page)

        try:
            yield page
        finally:
            await page.close()

    async def fetch_page(self, url: str, wait_for: Optional[str] = None) -> str:
        """
        Fetch a page and return its HTML content.

        Args:
            url: URL to fetch
            wait_for: Optional CSS selector to wait for before returning

        Returns:
            HTML content of the page
        """
        async with self.new_page() as page:
            logger.debug(f"Fetching: {url}")

            response = await page.goto(url, wait_until="domcontentloaded")

            if response is None:
                raise RuntimeError(f"Failed to load page: {url}")

            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status} for {url}")

            # Wait for specific element if requested
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=self.config.timeout_ms)

            # Small delay to let JavaScript render
            await asyncio.sleep(0.5)

            content = await page.content()
            logger.debug(f"Fetched {len(content)} bytes from {url}")

            return content


async def test_browser():
    """Test browser functionality."""
    config = ScraperConfig(headless=True)

    async with BrowserManager(config) as manager:
        url = "https://race.netkeiba.com/"
        content = await manager.fetch_page(url)
        print(f"Fetched {len(content)} bytes from {url}")
        print("Browser test successful!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_browser())
