"""Tests for scraper classes with mocked browser and cache."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.scraper.scrapers.base import BaseScraper
from src.scraper.scrapers.horse_scraper import HorseScraper
from src.scraper.scrapers.jockey_scraper import JockeyScraper
from src.scraper.scrapers.trainer_scraper import TrainerScraper
from src.scraper.config import ScraperConfig
from src.scraper.cache import FileCache
from src.scraper.rate_limiter import RateLimiter
from src.scraper.parsers.horse import HorseData
from src.scraper.parsers.jockey import JockeyData
from src.scraper.parsers.trainer import TrainerData


class TestBaseScraper:
    """Test BaseScraper abstract class behavior."""

    @pytest.fixture
    def mock_browser(self):
        """Create mock browser manager."""
        browser = MagicMock()
        browser.fetch_page = AsyncMock(return_value="<html></html>")
        return browser

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache that returns None (cache miss)."""
        cache = MagicMock(spec=FileCache)
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter that doesn't wait."""
        limiter = MagicMock(spec=RateLimiter)
        limiter.__aenter__ = AsyncMock(return_value=limiter)
        limiter.__aexit__ = AsyncMock(return_value=None)
        return limiter

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ScraperConfig(max_retries=2, retry_delay_seconds=0.01)


class TestHorseScraper:
    """Test HorseScraper with mocked dependencies."""

    @pytest.fixture
    def mock_browser(self):
        """Create mock browser manager."""
        browser = MagicMock()
        browser.fetch_page = AsyncMock(return_value="""
            <html>
            <div class="horse_title"><h1>テスト馬</h1></div>
            <table class="db_prof_table">
                <tr><th>生年月日</th><td>2020年4月15日</td></tr>
            </table>
            </html>
        """)
        return browser

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = MagicMock(spec=FileCache)
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = MagicMock(spec=RateLimiter)
        limiter.__aenter__ = AsyncMock(return_value=limiter)
        limiter.__aexit__ = AsyncMock(return_value=None)
        return limiter

    @pytest.fixture
    def scraper(self, mock_browser, mock_cache, mock_rate_limiter):
        """Create HorseScraper with mocked dependencies."""
        config = ScraperConfig(max_retries=2)
        return HorseScraper(
            browser=mock_browser,
            config=config,
            cache=mock_cache,
            rate_limiter=mock_rate_limiter,
        )

    def test_cache_category(self, scraper):
        """Cache category should be 'horse'."""
        assert scraper.cache_category == "horse"

    def test_get_url(self, scraper):
        """URL should include horse ID."""
        url = scraper.get_url("2019104975")
        assert "2019104975" in url
        assert "horse" in url

    @pytest.mark.asyncio
    async def test_scrape_cache_miss(self, scraper, mock_browser, mock_cache):
        """Scraping with cache miss should fetch and cache."""
        result = await scraper.scrape("2019104975")

        assert isinstance(result, HorseData)
        assert result.horse_id == "2019104975"
        mock_browser.fetch_page.assert_called_once()
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_cache_hit(self, scraper, mock_browser, mock_cache):
        """Scraping with cache hit should not fetch."""
        mock_cache.get.return_value = {
            "horse_id": "2019104975",
            "horse_name": "キャッシュ馬",
            "birth_date": "2020-04-15",
            "sex": "牡",
            "past_races": [],
        }

        result = await scraper.scrape("2019104975")

        assert result.horse_name == "キャッシュ馬"
        mock_browser.fetch_page.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_force_refresh(self, scraper, mock_browser, mock_cache):
        """Force refresh should bypass cache."""
        mock_cache.get.return_value = {
            "horse_id": "2019104975",
            "horse_name": "キャッシュ馬",
            "birth_date": None,
            "sex": "牡",
            "past_races": [],
        }

        result = await scraper.scrape("2019104975", force_refresh=True)

        # Should fetch even though cache has data
        mock_browser.fetch_page.assert_called_once()

    def test_to_cache_data(self, scraper):
        """Data should serialize to dict for caching."""
        horse = HorseData(
            horse_id="test",
            horse_name="テスト馬",
            birth_date="2020-01-01",
            sex="牡",
        )
        cache_data = scraper.to_cache_data(horse)

        assert isinstance(cache_data, dict)
        assert cache_data["horse_id"] == "test"
        assert cache_data["horse_name"] == "テスト馬"

    def test_from_cache_data(self, scraper):
        """Data should deserialize from cached dict."""
        cache_data = {
            "horse_id": "test",
            "horse_name": "テスト馬",
            "birth_date": "2020-01-01",
            "sex": "牡",
            "past_races": [],
        }
        horse = scraper.from_cache_data(cache_data)

        assert isinstance(horse, HorseData)
        assert horse.horse_id == "test"
        assert horse.horse_name == "テスト馬"


class TestJockeyScraper:
    """Test JockeyScraper with mocked dependencies."""

    @pytest.fixture
    def mock_browser(self):
        """Create mock browser manager."""
        browser = MagicMock()
        browser.fetch_page = AsyncMock(return_value="""
            <html>
            <div class="Name">ルメール<span class="Name_En">C.Lemaire</span></div>
            <table class="ResultsByYears">
                <tbody>
                    <tr><td>累計</td><td>-</td><td>100</td><td>80</td><td>60</td><td>200</td><td>440</td></tr>
                </tbody>
            </table>
            </html>
        """)
        return browser

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = MagicMock(spec=FileCache)
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = MagicMock(spec=RateLimiter)
        limiter.__aenter__ = AsyncMock(return_value=limiter)
        limiter.__aexit__ = AsyncMock(return_value=None)
        return limiter

    @pytest.fixture
    def scraper(self, mock_browser, mock_cache, mock_rate_limiter):
        """Create JockeyScraper with mocked dependencies."""
        config = ScraperConfig(max_retries=2)
        return JockeyScraper(
            browser=mock_browser,
            config=config,
            cache=mock_cache,
            rate_limiter=mock_rate_limiter,
        )

    def test_cache_category(self, scraper):
        """Cache category should be 'jockey'."""
        assert scraper.cache_category == "jockey"

    def test_get_url(self, scraper):
        """URL should include jockey ID."""
        url = scraper.get_url("05339")
        assert "05339" in url
        assert "jockey" in url

    @pytest.mark.asyncio
    async def test_scrape_returns_jockey_data(self, scraper):
        """Scraping should return JockeyData."""
        result = await scraper.scrape("05339")

        assert isinstance(result, JockeyData)
        assert result.jockey_id == "05339"


class TestTrainerScraper:
    """Test TrainerScraper with mocked dependencies."""

    @pytest.fixture
    def mock_browser(self):
        """Create mock browser manager."""
        browser = MagicMock()
        browser.fetch_page = AsyncMock(return_value="""
            <html>
            <div class="Name">矢作芳人<span class="Name_En">Y.Yahagi</span></div>
            <table class="ResultsByYears">
                <tbody>
                    <tr><td>累計</td><td>-</td><td>50</td><td>40</td><td>30</td><td>100</td><td>220</td></tr>
                </tbody>
            </table>
            </html>
        """)
        return browser

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = MagicMock(spec=FileCache)
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = MagicMock(spec=RateLimiter)
        limiter.__aenter__ = AsyncMock(return_value=limiter)
        limiter.__aexit__ = AsyncMock(return_value=None)
        return limiter

    @pytest.fixture
    def scraper(self, mock_browser, mock_cache, mock_rate_limiter):
        """Create TrainerScraper with mocked dependencies."""
        config = ScraperConfig(max_retries=2)
        return TrainerScraper(
            browser=mock_browser,
            config=config,
            cache=mock_cache,
            rate_limiter=mock_rate_limiter,
        )

    def test_cache_category(self, scraper):
        """Cache category should be 'trainer'."""
        assert scraper.cache_category == "trainer"

    def test_get_url(self, scraper):
        """URL should include trainer ID."""
        url = scraper.get_url("01061")
        assert "01061" in url
        assert "trainer" in url

    @pytest.mark.asyncio
    async def test_scrape_returns_trainer_data(self, scraper):
        """Scraping should return TrainerData."""
        result = await scraper.scrape("01061")

        assert isinstance(result, TrainerData)
        assert result.trainer_id == "01061"


class TestScraperRetryLogic:
    """Test retry logic in scrapers."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache that always misses."""
        cache = MagicMock(spec=FileCache)
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = MagicMock(spec=RateLimiter)
        limiter.__aenter__ = AsyncMock(return_value=limiter)
        limiter.__aexit__ = AsyncMock(return_value=None)
        return limiter

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, mock_cache, mock_rate_limiter):
        """Scraper should retry on fetch failure."""
        mock_browser = MagicMock()
        mock_browser.fetch_page = AsyncMock(
            side_effect=[
                Exception("Network error"),
                "<html><div class='horse_title'><h1>馬</h1></div></html>",
            ]
        )

        config = ScraperConfig(max_retries=3, retry_delay_seconds=0.01)
        scraper = HorseScraper(
            browser=mock_browser,
            config=config,
            cache=mock_cache,
            rate_limiter=mock_rate_limiter,
        )

        result = await scraper.scrape("test")

        assert result.horse_id == "test"
        assert mock_browser.fetch_page.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, mock_cache, mock_rate_limiter):
        """Should raise error after max retries exceeded."""
        mock_browser = MagicMock()
        mock_browser.fetch_page = AsyncMock(side_effect=Exception("Always fails"))

        config = ScraperConfig(max_retries=2, retry_delay_seconds=0.01)
        scraper = HorseScraper(
            browser=mock_browser,
            config=config,
            cache=mock_cache,
            rate_limiter=mock_rate_limiter,
        )

        with pytest.raises(RuntimeError, match="Failed to fetch"):
            await scraper.scrape("test")

        assert mock_browser.fetch_page.call_count == 2


class TestScrapeManyLogic:
    """Test scrape_many method."""

    @pytest.fixture
    def mock_browser(self):
        """Create mock browser."""
        browser = MagicMock()
        browser.fetch_page = AsyncMock(
            return_value="<html><div class='horse_title'><h1>馬</h1></div></html>"
        )
        return browser

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = MagicMock(spec=FileCache)
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock()
        return cache

    @pytest.fixture
    def mock_rate_limiter(self):
        """Create mock rate limiter."""
        limiter = MagicMock(spec=RateLimiter)
        limiter.__aenter__ = AsyncMock(return_value=limiter)
        limiter.__aexit__ = AsyncMock(return_value=None)
        return limiter

    @pytest.mark.asyncio
    async def test_scrape_many_sequential(
        self, mock_browser, mock_cache, mock_rate_limiter
    ):
        """Scrape many should fetch all IDs sequentially."""
        config = ScraperConfig(max_retries=2)
        scraper = HorseScraper(
            browser=mock_browser,
            config=config,
            cache=mock_cache,
            rate_limiter=mock_rate_limiter,
        )

        ids = ["horse1", "horse2", "horse3"]
        results = await scraper.scrape_many(ids, concurrency=1)

        assert len(results) == 3
        assert mock_browser.fetch_page.call_count == 3

    @pytest.mark.asyncio
    async def test_scrape_many_handles_errors(
        self, mock_cache, mock_rate_limiter
    ):
        """Scrape many should continue on individual errors."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Error on second call")
            return "<html><div class='horse_title'><h1>馬</h1></div></html>"

        mock_browser = MagicMock()
        mock_browser.fetch_page = AsyncMock(side_effect=side_effect)

        config = ScraperConfig(max_retries=1, retry_delay_seconds=0.01)
        scraper = HorseScraper(
            browser=mock_browser,
            config=config,
            cache=mock_cache,
            rate_limiter=mock_rate_limiter,
        )

        ids = ["horse1", "horse2", "horse3"]
        results = await scraper.scrape_many(ids, concurrency=1)

        # Should get results for horse1 and horse3 (horse2 failed)
        assert len(results) == 2
