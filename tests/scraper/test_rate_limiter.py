"""Tests for rate limiter functionality."""

import pytest
import asyncio

from src.scraper.rate_limiter import RateLimiter
from src.scraper.config import ScraperConfig


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_basic(self):
        """Test basic token acquisition."""
        limiter = RateLimiter(
            requests_per_minute=60,
            min_delay=0.01,
            max_delay=0.02,
        )
        
        # Should not raise
        await limiter.acquire()
        
        assert limiter.available_tokens < 60

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        limiter = RateLimiter(
            requests_per_minute=60,
            min_delay=0.01,
            max_delay=0.02,
        )
        
        async with limiter:
            pass
        
        assert limiter.available_tokens < 60

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting works."""
        import time
        
        limiter = RateLimiter(
            requests_per_minute=10,
            min_delay=0.05,
            max_delay=0.1,
        )
        
        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        
        # Should take at least 3 * 0.05 = 0.15 seconds
        assert elapsed >= 0.15

    def test_from_config(self):
        """Test creating from ScraperConfig."""
        config = ScraperConfig(
            requests_per_minute=30,
            min_delay_seconds=2.0,
            max_delay_seconds=4.0,
        )
        
        limiter = RateLimiter.from_config(config)
        
        assert limiter.requests_per_minute == 30
        assert limiter.min_delay == 2.0
        assert limiter.max_delay == 4.0
