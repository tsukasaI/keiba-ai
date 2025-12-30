"""
Keiba AI - Rate Limiter

Async token bucket rate limiter to avoid IP blocks when scraping.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import ScraperConfig


@dataclass
class RateLimiter:
    """
    Async token bucket rate limiter with random delays.

    Implements a token bucket algorithm where tokens are replenished
    at a fixed rate. Also adds random delays between requests.

    Usage:
        limiter = RateLimiter.from_config(config)
        async with limiter:
            # Make request here
            pass
    """

    requests_per_minute: int = 20
    min_delay: float = 1.5
    max_delay: float = 3.0
    _tokens: float = field(default=0.0, init=False)
    _last_update: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        # Start with full bucket
        self._tokens = float(self.requests_per_minute)
        self._last_update = time.monotonic()

    @classmethod
    def from_config(cls, config: Optional[ScraperConfig] = None) -> "RateLimiter":
        """Create rate limiter from scraper config."""
        config = config or ScraperConfig()
        return cls(
            requests_per_minute=config.requests_per_minute,
            min_delay=config.min_delay_seconds,
            max_delay=config.max_delay_seconds,
        )

    @property
    def _replenish_rate(self) -> float:
        """Tokens per second."""
        return self.requests_per_minute / 60.0

    def _replenish(self) -> None:
        """Replenish tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            float(self.requests_per_minute),
            self._tokens + elapsed * self._replenish_rate,
        )
        self._last_update = now

    async def acquire(self) -> None:
        """
        Acquire a token, waiting if necessary.

        This method is thread-safe and will wait if no tokens are available.
        """
        async with self._lock:
            self._replenish()

            # Wait if no tokens available
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._replenish_rate
                await asyncio.sleep(wait_time)
                self._replenish()

            # Consume token
            self._tokens -= 1.0

            # Add random delay
            delay = random.uniform(self.min_delay, self.max_delay)
            await asyncio.sleep(delay)

    async def __aenter__(self) -> "RateLimiter":
        """Acquire token on context enter."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Nothing to do on exit."""
        pass

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens (for debugging)."""
        self._replenish()
        return self._tokens


async def test_rate_limiter():
    """Test rate limiter functionality."""
    limiter = RateLimiter(requests_per_minute=10, min_delay=0.1, max_delay=0.2)

    print(f"Starting with {limiter.available_tokens:.1f} tokens")

    for i in range(5):
        start = time.monotonic()
        async with limiter:
            elapsed = time.monotonic() - start
            print(f"Request {i + 1}: waited {elapsed:.2f}s, tokens: {limiter.available_tokens:.1f}")

    print("Rate limiter test successful!")


if __name__ == "__main__":
    asyncio.run(test_rate_limiter())
