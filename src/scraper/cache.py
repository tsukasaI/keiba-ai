"""
Keiba AI - Caching Layer

JSON file-based caching with TTL for scraped data.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import ScraperConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    data: Any
    timestamp: float
    ttl_seconds: float

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() > self.timestamp + self.ttl_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "data": self.data,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            data=d["data"],
            timestamp=d["timestamp"],
            ttl_seconds=d["ttl_seconds"],
        )


@dataclass
class FileCache:
    """
    File-based cache with TTL support.

    Stores each cache entry as a separate JSON file, allowing for
    easy inspection and cleanup.

    Usage:
        cache = FileCache.from_config(config)

        # Check and get
        if data := cache.get("race_card", race_id):
            return data

        # Fetch and cache
        data = await scrape_race_card(race_id)
        cache.set("race_card", race_id, data, ttl_hours=24)
    """

    cache_dir: Path
    enabled: bool = True
    _ttl_defaults: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Optional[ScraperConfig] = None) -> "FileCache":
        """Create cache from scraper config."""
        config = config or ScraperConfig()
        cache = cls(
            cache_dir=config.cache_dir,
            enabled=config.cache_enabled,
        )
        # Set default TTLs for different data types
        cache._ttl_defaults = {
            "race_card": config.cache_ttl_hours * 3600,
            "race_result": 365 * 24 * 3600,  # Results don't change
            "horse": config.horse_cache_ttl_days * 24 * 3600,
            "jockey": config.jockey_cache_ttl_days * 24 * 3600,
            "trainer": config.trainer_cache_ttl_days * 24 * 3600,
            "odds": 5 * 60,  # Odds change frequently, 5 min cache
        }
        return cache

    def _get_cache_path(self, category: str, key: str) -> Path:
        """Get file path for a cache entry."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        safe_key = "".join(c if c.isalnum() else "_" for c in key)[:50]
        filename = f"{safe_key}_{key_hash}.json"
        return self.cache_dir / category / filename

    def get(self, category: str, key: str) -> Optional[Any]:
        """
        Get data from cache if it exists and is not expired.

        Args:
            category: Cache category (e.g., "race_card", "horse")
            key: Cache key (e.g., race_id, horse_id)

        Returns:
            Cached data or None if not found or expired
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(category, key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {category}/{key}")
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                entry = CacheEntry.from_dict(json.load(f))

            if entry.is_expired():
                logger.debug(f"Cache expired: {category}/{key}")
                cache_path.unlink()  # Remove expired entry
                return None

            logger.debug(f"Cache hit: {category}/{key}")
            return entry.data

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache read error for {category}/{key}: {e}")
            cache_path.unlink()  # Remove corrupted entry
            return None

    def set(
        self,
        category: str,
        key: str,
        data: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """
        Store data in cache.

        Args:
            category: Cache category
            key: Cache key
            data: Data to cache (must be JSON serializable)
            ttl_seconds: Time to live in seconds (uses category default if not specified)
        """
        if not self.enabled:
            return

        if ttl_seconds is None:
            ttl_seconds = self._ttl_defaults.get(category, 24 * 3600)

        cache_path = self._get_cache_path(category, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds,
        )

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached: {category}/{key}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Cache write error for {category}/{key}: {e}")

    def delete(self, category: str, key: str) -> bool:
        """
        Delete a cache entry.

        Returns:
            True if entry was deleted, False if it didn't exist
        """
        cache_path = self._get_cache_path(category, key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear_category(self, category: str) -> int:
        """
        Clear all entries in a category.

        Returns:
            Number of entries deleted
        """
        category_dir = self.cache_dir / category
        if not category_dir.exists():
            return 0

        count = 0
        for cache_file in category_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        return count

    def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        count = 0
        for category_dir in self.cache_dir.iterdir():
            if category_dir.is_dir():
                count += self.clear_category(category_dir.name)
        return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("**/*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = CacheEntry.from_dict(json.load(f))
                if entry.is_expired():
                    cache_file.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError):
                cache_file.unlink()
                count += 1
        return count

    def stats(self) -> dict:
        """Get cache statistics."""
        stats = {"total_entries": 0, "total_size_bytes": 0, "by_category": {}}

        for category_dir in self.cache_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            files = list(category_dir.glob("*.json"))
            size = sum(f.stat().st_size for f in files)

            stats["by_category"][category] = {
                "entries": len(files),
                "size_bytes": size,
            }
            stats["total_entries"] += len(files)
            stats["total_size_bytes"] += size

        return stats


def test_cache():
    """Test cache functionality."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FileCache(cache_dir=Path(tmpdir) / "cache")

        # Test set/get
        cache.set("test", "key1", {"foo": "bar"}, ttl_seconds=3600)
        data = cache.get("test", "key1")
        assert data == {"foo": "bar"}, f"Expected {{'foo': 'bar'}}, got {data}"

        # Test miss
        assert cache.get("test", "nonexistent") is None

        # Test expiry
        cache.set("test", "expired", {"old": "data"}, ttl_seconds=0)
        assert cache.get("test", "expired") is None

        # Test stats
        stats = cache.stats()
        print(f"Cache stats: {stats}")

        print("Cache test successful!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_cache()
