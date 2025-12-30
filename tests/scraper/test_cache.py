"""Tests for caching functionality."""

import pytest
import tempfile
from pathlib import Path

from src.scraper.cache import FileCache, CacheEntry


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_is_expired_fresh(self):
        """Test that fresh entries are not expired."""
        import time
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=time.time(),
            ttl_seconds=3600,
        )
        assert not entry.is_expired()

    def test_is_expired_old(self):
        """Test that old entries are expired."""
        import time
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl_seconds=3600,  # 1 hour TTL
        )
        assert entry.is_expired()

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        import time
        original = CacheEntry(
            data={"key": "value"},
            timestamp=time.time(),
            ttl_seconds=3600,
        )
        
        d = original.to_dict()
        restored = CacheEntry.from_dict(d)
        
        assert restored.data == original.data
        assert restored.timestamp == original.timestamp
        assert restored.ttl_seconds == original.ttl_seconds


class TestFileCache:
    """Tests for FileCache class."""

    def test_set_get(self):
        """Test basic set and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=Path(tmpdir) / "cache")
            
            cache.set("test", "key1", {"foo": "bar"}, ttl_seconds=3600)
            result = cache.get("test", "key1")
            
            assert result == {"foo": "bar"}

    def test_get_missing(self):
        """Test getting non-existent key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=Path(tmpdir) / "cache")
            
            result = cache.get("test", "nonexistent")
            
            assert result is None

    def test_get_expired(self):
        """Test that expired entries return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=Path(tmpdir) / "cache")
            
            cache.set("test", "expired", {"old": "data"}, ttl_seconds=0)
            result = cache.get("test", "expired")
            
            assert result is None

    def test_delete(self):
        """Test deleting cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=Path(tmpdir) / "cache")
            
            cache.set("test", "key1", {"data": True}, ttl_seconds=3600)
            assert cache.delete("test", "key1") is True
            assert cache.get("test", "key1") is None
            assert cache.delete("test", "key1") is False

    def test_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=Path(tmpdir) / "cache")
            
            cache.set("cat1", "key1", {"data": 1}, ttl_seconds=3600)
            cache.set("cat1", "key2", {"data": 2}, ttl_seconds=3600)
            cache.set("cat2", "key1", {"data": 3}, ttl_seconds=3600)
            
            stats = cache.stats()
            
            assert stats["total_entries"] == 3
            assert stats["by_category"]["cat1"]["entries"] == 2
            assert stats["by_category"]["cat2"]["entries"] == 1

    def test_disabled_cache(self):
        """Test that disabled cache doesn't store anything."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(cache_dir=Path(tmpdir) / "cache", enabled=False)
            
            cache.set("test", "key1", {"data": True}, ttl_seconds=3600)
            result = cache.get("test", "key1")
            
            assert result is None
