"""Tests for scraper configuration."""

import pytest
from src.scraper.config import ScraperConfig, RACECOURSE_CODES


def test_scraper_config_defaults():
    """Test default configuration values."""
    config = ScraperConfig()
    
    assert config.requests_per_minute == 20
    assert config.min_delay_seconds == 1.5
    assert config.max_delay_seconds == 3.0
    assert config.headless is True
    assert config.cache_enabled is True


def test_get_race_card_url():
    """Test race card URL generation."""
    config = ScraperConfig()
    url = config.get_race_card_url("202506050811")
    
    assert "race.netkeiba.com" in url
    assert "shutuba.html" in url
    assert "202506050811" in url


def test_get_horse_url():
    """Test horse profile URL generation."""
    config = ScraperConfig()
    url = config.get_horse_url("2019104975")
    
    assert "db.netkeiba.com" in url
    assert "/horse/" in url
    assert "2019104975" in url


def test_racecourse_codes():
    """Test racecourse code mappings."""
    assert "01" in RACECOURSE_CODES
    assert RACECOURSE_CODES["01"] == "札幌"
    assert RACECOURSE_CODES["05"] == "東京"
    assert RACECOURSE_CODES["06"] == "中山"
