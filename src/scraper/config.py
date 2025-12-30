"""
Keiba AI - Scraper Configuration

Configuration for netkeiba.com scraper including URLs, rate limits, and cache settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ScraperConfig:
    """Configuration for the netkeiba scraper."""

    # Base URLs
    BASE_URL: str = "https://race.netkeiba.com"
    DB_URL: str = "https://db.netkeiba.com"

    # URL patterns
    RACE_CARD_URL: str = "{base}/race/shutuba.html?race_id={race_id}"
    RACE_RESULT_URL: str = "{base}/race/result.html?race_id={race_id}"
    HORSE_URL: str = "{db}/horse/{horse_id}/"
    JOCKEY_URL: str = "{db}/jockey/{jockey_id}/"
    TRAINER_URL: str = "{db}/trainer/{trainer_id}/"
    ODDS_URL: str = "{base}/odds/index.html?race_id={race_id}"

    # Rate limiting
    requests_per_minute: int = 20
    min_delay_seconds: float = 1.5
    max_delay_seconds: float = 3.0

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 5.0

    # Cache configuration
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/scraper"))
    cache_ttl_hours: int = 24  # Cache expiry for race cards (before race)
    horse_cache_ttl_days: int = 7  # Horse profiles change less frequently
    jockey_cache_ttl_days: int = 7
    trainer_cache_ttl_days: int = 7

    # Browser configuration
    headless: bool = True
    timeout_ms: int = 30000
    user_agent: Optional[str] = None  # Use playwright-stealth default

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("data/scraped"))

    def get_race_card_url(self, race_id: str) -> str:
        """Get URL for race card (shutuba) page."""
        return self.RACE_CARD_URL.format(base=self.BASE_URL, race_id=race_id)

    def get_race_result_url(self, race_id: str) -> str:
        """Get URL for race result page."""
        return self.RACE_RESULT_URL.format(base=self.BASE_URL, race_id=race_id)

    def get_horse_url(self, horse_id: str) -> str:
        """Get URL for horse profile page."""
        return self.HORSE_URL.format(db=self.DB_URL, horse_id=horse_id)

    def get_jockey_url(self, jockey_id: str) -> str:
        """Get URL for jockey profile page."""
        return self.JOCKEY_URL.format(db=self.DB_URL, jockey_id=jockey_id)

    def get_trainer_url(self, trainer_id: str) -> str:
        """Get URL for trainer profile page."""
        return self.TRAINER_URL.format(db=self.DB_URL, trainer_id=trainer_id)

    def get_odds_url(self, race_id: str) -> str:
        """Get URL for odds page."""
        return self.ODDS_URL.format(base=self.BASE_URL, race_id=race_id)


# Race ID format: YYYYMMDDRRNN
# YYYY: Year (e.g., 2025)
# MM: Month (01-12)
# DD: Day (01-31)
# RR: Racecourse code (01-10)
# NN: Race number (01-12)

RACECOURSE_CODES = {
    "01": "札幌",  # Sapporo
    "02": "函館",  # Hakodate
    "03": "福島",  # Fukushima
    "04": "新潟",  # Niigata
    "05": "東京",  # Tokyo
    "06": "中山",  # Nakayama
    "07": "中京",  # Chukyo
    "08": "京都",  # Kyoto
    "09": "阪神",  # Hanshin
    "10": "小倉",  # Kokura
}

RACECOURSE_CODES_REVERSE = {v: k for k, v in RACECOURSE_CODES.items()}

# Track condition encoding (matches model config)
TRACK_CONDITION_MAP = {
    "良": 0,    # Good
    "稍重": 1,  # Yielding
    "重": 2,    # Soft
    "不良": 3,  # Heavy
}

# Horse sex encoding (matches model config)
HORSE_SEX_MAP = {
    "牡": 0,  # Male (stallion/colt)
    "牝": 1,  # Female (mare/filly)
    "セ": 2,  # Gelding
}

# Default values for missing data (from src/models/config.py)
MISSING_DEFAULTS = {
    "last_position": 10.0,
    "horse_weight": 480.0,
    "jockey_win_rate": 0.07,
    "trainer_win_rate": 0.07,
    "jockey_place_rate": 0.21,
    "avg_position_last_3": 10.0,
    "avg_position_last_5": 10.0,
    "win_rate_last_3": 0.0,
    "win_rate_last_5": 0.0,
    "place_rate_last_3": 0.0,
    "place_rate_last_5": 0.0,
    "jockey_races": 100,
    "trainer_races": 100,
    "career_races": 0,
}
