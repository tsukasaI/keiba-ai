"""
Keiba AI - Scraper CLI

Command-line interface for scraping netkeiba.com.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .browser import BrowserManager
from .cache import FileCache
from .config import ScraperConfig
from .pipeline.api_formatter import ApiFormatter
from .rate_limiter import RateLimiter
from .scrapers.race_card_scraper import RaceCardScraper
from .scrapers.horse_scraper import HorseScraper
from .scrapers.jockey_scraper import JockeyScraper
from .scrapers.trainer_scraper import TrainerScraper

logger = logging.getLogger(__name__)


class RaceScraper:
    """
    High-level scraper that combines all scrapers.

    Scrapes race card, then all horses, jockeys, and trainers.
    """

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self._cache = FileCache.from_config(self.config)
        self._rate_limiter = RateLimiter.from_config(self.config)
        self._formatter = ApiFormatter()

    async def scrape_race(
        self,
        race_id: str,
        include_profiles: bool = True,
        force_refresh: bool = False,
    ) -> dict:
        """
        Scrape a complete race with all supplemental data.

        Args:
            race_id: Race ID (e.g., "202506050811")
            include_profiles: If True, scrape horse/jockey/trainer profiles
            force_refresh: If True, bypass cache

        Returns:
            Dictionary with race data and API request
        """
        async with BrowserManager(self.config) as browser:
            # Initialize scrapers with shared resources
            race_scraper = RaceCardScraper(
                browser, self.config, self._cache, self._rate_limiter
            )
            horse_scraper = HorseScraper(
                browser, self.config, self._cache, self._rate_limiter
            )
            jockey_scraper = JockeyScraper(
                browser, self.config, self._cache, self._rate_limiter
            )
            trainer_scraper = TrainerScraper(
                browser, self.config, self._cache, self._rate_limiter
            )

            # Scrape race card
            logger.info(f"Scraping race card: {race_id}")
            race_data = await race_scraper.scrape(race_id, force_refresh)
            logger.info(
                f"Found {len(race_data.entries)} horses in {race_data.race_name}"
            )

            # Collect IDs
            horse_ids = [e.horse_id for e in race_data.entries if e.horse_id]
            jockey_ids = [e.jockey_id for e in race_data.entries if e.jockey_id]
            trainer_ids = [e.trainer_id for e in race_data.entries if e.trainer_id]

            horses = {}
            jockeys = {}
            trainers = {}

            if include_profiles:
                # Scrape horse profiles
                logger.info(f"Scraping {len(horse_ids)} horse profiles...")
                for horse_id in horse_ids:
                    try:
                        horses[horse_id] = await horse_scraper.scrape(
                            horse_id, force_refresh
                        )
                    except Exception as e:
                        logger.warning(f"Failed to scrape horse {horse_id}: {e}")

                # Scrape jockey profiles
                logger.info(f"Scraping {len(jockey_ids)} jockey profiles...")
                for jockey_id in jockey_ids:
                    try:
                        jockeys[jockey_id] = await jockey_scraper.scrape(
                            jockey_id, force_refresh
                        )
                    except Exception as e:
                        logger.warning(f"Failed to scrape jockey {jockey_id}: {e}")

                # Scrape trainer profiles
                logger.info(f"Scraping {len(trainer_ids)} trainer profiles...")
                for trainer_id in trainer_ids:
                    try:
                        trainers[trainer_id] = await trainer_scraper.scrape(
                            trainer_id, force_refresh
                        )
                    except Exception as e:
                        logger.warning(f"Failed to scrape trainer {trainer_id}: {e}")

            # Build API request
            request = self._formatter.build_request(
                race=race_data,
                horses=horses,
                jockeys=jockeys,
                trainers=trainers,
            )

            return {
                "race": race_data,
                "horses": horses,
                "jockeys": jockeys,
                "trainers": trainers,
                "api_request": request,
            }


async def cmd_scrape_race(args):
    """Handle scrape-race command."""
    config = ScraperConfig(
        headless=not args.visible,
        cache_enabled=not args.no_cache,
    )

    scraper = RaceScraper(config)
    result = await scraper.scrape_race(
        args.race_id,
        include_profiles=not args.no_profiles,
        force_refresh=args.force,
    )

    # Display race info
    formatter = ApiFormatter()
    print(formatter.format_for_display(result["race"]))
    print()

    # Output API request
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["api_request"].to_json())
        print(f"API request saved to: {output_path}")
    else:
        print("API Request JSON:")
        print(result["api_request"].to_json())


async def cmd_predict_race(args):
    """Handle predict-race command."""
    import urllib.request
    import urllib.error

    config = ScraperConfig(
        headless=not args.visible,
        cache_enabled=not args.no_cache,
    )

    scraper = RaceScraper(config)
    result = await scraper.scrape_race(
        args.race_id,
        include_profiles=not args.no_profiles,
        force_refresh=args.force,
    )

    # Display race info
    formatter = ApiFormatter()
    print(formatter.format_for_display(result["race"]))
    print()

    # Send to API
    api_url = f"{args.api_url}/predict"
    request_json = result["api_request"].to_json()

    print(f"Sending prediction request to {api_url}...")

    try:
        req = urllib.request.Request(
            api_url,
            data=request_json.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        print("\n=== Prediction Results ===")
        print(json.dumps(response_data, ensure_ascii=False, indent=2))

    except urllib.error.URLError as e:
        print(f"Error: Failed to connect to API: {e}")
        sys.exit(1)


async def cmd_cache_stats(args):
    """Handle cache-stats command."""
    config = ScraperConfig()
    cache = FileCache.from_config(config)

    stats = cache.stats()
    print("Cache Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total size: {stats['total_size_bytes'] / 1024:.1f} KB")
    print()
    print("By category:")
    for category, cat_stats in stats["by_category"].items():
        print(f"  {category}: {cat_stats['entries']} entries ({cat_stats['size_bytes'] / 1024:.1f} KB)")


async def cmd_cache_clear(args):
    """Handle cache-clear command."""
    config = ScraperConfig()
    cache = FileCache.from_config(config)

    if args.category:
        count = cache.clear_category(args.category)
        print(f"Cleared {count} entries from {args.category}")
    else:
        count = cache.clear_all()
        print(f"Cleared {count} entries from all categories")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Keiba AI - Netkeiba Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scrape-race command
    scrape_parser = subparsers.add_parser(
        "scrape-race", help="Scrape a single race"
    )
    scrape_parser.add_argument("race_id", help="Race ID (e.g., 202506050811)")
    scrape_parser.add_argument(
        "-o", "--output", help="Output file path for API request JSON"
    )
    scrape_parser.add_argument(
        "--no-profiles", action="store_true",
        help="Skip scraping horse/jockey/trainer profiles"
    )
    scrape_parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching"
    )
    scrape_parser.add_argument(
        "--force", action="store_true", help="Force refresh (ignore cache)"
    )
    scrape_parser.add_argument(
        "--visible", action="store_true", help="Show browser window"
    )

    # predict-race command
    predict_parser = subparsers.add_parser(
        "predict-race", help="Scrape and predict a race"
    )
    predict_parser.add_argument("race_id", help="Race ID (e.g., 202506050811)")
    predict_parser.add_argument(
        "--api-url", default="http://localhost:8080",
        help="API URL (default: http://localhost:8080)"
    )
    predict_parser.add_argument(
        "--no-profiles", action="store_true",
        help="Skip scraping horse/jockey/trainer profiles"
    )
    predict_parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching"
    )
    predict_parser.add_argument(
        "--force", action="store_true", help="Force refresh (ignore cache)"
    )
    predict_parser.add_argument(
        "--visible", action="store_true", help="Show browser window"
    )

    # cache-stats command
    subparsers.add_parser("cache-stats", help="Show cache statistics")

    # cache-clear command
    clear_parser = subparsers.add_parser("cache-clear", help="Clear cache")
    clear_parser.add_argument(
        "--category", help="Category to clear (default: all)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Run command
    if args.command == "scrape-race":
        asyncio.run(cmd_scrape_race(args))
    elif args.command == "predict-race":
        asyncio.run(cmd_predict_race(args))
    elif args.command == "cache-stats":
        asyncio.run(cmd_cache_stats(args))
    elif args.command == "cache-clear":
        asyncio.run(cmd_cache_clear(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
