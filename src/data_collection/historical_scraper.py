"""
Historical race data scraper orchestrator.

Orchestrates the Rust CLI (`keiba-api scrape-historical`) to collect
historical race data from netkeiba.com into SQLite database.

Features:
- Date range management with automatic chunking
- Resume capability from last scraped date
- Progress tracking and logging
- Configurable rate limiting via CLI

Usage:
    # Scrape 2024 data
    python src/data_collection/historical_scraper.py --start 2024-01-01 --end 2024-12-31

    # Resume from last scraped date
    python src/data_collection/historical_scraper.py --resume

    # Include odds data (slower but more complete)
    python src/data_collection/historical_scraper.py --start 2024-01-01 --end 2024-12-31 --include-odds
"""

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import HISTORICAL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_rust_cli_path() -> Path:
    """Get path to Rust CLI binary."""
    project_root = Path(__file__).parent.parent.parent
    release_path = project_root / "src" / "api" / "target" / "release" / "keiba-api"
    debug_path = project_root / "src" / "api" / "target" / "debug" / "keiba-api"

    if release_path.exists():
        return release_path
    elif debug_path.exists():
        return debug_path
    else:
        raise FileNotFoundError(
            f"Rust CLI not found. Build with: cd src/api && cargo build --release\n"
            f"Searched: {release_path}, {debug_path}"
        )


def get_db_path() -> Path:
    """Get path to SQLite database."""
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / HISTORICAL_CONFIG["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def generate_race_dates(start_date: date, end_date: date) -> list[date]:
    """Generate list of JRA race dates (weekends and some weekdays).

    JRA races are held on weekends (Sat/Sun) and occasionally weekdays
    for special meetings. We'll generate all dates and let the scraper
    skip non-race days.
    """
    dates = []
    current = start_date
    while current <= end_date:
        # JRA races are mainly on weekends
        if current.weekday() in (5, 6):  # Saturday, Sunday
            dates.append(current)
        current += timedelta(days=1)
    return dates


def scrape_date(cli_path: Path, db_path: Path, race_date: date,
                include_odds: bool = False, force: bool = False,
                verbose: bool = False) -> bool:
    """Scrape races for a single date using Rust CLI."""
    cmd = [
        str(cli_path),
        "scrape-historical",
        "--date", race_date.strftime("%Y-%m-%d"),
        "--db", str(db_path),
    ]

    if include_odds:
        cmd.append("--include-odds")
    if force:
        cmd.append("--force")
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=HISTORICAL_CONFIG["timeout_per_date"],
        )

        if result.returncode == 0:
            logger.info(f"Scraped {race_date}: {result.stdout.strip()}")
            return True
        else:
            if "No races found" in result.stderr:
                logger.debug(f"No races on {race_date}")
                return True
            logger.error(f"Failed {race_date}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout scraping {race_date}")
        return False
    except Exception as e:
        logger.error(f"Error scraping {race_date}: {e}")
        return False


def scrape_date_range(
    start_date: date,
    end_date: date,
    include_odds: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """Scrape races for a date range."""
    cli_path = get_rust_cli_path()
    db_path = get_db_path()

    logger.info(f"CLI: {cli_path}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Include odds: {include_odds}")

    dates = generate_race_dates(start_date, end_date)
    logger.info(f"Total race dates to process: {len(dates)}")

    stats = {"success": 0, "failed": 0, "skipped": 0}

    for i, race_date in enumerate(dates, 1):
        logger.info(f"[{i}/{len(dates)}] Processing {race_date}...")

        success = scrape_date(
            cli_path, db_path, race_date,
            include_odds=include_odds,
            force=force,
            verbose=verbose,
        )

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


def get_last_scraped_date(db_path: Path) -> date | None:
    """Get the last scraped date from database."""
    if not db_path.exists():
        return None

    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(race_date) FROM races")
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return datetime.strptime(result[0], "%Y-%m-%d").date()
        return None
    except Exception as e:
        logger.warning(f"Could not read last date from DB: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Historical race data scraper orchestrator"
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today(),
        help="End date (YYYY-MM-DD), default: today",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last scraped date",
    )
    parser.add_argument(
        "--include-odds",
        action="store_true",
        help="Include odds data (slower)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scrape existing data",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Determine start date
    if args.resume:
        db_path = get_db_path()
        last_date = get_last_scraped_date(db_path)
        if last_date:
            start_date = last_date + timedelta(days=1)
            logger.info(f"Resuming from {start_date} (last scraped: {last_date})")
        else:
            start_date = HISTORICAL_CONFIG["default_start_date"]
            logger.info(f"No previous data, starting from {start_date}")
    elif args.start:
        start_date = args.start
    else:
        start_date = HISTORICAL_CONFIG["default_start_date"]

    end_date = args.end

    if start_date > end_date:
        logger.info("Start date is after end date. Nothing to scrape.")
        return

    stats = scrape_date_range(
        start_date,
        end_date,
        include_odds=args.include_odds,
        force=args.force,
        verbose=args.verbose,
    )

    logger.info("=" * 50)
    logger.info("Scraping complete!")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Database: {get_db_path()}")


if __name__ == "__main__":
    main()
