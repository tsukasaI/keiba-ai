#!/usr/bin/env python
"""Scrape odds for existing races in the database."""

import sqlite3
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data/historical/keiba.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def get_race_ids(conn, start_date: str, end_date: str) -> list:
    """Get race IDs in date range."""
    cursor = conn.execute(
        """
        SELECT race_id FROM races
        WHERE race_date >= ? AND race_date <= ?
        ORDER BY race_date
        """,
        (start_date, end_date),
    )
    return [row[0] for row in cursor.fetchall()]


def parse_exacta_odds(html: str) -> dict:
    """Parse exacta odds from netkeiba odds page."""
    soup = BeautifulSoup(html, "html.parser")
    odds = {}

    # Find all odds tables
    tables = soup.find_all("table", class_="Odds_Table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                # Extract combination and odds
                combo_elem = cells[0].find("span")
                odds_elem = cells[1]
                if combo_elem and odds_elem:
                    combo_text = combo_elem.get_text(strip=True)
                    odds_text = odds_elem.get_text(strip=True)
                    try:
                        # Format: "1-2" -> odds value
                        if "-" in combo_text:
                            odds_val = float(odds_text.replace(",", ""))
                            odds[combo_text] = odds_val
                    except ValueError:
                        pass

    return odds


def fetch_exacta_odds(race_id: str) -> dict:
    """Fetch exacta odds for a race from netkeiba."""
    url = f"https://db.netkeiba.com/odds/index.html?pid=old_exact&id={race_id}&type=b1"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return parse_exacta_odds(resp.text)
    except Exception as e:
        print(f"  Error fetching {race_id}: {e}")
        return {}


def insert_odds(conn, race_id: str, bet_type: str, odds_dict: dict):
    """Insert odds into database."""
    for combo, odds in odds_dict.items():
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO odds_snapshots
                (race_id, bet_type, combination, odds)
                VALUES (?, ?, ?, ?)
                """,
                (race_id, bet_type, combo, odds),
            )
        except Exception as e:
            print(f"  Error inserting {race_id} {combo}: {e}")
    conn.commit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape odds for existing races")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", default=str(DB_PATH), help="Database path")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    race_ids = get_race_ids(conn, args.start, args.end)

    print(f"Found {len(race_ids)} races in {args.start} to {args.end}")

    total_odds = 0
    for i, race_id in enumerate(race_ids):
        print(f"[{i+1}/{len(race_ids)}] Fetching odds for {race_id}...", end="")

        odds = fetch_exacta_odds(race_id)
        if odds:
            insert_odds(conn, race_id, "exacta", odds)
            total_odds += len(odds)
            print(f" {len(odds)} combinations")
        else:
            print(" no odds found")

        # Rate limiting
        time.sleep(0.5)

    conn.close()
    print(f"\nDone! Scraped {total_odds} total odds entries.")


if __name__ == "__main__":
    main()
