"""Task 0.1 CLI — per-year market baseline pseudo-R2 from the netkeiba SQLite DB.

Usage:
    uv run python scripts/eval_market_baseline.py --db data/historical/keiba.db

Prints, per year: races total/used/dropped (by reason), entries used,
R2_market at beta=1, and R2_market recalibrated with beta fit on the previous
year (spec: docs/V2_TASKS.md sections 2.1-2.3). Sanity range: [0.05, 0.30].
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.market_eval import (  # noqa: E402
    JRA_ONLY_SQL,
    MIN_RUNNERS,
    PreparedRace,
    fit_power_recalibration,
    mcfadden_r2,
    power_probs,
    prepare_race,
    winner_prob,
)

SANITY_LO, SANITY_HI = 0.05, 0.30
MIN_FIT_RACES = 100  # don't fit beta on fewer races than this


@dataclass
class YearData:
    total: int = 0
    races: list[PreparedRace] = field(default_factory=list)
    drops: Counter = field(default_factory=Counter)

    @property
    def used(self) -> int:
        return len(self.races)

    @property
    def entries_used(self) -> int:
        return sum(r.n_runners for r in self.races)


def load_years(db_path: str, min_runners: int) -> dict[str, YearData]:
    """Group entry rows by race, apply drop rules, bucket by year."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        rows = conn.execute(
            f"""
            SELECT r.race_id, substr(r.race_date, 1, 4) AS year,
                   e.win_odds, e.finish_position
            FROM races r
            JOIN race_entries e ON e.race_id = r.race_id
            WHERE {JRA_ONLY_SQL.format(alias="r")}
            ORDER BY r.race_id
            """
        ).fetchall()
    finally:
        conn.close()

    by_race: dict[str, tuple[str, list, list]] = {}
    for race_id, year, odds, finish in rows:
        rec = by_race.setdefault(race_id, (year, [], []))
        rec[1].append(odds)
        rec[2].append(finish)

    years: dict[str, YearData] = {}
    for _race_id, (year, odds, finish) in by_race.items():
        yd = years.setdefault(year, YearData())
        yd.total += 1
        prepared, reason = prepare_race(odds, finish, min_runners=min_runners)
        if prepared is None:
            yd.drops[reason] += 1
        else:
            yd.races.append(prepared)
    return years


def r2_for_races(races: list[PreparedRace], beta: float = 1.0) -> float:
    wp = [winner_prob(power_probs(r.pi, beta), r.winner_mask) for r in races]
    ns = [r.n_runners for r in races]
    ks = [r.n_winners for r in races]
    return mcfadden_r2(wp, ns, ks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/historical/keiba.db")
    parser.add_argument("--min-runners", type=int, default=MIN_RUNNERS)
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"error: DB not found: {args.db}", file=sys.stderr)
        return 1

    years = load_years(args.db, args.min_runners)
    if not years:
        print("error: no races found in DB", file=sys.stderr)
        return 1

    header = (
        f"{'year':<6}{'total':>7}{'used':>7}{'dropped':>9}  "
        f"{'entries':>8}  {'R2(beta=1)':>11}  {'beta(prev)':>11}  {'R2(recal)':>10}"
    )
    print(header)
    print("-" * len(header))

    prev_races: list[PreparedRace] | None = None
    warnings: list[str] = []
    for year in sorted(years):
        yd = years[year]
        r2_raw = r2_for_races(yd.races) if yd.races else float("nan")

        beta_s, r2_recal_s = "-", "-"
        if prev_races is not None and len(prev_races) >= MIN_FIT_RACES and yd.races:
            beta = fit_power_recalibration(prev_races)
            r2_recal = r2_for_races(yd.races, beta)
            beta_s, r2_recal_s = f"{beta:.3f}", f"{r2_recal:.4f}"

        drop_detail = ", ".join(f"{k}={v}" for k, v in sorted(yd.drops.items()))
        print(
            f"{year:<6}{yd.total:>7}{yd.used:>7}{sum(yd.drops.values()):>9}  "
            f"{yd.entries_used:>8}  {r2_raw:>11.4f}  {beta_s:>11}  {r2_recal_s:>10}"
            + (f"   [{drop_detail}]" if drop_detail else "")
        )
        if yd.used >= MIN_FIT_RACES and not (SANITY_LO <= r2_raw <= SANITY_HI):
            warnings.append(
                f"year {year}: R2(beta=1)={r2_raw:.4f} outside sanity range "
                f"[{SANITY_LO}, {SANITY_HI}] — investigate before proceeding"
            )
        prev_races = yd.races

    print()
    print(f"sanity range for R2_market: [{SANITY_LO}, {SANITY_HI}] (lit. anchors: HK 0.13-0.19)")
    for w in warnings:
        print(f"WARNING: {w}")
    return 2 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
