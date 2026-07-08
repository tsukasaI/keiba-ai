"""Odds-free, leakage-proof feature builder (V2, Task 1.1).

Spec: docs/V2_TASKS.md section 6. Builds a per-(race_id, horse_id) feature
table from the netkeiba SQLite DB under the section-2.8 leakage rules:

- Every feature is computable strictly before the race's date. This is
  enforced by construction: races are processed in date order, and a date's
  results are ingested into the running accumulators only AFTER features for
  every race on that date have been emitted (so same-day races never see each
  other, and a race never sees itself).
- The current race's odds/popularity are never read (they are not even
  selected — see RACES_SQL / ENTRIES_SQL, asserted by tests).
- First-time values (debut horse, unseen jockey) are NaN, not 0.

Usage:
    uv run python -m src.preprocessing.features_v2 \
        --db data/historical/keiba.db --out data/processed/features_v2.parquet
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.models.market_eval import JRA_ONLY_SQL

# --- SQL (leakage rule: win_odds / popularity are deliberately NOT selected) ---

RACES_SQL = f"""
SELECT race_id, race_date, racecourse, race_number, distance, surface,
       track_condition, grade
FROM races r
WHERE {JRA_ONLY_SQL.format(alias="r")}
ORDER BY race_date, race_id
"""

ENTRIES_SQL = f"""
SELECT e.race_id, e.post_position, e.horse_id, e.horse_age, e.horse_sex,
       e.weight_carried, e.horse_weight, e.weight_change,
       e.jockey_id, e.trainer_id,
       e.finish_position, e.finish_time, e.last_3f,
       e.corner_1, e.corner_2, e.corner_3, e.corner_4
FROM race_entries e
JOIN races r ON r.race_id = e.race_id
WHERE {JRA_ONLY_SQL.format(alias="r")}
ORDER BY e.race_id, e.post_position
"""

# --- categorical encodings (values observed in the DB) ---

SEX_CODE = {"牡": 1.0, "牝": 2.0, "セ": 3.0}
SURFACE_CODE = {"turf": 0.0, "dirt": 1.0}
TRACK_COND_CODE = {"良": 0.0, "稍重": 1.0, "重": 2.0, "不良": 3.0}
GRADE_LEVEL = {"G1": 5.0, "G2": 4.0, "G3": 3.0, "L": 2.0, "OP": 1.0}

RECENT_N = 3  # "last-3 runs" window for form features
SPEED_MIN_KEY_RUNS = 10  # min prior runs on a (course,dist,surf,cond) key for a z-score

FEATURE_COLUMNS = [
    # race context
    "field_size",
    "post_position",
    "post_position_ratio",
    "distance",
    "surface_code",
    "track_cond_code",
    "grade_level",
    "month",
    # horse card
    "horse_age",
    "horse_sex_code",
    "weight_carried",
    "horse_weight",
    "weight_change",
    # horse history
    "days_since_last_run",
    "career_starts",
    "career_win_rate",
    "career_place_rate",
    "last1_norm_finish",
    "last3_avg_norm_finish",
    "last3_avg_last3f_pctl",
    "last3_avg_corner_pctl",
    "last3_avg_speed_z",
    "best_speed_z",
    "class_move",
    "dist_band_starts",
    "dist_band_win_rate",
    "dist_band_place_rate",
    "surface_starts",
    "surface_win_rate",
    "surface_place_rate",
    "course_starts",
    "course_win_rate",
    "course_place_rate",
    # jockey / trainer
    "jockey_starts",
    "jockey_win_rate",
    "jockey_place_rate",
    "jockey_course_win_rate",
    "jockey_surface_win_rate",
    "jockey_switch",
    "trainer_starts",
    "trainer_win_rate",
    "trainer_place_rate",
]

NAN = float("nan")


def distance_band(distance: float) -> int:
    if not distance or distance <= 0:
        return -1
    if distance < 1400:
        return 0
    if distance < 1800:
        return 1
    if distance < 2200:
        return 2
    return 3


@dataclass
class Tally:
    starts: int = 0
    wins: int = 0
    places: int = 0

    def add(self, finish_position: int) -> None:
        self.starts += 1
        if finish_position == 1:
            self.wins += 1
        if finish_position <= 3:
            self.places += 1

    @property
    def win_rate(self) -> float:
        return self.wins / self.starts if self.starts else NAN

    @property
    def place_rate(self) -> float:
        return self.places / self.starts if self.starts else NAN


@dataclass
class HorseState:
    career: Tally = field(default_factory=Tally)
    by_surface: dict = field(default_factory=lambda: defaultdict(Tally))
    by_band: dict = field(default_factory=lambda: defaultdict(Tally))
    by_course: dict = field(default_factory=lambda: defaultdict(Tally))
    recent_norm_finish: deque = field(default_factory=lambda: deque(maxlen=RECENT_N))
    recent_last3f_pctl: deque = field(default_factory=lambda: deque(maxlen=RECENT_N))
    recent_corner_pctl: deque = field(default_factory=lambda: deque(maxlen=RECENT_N))
    recent_speed_z: deque = field(default_factory=lambda: deque(maxlen=RECENT_N))
    recent_grade: deque = field(default_factory=lambda: deque(maxlen=RECENT_N))
    best_speed_z: float = NAN
    last_run_date: date | None = None
    last_jockey_id: str | None = None


@dataclass
class Welford:
    """Running mean/std for speed-figure baselines."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n >= 2 else NAN


def _nanmean(values) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else NAN


def _pct_ranks(values: list[float]) -> list[float]:
    """Percentile ranks in (0, 1], NaN-preserving, tie-aware (average rank)."""
    return list(pd.Series(values, dtype=float).rank(pct=True))


class FeatureBuilder:
    """Single pass over races in date order: emit features, then ingest results."""

    def __init__(self) -> None:
        self.horses: dict[str, HorseState] = defaultdict(HorseState)
        self.jockeys: dict[str, Tally] = defaultdict(Tally)
        self.jockey_course: dict[tuple, Tally] = defaultdict(Tally)
        self.jockey_surface: dict[tuple, Tally] = defaultdict(Tally)
        self.trainers: dict[str, Tally] = defaultdict(Tally)
        self.speed_baseline: dict[tuple, Welford] = defaultdict(Welford)

    # -- emit ---------------------------------------------------------------

    def emit_race(self, race: dict, entries: list[dict]) -> list[dict]:
        field_size = len(entries)
        race_day = race["date_obj"]
        surface_code = SURFACE_CODE.get(race["surface"], NAN)
        cond_code = TRACK_COND_CODE.get(race["track_condition"], NAN)
        grade_level = GRADE_LEVEL.get(race["grade"], 0.0)
        band = distance_band(race["distance"])
        course = race["racecourse"]

        rows = []
        for e in entries:
            hs = self.horses.get(e["horse_id"])
            jid, tid = e["jockey_id"], e["trainer_id"]
            jockey = self.jockeys.get(jid) if jid else None
            trainer = self.trainers.get(tid) if tid else None
            j_course = self.jockey_course.get((jid, course)) if jid else None
            j_surface = self.jockey_surface.get((jid, race["surface"])) if jid else None

            row = {
                "race_id": race["race_id"],
                "race_date": race["race_date"],
                "horse_id": e["horse_id"],
                "finish_position": e["finish_position"],
                "label_win": (
                    float(e["finish_position"] == 1)
                    if e["finish_position"] is not None
                    else NAN
                ),
                # race context
                "field_size": float(field_size),
                "post_position": float(e["post_position"]),
                "post_position_ratio": float(e["post_position"]) / field_size,
                "distance": float(race["distance"] or NAN),
                "surface_code": surface_code,
                "track_cond_code": cond_code,
                "grade_level": grade_level,
                "month": float(race_day.month),
                # horse card
                "horse_age": _f(e["horse_age"]),
                "horse_sex_code": SEX_CODE.get(e["horse_sex"], NAN),
                "weight_carried": _f(e["weight_carried"]),
                "horse_weight": _f(e["horse_weight"]),
                "weight_change": _f(e["weight_change"]),
            }

            if hs is None:
                row.update(
                    {
                        "days_since_last_run": NAN,
                        "career_starts": 0.0,
                        "career_win_rate": NAN,
                        "career_place_rate": NAN,
                        "last1_norm_finish": NAN,
                        "last3_avg_norm_finish": NAN,
                        "last3_avg_last3f_pctl": NAN,
                        "last3_avg_corner_pctl": NAN,
                        "last3_avg_speed_z": NAN,
                        "best_speed_z": NAN,
                        "class_move": NAN,
                        "dist_band_starts": 0.0,
                        "dist_band_win_rate": NAN,
                        "dist_band_place_rate": NAN,
                        "surface_starts": 0.0,
                        "surface_win_rate": NAN,
                        "surface_place_rate": NAN,
                        "course_starts": 0.0,
                        "course_win_rate": NAN,
                        "course_place_rate": NAN,
                        "jockey_switch": NAN,
                    }
                )
            else:
                sb = hs.by_surface.get(race["surface"], Tally())
                bb = hs.by_band.get(band, Tally())
                cb = hs.by_course.get(course, Tally())
                row.update(
                    {
                        "days_since_last_run": (
                            float((race_day - hs.last_run_date).days)
                            if hs.last_run_date
                            else NAN
                        ),
                        "career_starts": float(hs.career.starts),
                        "career_win_rate": hs.career.win_rate,
                        "career_place_rate": hs.career.place_rate,
                        "last1_norm_finish": (
                            hs.recent_norm_finish[-1] if hs.recent_norm_finish else NAN
                        ),
                        "last3_avg_norm_finish": _nanmean(hs.recent_norm_finish),
                        "last3_avg_last3f_pctl": _nanmean(hs.recent_last3f_pctl),
                        "last3_avg_corner_pctl": _nanmean(hs.recent_corner_pctl),
                        "last3_avg_speed_z": _nanmean(hs.recent_speed_z),
                        "best_speed_z": hs.best_speed_z,
                        "class_move": (
                            grade_level - _nanmean(hs.recent_grade)
                            if hs.recent_grade
                            else NAN
                        ),
                        "dist_band_starts": float(bb.starts),
                        "dist_band_win_rate": bb.win_rate,
                        "dist_band_place_rate": bb.place_rate,
                        "surface_starts": float(sb.starts),
                        "surface_win_rate": sb.win_rate,
                        "surface_place_rate": sb.place_rate,
                        "course_starts": float(cb.starts),
                        "course_win_rate": cb.win_rate,
                        "course_place_rate": cb.place_rate,
                        "jockey_switch": (
                            float(hs.last_jockey_id != jid)
                            if hs.last_jockey_id and jid
                            else NAN
                        ),
                    }
                )

            row.update(
                {
                    "jockey_starts": float(jockey.starts) if jockey else 0.0,
                    "jockey_win_rate": jockey.win_rate if jockey else NAN,
                    "jockey_place_rate": jockey.place_rate if jockey else NAN,
                    "jockey_course_win_rate": j_course.win_rate if j_course else NAN,
                    "jockey_surface_win_rate": j_surface.win_rate if j_surface else NAN,
                    "trainer_starts": float(trainer.starts) if trainer else 0.0,
                    "trainer_win_rate": trainer.win_rate if trainer else NAN,
                    "trainer_place_rate": trainer.place_rate if trainer else NAN,
                }
            )
            rows.append(row)
        return rows

    # -- ingest -------------------------------------------------------------

    def ingest_race(self, race: dict, entries: list[dict]) -> None:
        runners = [e for e in entries if e["finish_position"] is not None]
        if not runners:
            return
        n = len(runners)
        race_day = race["date_obj"]
        course = race["racecourse"]
        band = distance_band(race["distance"])
        grade_level = GRADE_LEVEL.get(race["grade"], 0.0)
        speed_key = (
            course,
            race["distance"],
            race["surface"],
            race["track_condition"],
        )

        last3f_pctl = _pct_ranks([_f(e["last_3f"]) for e in runners])
        corner_pctl = _pct_ranks(
            [
                _nanmean([_f(e[c]) for c in ("corner_1", "corner_2", "corner_3", "corner_4")])
                for e in runners
            ]
        )

        # Speed z-scores use the baseline as of BEFORE this race; the baseline
        # is updated with this race's times only after all z are computed.
        baseline = self.speed_baseline[speed_key]
        zs = []
        for e in runners:
            t = _f(e["finish_time"])
            if (
                not math.isnan(t)
                and baseline.n >= SPEED_MIN_KEY_RUNS
                and not math.isnan(baseline.std)
                and baseline.std > 1e-9
            ):
                zs.append((baseline.mean - t) / baseline.std)  # positive = faster
            else:
                zs.append(NAN)
        for e in runners:
            t = _f(e["finish_time"])
            if not math.isnan(t):
                baseline.add(t)

        for e, l3f_p, cor_p, z in zip(runners, last3f_pctl, corner_pctl, zs):
            fp = int(e["finish_position"])
            hs = self.horses[e["horse_id"]]
            hs.career.add(fp)
            hs.by_surface[race["surface"]].add(fp)
            hs.by_band[band].add(fp)
            hs.by_course[course].add(fp)
            hs.recent_norm_finish.append(fp / n)
            hs.recent_last3f_pctl.append(l3f_p if l3f_p == l3f_p else NAN)
            hs.recent_corner_pctl.append(cor_p if cor_p == cor_p else NAN)
            hs.recent_speed_z.append(z)
            hs.recent_grade.append(grade_level)
            if not math.isnan(z) and (math.isnan(hs.best_speed_z) or z > hs.best_speed_z):
                hs.best_speed_z = z
            hs.last_run_date = race_day
            hs.last_jockey_id = e["jockey_id"] or hs.last_jockey_id

            if e["jockey_id"]:
                self.jockeys[e["jockey_id"]].add(fp)
                self.jockey_course[(e["jockey_id"], course)].add(fp)
                self.jockey_surface[(e["jockey_id"], race["surface"])].add(fp)
            if e["trainer_id"]:
                self.trainers[e["trainer_id"]].add(fp)


def _f(value) -> float:
    """To float, None -> NaN."""
    return float(value) if value is not None else NAN


def build_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build the full feature table (all JRA races in the DB, date order)."""
    races = [
        dict(
            zip(
                (
                    "race_id",
                    "race_date",
                    "racecourse",
                    "race_number",
                    "distance",
                    "surface",
                    "track_condition",
                    "grade",
                ),
                row,
            )
        )
        for row in conn.execute(RACES_SQL)
    ]
    entries_by_race: dict[str, list[dict]] = defaultdict(list)
    entry_cols = (
        "race_id",
        "post_position",
        "horse_id",
        "horse_age",
        "horse_sex",
        "weight_carried",
        "horse_weight",
        "weight_change",
        "jockey_id",
        "trainer_id",
        "finish_position",
        "finish_time",
        "last_3f",
        "corner_1",
        "corner_2",
        "corner_3",
        "corner_4",
    )
    for row in conn.execute(ENTRIES_SQL):
        rec = dict(zip(entry_cols, row))
        entries_by_race[rec["race_id"]].append(rec)

    races_by_date: dict[str, list[dict]] = defaultdict(list)
    for race in races:
        if not entries_by_race.get(race["race_id"]):
            continue
        race["date_obj"] = datetime.strptime(race["race_date"], "%Y-%m-%d").date()
        races_by_date[race["race_date"]].append(race)

    builder = FeatureBuilder()
    all_rows: list[dict] = []
    for race_date in sorted(races_by_date):
        day_races = sorted(races_by_date[race_date], key=lambda r: r["race_id"])
        # emit every race of the day first, then ingest the day's results:
        # same-day races (and the race itself) can never leak into features.
        for race in day_races:
            all_rows.extend(builder.emit_race(race, entries_by_race[race["race_id"]]))
        for race in day_races:
            builder.ingest_race(race, entries_by_race[race["race_id"]])

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df[
            ["race_id", "race_date", "horse_id", "finish_position", "label_win"]
            + FEATURE_COLUMNS
        ]
    return df


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/historical/keiba.db")
    parser.add_argument("--out", default="data/processed/features_v2.parquet")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        df = build_features(conn)
    finally:
        conn.close()

    df.to_parquet(args.out, index=False)
    n_races = df["race_id"].nunique() if not df.empty else 0
    print(f"wrote {len(df)} rows ({n_races} races) to {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
