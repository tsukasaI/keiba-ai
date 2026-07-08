"""Tests for src/preprocessing/features_v2.py (V2 Task 1.1).

The leakage property tests are the point of this file: features for a race
must be identical when rebuilt from a DB that contains (a) only strictly
earlier races plus the target race, and (b) the same but with the target
race's own results nulled out. Same-day sibling races and NAR races must not
influence features either.
"""

import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features_v2 import (
    ENTRIES_SQL,
    FEATURE_COLUMNS,
    RACES_SQL,
    build_features,
)

SCHEMA = """
CREATE TABLE races (
    race_id TEXT PRIMARY KEY, race_date TEXT NOT NULL, racecourse TEXT NOT NULL,
    race_number INTEGER NOT NULL, race_name TEXT, distance INTEGER NOT NULL,
    surface TEXT NOT NULL, track_condition TEXT, grade TEXT,
    field_size INTEGER, weather TEXT, created_at TEXT
);
CREATE TABLE race_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, race_id TEXT NOT NULL,
    post_position INTEGER NOT NULL, horse_id TEXT NOT NULL, horse_name TEXT,
    horse_age INTEGER, horse_sex TEXT, weight_carried REAL, horse_weight INTEGER,
    weight_change INTEGER, jockey_id TEXT, jockey_name TEXT, trainer_id TEXT,
    trainer_name TEXT, finish_position INTEGER, finish_time REAL, margin TEXT,
    last_3f REAL, corner_1 INTEGER, corner_2 INTEGER, corner_3 INTEGER,
    corner_4 INTEGER, win_odds REAL, popularity INTEGER
);
"""

RACE_A = "202406010101"  # day 1, Nakayama
RACE_B = "202406010102"  # day 1, Nakayama (same-day sibling)
RACE_N = "202444010101"  # day 1, NAR track (course code 44) — must be ignored
RACE_C = "202406010201"  # day 2, the target race
DAY1, DAY2 = "2024-01-06", "2024-01-07"


def _races():
    common = dict(racecourse="中山", distance=1600, surface="turf", track_condition="良")
    return [
        dict(race_id=RACE_A, race_date=DAY1, race_number=1, grade="", **common),
        dict(race_id=RACE_B, race_date=DAY1, race_number=2, grade="", **common),
        dict(race_id=RACE_N, race_date=DAY1, race_number=1, grade="",
             racecourse="大井", distance=1200, surface="dirt", track_condition="良"),
        dict(race_id=RACE_C, race_date=DAY2, race_number=1, grade="", **common),
    ]


def _entry(race_id, post, horse, jockey, trainer, fp, time=None, l3f=None, corner=None):
    return dict(
        race_id=race_id, post_position=post, horse_id=horse, horse_age=4,
        horse_sex="牡", weight_carried=57.0, horse_weight=480, weight_change=0,
        jockey_id=jockey, trainer_id=trainer, finish_position=fp,
        finish_time=time, last_3f=l3f,
        corner_1=corner, corner_2=corner, corner_3=corner, corner_4=corner,
        win_odds=2.5, popularity=1,
    )


def _entries():
    rows = []
    # Race A: H1(J1) wins; clean 5-runner race with hand-computable percentiles.
    for i, (horse, jockey, fp) in enumerate(
        [("H1", "J1", 1), ("H2", "J2", 2), ("H3", "J3", 3), ("H4", "J4", 4), ("H5", "J5", 5)],
        start=1,
    ):
        rows.append(_entry(RACE_A, i, horse, jockey, f"T{i}", fp,
                           time=95.0 + fp, l3f=32.0 + fp, corner=fp))
    # Race B (same day): J1 rides again and wins on H6.
    for i, (horse, jockey, fp) in enumerate(
        [("H6", "J1", 1), ("H7", "J7", 2), ("H8", "J8", 3), ("H9", "J9", 4), ("H10", "J10", 5)],
        start=1,
    ):
        rows.append(_entry(RACE_B, i, horse, jockey, f"T{i}", fp,
                           time=95.0 + fp, l3f=32.0 + fp, corner=fp))
    # NAR race: H1 "wins" it — must NOT count toward H1's history.
    rows.append(_entry(RACE_N, 1, "H1", "J1", "T1", 1, time=70.0))
    rows.append(_entry(RACE_N, 2, "HX", "JX", "TX", 2, time=71.0))
    # Race C (day 2): H1 switches to J2; H6 keeps J1; H11 debuts; H3 scratched.
    rows.append(_entry(RACE_C, 1, "H1", "J2", "T1", 1, time=96.0, l3f=33.0, corner=1))
    rows.append(_entry(RACE_C, 2, "H6", "J1", "T6", 2, time=96.5, l3f=33.5, corner=2))
    rows.append(_entry(RACE_C, 3, "H11", "J3", "T11", 3, time=97.0, l3f=34.0, corner=3))
    rows.append(_entry(RACE_C, 4, "H2", "J4", "T2", 4, time=97.5, l3f=34.5, corner=4))
    rows.append(_entry(RACE_C, 5, "H3", "J5", "T3", None))  # scratched
    return rows


def make_db(races=None, entries=None) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    for r in races if races is not None else _races():
        conn.execute(
            """INSERT INTO races (race_id, race_date, racecourse, race_number,
               distance, surface, track_condition, grade)
               VALUES (:race_id, :race_date, :racecourse, :race_number,
               :distance, :surface, :track_condition, :grade)""",
            r,
        )
    for e in entries if entries is not None else _entries():
        conn.execute(
            """INSERT INTO race_entries (race_id, post_position, horse_id,
               horse_age, horse_sex, weight_carried, horse_weight, weight_change,
               jockey_id, trainer_id, finish_position, finish_time, last_3f,
               corner_1, corner_2, corner_3, corner_4, win_odds, popularity)
               VALUES (:race_id, :post_position, :horse_id, :horse_age,
               :horse_sex, :weight_carried, :horse_weight, :weight_change,
               :jockey_id, :trainer_id, :finish_position, :finish_time, :last_3f,
               :corner_1, :corner_2, :corner_3, :corner_4, :win_odds, :popularity)""",
            e,
        )
    conn.commit()
    return conn


@pytest.fixture(scope="module")
def full_features() -> pd.DataFrame:
    return build_features(make_db())


def _row(df, race_id, horse_id) -> pd.Series:
    sel = df[(df["race_id"] == race_id) & (df["horse_id"] == horse_id)]
    assert len(sel) == 1
    return sel.iloc[0]


class TestStaticNoOdds:
    def test_sql_never_selects_odds_or_popularity(self):
        for sql in (RACES_SQL, ENTRIES_SQL):
            assert "win_odds" not in sql
            assert "popularity" not in sql

    def test_no_odds_named_feature(self):
        for col in FEATURE_COLUMNS:
            assert "odds" not in col.lower()
            assert "popular" not in col.lower()


class TestBasicCorrectness:
    def test_history_features_for_h1_at_race_c(self, full_features):
        row = _row(full_features, RACE_C, "H1")
        assert row["career_starts"] == 1.0  # NAR win must NOT count
        assert row["career_win_rate"] == 1.0
        assert row["career_place_rate"] == 1.0
        assert row["days_since_last_run"] == 1.0
        assert row["last1_norm_finish"] == pytest.approx(1 / 5)
        assert row["last3_avg_norm_finish"] == pytest.approx(1 / 5)
        assert row["last3_avg_last3f_pctl"] == pytest.approx(1 / 5)
        assert row["last3_avg_corner_pctl"] == pytest.approx(1 / 5)
        assert row["jockey_switch"] == 1.0  # J1 -> J2
        assert row["surface_win_rate"] == 1.0
        assert row["course_win_rate"] == 1.0
        assert row["class_move"] == 0.0

    def test_jockey_stats_accumulate_across_days(self, full_features):
        # J1 won both day-1 races; H6 keeps J1 at race C.
        row = _row(full_features, RACE_C, "H6")
        assert row["jockey_starts"] == 2.0
        assert row["jockey_win_rate"] == 1.0
        assert row["jockey_switch"] == 0.0
        # J2 rode H2 to 2nd on day 1.
        row_h1 = _row(full_features, RACE_C, "H1")
        assert row_h1["jockey_starts"] == 1.0
        assert row_h1["jockey_win_rate"] == 0.0
        assert row_h1["jockey_place_rate"] == 1.0

    def test_debut_horse_is_nan_not_zero(self, full_features):
        row = _row(full_features, RACE_C, "H11")
        assert row["career_starts"] == 0.0
        assert math.isnan(row["career_win_rate"])
        assert math.isnan(row["days_since_last_run"])
        assert math.isnan(row["jockey_switch"])
        assert math.isnan(row["last1_norm_finish"])

    def test_scratched_entry_gets_row_with_nan_label(self, full_features):
        row = _row(full_features, RACE_C, "H3")
        assert math.isnan(row["label_win"])
        assert row["career_starts"] == 1.0  # ran on day 1

    def test_same_day_sibling_race_not_visible(self, full_features):
        # Race B ran after race A on the same day (by race_id order), but J1's
        # race-A win must not be visible to race B's features.
        row = _row(full_features, RACE_B, "H6")
        assert row["jockey_starts"] == 0.0
        assert math.isnan(row["jockey_win_rate"])

    def test_nar_race_emits_no_rows(self, full_features):
        assert (full_features["race_id"] == RACE_N).sum() == 0


class TestLeakageProperty:
    """Rebuilding from a truncated DB must reproduce the target race's features."""

    def _compare_race_c(self, full_features, truncated_conn):
        df_trunc = build_features(truncated_conn)
        a = (
            full_features[full_features["race_id"] == RACE_C]
            .sort_values("horse_id")
            .reset_index(drop=True)
        )
        b = (
            df_trunc[df_trunc["race_id"] == RACE_C]
            .sort_values("horse_id")
            .reset_index(drop=True)
        )
        assert list(a["horse_id"]) == list(b["horse_id"])
        np.testing.assert_allclose(
            a[FEATURE_COLUMNS].to_numpy(dtype=float),
            b[FEATURE_COLUMNS].to_numpy(dtype=float),
            equal_nan=True,
            err_msg="features changed when future/self data was removed — leakage",
        )

    def test_strictly_earlier_plus_target_only(self, full_features):
        races = [r for r in _races() if r["race_date"] < DAY2 or r["race_id"] == RACE_C]
        keep = {r["race_id"] for r in races}
        entries = [e for e in _entries() if e["race_id"] in keep]
        self._compare_race_c(full_features, make_db(races, entries))

    def test_target_own_results_nulled(self, full_features):
        entries = []
        for e in _entries():
            e = dict(e)
            if e["race_id"] == RACE_C:
                e.update(
                    finish_position=None, finish_time=None, last_3f=None,
                    corner_1=None, corner_2=None, corner_3=None, corner_4=None,
                )
            entries.append(e)
        self._compare_race_c(full_features, make_db(_races(), entries))
