"""Task 1.3 CLI — walk-forward delta-R2 of (fundamental + market blend) vs market.

This is the GATE 1 instrument (docs/V2_TASKS.md sections 2.4, 2.5, 4): for each
eval year Y it trains the odds-free fundamental model on years <= Y-2, fits the
blend (alpha, beta) and the market recalibration beta on year Y-1, and reports
McFadden pseudo-R2 on year Y for the recalibrated market baseline vs the blend,
plus a race-resampling bootstrap 95% CI for the pooled delta-R2.

Usage:
    uv run python scripts/eval_delta_r2.py --db data/historical/keiba.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.market_eval import (  # noqa: E402
    JRA_ONLY_SQL,
    MIN_RUNNERS,
    blend_probs,
    bootstrap_delta_r2,
    fit_blend,
    fit_power_recalibration,
    mcfadden_r2,
    power_probs,
    prepare_race,
    winner_prob,
    PreparedRace,
)
from src.models.fundamental_model import (  # noqa: E402
    add_year,
    predict_raw,
    train_fundamental,
    walk_forward_splits,
)

MIN_FIT_RACES = 100  # skip an eval year whose blend-fit year has fewer usable races


@dataclass
class AlignedRace:
    """One eval-ready race: market probs, fundamental probs, outcome."""

    race_id: str
    year: int
    pi: np.ndarray
    f: np.ndarray
    winner_mask: np.ndarray

    @property
    def n_runners(self) -> int:
        return len(self.pi)

    @property
    def n_winners(self) -> int:
        return int(self.winner_mask.sum())


def load_odds_rows(db_path: str) -> dict[str, dict]:
    """Per race: aligned horse_id / win_odds / finish_position arrays + year."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        rows = conn.execute(
            f"""
            SELECT r.race_id, substr(r.race_date, 1, 4), e.horse_id,
                   e.win_odds, e.finish_position
            FROM races r
            JOIN race_entries e ON e.race_id = r.race_id
            WHERE {JRA_ONLY_SQL.format(alias="r")}
            ORDER BY r.race_id, e.post_position
            """
        ).fetchall()
    finally:
        conn.close()
    races: dict[str, dict] = {}
    for race_id, year, horse_id, odds, fp in rows:
        rec = races.setdefault(
            race_id, {"year": int(year), "horse_ids": [], "odds": [], "fp": []}
        )
        rec["horse_ids"].append(horse_id)
        rec["odds"].append(odds)
        rec["fp"].append(fp)
    return races


def align_races(
    race_rows: dict[str, dict],
    preds: dict[tuple[str, str], float],
    years: set[int],
) -> tuple[list[AlignedRace], dict[str, int]]:
    """Apply the section-2.1 drop rules and attach race-normalized f."""
    aligned: list[AlignedRace] = []
    drops: dict[str, int] = defaultdict(int)
    for race_id, rec in race_rows.items():
        if rec["year"] not in years:
            continue
        prepared, reason = prepare_race(rec["odds"], rec["fp"], min_runners=MIN_RUNNERS)
        if prepared is None:
            drops[reason] += 1
            continue
        runner_mask = _runner_mask(rec)
        runner_ids = [h for h, keep in zip(rec["horse_ids"], runner_mask) if keep]
        raw = np.array([preds.get((race_id, h), np.nan) for h in runner_ids])
        if np.isnan(raw).any():
            drops["no_prediction"] += 1
            continue
        total = raw.sum()
        f = raw / total if total > 0 else np.full(len(raw), 1.0 / len(raw))
        aligned.append(
            AlignedRace(
                race_id=race_id,
                year=rec["year"],
                pi=prepared.pi,
                f=f,
                winner_mask=prepared.winner_mask,
            )
        )
    return aligned, dict(drops)


def _runner_mask(rec: dict) -> list[bool]:
    """Same runner rule as prepare_race: valid odds AND a finish position."""
    mask = []
    for odds, fp in zip(rec["odds"], rec["fp"]):
        ok_odds = odds is not None and np.isfinite(odds) and odds > 1.0
        mask.append(bool(ok_odds and fp is not None))
    return mask


def year_metrics(
    races: list[AlignedRace], alpha: float, beta_blend: float, beta_market: float
) -> dict:
    market_wp, blend_wp, fund_wp, ns, ks = [], [], [], [], []
    for r in races:
        market_wp.append(winner_prob(power_probs(r.pi, beta_market), r.winner_mask))
        blend_wp.append(
            winner_prob(blend_probs(r.f, r.pi, alpha, beta_blend), r.winner_mask)
        )
        fund_wp.append(winner_prob(r.f, r.winner_mask))
        ns.append(r.n_runners)
        ks.append(r.n_winners)
    r2_market = mcfadden_r2(market_wp, ns, ks)
    r2_blend = mcfadden_r2(blend_wp, ns, ks)
    return {
        "n_races": len(races),
        "r2_fundamental": mcfadden_r2(fund_wp, ns, ks),
        "r2_market": r2_market,
        "r2_blend": r2_blend,
        "delta_r2": r2_blend - r2_market,
        "market_wp": np.array(market_wp),
        "blend_wp": np.array(blend_wp),
        "field_sizes": np.array(ns, dtype=float),
        "n_winners": np.array(ks, dtype=float),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/historical/keiba.db")
    parser.add_argument("--features", default="data/processed/features_v2.parquet")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild the feature parquet from the DB before evaluating",
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--out-dir", default="data/models/v2")
    args = parser.parse_args()

    import pandas as pd

    features_path = Path(args.features)
    if args.rebuild_features or not features_path.exists():
        from src.preprocessing.features_v2 import build_features

        print(f"building features from {args.db} ...")
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            df = build_features(conn)
        finally:
            conn.close()
        features_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(features_path, index=False)
    features = add_year(pd.read_parquet(features_path))

    race_rows = load_odds_rows(args.db)
    years = sorted(int(y) for y in features["year"].unique())
    splits = walk_forward_splits(years)
    if not splits:
        print(f"error: no walk-forward split possible from years {years}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_year = []
    pooled = {"market_wp": [], "blend_wp": [], "ns": [], "ks": []}
    for eval_year, blend_year, train_years in splits:
        df_train = features[features["year"].isin(train_years)]
        df_blend = features[features["year"] == blend_year]
        df_eval = features[features["year"] == eval_year]
        if df_train.empty or df_blend.empty or df_eval.empty:
            continue

        booster = train_fundamental(df_train, df_blend)
        pred_rows = pd.concat([df_blend, df_eval], ignore_index=True)
        raw = predict_raw(booster, pred_rows)
        preds = {
            (rid, hid): p
            for rid, hid, p in zip(pred_rows["race_id"], pred_rows["horse_id"], raw)
        }

        blend_races, blend_drops = align_races(race_rows, preds, {blend_year})
        eval_races, eval_drops = align_races(race_rows, preds, {eval_year})
        if len(blend_races) < MIN_FIT_RACES or not eval_races:
            print(
                f"skip eval {eval_year}: blend-fit races={len(blend_races)} "
                f"(need >= {MIN_FIT_RACES}), eval races={len(eval_races)}"
            )
            continue

        beta_market = fit_power_recalibration(
            [PreparedRace(pi=r.pi, winner_mask=r.winner_mask) for r in blend_races]
        )
        alpha, beta_blend = fit_blend(
            [r.f for r in blend_races],
            [r.pi for r in blend_races],
            [r.winner_mask for r in blend_races],
        )

        m = year_metrics(eval_races, alpha, beta_blend, beta_market)
        _, lo, hi = bootstrap_delta_r2(
            m["blend_wp"], m["market_wp"], m["field_sizes"], m["n_winners"],
            n_boot=args.n_boot, seed=0,
        )
        booster.save_model(str(out_dir / f"fundamental_eval{eval_year}.txt"))
        per_year.append(
            {
                "eval_year": eval_year,
                "blend_year": blend_year,
                "train_years": train_years,
                "n_races": m["n_races"],
                "alpha": alpha,
                "beta_blend": beta_blend,
                "beta_market": beta_market,
                "r2_fundamental": m["r2_fundamental"],
                "r2_market": m["r2_market"],
                "r2_blend": m["r2_blend"],
                "delta_r2": m["delta_r2"],
                "delta_r2_ci95": [lo, hi],
                "drops": {"blend": blend_drops, "eval": eval_drops},
            }
        )
        pooled["market_wp"].append(m["market_wp"])
        pooled["blend_wp"].append(m["blend_wp"])
        pooled["ns"].append(m["field_sizes"])
        pooled["ks"].append(m["n_winners"])

    if not per_year:
        print("error: no eval year had enough data", file=sys.stderr)
        return 1

    point, lo, hi = bootstrap_delta_r2(
        np.concatenate(pooled["blend_wp"]),
        np.concatenate(pooled["market_wp"]),
        np.concatenate(pooled["ns"]),
        np.concatenate(pooled["ks"]),
        n_boot=args.n_boot,
        seed=0,
    )

    header = (
        f"{'eval':<6}{'races':>6}  {'R2_fund':>8}  {'R2_mkt':>8}  {'R2_blend':>9}  "
        f"{'dR2':>8}  {'dR2 95% CI':>20}  {'alpha':>6}  {'beta':>6}"
    )
    print(header)
    print("-" * len(header))
    for m in per_year:
        ci = m["delta_r2_ci95"]
        print(
            f"{m['eval_year']:<6}{m['n_races']:>6}  {m['r2_fundamental']:>8.4f}  "
            f"{m['r2_market']:>8.4f}  {m['r2_blend']:>9.4f}  {m['delta_r2']:>+8.4f}  "
            f"[{ci[0]:>+8.4f}, {ci[1]:>+8.4f}]  {m['alpha']:>6.3f}  {m['beta_blend']:>6.3f}"
        )
    print()
    print(
        f"pooled delta-R2 = {point:+.4f}, bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}] "
        f"(n_boot={args.n_boot})"
    )
    gate = (
        "PASS (preliminary)"
        if all(m["delta_r2"] > 0 for m in per_year) and lo > 0
        else "FAIL (preliminary)"
    )
    print(f"GATE 1 (delta-R2 > 0 every year AND pooled CI excludes 0): {gate}")
    print(
        "caveats: historical final win odds (not bet-time) => market baseline is "
        "optimistic; data may be mid-backfill (Task 0.2) => rerun after completion."
    )

    report = {
        "pooled": {"delta_r2": point, "ci95": [lo, hi], "n_boot": args.n_boot},
        "gate1_preliminary": gate,
        "per_year": [
            {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
            for m in per_year
        ],
    }
    (out_dir / "delta_r2_report.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    lines = [
        "# delta-R2 report (walk-forward)",
        "",
        "| eval | races | R2_fund | R2_market | R2_blend | dR2 | dR2 95% CI | alpha | beta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in per_year:
        ci = m["delta_r2_ci95"]
        lines.append(
            f"| {m['eval_year']} | {m['n_races']} | {m['r2_fundamental']:.4f} "
            f"| {m['r2_market']:.4f} | {m['r2_blend']:.4f} | {m['delta_r2']:+.4f} "
            f"| [{ci[0]:+.4f}, {ci[1]:+.4f}] | {m['alpha']:.3f} | {m['beta_blend']:.3f} |"
        )
    lines += [
        "",
        f"pooled delta-R2 = {point:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]",
        f"GATE 1: {gate}",
    ]
    (out_dir / "delta_r2_report.md").write_text("\n".join(lines))
    print(f"report written to {out_dir}/delta_r2_report.{{md,json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
