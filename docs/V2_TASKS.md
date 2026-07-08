# V2 Tasks — Executable Breakdown of STRATEGY_V2

This file is the **execution spec** for the V2 rebuild. The rationale and evidence live in
`docs/STRATEGY_V2.md` (Japanese). This file is written so that a fresh Claude session
(any model tier) can continue the build **without re-researching or re-deriving decisions**.

## How to use this file (read this first)

1. Read `docs/STRATEGY_V2.md` once for the why. This file is the how.
2. Find the task marked **▶ NEXT**. Do it end-to-end: code + tests + the task's
   Verify block. Then: check the box, move **▶ NEXT** to the following task, fill in
   the Status Ledger (§9), and update `HANDOVER.md`.
3. Tasks are strictly ordered unless marked `[parallel-ok]`.
4. **Settled Decisions (§1) are settled.** Do not re-open them unless a GATE fails.
   The formulas in §2 are the spec — implement them as written; do not substitute
   alternatives from memory.
5. Verification discipline (repo rule): `PYTHONPATH=. uv run pytest tests/ -v` and
   `cd src/api && cargo test` must be green before declaring any task done.
6. When something here contradicts reality (a path moved, a schema changed), trust
   reality, fix the doc, and note it in `HANDOVER.md`.

---

## 1. Settled Decisions (do not re-litigate)

| # | Decision | Rationale (short) |
|---|---|---|
| D1 | The fundamental model contains **no odds/popularity-derived features**. Market info enters only at the blend stage (§2.4). | Odds-as-feature makes the model mirror the market (deployed CatBoost: odds = top feature, 23%) — no separable edge. Benter's central lesson. |
| D2 | The model-quality metric is **ΔR² vs the market** (§2.2–2.5), not backtest ROI. ROI appears only in paper/live reports with confidence intervals. | ΔR² is computable from data we already have, and is orders of magnitude lower-variance than ROI. ΔR² ≤ 0 ⇒ no betting scheme can win long-run. |
| D3 | Bet-type order: validate on **win/quinella/wide** (takeout 20–22.5%, verifiable sample sizes); extend to exacta/trio after GATE 2; trifecta last if ever. | Verifying an edge on trifecta needs 10⁴–10⁵ bets (fat-tailed payouts). Win-level bets verify in ~10³. |
| D4 | EV threshold starts at **1.1** (config), never 1.0. Longshot exclusion active (§2.7). | EV filters preferentially select the model's most over-estimated probabilities (winner's curse). |
| D5 | Kelly fraction **0.25**, with per-bet and per-day caps (§2.7). | Estimation error compounds nonlinearly in Kelly; practitioner standard is ¼–½. |
| D6 | JRA-VAN purchase is **deferred until GATE 1 passes**. | If ΔR² ≤ 0 on free data, paying for data is premature; if it passes, JRA-VAN features are the first enrichment lever. |
| D7 | Real-money betting is **out of scope until GATE 2 passes** (§4). IPAT automation out of scope until Phase 4 explicitly unlocks it. | Only verified paper evidence justifies real stakes. |
| D8 | Python (uv) for research/training; Rust for the live/paper pipeline. Keep the existing scraper, SQLite storage, and paper-record/settle/report loop; **rebuild the model and evaluation layers**. | The capture infra matches what the research says is the only viable odds source; the model layer is the broken part. |
| D9 | Walk-forward time splits everywhere. Never random splits. Never fit anything on the eval year. | Leakage is the #1 documented failure mode in this domain. |
| D10 | Primary training data = netkeiba SQLite (2022–2025). Kaggle (1986–2021) is optional **train-only** augmentation later — its race_ids do not join netkeiba's (known 0-overlap trap, see `docs/TODO_IMPROVEMENTS.md`). | Avoids the cross-dataset join failure that already burned one calibration attempt. |

---

## 2. Exact formulas (the spec)

Notation: race r has runners i = 1..n_r (only horses that actually ran: `win_odds` present
and > 1.0 **and** `finish_position` present). w_r = winner(s) of r.

### 2.1 Market-implied win probability

Decimal win odds O_i (単勝, includes ~20% takeout):

```
π_i = (1/O_i) / Σ_{j∈r} (1/O_j)
```

Normalization removes takeout/breakage. Rules:
- Drop a **horse** from the race if O missing or ≤ 1.0 (scratch or parse gap).
- Drop the **race** if the winner is dropped, or if fewer than 5 runners remain. Count and report drops.
- Dead heat (同着, multiple `finish_position = 1`): the "winner probability" is the **sum** of π over all winners.

### 2.2 Pseudo-R² (McFadden, winner log-likelihood)

For any within-race probability assignment p:

```
LL(p) = Σ_r ln p_{w_r}          LL_0 = Σ_r ln(1/n_r)          R²(p) = 1 − LL(p)/LL_0
```

Clip every probability to [1e-6, 1.0] before ln; renormalize within race after clipping.
Sanity anchors (from literature, Hong Kong): R²(market) ≈ 0.13 (1986–93), ≈ 0.19 (2016–23).
If our market R² lands far outside **[0.05, 0.30]**, suspect a data or formula bug before
believing it.

### 2.3 Market-only baseline (recalibrated — this is what "the market" means in ΔR²)

```
b_i(β) = π_i^β / Σ_j π_j^β        β fit by MLE (maximize LL(b)) on the blend-fit slice
```

R²_market = R²(b). Using the recalibrated baseline prevents claiming ΔR² > 0 from mere
recalibration of the market. Expect β near 1.

### 2.4 Fundamental probability and blend

Fundamental model (D1: no odds features) produces per-horse win estimate p̂_i.
- **Primary path**: LightGBM binary classifier (label: won = 1), then within-race
  normalize: `f_i = p̂_i / Σ_j p̂_j`.
- **Optional experiment**: LightGBM `lambdarank` scores s_i, `f_i = softmax(s_i / T)` with
  temperature T fit by MLE on the blend-fit slice. Keep whichever gives higher eval-year ΔR².

Blend (2-parameter conditional logit):

```
c_i(α, β) = exp(α·ln f_i + β·ln π_i) / Σ_j exp(α·ln f_j + β·ln π_j)
```

Fit (α, β) by MLE on the blend-fit slice (scipy.optimize.minimize on negative LL,
x0 = (0.5, 1.0); Nelder-Mead or L-BFGS-B both fine). Expect α ∈ (0, 1.5), β ∈ (0.5, 1.5).
α ≤ 0 means the fundamental model adds nothing — consistent with ΔR² ≈ 0.

### 2.5 ΔR², walk-forward protocol, and the confidence interval

```
ΔR² = R²(c) − R²(b)     — both evaluated on the SAME eval-year races
```

Walk-forward protocol (netkeiba data, after Task 0.2 completes):

| Eval year Y | Fundamental trained on | Blend/β/T fit on |
|---|---|---|
| 2024 | 2022 | 2023 |
| 2025 | 2022–2023 | 2024 |
| 2026+ (live) | all ≤ Y−2 | Y−1 |

**Bootstrap CI**: resample eval races with replacement (pooled eval years), B = 10,000;
recompute ΔR² per resample (LL_0 recomputed per resample too); report the 2.5/97.5
percentiles.

### 2.6 Corrected Harville (Phase 2 — exotic probabilities)

Raw Harville systematically overestimates favorites' 2nd/3rd chances. Corrected form:

```
P(1st = i)              = c_i
P(2nd = j | 1st = i)    = c_j^γ / Σ_{k≠i} c_k^γ
P(3rd = k | 1st=i,2nd=j)= c_k^δ / Σ_{l≠i,j} c_l^δ
```

Fit γ by MLE on observed (1st, 2nd) pairs over training years (γ = 1 recovers raw
Harville); fit δ on (1st, 2nd, 3rd) triples. Literature anchors (HK): γ ≈ 0.81, δ ≈ 0.65.
Expect JRA values somewhere in 0.6–1.0 — **refit on our data; never reuse literature
constants**. Unit test: γ = δ = 1 must reproduce the existing `exacta.rs`/`trifecta.rs`
outputs exactly.

### 2.7 EV and staking (Phase 2 — bet engine v2)

```
EV(combo) = P_corrected(combo) × O_combo        bet iff EV ≥ 1.1        (config: ev_threshold)
```

Additional filters (all config, tune only on paper evidence):
- **Longshot exclusion**: skip any combo where a leg's blended win prob c_i < 0.02.
- **Stake**: quarter-Kelly `stake_fraction = 0.25 × max(0, (P·O − 1)/(O − 1))`,
  capped at 1% of bankroll per bet and 5% per race day; round to ¥100, min ¥100.
- Record as late as safely possible (target: post time − 2–4 min; the pipeline takes
  ~30–40 s and all fetch paths are timeout-bounded since commit `6b011ef`).

### 2.8 Leakage rules (hard constraints, apply to every feature)

1. Every feature must be computable **strictly before** the race's date
   (`race_date` comparison; same-day earlier races are also excluded — simpler and safe).
2. No feature may derive from the current race's odds or popularity (D1). `win_odds`
   and `popularity` columns of the current race are forbidden inputs to the fundamental model.
3. Aggregates (jockey win rate, horse form, …) must be as-of joins over strictly-earlier
   races only.
4. Time splits only (D9).
5. First-time values (debut horse, new jockey) are **NaN, not 0** — LightGBM handles NaN natively.

---

## 3. Repo execution notes (for any session)

- Python: run from repo root, `PYTHONPATH=. uv run pytest tests/ -v`. Scripts:
  `uv run python scripts/<name>.py`.
- Rust: `cd src/api && cargo build --release && cargo test`. Binary:
  `src/api/target/release/keiba-api`.
- SQLite: `data/historical/keiba.db`. Key tables: `races` (race_id, race_date, racecourse,
  distance, surface, track_condition, grade, field_size, weather), `race_entries`
  (race_id, post_position, horse_id, …, finish_position, finish_time, last_3f,
  corner_1..4, win_odds, popularity), `odds_snapshots`, `paper_bets`, `race_payouts`.
- Kaggle raw CSVs: `data/raw/19860105-20210731_{race_result,odds,laptime}.csv`,
  `data/raw/20020615-20210731_corner_passing_order.csv`. Existing loaders:
  `src/models/data_loader.py`, `src/models/odds_loader.py` — **inspect before writing new
  parsers; do not guess column names.**
- Hooks: `grep`/`find` are blocked — use `rg`/`fd`. `gh pr create`/`gh pr merge`,
  `git restore`, `git checkout --` are blocked — present the command for manual run.
  `rustfmt main.rs` reformats the whole crate (mod recursion) — format deliberately.
- Commits: English, Contextual Commits format (see user CLAUDE.md). Do not commit unless
  the user asked.
- `db.netkeiba.com` lags ~1 day for results; `race.netkeiba.com` is immediate. The lag
  only matters for same-day work; past-year scraping is unaffected.
- End of session: update `HANDOVER.md` (state, next task, open issues).

---

## 4. Gates

**GATE 1 (after Task 1.3)** — Pass iff:
(a) ΔR² > 0 in **every** walk-forward eval year individually, **and**
(b) the pooled bootstrap 95% CI for ΔR² excludes 0.
On first fail: one retry allowed after feature enrichment (Task 1.4; consider early
JRA-VAN). On second fail: **stop — no betting phases.** Record the outcome in §9 either way.

**GATE 2 (during Phase 3)** — Pass iff, over ≥ 1,000 settled paper bets on
win/quinella/wide combined:
(a) pooled ROI point estimate > 0, **and**
(b) bootstrap 95% CI lower bound > −22.5% (i.e., statistically better than random betting
at the same takeout), **and**
(c) live ΔR² (computed against bet-time odds, §Phase 3) > 0 with CI excluding 0.
Note: (b) is deliberately weak — 1,000 bets cannot prove ROI > 0. That is why Phase 4
starts small. Do not weaken (a) or (c).

**Exit rules** (agreed in STRATEGY_V2 §5): GATE 1 failed twice → no real betting, ever,
on this model line. GATE 2 unmet → stay in paper. Phase 4 monthly drawdown limit hit →
stop, return to paper.

---

## 5. Phase 0 — Foundation

### ✅ Task 0.1 (done 2026-07-08): Market baseline + ΔR² evaluation harness

**Context.** Everything downstream is judged by this harness (D2). It must exist before
any model work. Needs only `finish_position` + `win_odds` — both already in SQLite
(coverage 2022: 35,000/38,314 entries with valid odds; missing = scratches/parse gaps,
handled by §2.1 drop rules).

**Deliverables.**
- `src/models/market_eval.py` — pure functions, no I/O:
  - `implied_probs(odds: np.ndarray) -> np.ndarray` (§2.1)
  - `mcfadden_r2(prob_winner: np.ndarray, field_sizes: np.ndarray) -> float` (§2.2)
  - `fit_power_recalibration(pi_by_race, winner_idx) -> float` (β, §2.3)
  - `fit_blend(f_by_race, pi_by_race, winner_idx) -> tuple[float, float]` (α, β, §2.4)
  - `bootstrap_delta_r2(...) -> tuple[float, float, float]` (point, lo, hi, §2.5)
  - The API must accept fundamental probs `f` as an optional input now (used from Phase 1),
    and work in market-only mode today.
- `scripts/eval_market_baseline.py` — CLI:
  `uv run python scripts/eval_market_baseline.py --db data/historical/keiba.db`
  prints a per-year table: races total / used / dropped (+why), n_entries used,
  R²_market with β = 1, and R²_market with β fit on the previous year (skip β-fit for the
  earliest year). Optional `--kaggle` mode may reuse `data_loader.py`/`odds_loader.py`
  (verify columns first); netkeiba mode is the required part.
- `tests/test_market_eval.py` — minimum cases:
  - odds [2.0, 2.0] → π = [0.5, 0.5]; single 2-horse race → R² = 0 exactly.
  - uniform odds over many races → R² = 0; perfect prediction (p_winner → 1) → R² → 1.
  - takeout invariance: scaling all 1/O by a constant leaves π unchanged.
  - drop rules: horse with O ≤ 1.0 excluded; race dropped when winner has no odds;
    dead-heat sums winner probs; < 5 runners → race dropped.
  - `fit_blend` on synthetic data where truth = market (α ≈ 0 recovered) and where
    truth = fundamental (β ≈ 0 recovered).

**Verify.**
```bash
PYTHONPATH=. uv run pytest tests/test_market_eval.py -v          # green
uv run python scripts/eval_market_baseline.py --db data/historical/keiba.db
# R²_market per year within [0.05, 0.30]; if outside, debug before proceeding.
```
Then fill §9 ledger with the per-year R²_market numbers.

**Gotchas.** Races table has `field_size` but trust the entry rows actually loaded, not
the declared field size. (The old LightGBM `libomp.dylib` dlopen failure was fixed
2026-07-08 via `brew install libomp` — both test suites run green: 219 Python / 116 Rust.)

### ▶ NEXT — Task 0.2 `[parallel-ok]` — Historical data completion (2023 / 2024 gaps / 2025)

_Status 2026-07-08 09:01 JST: scrape relaunched in background (2023 → 2025 → 2024) after
the first attempt died: a timed-out `keiba-api` run leaked its headless Chrome, which
held chromiumoxide's shared profile lock, and every later date failed in ~3 s with
"SingletonLock: File exists" (104/104 failures). The orchestrator is now hardened
(commit `c4c9545`): orphan-browser cleanup at start + after any failed date with one
retry, and it iterates **all** dates (the old weekend-only filter silently skipped JRA
holiday meetings). If the scrape died with the session, just re-run the commands below —
already-scraped races are skipped. What remains for this task: wait for completion, run
the verification queries, **re-scrape with `--force` the dates of races that have a
`races` row but no `race_entries` rows** (2022: 155, 2023: 28, 2024: 8 as of 2026-07-08 —
find them via the no-entries query below), then re-run the Task 0.1 CLI and update the
ledger._

```bash
# find dates whose races lack entry rows (need --force re-scrape):
sqlite3 data/historical/keiba.db "SELECT DISTINCT race_date FROM races r WHERE NOT EXISTS (SELECT 1 FROM race_entries e WHERE e.race_id = r.race_id) ORDER BY race_date;"
```

**Context.** Current per-year race counts: 2022: 3,488 (complete), 2023: 460, 2024: 2,735,
2025: 0. JRA runs ~3,400–3,500/year. The walk-forward protocol (§2.5) needs 2023 and 2025
filled.

**Steps.**
1. Long-running, rate-limited scrape (run in background, resumable):
   ```bash
   uv run python src/data_collection/historical_scraper.py --start 2023-01-01 --end 2023-12-31
   uv run python src/data_collection/historical_scraper.py --start 2024-01-01 --end 2024-12-31
   uv run python src/data_collection/historical_scraper.py --start 2025-01-01 --end 2025-12-31
   # or: src/api/target/release/keiba-api scrape-historical --start ... --end ... (--force not needed; existing races are skipped)
   ```
2. Verify counts:
   ```bash
   sqlite3 data/historical/keiba.db "SELECT substr(race_date,1,4) y, COUNT(*) FROM races GROUP BY y;"
   # target: ≥ 3,300 for each of 2023/2024/2025
   sqlite3 data/historical/keiba.db "SELECT substr(r.race_date,1,4) y, ROUND(1.0*SUM(e.win_odds IS NOT NULL AND e.win_odds>1.0)/COUNT(*),3) FROM race_entries e JOIN races r USING(race_id) GROUP BY y;"
   # odds coverage ≥ 0.90 per year
   ```
3. Update §9 ledger; re-run Task 0.1's CLI to add the new years' R²_market.

---

## 6. Phase 1 — Odds-free fundamental model

### Task 1.1 — Feature builder v2 (odds-free, leakage-proof)

**Deliverables.**
- `src/preprocessing/features_v2.py`: builds a per-(race_id, horse_id) feature table from
  SQLite only, under §2.8 rules. Output: `data/processed/features_v2.parquet` with
  `race_id, race_date, horse_id, post_position, label_win, finish_position` + features.
- Feature set v1 (all computable from current schema; NaN where history absent):
  - Horse: age, sex, weight_carried, horse_weight, weight_change, days_since_last_run,
    career starts/win-rate/place-rate (as-of), last-3 finish positions normalized by field
    size, mean last_3f percentile (as-of), running style (mean corner_1..4 position
    percentile), distance-band / surface / racecourse win- and place-rates (as-of),
    grade-level of current race vs mean grade of last 3 runs (class movement).
  - Jockey: as-of win/place rate overall, by racecourse, by surface; jockey-switch flag.
  - Trainer: as-of win/place rate.
  - Race context: field_size, post_position (raw + /field_size), distance, surface,
    track_condition, grade, month.
  - Speed figure v1: z-score of finish_time within (racecourse, distance, surface,
    track_condition) using **as-of** means/stds; mean of last 3 runs.
- **Leakage property test** (required): for K = 20 sampled races, rebuild features from a
  DB copy containing only rows with `race_date` strictly earlier — results must be
  identical to the full-DB build for those races. Plus a static check that `win_odds` /
  `popularity` columns are never read for the current race.

**Verify.** pytest green; feature table row count ≈ entries count for covered years;
spot-check 2–3 horses' as-of stats by hand against SQL.

### Task 1.2 — Fundamental model training

**Deliverables.**
- `src/models/fundamental_model.py`: LightGBM binary (primary path, §2.4) with the §2.5
  walk-forward splits; saves per-eval-year models to `data/models/v2/`.
  (LightGBM works locally as of 2026-07-08 — libomp installed. CatBoost binary remains
  the fallback if anything regresses; note any swap in §9.)
- Report standalone R²_fund per eval year (expect it **below** R²_market — that is normal
  and fine; the blend is where value shows).
- Optional: `lambdarank` variant behind a flag; keep the better eval-year ΔR².

### Task 1.3 — Blend + walk-forward ΔR² report → **GATE 1**

**Deliverables.**
- Blend fit + eval per §2.4/§2.5 wired into `scripts/eval_market_baseline.py` (or a new
  `scripts/eval_delta_r2.py`); emits `data/models/v2/delta_r2_report.{md,json}` with:
  per-year R²_market / R²_blend / ΔR², fitted (α, β, T), pooled bootstrap CI, drop counts.
- Fill §9; evaluate GATE 1 per §4 and record PASS/FAIL + evidence.

### Task 1.4 — (only if GATE 1 fails once) Feature enrichment retry

Levers, in order: (1) fix whatever the per-year error analysis shows (calibration by
grade/field-size buckets); (2) add pace/position dynamics (corner deltas), better speed
figure (beaten lengths via `margin`, going-adjusted); (3) sire stats (infra exists:
`scripts/generate_sire_stats.py`, `src/api/src/scraper/sire_stats.rs`); (4) early JRA-VAN
(D6 override, user approval required — it's a purchase). Then re-run 1.2–1.3 once.

---

## 7. Phase 2 — Exotic pricing + bet engine v2 (requires GATE 1 PASS)

### Task 2.1 — Corrected Harville (γ, δ) fit + Rust port
- Python: fit γ, δ per §2.6 on training years; store in `data/models/v2/harville.json`.
- Rust: extend `exacta.rs`/`trifecta.rs`/`quinella.rs`/`trio.rs`/`wide.rs` to accept
  (γ, δ); default 1.0 = current behavior (unit test: exact equality at 1.0).
- Load from calibration config in `live`/`paper-record` paths.

### Task 2.2 — Win/quinella/wide support in the paper loop
- Odds fetch: current API usage is `type=1&action=update` for win odds (see
  `src/api/src/scraper/parsers/odds.rs` and the `live-pipeline-debugging` skill).
  Quinella/wide type codes are **unknown — investigate** the odds API on a live race day
  (inspect `race.netkeiba.com/odds/` pages' XHR; do not guess).
- Payouts: `parse_payouts` (`scraper/historical/race_result.rs`) already parses the
  pay_table — extend to 単勝/馬連/ワイド rows (they are in the same table).
- `paper-record`/`paper-settle`: accept the new bet types end-to-end. Schema already
  stores `bet_type` as TEXT — verify no enum bottleneck in `cli.rs`/`types.rs`.

### Task 2.3 — Bet engine v2
- EV threshold default 1.1; longshot exclusion (§2.7); quarter-Kelly with per-bet /
  per-day caps; all in config with the §2.7 defaults; unit tests for each filter.

### Task 2.4 — JRA-VAN decision point (user decision, not autonomous)
- Present to the user: cost (~¥2,090/mo), Windows VM requirement, JVLinkToSQLite /
  jrvltsql bridges, what it adds (feature quality; official time-series odds up to 馬連).
  Proceed only on explicit approval.

---

## 8. Phase 3 — Full automation + verified sample (requires Phase 2)

### Task 3.1 — Race-day automation
- One command (`keiba-api race-day` or an orchestrator script) that: discovers today's
  races + post times from `race.netkeiba.com/top/race_list.html?kaisai_date=YYYYMMDD`,
  schedules `paper-record` at post − 3 min per race, runs `paper-settle` in the evening,
  and appends to a weekly report. Schedule via launchd/cron on this Mac (or a small VPS
  if uptime becomes the constraint — user decision).
- Every fetch path is already timeout-bounded (`6b011ef`) — keep it that way; the 70-min
  hang class is a known past failure (see `live-pipeline-debugging` skill).

### Task 3.2 — Statistical reporting + live ΔR²
- `paper-report` additions: bootstrap 95% CI on ROI (pooled + per bet type), slippage
  analysis (recorded odds vs official payout-implied final odds), and **live ΔR²**:
  R²(blend) − R²(recalibrated market) computed on bet-time win odds snapshots — this is
  the un-leaked version of GATE 1's number and the (c) criterion of GATE 2.

### Task 3.3 — All-races odds archive
- Extend the daily loop to snapshot win + quinella + wide (+ exacta if cheap) odds for
  **every** race shortly before post, prediction or not, into `odds_snapshots`.
  This dataset does not exist anywhere commercially (JRA-VAN's archived time-series stops
  at 馬連) — it compounds into the project's proprietary asset and feeds
  `backtest-historical` with real odds.

Then run the loop until **GATE 2** (§4) resolves.

---

## 9. Status Ledger (append-only; newest on top)

| Date | Event / numbers | By |
|---|---|---|
| 2026-07-08 | Task 0.2 in progress: first scrape run leaked a browser → SingletonLock cascade (2023 stopped at 1,402 races; 2025/2024 passes all failed). Hardened orchestrator (`c4c9545`: orphan cleanup + retry + all-dates iteration incl. holidays) and relaunched; DB at 7,630 races and growing. | Fable session |
| 2026-07-08 | **Task 0.1 DONE** (`market_eval.py` + CLI + 23 tests). R²_market(β=1): 2022 = **0.2732** (3,262 races used), 2023 = **0.2438** (427, partial year), 2024 = **0.2444** (2,658). β(prev-year fit) ≈ 0.94–1.00 (market ~perfectly calibrated). All within sanity range — but well above the HK anchors (0.13–0.19): the JRA win market is highly efficient, so the ΔR² bar is high. Data note: races lacking `race_entries`: 2022=155, 2023=28, 2024=8 → fold into Task 0.2 (`--force` re-scrape of those dates). | Fable session |
| 2026-07-08 | Env verified green: 219 Python / 116 Rust tests pass. Fixed pytest-collection breakage (`brew install libomp`). | Fable session |
| 2026-07-07 | V2 direction adopted; docs written. DB: 2022=3,488 / 2023=460 / 2024=2,735 / 2025=0 races. Only verified betting sample to date: 63 exacta paper bets (2026-05-31), all lost, ¥7,000 staked / ¥0 returned. | Fable session |

Fill-in targets (add rows as they land):
- Task 0.1: R²_market per year = ?
- Task 1.3: per-year ΔR², (α, β), pooled CI → GATE 1 PASS/FAIL
- GATE 2 evidence when reached.

---

## 10. Phase 4 — Real money (LOCKED until GATE 2 PASS)

Sketch only, by design: start manual (weekend, minutes/day), bankroll fixed in advance,
quarter-Kelly, monthly drawdown stop (−30% → halt and return to paper), weekly
paper-vs-real slippage review. netkeiba IPAT連携 for semi-automation only after manual
weeks look sane; full automation is a separate user decision (gray zone, no ToS
protection — see STRATEGY_V2 §3.5).
