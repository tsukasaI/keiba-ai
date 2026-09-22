# Handover — V2 Rebuild kickoff

_Last updated: 2026-07-08 (Fable session: research + V2 design). Previous handover
content (paper-trading debug session, 2026-05-31) is superseded: that branch was merged
to `main` (`3ac3ec9`), and the 63 pending bets were settled._

> **2026-09-22 addendum (research-only session, no code changes).** Project restarted.
> Read `docs/RESEARCH_2026-09.md` before resuming: it refreshes the July research with
> verified new evidence (late odds movement is a JRA-proven signal; FLB exists in JRA;
> keibaodds.com has free win-odds time series back to ~2023) and lists proposed V2
> amendments in its §6 that the user has **not yet approved**. Do not apply them or
> change ▶ NEXT until the user decides. Ledger row added in V2_TASKS §9.

## Where the project stands

- **Direction reset (this session).** Three research passes (proven quantitative betting
  methods / Japanese keiba-AI + data landscape / ML pitfalls) were synthesized into a
  zero-based redesign:
  - `docs/STRATEGY_V2.md` — design + roadmap + evidence (Japanese, for the user)
  - `docs/V2_TASKS.md` — **executable task breakdown for autonomous sessions** (English)
  - `CLAUDE.md` — new "Current Focus: V2 Rebuild" section pointing at both
- **Key design decisions** (details + rationale in V2_TASKS §1, settled — don't re-open):
  odds-free fundamental model → 2-param market blend (Benter), ΔR² vs market as the
  go/no-go metric (not backtest ROI), corrected Harville (fit γ/δ), EV threshold 1.1 +
  longshot exclusion + quarter-Kelly, validate on win/quinella/wide before exotics,
  JRA-VAN deferred until GATE 1 passes, real money locked until GATE 2.
- **Key research fact**: full-combination pre-race odds history does not exist anywhere
  (JRA-VAN's archived time-series stops at 馬連 — official dev-forum confirmed), so
  self-capture (the existing paper-record direction) is the only path; the daily odds
  archive (Task 3.3) becomes a proprietary asset.

## Ground truth (verified this session, 2026-07-07)

- `paper_bets`: 63 exacta bets recorded 2026-05-31, **all settled lost** — ¥7,000 staked,
  ¥0 returned. Only verified betting sample to date. (Not statistically damning alone —
  but the v1 model was structurally broken: odds = top feature i.e. market mirror,
  uncorrected Harville, EV threshold 1.0.)
- SQLite `data/historical/keiba.db` races per year: 2022 = 3,488 (complete),
  2023 = 460, 2024 = 2,735, 2025 = 0, 2026 = 5. JRA runs ~3,400+/yr → 2023/2025 have
  big holes (Task 0.2 fills them).
- `race_entries` has everything the eval harness needs: `finish_position`, `win_odds`
  (coverage ~91%, gaps = scratches), `last_3f`, `corner_1..4`, `popularity`.
- Working tree: docs-only changes this session (`CLAUDE.md` edit + 2 new docs +
  this file). **Not committed** — user hasn't asked.

## ▶ NEXT task

**Finish Task 0.2, then run the formal GATE 1 evaluation.** Concretely:

1. Check the background scrape finished (JRA-only, order 2023 → 2022 → 2025 → 2024;
   relaunched 2026-07-08 after two failures — see below). If it died: re-run the same
   commands (V2_TASKS Task 0.2); already-scraped races are skipped.
2. Verify per-year **JRA** counts ≈ 3,300–3,500 (`substr(race_id,5,2)` 01-10 only);
   `--force` re-scrape dates whose races lack `race_entries` rows.
3. `uv run python scripts/eval_market_baseline.py` (updated per-year baselines) and
   `uv run python scripts/eval_delta_r2.py --db data/historical/keiba.db
   --rebuild-features` → record the **formal GATE 1 verdict** in V2_TASKS §9 ledger.

**Done this session (2026-07-08): Phase 1 pipeline complete (Tasks 1.1–1.3).**
- `src/preprocessing/features_v2.py` — 41 odds-free features, emit-then-ingest by date
  (leakage-proof by construction), 10 tests incl. leakage property tests.
- `src/models/fundamental_model.py` — walk-forward LightGBM binary + race-normalize.
- `scripts/eval_delta_r2.py` — blend fit + ΔR² + bootstrap CI + report
  (`data/models/v2/delta_r2_report.{md,json}`).
- First preliminary result (train=2022 only, ~930 races): ΔR² = −0.0000
  CI [−0.0004, +0.0004], α = 0.062 → fundamental adds nothing YET; blend correctly
  ignores it. **Not a formal GATE 1 attempt** — data was mid-backfill.

**Data landmines discovered (all fixed, read before touching the scraper):**
- **NAR pollution**: db.netkeiba date pages mix JRA + NAR; DB was majority NAR
  (true JRA 2022=972 / 2023=645 / 2024=2,495). Scraper now JRA-only (`d131aca`);
  Python filters via `JRA_ONLY_SQL` (market_eval.py). NAR rows still in DB —
  deletion = user decision.
- **Leaked-browser SingletonLock cascade**: a timed-out keiba-api leaves its headless
  Chrome holding chromiumoxide's profile lock → all later launches die in ~3 s.
  Orchestrator self-cleans + retries now (`c4c9545`).
- Old scrape rows have partially garbage `race_date` (e.g. 70 races on 2022-01-01) —
  from the pre-`3c3607e` date parser. The relaunched full-year passes rewrite… **no,
  they skip existing race_ids** — if wrong-date JRA rows matter for walk-forward
  ordering, `--force` re-scrape 2022 fully (decide when verifying counts).

## Verify environment quickly

```bash
PYTHONPATH=. uv run pytest tests/ -v        # expect: 219 passed (verified 2026-07-08)
cd src/api && cargo test                     # expect: 116 passed (verified 2026-07-08)
```

The long-standing local quirk (LightGBM `libomp.dylib` dlopen failure, which broke pytest
collection entirely) was **fixed 2026-07-08 via `brew install libomp`**. Both suites run
green now; LightGBM training (Task 1.2) is unblocked.
