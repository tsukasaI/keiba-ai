---
name: validation-and-qa
description: >
  How to make trustworthy ROI claims in keiba-ai: forward paper-trading is the
  only verified-ROI mechanism here, because historical full-combination
  pre-race odds do not exist for this domain. Use when: ROI検証, paper-trading
  運用 (record/settle/report), "この戦略のROIは本物か", backtest結果の解釈,
  EV戦略の評価. NOT for: liveパイプラインのバグ調査 — see sibling skill
  `live-pipeline-debugging` (same repo) for that.
allowed-tools: Bash, Read, Glob, Grep
---

# validation-and-qa — trustworthy ROI in keiba-ai

## Purpose

Any ROI number in this repo is only as trustworthy as the odds it was computed
against. This skill exists so that ROI claims get the right validity label
before they're repeated (to the user, in a commit, in a doc) — and so
paper-trading, the one mechanism that produces a *verified* number, is run
correctly.

SSoT: `docs/PAPER_TRADING.md`. This skill is a procedural summary + pointer —
read that file for full detail; don't restate it from memory here.

## When NOT to use

- The live pipeline produced zero bets, wrong bets, or a crash → that's
  `live-pipeline-debugging`, not this skill.
- Purely asking "what does EV/Kelly mean" with no ROI claim involved → answer
  directly, no skill needed.

## Validity hierarchy (why paper-trading > backtest)

1. **Paper-trading ROI** (`paper-record`→`paper-settle`→`paper-report`) —
   verified. Captures real pre-race combination odds before post time, settles
   against the official payout (haraimodoshi). The only trustworthy figure
   this project can produce.
2. **Backtest ROI** (`backtest-historical`, the historical +19.3% figure) —
   optimistic, not verified. Neither data source has real pre-race
   full-combination odds: Kaggle only has winning-combination (post-race)
   payouts, and netkeiba drops full-combination odds once a race finishes
   (`db.netkeiba.com/odds/<id>/umatan/` → HTTP 404). Backtest therefore
   estimates combination odds from win odds when real ones are absent (see
   `CLAUDE.md` "Known Issues & Limitations" → "No Real Combination Odds
   Anywhere").

Any backtest ROI reported to the user must carry this caveat, not be quoted
bare.

## Paper-trading cycle procedure

Binary: `keiba-api` (built from `src/api`, package name `keiba-api`).

```bash
cd src/api && cargo build --release
BIN=./target/release/keiba-api
DB=../../data/historical/keiba.db

# 1. record — BEFORE post time (real combo odds only exist while betting is open)
$BIN paper-record <RACE_ID> --bet-type exacta --ev-threshold 1.0 --db "$DB" --verbose
#    Expected: live odds fetched, EV→post-position map→Kelly stake logged, OR
#    "No bets meet EV threshold" — a VALID outcome (efficient market), not a bug.

# 2. settle — AFTER the race finishes and results publish
$BIN paper-settle --db "$DB" --verbose
#    Timing: race.netkeiba.com publishes finish order + payout IMMEDIATELY;
#    db.netkeiba.com lags 1+ day. paper-settle prefers race.netkeiba (live) and
#    falls back to db.netkeiba for the independent finish-order cross-check.
#    Settling too early against a stale db.netkeiba view is the main failure mode.

# 3. report — anytime, review verified performance
$BIN paper-report --db "$DB" --bankroll 100000
#    Outputs: total staked/returned, net P&L, ROI, hit rate (overall + per bet
#    type), cumulative bankroll trajectory, pending-bet count.
```

Bet types supported: exacta (馬単), trifecta (三連単) only — the two the odds
API and official-payout parser cover.

## Past-failure checklist before trusting any live-pipeline number

Before treating a `paper-record`/`live` bet count or EV figure as meaningful,
confirm these two historical bugs (both fixed, but re-check if numbers look
implausible again — e.g. zero bets across many races, or every stake at the
minimum unit):

- **Key-mismatch class**: predictions were once keyed by netkeiba `horse_id`
  while odds were keyed by post position, so the EV join silently always
  failed (`live` always reported "No bets"). Now translated via an
  `id_to_post` map before the odds lookup.
- **Odds-unit class**: EV used decimal odds correctly, but Kelly sizing
  divided by 100, collapsing every stake to the minimum unit. Now both use
  decimal odds via shared `full_kelly_fraction`/`kelly_stake` helpers.

If either symptom reappears (systematic "no bets" or all-minimum stakes),
suspect a regression of the same class, not a market-efficiency explanation.

## Before quoting any ROI number

1. Run the test gates — a red suite invalidates any number computed on top of it:
   - `PYTHONPATH=. uv run pytest tests/ -v`
   - `cd src/api && cargo test`
2. State which tier (paper-trading vs backtest) the number came from, using
   the validity hierarchy above.
3. Do not scrape, place bets, or paper-trade yourself to "check" a number —
   this skill verifies from source/docs only. Mark anything not directly run
   in this session as **未実行・ファイル根拠で検証**.

## Re-verify (docs and code drift here — don't trust a stale read)

- Re-read `docs/PAPER_TRADING.md` for the current loop/schema — it is the SSoT.
- Re-check `src/api/src/cli.rs` (`PaperRecord`/`PaperSettle`/`PaperReport`
  variants) for the exact current flags before running a command.
- Check `HANDOVER.md` (repo root, untracked — a live paused-session note, not
  committed history) for the feature's current state: as of the last
  verification pass it described 63 pending bets awaiting settlement, blocked
  on `db.netkeiba` lag. That specific blocker was already fixed on `main`
  (commit `46e0dab`, settling from `race.netkeiba` instead) — treat
  `HANDOVER.md`'s narrative as possibly stale relative to `git log`, and diff
  the two before repeating either as current fact.
- `CLAUDE.md`'s "No Real Combination Odds Anywhere" section is similarly
  hand-maintained prose, not generated — cross-check its "no real races
  recorded yet" claim against `HANDOVER.md`/the `paper_bets` table before
  repeating it; it may be behind actual recorded bets.

## Sibling references

- `live-pipeline-debugging` (this repo, same `.claude/skills/` directory) —
  bug-hunting in the `live` prediction/betting pipeline itself, as opposed to
  interpreting or trusting the ROI numbers it produces.
- `boatrace-ai`'s `validation-and-qa`
  (`~/engineer/gamble_support/boatrace-ai/.claude/skills/validation-and-qa/`)
  — the same "optimistic backtest vs. verified ROI" problem, solved there via
  historical pre-deadline odds snapshots instead of forward paper-trading
  (boatrace odds are retained pre-deadline where netkeiba's are not). At the
  time of writing that sibling file was empty/still being authored — confirm
  its content directly rather than assuming this description before citing it.
