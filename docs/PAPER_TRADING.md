# Forward Paper-Trading

This is the project's path to a **verified ROI** number. Every prior ROI figure
(e.g. the +19.3% backtest) came from post-race or estimated odds and is at best
indicative. Paper-trading captures **real pre-race combination odds before post
time**, logs the bets the EV strategy would place, and settles them against the
**official payouts** after the race — producing the first trustworthy ROI the
project has measured, with no real money at risk.

## The loop

```
paper-record  →  (race runs)  →  paper-settle  →  paper-report
```

1. **`paper-record <RACE_ID>`** — BEFORE post time. Runs the live prediction
   (`predict_live`), captures real combination odds from netkeiba's odds API, and
   writes each EV bet to the `paper_bets` table (status `pending`) with a
   Kelly-sized stake. Also snapshots the bet odds into `odds_snapshots` (which
   `backtest-historical` reads, so this incrementally improves historical backtest
   accuracy for free). Idempotent per `(race_id, bet_type, combination)`.

2. **`paper-settle`** — AFTER the race. For each race with pending bets whose
   official payouts aren't yet stored, scrapes the result page once (reusing the
   historical result parser), persists its payout table into `race_payouts`, then
   settles every pending bet: a combination present in the official payout **wins**
   (`payout = stake/100 × payout_per_100`), otherwise it **loses**. Races that
   haven't finished leave their bets pending. The finish order in `race_entries`
   is used as an independent cross-check (warns on mismatch).

3. **`paper-report`** — Aggregates settled bets into total staked/returned, net
   P&L, ROI, and hit rate (overall and per bet type), plus a cumulative bankroll
   trajectory and a count of bets still pending.

## Usage

```bash
cd src/api && cargo build --release
BIN=./target/release/keiba-api

# Before post time (real pre-race odds are only available then):
$BIN paper-record 202506050811 --bet-type exacta --ev-threshold 1.0 \
     --db ../../data/historical/keiba.db

# After the race has finished and results are published:
$BIN paper-settle --db ../../data/historical/keiba.db --verbose

# Review verified performance at any time:
$BIN paper-report --db ../../data/historical/keiba.db --bankroll 100000
```

## Timing constraint (important)

`paper-record` **must** run before post time. netkeiba's odds API only returns
real combination odds while betting is open; after a race finishes those odds are
gone (only winning-combination payouts remain). Each recorded bet stores
`recorded_at` and the API's `official_datetime` so a stale capture is detectable.

## Settlement accuracy and its limit

Settlement uses the **official payout** (払戻 / haraimodoshi) — the real amount
returned — so a won bet's P&L is exact. One honest caveat remains: pari-mutuel
payouts are set by the **final** odds, while the recorded EV decision and stake
were made at the **bet-time** odds. Those differ (slippage). We therefore:

- settle P&L at the official payout (correct money), and
- also store the bet-time odds, EV, and Kelly fraction on each bet, so the
  bet-time-vs-final slippage can be analysed later.

This is strictly more honest than settling at the recorded odds (which would make
ROI an approximation again).

## Bet types

v1 covers **exacta (馬単)** and **trifecta (三連単)** — the two the odds API and
the official-payout parser support.

## Schema

- `paper_bets(race_id, race_date, bet_type, combination, probability, odds,
  expected_value, kelly_fraction, stake, recorded_at, odds_official_datetime,
  status, payout, settled_at)` — one row per recorded EV bet. No foreign key, so
  bets can be recorded before the race row exists.
- `race_payouts(race_id, bet_type, combination, payout_per_100)` — official payout
  per winning combination, populated by `paper-settle`.

Combination strings are zero-padded post positions (`"03-04"`, `"03-04-05"`),
consistent across `paper_bets`, `race_payouts`, and `odds_snapshots`.

## Prerequisite bug fixes

Building this surfaced two latent bugs in the `live` betting pipeline that meant
it had never produced a correct bet, both fixed alongside this feature:

- **Combination keys never matched.** Predictions were keyed by netkeiba horse_id
  while odds were keyed by post position, so the EV join always failed and `live`
  always reported "No bets". EV combos are now translated horse_id → post position
  before the odds lookup.
- **EV and Kelly disagreed on the odds unit.** Odds are decimal (e.g. `12.5`); EV
  used them directly (correct) but Kelly sizing divided by 100, collapsing every
  stake to the minimum unit. Kelly now uses decimal odds consistently
  (`full_kelly_fraction` / `kelly_stake` helpers).
