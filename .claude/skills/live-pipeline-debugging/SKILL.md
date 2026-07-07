---
name: live-pipeline-debugging
description: >
  Debugging playbook for the live prediction pipeline (scrape -> predict -> EV ->
  bet selection), built from past real failures ("failure archaeology") — check
  these known classes before debugging from scratch.
  Use when: live予測が空/おかしい, NoOddsFound, スクレイパーのハング,
  EVが変な馬券を選ぶ, chromiumoxide/timeout問題, netkeiba API調査.
  NOT for: ROI妥当性の判断 — see sibling `validation-and-qa` (paper-trading
  ROI verification cycle; SSoT `docs/PAPER_TRADING.md`).
allowed-tools: Read, Grep, Glob, Bash
---

# Live pipeline debugging

## Purpose

Before debugging the live pipeline (`keiba-api live` / `paper-record`) from
scratch, check whether the symptom matches one of the known failure classes
below. Each entry is a real past bug, not a hypothetical — cited to a commit
SHA and file/line so it can be re-verified against current source.

## When NOT to use

- Judging whether the EV strategy / ROI is *statistically* sound — that's
  `validation-and-qa` (sibling skill; covers the paper-trading ROI
  verification cycle, SSoT `docs/PAPER_TRADING.md`).
- Anything requiring an actual live scrape or browser run. This skill is
  file/history-grounded only; runtime claims below are marked 未実行.

## Known failure classes

### 1. Key mismatch: predictions keyed by horse_id, odds keyed by post position
- **Symptom**: live EV pipeline never selects a bet ("No bets meet EV
  threshold" on every race, even efficient-market ones aside).
- **Root cause**: predictions/model output indexed by netkeiba `horse_id`;
  odds API and official payouts are indexed by post position (zero-padded,
  e.g. `"03"`). Join always empty.
- **Fixed**: commit `7d52f88` (learned: "predictions keyed by horse_id never
  joined post-position-keyed odds"). Join lives in
  `src/api/src/cli.rs:1177-1191` (`id_to_post: HashMap<String, u8>`, built
  from `entry.horse_id` -> `entry.post_position`), consumed at
  `cli.rs:1255`/`1260` (`calculate_ev_trifecta`/`calculate_ev_exacta`).
- **Regression check**: `cargo test test_calculate_ev_exacta_maps_ids_to_post_positions`
  (`cli.rs:2982`) and `test_calculate_ev_exacta_skips_unmapped_ids`
  (`cli.rs:3006`). 未実行 this session — verify by running `cargo test` in
  `src/api/`.

### 2. Kelly odds-unit bug
- **Symptom**: every bet stake collapses to the minimum unit even with a
  large edge.
- **Root cause**: EV math used decimal odds directly into a Kelly formula
  expecting a different unit, then divided by 100 again downstream.
- **Fixed**: same commit `7d52f88` (learned: "Kelly divided decimal odds by
  100, every stake collapsed to the minimum"). Unified in
  `full_kelly_fraction` (`cli.rs:1667-1674`, `b = decimal_odds - 1.0`) and
  `kelly_stake` (`cli.rs:1678-1682`).
- **Regression check**: `test_full_kelly_fraction` (`cli.rs:2963`),
  `test_kelly_stake_rounds_and_floors` (`cli.rs:2973`). 未実行.

### 3. netkeiba data quirks
- **No odds in shutuba HTML**: the scraped shutuba (race-card) page carries
  no odds — netkeiba loads them via AJAX. Win (単勝) odds require the
  `type=1&action=update` JSON API (`src/api/src/scraper/mod.rs:65`, parsed by
  `test_parse_win` in `odds.rs:278`). Learned in commit `991b6ea`.
- **No full-combination odds retention**: `db.netkeiba.com/odds/<id>/umatan/`
  returns HTTP 404 — per `CLAUDE.md:352-354`, that endpoint simply does not
  exist, and separately netkeiba does not retain full-combination odds after
  a race finishes (only winning-combination payouts remain). The
  `--include-odds` flag that once targeted this endpoint was removed for
  that reason.
- **Results lag**: `db.netkeiba.com` publishes same-day results with 1+ day
  lag; `race.netkeiba.com` publishes finish order + payout immediately.
  Fixed by adding a live-site settle path in commit `46e0dab`
  (`RaceResultParser::parse_payouts_live`, `race_result.rs`).
- **Regression check**: `test_parse_payouts_live` (`race_result.rs:905`),
  `test_parse_payouts_live_absent` (`:920`). 未実行.

### 4. Headless-browser hangs (chromiumoxide)
- **Symptom**: `paper-record`/`live` hangs indefinitely on large-field races
  (observed 70+ min, then 5h42m at 0% CPU per commit message).
- **Root cause**: `config.timeout` only wrapped DOM-ready polling, not page
  navigation (`new_page`) or `page.content()`; reqwest odds/payout fetches
  used `Client::new()` with no timeout at all.
- **Fixed**: commit `6b011ef`. `fetch_page_inner` split out, whole fetch
  wrapped in a hard deadline at `browser.rs:155-156`
  (`timeout(deadline, self.fetch_page_inner(...))`, deadline =
  `config.timeout + 20s`); `http_client()` (`cli.rs:1566`) adds a 15s reqwest
  timeout; `predict_live_bounded` (`cli.rs:1538`) wraps the whole prediction
  in a 300s wall-clock deadline. Constraint from the commit:
  `tokio::time::timeout` fires regardless of *where* the inner future stalls.
- **Regression check**: **no unit test** — `browser.rs` has zero `#[test]`
  functions (confirmed by grep). Only verification on record: the manual
  timing note in `6b011ef` ("日本ダービー, 18 horses, 49 uncached profiles ...
  119s"), which is historical, not repeatable. Re-verify by timing a
  known-large-field race manually — 未実行 here.
- Chrome dependency: `README.md:122` and `src/api/Cargo.toml:46`
  (`chromiumoxide = "0.7"`).

### 5. CatBoost ONNX ZipMap vs. tensor
- **Symptom**: live predictions ignore the market (odds feature near-zero
  importance) even after odds are fed correctly — because the *wrong model*
  was deployed, not because of a parsing bug.
- **Root cause**: `export_onnx.py` loaded the CatBoost pickle but called the
  LightGBM converter; separately, CatBoost's native ONNX export emits
  probabilities as a ZipMap (sequence-of-maps), but `src/api/src/model.rs`
  reads a raw tensor.
- **Fixed**: commit `52d5f90`. `_strip_zipmap` (`scripts/export_onnx.py:44`)
  strips the ZipMap node so output index 1 is a raw `[N,18]` float tensor;
  `export_catboost` (`:76`) uses it.
- **Regression check**: the parity gate baked into the export script itself
  — compares ONNX output to `predict_proba` on real feature rows, threshold
  `max abs diff < 1e-4` (`export_onnx.py:141-144`, prints `PARITY PASS/FAIL`).
  This runs at export time, not as a `cargo test` or pytest — re-run
  `python scripts/export_onnx.py` after any model/export change. 未実行 here.

### 6. Environment gotchas (HANDOVER.md)
- Claude Code hooks in this environment block `gh pr create`/`gh pr merge`,
  `git restore`, `git checkout --`, and blanket `grep`/`find` — use `rg`/`fd`,
  and surface blocked git/gh commands for the user to run manually. Source:
  `HANDOVER.md` ("Gotchas discovered" section, line ~140).
- `rustfmt <file>` recurses into `mod` children — formatting `main.rs`
  reformats the whole crate; that's why formatting was committed separately
  in this project's history (`HANDOVER.md`).

## Diagnostic commands

- **Test scope**: `cd src/api && cargo test` runs the full Rust suite
  (unit tests embedded per-module, e.g. `cli.rs`, `odds.rs`,
  `race_result.rs`). `HANDOVER.md` reports two different counts across
  sessions ("109 passed" vs "113 Rust tests pass" later) — treat both as
  historical, not current; count drifts with each change. 未実行 this
  session — don't trust either number without re-running.
- **Live run syntax** (documented from `cli.rs:190-218`, do not execute —
  未実行・ファイル根拠で検証):
  ```bash
  cd src/api && cargo build --release
  ./target/release/keiba-api live <RACE_ID> \
    --bet-type exacta|trifecta \
    --ev-threshold 1.0 \
    [--calibration <path>] [--force] [--verbose]
  ```
  Must run before post time — combination odds only exist while betting is
  open (per `docs/PAPER_TRADING.md`, `CLAUDE.md:426`).
- **Paper-trading loop** (`cli.rs:283` `PaperRecord`, and `paper-settle`/
  `paper-report`): see `HANDOVER.md` "TO RESUME" block for exact invocation;
  do not run against a live race window without explicit user go-ahead.

## Re-verify (before trusting this file)

1. Re-read `HANDOVER.md` in full — it is a living document and the most
   recent session's notes may supersede a failure class above or add a new
   one.
2. `rg -n "id_to_post|full_kelly_fraction|kelly_stake" src/api/src/cli.rs` —
   confirm the line numbers cited under classes 1-2 still match (they will
   drift as the file grows).
3. `rg -n "predict_live_bounded|fn http_client" src/api/src/cli.rs` and
   `rg -n "timeout" src/api/src/scraper/browser.rs` — confirm class 4's
   deadline wrapping is still in place.
4. Confirm the Chrome dependency is still declared:
   `rg -n "chromiumoxide" src/api/Cargo.toml` and `rg -n "Chrome" README.md`.
5. `git log --oneline -1` on the SHAs cited above (`7d52f88`, `991b6ea`,
   `46e0dab`, `6b011ef`, `52d5f90`) to confirm they're still reachable from
   `main` and haven't been reverted.
