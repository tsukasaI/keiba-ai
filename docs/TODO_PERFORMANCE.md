# Performance Improvement TODOs

This document outlines identified performance bottlenecks and recommended optimizations.

## Rust (Scraper & Inference)

### Critical: Browser Overhead in Live Command

**Location**: `src/api/src/cli.rs:600-727`, `src/api/src/scraper/browser.rs`

**Issue**:
- Browser launched twice per `live` command (once for race card, once for profiles)
- 2 second startup delay (`browser.rs:61`)
- 2 second page load wait (`browser.rs:91`)
- Sequential profile fetching (one-by-one for each horse/jockey/trainer)

**Current Impact**: For 18 horses → ~40-60 seconds total scraping time

**Recommendations**:
1. **Reuse single browser instance** - Launch browser once at start, reuse for all fetches
2. **Reduce wait times** - Use DOM readiness checks instead of fixed 2s sleeps
3. **Parallel page fetching** - Fetch multiple profiles concurrently (respect rate limits)
4. **Connection pooling** - Keep browser pages warm for reuse

**Priority**: HIGH (directly affects user-facing latency)

---

### Medium: Trifecta Combinatorial Explosion

**Location**: `src/api/src/trifecta.rs:17-75`

**Issue**:
- O(n³) loop for n horses: 18 horses = 4,896 combinations
- Creates HashMap entries for all combinations above threshold
- Sorts ALL combinations even when only top N needed

**Current Impact**: Negligible for single race, but adds up in backtest

**Recommendations**:
1. **Use min-heap for top-K** - Track only top N during iteration instead of storing all
2. **Early pruning** - Already implemented (`line 47-49`), verify effectiveness
3. **Pre-allocate HashMap** - Use `HashMap::with_capacity(estimated_size)`

**Priority**: LOW (current performance is acceptable)

---

### Low: Model Session Mutex

**Location**: `src/api/src/model.rs:51-54`

**Issue**:
- Uses `Mutex` for ONNX session, serializing all predictions
- For API server mode, this blocks concurrent requests

**Current Impact**: Only matters for high-concurrency API usage

**Recommendations**:
1. **Use RwLock** - Allow concurrent reads (predictions are read-only)
2. **Session pool** - Create multiple sessions for parallel inference
3. **Async inference** - Use ort's async features if available

**Priority**: LOW (single-user CLI is current use case)

---

## Python (Training Pipeline)

### Medium: Walk-forward Backtest Sequential Execution

**Location**: `src/models/backtester.py`

**Issue**:
- Trains 6+ models sequentially during walk-forward
- Each period is independent but run serially

**Current Impact**: Full backtest takes several minutes

**Recommendations**:
1. **Parallel period execution** - Use `multiprocessing` or `joblib` for independent periods
2. **Model caching** - Cache trained models by period hash
3. **Incremental training** - Warm-start from previous period's model

**Priority**: MEDIUM (affects development iteration speed)

---

### Low: DataFrame Memory Usage

**Location**: `src/models/trainer.py:50-51`

**Issue**:
- `df.copy()` creates full DataFrame copy
- Multiple boolean masks create temporary arrays

**Current Impact**: Memory usage during training (not a blocker)

**Recommendations**:
1. **Use `.loc` views** - Avoid unnecessary copies where possible
2. **Downcast dtypes** - Use `float32` instead of `float64` for features
3. **Clear intermediate objects** - Explicit `del` and `gc.collect()`

**Priority**: LOW (current memory usage is acceptable)

---

### Low: Feature Engineering Vectorization

**Location**: `src/preprocessing/feature_engineering.py`

**Issue**:
- Some operations may use row-by-row iteration
- Could be replaced with vectorized pandas/numpy operations

**Current Impact**: One-time preprocessing, not in critical path

**Recommendations**:
1. **Profile with `line_profiler`** - Identify slow functions
2. **Use `.apply()` with vectorized functions**
3. **Consider `numba` JIT** for complex calculations

**Priority**: LOW (one-time operation)

---

## Quick Wins

### Completed

1. **✅ Reuse browser instance in `run_live`** - Single browser for race card + all profiles
   - Changed: `cli.rs` - launch browser once at start, pass to `fetch_race_card_with_browser`
   - Impact: Saves ~3-4 seconds (avoided double browser launch)

2. **✅ Reduce page load waits** - Changed in `browser.rs`
   - Browser startup: 2s → 1s
   - Page load wait: 2s → 1.5s
   - Impact: Saves ~1 second per operation

### Deferred

3. **Python parallel backtest** - Deferred due to complexity
   - LightGBM has internal parallelization that conflicts with process-level parallelism
   - Would require careful thread/process management
   - Lower priority since it only affects development iteration, not production

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| `live` command latency | ~60s | <30s |
| Per-profile fetch time | ~3s | ~1.5s |
| Full backtest (6 periods) | ~5min | ~2min |
| ONNX inference per race | ~10ms | ~5ms |

## How to Measure

```bash
# Rust CLI timing
time ./target/release/keiba-api live 202506050811 --verbose

# Python profiling
python -m cProfile -s cumtime scripts/retrain.py --validate-only

# Rust profiling (requires flamegraph)
cargo flamegraph --bin keiba-api -- live 202506050811
```
