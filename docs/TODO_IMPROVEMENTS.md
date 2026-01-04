# Keiba-AI Improvement Roadmap

This document outlines identified improvements for UI/UX, model quality, and system reliability.

## Critical Issues (Must Fix)

### 1. Feature Count Mismatch in JSON API

**Status**: ✅ FIXED

**Problem**:
- ONNX model expects 39 features (`NUM_FEATURES = 39` in `model.rs`)
- Live scraper correctly produces 39 features (`scraper/feature_builder.rs`)
- JSON API only accepts 23 features (`types.rs:HorseFeatures`)
- `predict` and `backtest` commands use wrong feature struct

**Locations**:
- `src/api/src/types.rs:8-62` - 23-feature HorseFeatures
- `src/api/src/scraper/feature_builder.rs:10-103` - 39-feature HorseFeatures
- `src/api/src/model.rs:16` - `NUM_FEATURES = 39`

**Impact**: `predict` and `backtest` commands likely fail or produce incorrect results

**Fix**: Unify to single 39-feature HorseFeatures struct across all commands:
1. Update `types.rs` HorseFeatures to match 39-feature version
2. Update `backtest.rs` feature loading to handle all 39 columns
3. Update `scripts/prepare_backtest_data.py` to export 39 features
4. Add validation in model.rs to verify input dimensions

---

### 2. Post-Race Odds in Backtest (Data Leakage Risk)

**Status**: ✅ DOCUMENTED

**Problem**:
- Kaggle dataset only contains winning combination odds (post-race)
- Backtest uses these odds for expected value calculation
- Hit rates are artificially high because losing combinations have no odds data

**Impact**: Backtest ROI of +19.3% may be overly optimistic

**Mitigation Implemented**:
1. ✅ **Documentation**: Added warning in `backtest.rs` module documentation
2. ✅ **CLI Warning**: Backtest output now displays warning about data limitation
3. **Future**: Integrate JRA-VAN for real pre-race odds data

**Backtest Output Now Shows**:
```
WARNING: Kaggle dataset contains only winning combination odds (post-race).
         Hit rates and ROI may be overly optimistic. Use JRA-VAN for accurate results.
```

---

## UI/UX Improvements

### Priority 1: Faster Live Prediction

**Status**: ✅ IMPLEMENTED (Parallel Fetching)

**Current State**: ~30-40 seconds for 18 horses (was ~60 seconds)
**Target**: <30 seconds

| Optimization | Status | Savings |
|--------------|--------|---------|
| Parallel profile fetching | ✅ Done | 20-30s |
| DOM readiness detection | Pending | 5-10s |
| Connection pooling | Pending | 5s |
| Aggressive caching | ✅ Existing | Variable |

**Implementation Details**:
- Uses `futures::stream::buffer_unordered(4)` for concurrent fetches
- Semaphore limits concurrency to 4 parallel browser pages
- Rate limiter still respected to avoid IP blocks
- Cache check done upfront to minimize network calls

---

### Priority 2: Input Validation & Error Messages

**Status**: ✅ IMPLEMENTED

**Implemented**:
- ✅ Race ID format validation (YYYYRRCCNNDD, with racecourse codes, etc.)
- ✅ Bet type validation (exacta, trifecta, quinella, trio, wide, all)
- ✅ Model file existence check with helpful error message
- ✅ 8 new unit tests for validation functions

**Validation Examples**:
```rust
// Race ID validation - validates format and components
validate_race_id("202506050811")  // OK
validate_race_id("202500050811")  // Error: "Racecourse code must be 01-10..."

// Model loading - clear error messages
// "Model file not found: 'path'\n
//  Run 'python scripts/retrain.py --export-only' to export ONNX model"
```

---

### Priority 3: Enhanced Output Display

**Status**: ✅ IMPLEMENTED

**Implemented Features**:
- ✅ Kelly criterion bet amounts shown for each bet
- ✅ Edge detection vs implied odds (green for positive edge)
- ✅ Bankroll display and percentage calculations
- ✅ Table format with headers for clarity
- ✅ Total suggested bets summary

**Example Output**:
```
Win Probabilities (vs Implied Odds):
  horse1: 15.23% - ホースA [+2.1% edge]
  horse2: 12.45% - ホースB [-0.8%]

Recommended Bets (EV > 1.0):
  ───────────────────────────────────────────────────────────────────────────
    #      Combo    Prob%     Odds       EV    Kelly%       Bet (¥)
  ───────────────────────────────────────────────────────────────────────────
    1.      01-03   0.0234     44.0    1.234     1.25%       ¥  1200
    2.      02-05   0.0189     55.0    1.189     0.95%       ¥  1000
  ───────────────────────────────────────────────────────────────────────────

  Total suggested bets: ¥  2200 (2.2% of bankroll)
```

---

### Priority 4: Retry Logic & Resilience

**Status**: ✅ IMPLEMENTED

**Implemented Features**:
- ✅ `src/api/src/retry.rs` - Retry module with exponential backoff
- ✅ `RetryConfig` - Configurable retry behavior (max retries, delays)
- ✅ Browser launch with retry
- ✅ Page fetch with retry option
- ✅ 5 unit tests for retry logic

**Usage**:
```rust
use crate::retry::{retry_anyhow, RetryConfig};

// Use default network retry config
let config = RetryConfig::network();
let result = retry_anyhow(&config, "fetch data", || async {
    fetch_data().await
}).await?;
```

---

## Model Improvements

### Priority 1: Blood Features (Sire/Broodmare)

**Status**: ✅ IMPLEMENTED (Infrastructure Ready)

**What's Implemented**:
- `scripts/generate_sire_stats.py` - Generates sire statistics from cache/external data
- `src/api/src/scraper/sire_stats.rs` - Rust sire stats loader with caching
- `src/api/src/scraper/feature_builder.rs` - Blood features added to HorseFeatures struct
- `data/models/sire_stats.json` - Pre-computed sire performance data

**Implemented Features**:
| Feature | Description | Status |
|---------|-------------|--------|
| sire_win_rate | Sire's offspring win rate | ✅ Ready |
| sire_place_rate | Sire's offspring place rate | ✅ Ready |
| broodmare_sire_win_rate | BMS offspring win rate | ✅ Ready |
| broodmare_sire_place_rate | BMS offspring place rate | ✅ Ready |

**Usage**:
```bash
# Generate sire stats from cached horse profiles
python scripts/generate_sire_stats.py --from-cache

# Blood features are automatically computed during live prediction
# when horse profile includes sire/broodmare_sire data
```

**Next Steps** (To Enable Blood Features in Model):
1. Model currently uses 39 features (blood features excluded)
2. To use blood features, call `to_array_with_blood()` (43 features)
3. Retrain model with blood features when JRA-VAN data is available
4. Update model.rs `NUM_FEATURES` constant to 43

---

### Priority 2: Feature Importance Export

**Current State**: Model doesn't export feature importance

**Benefits**:
- Identify which features drive predictions
- Prune low-importance features
- Validate model learning sensible patterns
- Enable feature selection for efficiency

**Implementation**:
```python
# In retrain.py after training
importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

importance_df.to_csv('data/models/feature_importance.csv', index=False)
```

---

### Priority 3: Ensemble Optimization

**Status**: ✅ STACKING IMPLEMENTED

**Current State**: Supports three strategies:
- `simple_average`: Equal weight for all models
- `weighted_average`: Weights based on validation log loss
- `stacking`: Train meta-learner (RidgeCV) on cross-validated base predictions

**Stacking Implementation**:
- Uses K-fold CV (default 5 folds) to generate out-of-fold predictions
- Meta-features: flattened predictions from all base models (n_models × 18 classes)
- Meta-learner: RidgeCV with alpha selection per class (18 meta-learners)
- Base models retrained on full data after meta-learner fitting

**Usage**:
```bash
# Train with stacking
python scripts/retrain.py --model-type ensemble --stacking
```

**Remaining Improvements**:
1. ~~**Stacking**: Use meta-learner to combine base predictions~~ ✅ Done
2. **Dynamic weighting**: Weight models by recent performance (future)
3. **Diversity selection**: Choose ensemble members that disagree productively (future)

---

### Priority 4: Running Style Deep Features

**Current State**: Simple early/late position features

**Enhancement**:
- Pace scenario modeling (fast/slow early pace)
- Position relative to leader at each point
- Acceleration patterns in final furlong
- Corner position preference by track

---

## Testing & Reliability

### Priority 1: Scraper Unit Tests

**Status**: ✅ IMPLEMENTED

**Added Tests**:
- `race_card.rs`: 5 tests (parse_race_info, parse_entries, extract_grade, empty_html)
- `jockey.rs`: 4 tests (parse_profile, parse_rates, clean_name, empty_html)
- `trainer.rs`: 4 tests (parse_profile, parse_rates, clean_name, empty_html)

**Test Coverage**:
- Race info parsing (name, distance, surface, track condition, grade)
- Entry parsing (horse, jockey, trainer IDs and names)
- Stats table parsing (wins, seconds, thirds, total races, rates)
- Edge cases (empty HTML, missing data)

Total Rust tests: 53 → 72 (+19 new tests)

---

### Priority 2: Integration Tests

**Missing Tests**:
- Python → ONNX → Rust pipeline
- Feature engineering → Training → Export → Inference
- Live command end-to-end (with mock data)

**Proposed Structure**:
```
tests/
├── fixtures/
│   ├── race_card.html
│   ├── horse_profile.html
│   └── sample_race.json
├── integration/
│   ├── test_onnx_export.py
│   └── test_live_command.rs
└── e2e/
    └── test_full_pipeline.sh
```

---

### Priority 3: CI/CD Pipeline

**Current State**: No automated testing

**Proposed Workflow** (`.github/workflows/ci.yml`):
```yaml
jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: uv pip install -r requirements.txt
      - run: uv run pytest tests/ -v

  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo test --manifest-path src/api/Cargo.toml

  lint:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy -- -D warnings
      - run: ruff check src/ scripts/
```

---

## Data Source Improvements

### JRA-VAN Integration (Future)

**Benefits**:
- Real pre-race odds for accurate backtesting
- Training/workout data (key predictor)
- 2022+ data for recent patterns
- Official data quality

**Integration Points**:
1. Data download scripts
2. Feature engineering for workout features
3. Real-time odds API for live predictions
4. Historical odds for backtest accuracy

---

## Metrics & Monitoring

### Production Monitoring

| Metric | Tool | Threshold |
|--------|------|-----------|
| Prediction latency | Prometheus | <30s p95 |
| Scraper success rate | Custom | >95% |
| Model accuracy drift | Weekly validation | ±5% from baseline |
| ROI (paper trading) | Daily calculation | Track trend |

---

## Priority Matrix

| Item | Impact | Effort | Priority | Status |
|------|--------|--------|----------|--------|
| Feature count fix | Critical | Low | P0 | ✅ Done |
| Parallel profile fetch | High | Medium | P1 | ✅ Done |
| Input validation | High | Low | P1 | ✅ Done |
| Blood features | High | Medium | P2 | ✅ Done (infra) |
| Scraper tests | Medium | Low | P2 | ✅ Done |
| Feature importance export | Medium | Low | P2 | ✅ Done |
| Stacking ensemble | Medium | Medium | P2 | ✅ Done |
| Enhanced output | Medium | Medium | P3 | ✅ Done |
| Retry logic | Medium | Medium | P3 | ✅ Done |
| DOM readiness detection | Medium | Low | P3 | ✅ Done |
| JRA-VAN integration | High | High | P4 | Future |

---

## Version Roadmap

### v1.1 - Stability (COMPLETED)
- [x] Fix feature count mismatch (39 features unified)
- [x] Add input validation (race ID, bet type, model path)
- [x] Parallel profile fetching (4x concurrent)
- [x] Scraper unit tests (77 total Rust tests)
- [x] Document data leakage limitation (warning in backtest output)

### v1.2 - Performance (COMPLETED)
- [x] Parallel profile fetching (implemented)
- [x] DOM readiness detection (JavaScript-based)
- [x] Retry logic with exponential backoff
- [x] Enhanced output (Kelly sizing, edge detection)

### v1.3 - Model Quality (COMPLETED)
- [x] Blood features infrastructure (sire_stats.rs, generate_sire_stats.py)
- [x] Feature importance export
- [x] Enhanced ensemble (stacking with meta-learner)
- [ ] Improved calibration
- [ ] Retrain model with blood features (requires JRA-VAN data)

### v2.0 - Production (Future)
- [ ] JRA-VAN integration
- [ ] Real-time odds
- [ ] Monitoring & alerting
- [ ] Paper trading dashboard
