# Keiba-AI Improvement Roadmap

This document outlines identified improvements for UI/UX, model quality, and system reliability.

## Critical Issues (Must Fix)

### 1. Feature Count Mismatch in JSON API

**Status**: Bug - High Priority

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

**Status**: Known Limitation

**Problem**:
- Kaggle dataset only contains winning combination odds (post-race)
- Backtest uses these odds for expected value calculation
- Hit rates are artificially high because losing combinations have no odds data

**Impact**: Backtest ROI of +19.3% may be overly optimistic

**Mitigation Options**:
1. **Short-term**: Document limitation clearly, add warning in backtest output
2. **Medium-term**: Estimate pre-race odds using population statistics
3. **Long-term**: Integrate JRA-VAN for real pre-race odds data

---

## UI/UX Improvements

### Priority 1: Faster Live Prediction

**Current State**: ~60 seconds for 18 horses
**Target**: <30 seconds

| Optimization | Estimated Savings | Complexity |
|--------------|------------------|------------|
| Parallel profile fetching | 20-30s | Medium |
| DOM readiness detection | 5-10s | Low |
| Connection pooling | 5s | Medium |
| Aggressive caching | Variable | Low |

**Implementation Plan**:
1. Use `futures::stream::buffer_unordered` for concurrent profile fetches
2. Respect rate limiter (token bucket) to avoid blocks
3. Implement `MutationObserver` style wait instead of fixed delays
4. Pool browser pages instead of creating/destroying

---

### Priority 2: Input Validation & Error Messages

**Current Issues**:
- Invalid race ID silently fails during scraping
- Model file missing causes cryptic error
- Odds API format changes silently ignored
- Feature dimension mismatch not caught early

**Improvements**:
```rust
// Race ID validation
pub fn validate_race_id(id: &str) -> Result<RaceId, ValidationError> {
    // Format: YYYYRRCCNNDD (12 digits)
    // Validate year range, racecourse code, etc.
}

// Model loading with clear errors
pub fn load_model(path: &Path) -> Result<Model, ModelError> {
    if !path.exists() {
        return Err(ModelError::FileNotFound(path.display().to_string()));
    }
    // Validate input shape matches expected features
}
```

---

### Priority 3: Enhanced Output Display

**Current Output Limitations**:
- No Kelly criterion bet amounts shown
- No confidence intervals on predictions
- No comparison to public odds for edge detection
- No race result verification (has it already run?)

**Proposed Enhancements**:

```
┌─────────────────────────────────────────────────────────────┐
│ Race: 有馬記念 (G1) - Nakayama 2500m Turf                   │
│ Status: Pre-race (voting closes 15:35)                      │
├─────────────────────────────────────────────────────────────┤
│ Top Win Probabilities                                       │
│   #1 ドウデュース      23.4% (vs odds 21.2%) [+2.2% edge]  │
│   #2 ジャスティンパレス 18.7% (vs odds 19.5%) [-0.8%]      │
│   #3 スターズオンアース 12.1% (vs odds 11.8%) [+0.3%]      │
├─────────────────────────────────────────────────────────────┤
│ Recommended Bets (EV > 1.05)                                │
│   Exacta 1→3: EV=1.23 Prob=2.8% Odds=44.0                  │
│   Kelly: ¥2,300 (2.3% of ¥100k bankroll)                   │
│   95% CI: [1.08, 1.38]                                      │
└─────────────────────────────────────────────────────────────┘
```

---

### Priority 4: Retry Logic & Resilience

**Missing Recovery Mechanisms**:
- No retry on network failures
- No fallback for HTML structure changes
- No graceful degradation when some data unavailable

**Proposed Implementation**:
```rust
// Exponential backoff retry
async fn fetch_with_retry<F, T>(
    operation: F,
    max_retries: u32,
) -> Result<T, Error>
where
    F: Fn() -> Future<Output = Result<T, Error>>,
{
    for attempt in 0..max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt < max_retries - 1 => {
                let delay = Duration::from_millis(100 * 2_u64.pow(attempt));
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

---

## Model Improvements

### Priority 1: Blood Features (Sire/Broodmare)

**Current State**: Not implemented despite being documented in CLAUDE.md

**Why Important**:
- Sire lineage strongly predicts distance/surface aptitude
- Broodmare sire influences stamina inheritance
- Critical for JRA prediction accuracy

**Data Sources**:
- Already scraped in `horse.rs` parser (pedigree data available)
- Need to encode as features in `feature_builder.rs`

**Proposed Features**:
| Feature | Description | Encoding |
|---------|-------------|----------|
| sire_distance_aptitude | Sire's offspring avg winning distance | Normalized |
| sire_surface_aptitude | Sire's offspring turf/dirt win rate | 0-1 |
| broodmare_sire_stamina | BMS offspring avg stamina index | Normalized |
| sire_class_level | Sire's highest grade offspring | Ordinal |

**Implementation**: Add to feature_builder.rs, requires pedigree parsing enhancement

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

**Current State**: Simple weighted average ensemble

**Improvements**:
1. **Stacking**: Use meta-learner to combine base predictions
2. **Dynamic weighting**: Weight models by recent performance
3. **Diversity selection**: Choose ensemble members that disagree productively

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

**Current State**: No HTML parser tests

**Risk**: netkeiba HTML changes break scraper silently

**Test Plan**:
```rust
#[cfg(test)]
mod tests {
    const RACE_CARD_HTML: &str = include_str!("fixtures/race_card_sample.html");

    #[test]
    fn test_parse_race_card() {
        let result = parse_race_card(RACE_CARD_HTML);
        assert_eq!(result.race_name, "有馬記念");
        assert_eq!(result.horses.len(), 16);
    }
}
```

Store fixture HTML files in `src/api/tests/fixtures/`

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

| Item | Impact | Effort | Priority |
|------|--------|--------|----------|
| Feature count fix | Critical | Low | P0 |
| Parallel profile fetch | High | Medium | P1 |
| Input validation | High | Low | P1 |
| Blood features | High | Medium | P2 |
| Scraper tests | Medium | Low | P2 |
| Enhanced output | Medium | Medium | P3 |
| JRA-VAN integration | High | High | P4 |

---

## Version Roadmap

### v1.1 - Stability (Target: 2 weeks)
- [ ] Fix feature count mismatch
- [ ] Add input validation
- [ ] Scraper unit tests
- [ ] Document data leakage limitation

### v1.2 - Performance (Target: 4 weeks)
- [ ] Parallel profile fetching
- [ ] DOM readiness detection
- [ ] Connection pooling
- [ ] <30s live latency

### v1.3 - Model Quality (Target: 8 weeks)
- [ ] Blood features implementation
- [ ] Feature importance export
- [ ] Enhanced ensemble
- [ ] Improved calibration

### v2.0 - Production (Target: 12 weeks)
- [ ] JRA-VAN integration
- [ ] Real-time odds
- [ ] Monitoring & alerting
- [ ] Paper trading dashboard
