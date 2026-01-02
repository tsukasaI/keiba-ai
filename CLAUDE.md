# Claude Code Instructions

## Project Overview

This is a Japanese horse racing (Keiba) AI prediction system for JRA (Japan Racing Association) races. The goal is to predict Exacta (1st and 2nd place in exact order) outcomes and maximize ROI using expected value-based betting strategy.

## Tech Stack

- **Analysis/ML Training**: Python (use `uv` for package management)
- **Inference & Live Prediction**: Rust (single binary, no Python dependency)
- **Data Source**: Kaggle JRA Dataset (2019-2021) + JRA official (future: JRA-VAN paid data)

## Project Structure

```
keiba-ai/
├── config/settings.py        # Configuration
├── data/
│   ├── raw/                  # Raw data (Kaggle CSV files)
│   ├── processed/            # Processed/feature-engineered data
│   └── cache/                # Scraper cache (auto-created)
├── src/
│   ├── data_collection/      # Data download scripts
│   ├── preprocessing/        # Feature engineering
│   ├── models/               # ML models, backtesting, calibration
│   └── api/                  # Rust inference API & CLI
│       ├── src/
│       │   ├── main.rs       # Entry point (CLI + server)
│       │   ├── cli.rs        # CLI commands (serve, predict, backtest, live)
│       │   ├── routes.rs     # API handlers
│       │   ├── model.rs      # ONNX inference
│       │   ├── backtest.rs   # Walk-forward backtesting
│       │   ├── exacta.rs     # Exacta probability (Harville formula)
│       │   ├── trifecta.rs   # Trifecta probability
│       │   ├── quinella.rs   # Quinella probability
│       │   ├── trio.rs       # Trio probability
│       │   ├── wide.rs       # Wide probability
│       │   ├── betting.rs    # EV, Kelly criterion
│       │   ├── calibration.rs # Probability calibration
│       │   ├── config.rs     # Configuration
│       │   ├── types.rs      # Request/response types
│       │   └── scraper/      # Live race scraper (Rust)
│       │       ├── mod.rs           # Module definition
│       │       ├── browser.rs       # chromiumoxide browser automation
│       │       ├── cache.rs         # File-based cache with TTL
│       │       ├── rate_limiter.rs  # Token bucket rate limiter
│       │       ├── feature_builder.rs # 39 ML features
│       │       └── parsers/         # HTML/JSON parsers
│       │           ├── race_card.rs # Race card parser
│       │           ├── horse.rs     # Horse profile parser
│       │           ├── jockey.rs    # Jockey profile parser
│       │           ├── trainer.rs   # Trainer profile parser
│       │           └── odds.rs      # Odds API parser
│       └── scripts/
│           └── prepare_backtest_data.py
├── scripts/                  # Python scripts
│   ├── retrain.py            # Model retraining pipeline
│   ├── run_validation.py     # Validation backtest
│   └── export_onnx.py        # ONNX model export
├── tests/                    # Python unit tests (213 tests)
│   └── test_*.py             # Model/backtesting tests
└── notebooks/                # Jupyter exploration
```

## Development Phases

All phases completed:

- [x] **Phase 1**: Data Collection & Exploration - Kaggle dataset (2019-2021)
- [x] **Phase 2**: Model Building - LightGBM position probability model
- [x] **Phase 3**: Backtesting - Walk-forward validation (+19.3% ROI with calibration)
- [x] **Phase 4**: Rust Inference API - REST API with all 5 bet types
- [x] **Phase 5**: Live Race Scraper - netkeiba.com integration
- [x] **Phase 6**: Full Rust Migration - Single binary with `live` command (no Python dependency)

## Data Source

### Primary: Kaggle JRA Horse Racing Dataset
- URL: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
- Period: 1986-2021 (use 2019-2021 for this project)
- Format: CSV (pre-processed, easy to use)
- Contents: Race results, betting odds, lap times, corner passing orders

### Future: JRA-VAN DataLab (Paid)
- For 2022+ data when moving to production
- Official JRA data with more features (training data, etc.)

## Key Concepts

### Expected Value Strategy
```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

### Supported Bet Types

| Bet Type | Japanese | Description | Combinations (18 horses) |
|----------|----------|-------------|--------------------------|
| Exacta | 馬単 | 1st-2nd in exact order | 306 |
| Trifecta | 三連単 | 1st-2nd-3rd in exact order | 4,896 |
| Quinella | 馬連 | 1st-2nd any order | 153 |
| Trio | 三連複 | 1st-2nd-3rd any order | 816 |
| Wide | ワイド | 2 horses both in top 3 | 153 |

All bet types use the Harville formula for probability calculation.

### Model Output Design
The model should output **probability distribution for each horse's finishing position**, not just win probability. This allows easy extension to other bet types.

```python
# Example output per horse
{
    "horse_1": {"1st": 0.15, "2nd": 0.12, "3rd": 0.10, ...},
    "horse_2": {"1st": 0.08, "2nd": 0.10, "3rd": 0.12, ...},
    ...
}
```

## Features to Consider

### Basic Features (Available in Kaggle data)
| Feature | Description | Japanese Term |
|---------|-------------|---------------|
| Horse info | Age, sex, weight | horse_age, horse_sex, horse_weight |
| Jockey | Jockey name, win rate | jockey |
| Trainer | Trainer name, win rate | trainer |
| Odds | Win odds, place odds | odds |
| Post position | Gate number (1-18) | gate_number, post_position |
| Distance | Race distance (1000-3600m) | distance |
| Surface | Turf or Dirt | turf/dirt |
| Track condition | Good/Yielding/Soft/Heavy | track_condition |
| Weight carried | Handicap weight | weight_carried |
| Past performance | Previous race results | past_performance |

### Advanced Features (Derive from data)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| Running style | Front-runner/Stalker/Closer | From corner positions |
| Distance aptitude | Performance by distance | distance_aptitude |
| Surface aptitude | Turf vs Dirt performance | surface_aptitude |
| Track aptitude | Performance at specific tracks | track_aptitude |
| Class level | Race grade history | class_level |
| Form | Recent performance trend | form |

### Blood Features (Important in JRA)
| Feature | Description |
|---------|-------------|
| Sire | Father's lineage |
| Broodmare Sire | Mother's father |
| Sire line | Pedigree line |

**Note**: Training/workout data is NOT available in Kaggle dataset. Will be added when JRA-VAN is integrated.

## Commands

### Setup (using uv)
```bash
cd keiba-ai
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Download Kaggle Data
```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
kaggle datasets download -d takamotoki/jra-horse-racing-dataset
unzip jra-horse-racing-dataset.zip -d data/raw/
```

### Phase 1 Execution
```bash
# 1. Download data (or manually from Kaggle website)
uv run python src/data_collection/download_kaggle.py

# 2. Explore data
uv run jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Feature engineering
uv run python src/preprocessing/feature_engineering.py
```

### Run Rust CLI (Production)

```bash
# Build
cd src/api && cargo build --release

# Live prediction (main use case - single command, no Python needed)
./target/release/keiba-api live 202506050811
./target/release/keiba-api live 202506050811 --bet-type trifecta --ev-threshold 1.2 --verbose

# Start API server
./target/release/keiba-api serve --port 8080

# Run prediction from JSON file
./target/release/keiba-api predict race.json --bet-types all --format table

# Run backtest (from project root)
./target/release/keiba-api backtest \
  ./data/processed/backtest_features.parquet \
  --odds ./data/processed/exacta_odds.csv \
  --bet-type exacta \
  --calibration ./data/models/calibration.json \
  --ev-threshold 1.0

# CLI help
./target/release/keiba-api --help
./target/release/keiba-api live --help
```

### Model Retraining Pipeline

```bash
# Full pipeline: features → training → validation → export (LightGBM default)
python scripts/retrain.py

# Use different model types
python scripts/retrain.py --model-type catboost   # CatBoost (better for categorical features)
python scripts/retrain.py --model-type xgb        # XGBoost
python scripts/retrain.py --model-type ensemble   # Ensemble (LightGBM+XGBoost+CatBoost)

# Run hyperparameter optimization before training
python scripts/retrain.py --optimize              # 50 trials (default)
python scripts/retrain.py --optimize --n-trials 100  # More optimization trials

# Skip feature engineering (use existing features.parquet)
python scripts/retrain.py --skip-features

# Only run validation (requires trained model)
python scripts/retrain.py --validate-only

# Only export ONNX + calibration (requires trained model)
python scripts/retrain.py --export-only
```

#### Model Types

| Type | Description | Best For |
|------|-------------|----------|
| `lgbm` | LightGBM (default) | General purpose, fast training |
| `catboost` | CatBoost | Categorical features (jockey/trainer) |
| `xgb` | XGBoost | Alternative gradient boosting |
| `ensemble` | Weighted ensemble | Maximum accuracy, stable ROI |

Outputs:
- `data/models/position_model_39features.pkl` - Pickled model
- `data/models/position_model_{type}.pkl` - Model by type
- `data/models/position_model.onnx` - ONNX model for Rust (LightGBM only)
- `data/models/calibration.json` - Fitted calibration config
- `data/models/best_params_{type}.json` - Optimized hyperparameters (if --optimize used)

#### CLI Commands

| Command | Description |
|---------|-------------|
| `live` | **Live prediction** - Scrape race data and predict (single command) |
| `serve` | Start REST API server |
| `predict` | Run prediction on race JSON file |
| `backtest` | Run walk-forward backtest on historical data |

#### Live Command Options

```bash
keiba-api live <RACE_ID> [OPTIONS]

Arguments:
  <RACE_ID>  Race ID (e.g., 202506050811)

Options:
  -b, --bet-type <BET_TYPE>      Bet type: exacta, trifecta [default: exacta]
      --ev-threshold <THRESHOLD>  EV threshold for recommendations [default: 1.0]
  -o, --output <FILE>            Output file path (JSON)
      --calibration <FILE>       Calibration config file [default: data/models/calibration.json]
  -f, --force                    Force refresh (ignore cache)
  -v, --verbose                  Show detailed progress
```

#### Race ID Format

`YYYYRRCCNNDD` where:
- `YYYY` = Year (e.g., 2025)
- `RR` = Racecourse code (06=Nakayama, 05=Tokyo, etc.)
- `CC` = Meeting number
- `NN` = Day number
- `DD` = Race number (01-12)

Example: `202506050811` = 2025 Nakayama 5th meeting 8th day Race 11 (有馬記念)

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Race prediction (all 5 bet types) |

## Important Notes

1. **Time-series validation**: Never use future data to predict past races. Use `TimeSeriesSplit` for cross-validation.
2. **Data leakage**: Be careful not to include post-race information (actual odds at race time vs final odds).
3. **Horse racing specifics**:
   - Lane 1 advantage exists in boat racing, but not as strong in horse racing
   - Inside post positions have slight advantage in short races
   - Weather and track condition heavily affect results
4. **JRA specifics**:
   - 10 racecourses: Sapporo, Hakodate, Fukushima, Niigata, Nakayama, Tokyo, Chukyo, Kyoto, Hanshin, Kokura
   - Race grades: G1, G2, G3, Listed, Open, Conditions, Maiden

## Differences from Boat Racing

| Aspect | Boat Racing | Horse Racing |
|--------|-------------|--------------|
| Participants | 6 boats (fixed) | 8-18 horses (variable) |
| Post advantage | Lane 1 ~50% win rate | Slight inside advantage |
| Key factors | Motor, start timing | Blood, jockey, training |
| Data availability | Official free download | Paid (JRA-VAN) or scraping |
| Race frequency | Daily, 24 venues | Weekend mainly, 10 venues |

## Implemented Features

- ✅ All 5 bet types (Exacta, Trifecta, Quinella, Trio, Wide)
- ✅ Probability calibration (temperature scaling, binning)
- ✅ Walk-forward backtesting (Python + Rust CLI)
- ✅ Kelly criterion bet sizing
- ✅ Expected value filtering
- ✅ Live race scraper (netkeiba.com) - **Full Rust implementation**
- ✅ Single binary CLI (`live` command - no Python dependency)
- ✅ File-based cache with TTL (7 days for profiles, 24h for race card)
- ✅ Calibration in Rust CLI (`live`, `predict`, `serve` commands)
- ✅ Model retraining pipeline (`scripts/retrain.py`)
- ✅ Multiple model types (LightGBM, CatBoost, XGBoost, Ensemble)
- ✅ Hyperparameter optimization with Optuna (`--optimize` flag)
- ✅ Colored CLI output with progress bars
- ✅ Comprehensive test suite (213 Python + 61 Rust tests)

## Known Issues & Limitations

### Fixed: Feature Count Mismatch

~~The JSON API (`predict`, `backtest`) uses a 23-feature `HorseFeatures` struct (`types.rs`), but the ONNX model expects 39 features.~~

**Status**: ✅ Fixed. All commands now use unified 39-feature struct with `#[serde(default)]` for backward compatibility.

### Data Limitation: Post-Race Odds in Backtest

The Kaggle dataset only contains winning combination odds (post-race), not pre-race odds for all combinations. This makes backtest hit rates appear higher than real-world performance.

**Mitigation**: The reported +19.3% ROI should be considered optimistic. Real-world ROI will likely be lower.

### Performance: Live Command Latency

~~Current `live` command takes ~60 seconds for 18 horses due to sequential profile fetching.~~

**Status**: Improved to ~30-40 seconds with parallel fetching (4 concurrent). Target <30s still in progress.

**See**: `docs/TODO_PERFORMANCE.md` for further optimization plan.

### Missing Features: Blood Data

Sire/broodmare features are documented but not yet implemented. These are important predictors for JRA races.

**See**: `docs/TODO_IMPROVEMENTS.md` for implementation plan.

## Improvement Roadmap

See `docs/TODO_IMPROVEMENTS.md` for detailed improvement plans covering:
- UI/UX enhancements (input validation, error messages, output formatting)
- Model quality (blood features, feature importance, ensemble optimization)
- Performance (parallel fetching, connection pooling)
- Testing (scraper tests, integration tests, CI/CD)

## Future Extensions

- NAR (Regional racing) support
- JRA-VAN integration for real-time predictions with pre-race odds
- Production deployment (Docker, monitoring)
- Blood features (sire/broodmare aptitude)
- Real-time odds edge detection

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
- JRA Official: https://www.jra.go.jp/
- JRA-VAN (Paid): https://jra-van.jp/
