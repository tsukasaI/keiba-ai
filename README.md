# Keiba AI Prediction System (JRA)

AI system for predicting Exacta (1st-2nd place in order) outcomes in JRA (Japan Racing Association) races.
Aims to improve ROI using expected value-based betting strategy.

## Project Structure

```
keiba-ai/
├── config/
│   └── settings.py          # Configuration
├── data/
│   ├── raw/                  # Raw data (Kaggle CSV)
│   ├── processed/            # Processed data
│   └── cache/                # Scraper cache (auto-created)
├── src/
│   ├── data_collection/      # Data collection
│   │   └── download_kaggle.py
│   ├── preprocessing/        # Feature engineering
│   ├── models/               # Prediction models
│   ├── scraper/              # Live race scraper (Python, legacy)
│   │   ├── parsers/          # HTML parsers (race_card, horse, jockey, trainer)
│   │   ├── scrapers/         # Async scrapers with rate limiting
│   │   ├── pipeline/         # Feature building for API
│   │   └── cli.py            # Command-line interface
│   └── api/                  # Rust inference API & CLI (includes scraper)
├── tests/                    # Python unit tests
├── notebooks/                # Jupyter exploration
├── requirements.txt
├── CLAUDE.md                 # Claude Code instructions
└── README.md
```

## Setup

```bash
# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Data Collection

### Using Kaggle API

```bash
# 1. Set up Kaggle API authentication
#    Get API token from https://www.kaggle.com/settings
#    Save to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Run download
uv run python src/data_collection/download_kaggle.py
```

### Manual Download

1. Visit https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
2. Click "Download" button
3. Extract ZIP to `data/raw/`

## Data Source

- **Kaggle JRA Dataset**: JRA race data from 1986-2021
  - Race results, odds, lap times, corner passing positions
  - This project uses 2019-2021 data

## Development Phases

- [x] Phase 1: Data Collection & Exploration
- [x] Phase 2: Model Building
- [x] Phase 3: Backtesting (+19.3% ROI with calibration)
- [x] Phase 4: Rust Inference API
- [x] Phase 5: Live Race Scraper (netkeiba.com) - Python
- [x] Phase 6: Full Rust Migration (single binary, no Python dependency)

## Features

| Category | Features |
|----------|----------|
| Basic Info | horse_age, horse_sex, horse_weight, weight_carried, post_position |
| Jockey/Trainer | win_rate, place_rate |
| Race Conditions | distance, turf/dirt, track_condition, racecourse |
| Past Performance | last 5 race positions, win rate |
| Running Style | front-runner/stalker/closer (from corner positions) |
| Pedigree | sire, broodmare sire aptitude |

*Training/workout data will be added when JRA-VAN (paid data) is integrated.

## Strategy

**Expected Value Based**
```
expected_value = predicted_probability × odds
Buy only when expected_value > 1.0
```

## Differences from Boat Racing Project

| Aspect | Boat Racing | Horse Racing |
|--------|-------------|--------------|
| Participants | 6 boats (fixed) | 8-18 horses (variable) |
| Data | Official free download | Kaggle or paid (JRA-VAN) |
| Key Factors | Motor, start timing | Blood, jockey, training |

## Running Predictions

### Quick Start (Single Command)

```bash
# Build
cd src/api && cargo build --release

# Run live prediction (from project root)
./src/api/target/release/keiba-api live 202506050811

# With options
./src/api/target/release/keiba-api live 202506050811 \
    --bet-type trifecta \
    --ev-threshold 1.2 \
    --verbose
```

**Requirements**: Google Chrome installed (for headless browser automation)

### CLI Commands

| Command | Description |
|---------|-------------|
| `live` | **Live prediction** - Scrape and predict in one command |
| `serve` | Start REST API server |
| `predict` | Run prediction on JSON file |
| `backtest` | Walk-forward backtest on historical data |

### API Server Mode

```bash
# Start API server
./target/release/keiba-api serve --port 8080

# Run prediction from JSON file
./target/release/keiba-api predict race.json --bet-types all --format table
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Race prediction (all bet types) |

### Supported Bet Types

| Bet Type | Japanese | Description |
|----------|----------|-------------|
| Exacta | 馬単 | 1st-2nd in exact order |
| Trifecta | 三連単 | 1st-2nd-3rd in exact order |
| Quinella | 馬連 | 1st-2nd any order |
| Trio | 三連複 | 1st-2nd-3rd any order |
| Wide | ワイド | 2 horses both in top 3 |

## Live Race Scraper

### Rust Version (Recommended)

The `live` command scrapes race data and runs prediction in a single command:

```bash
./keiba-api live 202506050811 --verbose
```

**Features**:
- Single binary, no Python dependency
- Chrome DevTools Protocol via chromiumoxide
- File-based cache with TTL (7 days for profiles, 24h for race card)
- Rate limiting (60 req/min, 0.5-1.0s delay)

### Python Version (Legacy)

```bash
# Install Playwright browsers (one-time setup)
playwright install chromium

# Scrape a single race
uv run python -m src.scraper.cli scrape-race 202506050811

# Scrape and call prediction API
uv run python -m src.scraper.cli predict-race 202506050811 \
    --api-url http://localhost:8080
```

### Race ID Format

`YYYYRRCCNNDD` where:
- `YYYY` = Year (e.g., 2025)
- `RR` = Racecourse code (06=Nakayama, 05=Tokyo)
- `CC` = Meeting number
- `NN` = Day number
- `DD` = Race number (01-12)

Example: `202506050811` = 2025 Nakayama 5th meeting 8th day Race 11 (有馬記念)

### 23 Features Scraped

| Source | Features |
|--------|----------|
| Race Card | post_position, weight_carried, distance, surface, track_condition, odds |
| Horse Profile | age, sex, career races, last 3/5 race stats (avg position, win/place rate) |
| Jockey Profile | win_rate, place_rate, total races |
| Trainer Profile | win_rate, total races |

## Testing

### Python Tests

```bash
# Run all Python tests
PYTHONPATH=. uv run pytest tests/ -v

# Run specific test file
PYTHONPATH=. uv run pytest tests/test_backtester.py -v
```

**Test Coverage (290 tests)**

| Module | Tests | Description |
|--------|-------|-------------|
| `position_model.py` | 29 | Model training, prediction, save/load |
| `backtester.py` | 27 | Backtest logic, stratification, segment filtering |
| `odds_loader.py` | 21 | Odds loading, all bet types, lookups |
| `trainer.py` | 20 | CV training, time splits, evaluation |
| `data_loader.py` | 18 | Data loading, filtering, missing values |
| `calibrator.py` | 15 | Probability calibration methods |
| `evaluator.py` | 14 | Metrics calculation, exacta accuracy |
| `expected_value.py` | 12 | EV calculation, Kelly criterion |
| `exacta_calculator.py` | 10 | Exacta probability calculation |
| `types.py` | 10 | Data types, BacktestResults |
| `scraper/data_classes` | 43 | HorseData, JockeyData, TrainerData properties |
| `scraper/parsers` | 30 | HTML parsing for horse, jockey, trainer, race card |
| `scraper/scrapers` | 17 | Cache behavior, retry logic, scrape_many |
| `scraper/infra` | 24 | Cache, config, feature builder, rate limiter |

### Rust Tests (36 tests)

```bash
# Run Rust tests
cd src/api && cargo test

# Run with verbose output
cd src/api && cargo test -- --nocapture
```

| Module | Tests | Description |
|--------|-------|-------------|
| `calibration.rs` | 10 | Temperature scaling, binning, JSON loading |
| `betting.rs` | 5 | EV calculation, Kelly criterion |
| `backtest.rs` | 4 | Bet types, period metrics, drawdown |
| `exacta.rs` | 2 | Exacta probability calculation |
| `trifecta.rs` | 3 | Trifecta probability calculation |
| `quinella.rs` | 3 | Quinella probability calculation |
| `trio.rs` | 4 | Trio probability calculation |
| `wide.rs` | 4 | Wide probability calculation |
| `model.rs` | 1 | Feature names |

## Future Extensions

- Calibration integration in Rust API `/predict` endpoint
- Regional racing (NAR) support
- JRA-VAN integration (real-time predictions with pre-race odds)
- Production deployment (Docker, monitoring)
