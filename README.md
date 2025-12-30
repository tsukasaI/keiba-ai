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
│   ├── scraper/              # Live race scraper
│   │   ├── parsers/          # HTML parsers (race_card, horse, jockey, trainer)
│   │   ├── scrapers/         # Async scrapers with rate limiting
│   │   ├── pipeline/         # Feature building for API
│   │   └── cli.py            # Command-line interface
│   └── api/                  # Rust inference API
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
- [x] Phase 5: Live Race Scraper (netkeiba.com)

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

## Running the API

### CLI Commands

```bash
# Build
cd src/api && cargo build --release

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

Scrape live race data from netkeiba.com and get predictions.

### Install Browser

```bash
# Install Playwright browsers (one-time setup)
playwright install chromium
```

### CLI Commands

```bash
# Scrape a single race (outputs JSON with 23 features per horse)
uv run python -m src.scraper.cli scrape-race 202506050811

# Scrape and call prediction API
uv run python -m src.scraper.cli predict-race 202506050811 \
    --api-url http://localhost:8080

# Verbose mode (show scraping progress)
uv run python -m src.scraper.cli -v scrape-race 202506050811

# Clear cache
uv run python -m src.scraper.cli clear-cache
```

### Race ID Format

Race ID format: `YYYYRRCCNNDD` where:
- `YYYY` = Year (e.g., 2025)
- `RR` = Racecourse code (e.g., 06 = Nakayama)
- `CC` = Kai (meeting number)
- `NN` = Day number
- `DD` = Race number (01-12)

Example: `202506050811` = 2025 Nakayama 5th meeting 8th day Race 11 (有馬記念)

### Features Scraped

| Source | Features |
|--------|----------|
| Race Card | post_position, weight_carried, distance, surface, track_condition, odds |
| Horse Profile | age, sex, career races, last 5 race stats |
| Jockey Profile | win_rate, place_rate, career races |
| Trainer Profile | win_rate, place_rate, career races |

### Rate Limiting

- 20 requests/minute with 1.5-3s random delays
- Uses stealth browser to avoid detection
- JSON cache with 24h TTL (configurable)

## Future Extensions

- Regional racing (NAR) support
- JRA-VAN integration (real-time predictions)
- Probability calibration integration in API
