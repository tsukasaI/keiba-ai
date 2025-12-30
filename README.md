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
│   └── processed/            # Processed data
├── src/
│   ├── data_collection/      # Data collection
│   │   └── download_kaggle.py
│   ├── preprocessing/        # Feature engineering
│   ├── models/               # Prediction models (Phase 2)
│   └── api/                  # Rust inference API (Phase 4)
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

```bash
# From project root
cd src/api && cargo build --release
./target/release/keiba-api

# API endpoints
# GET  /health      - Health check
# GET  /model/info  - Model information
# POST /predict     - Race prediction
```

## Future Extensions

- Additional bet types (Trifecta, Trio, Wide)
- Regional racing (NAR) support
- JRA-VAN integration (real-time predictions)
