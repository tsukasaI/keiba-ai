# Claude Code Instructions

## Project Overview

This is a Japanese horse racing (Keiba) AI prediction system for JRA (Japan Racing Association) races. The goal is to predict Exacta (1st and 2nd place in exact order) outcomes and maximize ROI using expected value-based betting strategy.

## Tech Stack

- **Analysis/ML**: Python (use `uv` for package management)
- **Inference API**: Rust (implemented in Phase 4)
- **Data Source**: Kaggle JRA Dataset (2019-2021) + JRA official (future: JRA-VAN paid data)

## Project Structure

```
keiba-ai/
├── config/settings.py        # Configuration
├── data/
│   ├── raw/                  # Raw data (Kaggle CSV files)
│   └── processed/            # Processed/feature-engineered data
├── src/
│   ├── data_collection/      # Data download scripts
│   ├── preprocessing/        # Feature engineering
│   ├── models/               # ML models (Phase 2)
│   └── api/                  # Rust inference API (Phase 4)
└── notebooks/                # Jupyter exploration
```

## Development Phases

### Phase 1: Data Collection & Exploration
1. Download Kaggle JRA dataset (2019-2021, ~3 years)
2. Explore data structure and available features
3. Design feature engineering pipeline
4. Understand horse racing domain specifics

### Phase 2: Model Building
1. Predict probability distribution of finishing positions for each horse
2. Calculate Exacta probabilities
3. Implement expected value calculation

### Phase 3: Backtesting
1. Validate strategy on historical data (time-series split)
2. Calculate ROI for "expected value > 1.0" betting

### Phase 4: Inference API & UI
1. Build REST API in Rust
2. Simple dashboard for race day predictions

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

### Bet Type: Exacta (Umatan)
- Predict 1st and 2nd place in exact order
- Up to 306 combinations (18 horses × 17 remaining)
- Similar approach to boat racing exacta

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

### Phase 4: Run Rust API
```bash
# Build and run from project root
cd src/api && cargo build --release
./target/release/keiba-api

# Or run directly
cargo run --release
```

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

## Future Extensions (Post Phase 4)

- Additional bet types (Trifecta, Trio, Wide)
- Target profit calculation
- Kelly criterion for bet sizing
- NAR (Regional racing) support
- JRA-VAN integration for real-time predictions

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
- JRA Official: https://www.jra.go.jp/
- JRA-VAN (Paid): https://jra-van.jp/
