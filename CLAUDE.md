# Claude Code Instructions

## Project Overview

This is a Japanese horse racing (競馬/Keiba) AI prediction system for JRA (Japan Racing Association) races. The goal is to predict Umatan (馬単/Exacta - 1st and 2nd place in exact order) outcomes and maximize ROI using expected value-based betting strategy.

## Tech Stack

- **Analysis/ML**: Python (use `uv` for package management)
- **Inference API**: Go or Rust (to be implemented in Phase 4)
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
│   └── api/                  # Inference API (Phase 4)
└── notebooks/                # Jupyter exploration
```

## Development Phases

### Phase 1: Data Collection & Exploration (Current)
1. Download Kaggle JRA dataset (2019-2021, ~3 years)
2. Explore data structure and available features
3. Design feature engineering pipeline
4. Understand horse racing domain specifics

### Phase 2: Model Building
1. Predict probability distribution of finishing positions for each horse
2. Calculate Umatan (exacta) probabilities
3. Implement expected value calculation

### Phase 3: Backtesting
1. Validate strategy on historical data (time-series split)
2. Calculate ROI for "expected value > 1.0" betting

### Phase 4: Inference API & UI
1. Build REST API in Go or Rust
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

### Bet Type: 馬単 (Umatan/Exacta)
- Predict 1st and 2nd place in exact order
- Up to 306 combinations (18 horses × 17 remaining)
- Similar approach to boat racing 2連単

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
| Feature | Description | Notes |
|---------|-------------|-------|
| Horse info | Age, sex, weight | 馬齢、性別、馬体重 |
| Jockey | Jockey name, win rate | 騎手 |
| Trainer | Trainer name, win rate | 調教師 |
| Odds | Win odds, place odds | オッズ |
| Post position | Gate number (1-18) | 枠番、馬番 |
| Distance | Race distance (1000-3600m) | 距離 |
| Surface | Turf (芝) or Dirt (ダート) | 馬場 |
| Track condition | Good/Yielding/Soft/Heavy | 馬場状態 |
| Weight carried | Handicap weight (斤量) | 斤量 |
| Past performance | Previous race results | 過去成績 |

### Advanced Features (Derive from data)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| Running style | Front-runner/Stalker/Closer | 脚質 (from corner positions) |
| Distance aptitude | Performance by distance | 距離適性 |
| Surface aptitude | Turf vs Dirt performance | 芝/ダート適性 |
| Track aptitude | Performance at specific tracks | コース適性 |
| Class level | Race grade history | クラス |
| Form | Recent performance trend | 調子 |

### Blood Features (Important in JRA)
| Feature | Description |
|---------|-------------|
| Sire (父) | Father's lineage |
| Broodmare Sire (母父) | Mother's father |
| Sire line | Pedigree line (サイアーライン) |

**Note**: Training/workout data (調教データ) is NOT available in Kaggle dataset. Will be added when JRA-VAN is integrated.

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

## Important Notes

1. **Time-series validation**: Never use future data to predict past races. Use `TimeSeriesSplit` for cross-validation.
2. **Data leakage**: Be careful not to include post-race information (actual odds at race time vs final odds).
3. **Horse racing specifics**: 
   - 1号艇 advantage exists in boat racing, but not as strong in horse racing
   - Inside post positions have slight advantage in short races
   - Weather and track condition heavily affect results
4. **JRA specifics**:
   - 10 racecourses: 札幌、函館、福島、新潟、中山、東京、中京、京都、阪神、小倉
   - Race grades: G1, G2, G3, Listed, Open, Conditions, Maiden

## Differences from Boat Racing (競艇)

| Aspect | Boat Racing | Horse Racing |
|--------|-------------|--------------|
| Participants | 6 boats (fixed) | 8-18 horses (variable) |
| Post advantage | 1号艇 ~50% win rate | Slight inside advantage |
| Key factors | Motor, start timing | Blood, jockey, training |
| Data availability | Official free download | Paid (JRA-VAN) or scraping |
| Race frequency | Daily, 24 venues | Weekend mainly, 10 venues |

## Future Extensions (Post Phase 4)

- Additional bet types (三連単, 三連複, ワイド)
- Target profit calculation
- Kelly criterion for bet sizing
- NAR (地方競馬) support
- JRA-VAN integration for real-time predictions

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
- JRA Official: https://www.jra.go.jp/
- JRA-VAN (Paid): https://jra-van.jp/
