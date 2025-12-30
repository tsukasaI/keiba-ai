# Keiba-AI Inference API

Rust-based REST API and CLI for horse racing predictions with betting signals.

## Features

- ONNX model inference for position probability prediction
- Probability calibration (temperature scaling, binning)
- Multi-bet type support: Exacta, Trifecta, Quinella, Trio, Wide
- Expected value calculation and Kelly criterion betting signals
- Walk-forward backtesting CLI

## Installation

```bash
cd src/api
cargo build --release
```

## CLI Commands

### Start API Server

```bash
# Default: 127.0.0.1:8080
cargo run -- serve

# Custom host/port
cargo run -- serve -H 0.0.0.0 -p 3000
```

### Run Prediction

```bash
# JSON output
cargo run -- predict race.json

# Table output with specific bet types
cargo run -- predict race.json -b exacta,trifecta -f table

# Custom model path
cargo run -- predict race.json -m /path/to/model.onnx
```

### Run Backtest

```bash
# Basic exacta backtest
cargo run -- backtest features.parquet --odds exacta_odds.csv

# Backtest different bet types
cargo run -- backtest features.parquet --odds quinella_odds.csv --bet-type quinella
cargo run -- backtest features.parquet --odds trifecta_odds.csv --bet-type trifecta
cargo run -- backtest features.parquet --odds trio_odds.csv --bet-type trio
cargo run -- backtest features.parquet --odds wide_odds.csv --bet-type wide

# With calibration and custom EV threshold
cargo run -- backtest features.parquet \
  --odds exacta_odds.csv \
  --bet-type exacta \
  --calibration calibration.json \
  --ev-threshold 1.1 \
  --format table

# Walk-forward with custom periods
cargo run -- backtest features.parquet \
  --odds exacta_odds.csv \
  --periods 6 \
  --train-months 18 \
  --test-months 3
```

**Supported bet types:**
- `exacta` (馬単) - 1st and 2nd in exact order
- `quinella` (馬連) - 1st and 2nd in any order
- `trifecta` (三連単) - 1st, 2nd, and 3rd in exact order
- `trio` (三連複) - 1st, 2nd, and 3rd in any order
- `wide` (ワイド) - 2 horses in top 3

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{"status": "ok", "version": "0.1.0"}
```

### Model Info

```
GET /model/info
```

Response:
```json
{
  "model_path": "data/models/position_model.onnx",
  "num_features": 23,
  "feature_names": ["horse_age_num", "horse_sex_encoded", ...]
}
```

### Predict

```
POST /predict
Content-Type: application/json
```

Request:
```json
{
  "race_id": "202312100101",
  "horses": [
    {
      "horse_id": "1",
      "features": {
        "horse_age_num": 4,
        "horse_sex_encoded": 1,
        "post_position_num": 1,
        "weight_carried": 57.0,
        "horse_weight": 480,
        "jockey_win_rate": 0.15,
        "jockey_place_rate": 0.35,
        "trainer_win_rate": 0.12,
        "jockey_races": 500,
        "trainer_races": 300,
        "distance_num": 1600,
        "is_turf": 1,
        "is_dirt": 0,
        "track_condition_num": 1,
        "avg_position_last_3": 3.5,
        "avg_position_last_5": 4.2,
        "win_rate_last_3": 0.33,
        "win_rate_last_5": 0.20,
        "place_rate_last_3": 0.67,
        "place_rate_last_5": 0.60,
        "last_position": 2,
        "career_races": 15,
        "odds_log": 2.3
      }
    }
  ],
  "bet_types": ["exacta", "trifecta"],
  "exacta_odds": {"1-2": 1520, "1-3": 2340},
  "trifecta_odds": {"1-2-3": 15200}
}
```

Response:
```json
{
  "race_id": "202312100101",
  "predictions": {
    "win_probabilities": {"1": 0.15, "2": 0.12, "3": 0.10},
    "top_exactas": [
      {
        "first": "1",
        "second": "2",
        "probability": 0.018,
        "odds": 1520,
        "expected_value": 1.37,
        "edge": 0.37,
        "recommended": true
      }
    ],
    "top_trifectas": [...],
    "top_quinellas": [...],
    "top_trios": [...],
    "top_wides": [...]
  },
  "betting_signals": {
    "exacta": [
      {
        "combination": ["1", "2"],
        "bet_type": "exacta",
        "probability": 0.018,
        "odds": 1520,
        "expected_value": 1.37,
        "kelly_fraction": 0.05,
        "recommended_bet": 100
      }
    ],
    "trifecta": [...],
    "quinella": [...],
    "trio": [...],
    "wide": [...]
  }
}
```

## Configuration

### config.toml

```toml
[server]
host = "0.0.0.0"
port = 8080

[model]
path = "data/models/position_model.onnx"

[betting]
ev_threshold = 1.0
min_probability = 0.001
max_combinations = 50
bet_unit = 100
kelly_fraction = 0.25

[calibration]
enabled = true
config_file = "data/models/calibration.json"
```

### Environment Variables

Override config with `KEIBA_` prefix:

```bash
export KEIBA_SERVER_PORT=3000
export KEIBA_MODEL_PATH=/path/to/model.onnx
export KEIBA_BETTING_EV_THRESHOLD=1.1
```

## Calibration

### JSON Format

Temperature scaling:
```json
{"type": "temperature", "temperature": 1.15}
```

Binning calibration:
```json
{
  "type": "binning",
  "n_bins": 10,
  "bin_edges": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  "bin_values": [0.02, 0.08, 0.15, 0.25, 0.38, 0.52, 0.65, 0.78, 0.88, 0.95]
}
```

### Export from Python

```bash
# From pickle file
python src/models/export_calibration.py calibrator.pkl calibration.json

# Create directly
python src/models/export_calibration.py --temperature 1.15 calibration.json
```

## Data Formats

### Features Parquet (for backtest)

Required columns:
- `race_id`: String
- `race_date`: String (YYYY-MM-DD)
- `horse_num`: Int64
- `position`: Int64 (actual finishing position)
- Feature columns matching `FEATURE_NAMES` in config.rs

### Odds CSV (for backtest)

For 2-horse bets (exacta, quinella, wide):
```csv
race_id,first,second,odds
202312100101,1,2,1520
202312100101,1,3,2340
```

For 3-horse bets (trifecta, trio):
```csv
race_id,first,second,third,odds
202312100101,1,2,3,15200
202312100101,1,3,2,18500
```

## Development

```bash
# Run tests
cargo test

# Check compilation
cargo check

# Format code
cargo fmt

# Lint
cargo clippy
```
