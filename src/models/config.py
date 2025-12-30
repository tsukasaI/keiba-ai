"""
競馬AI予測システム - モデル設定

Model configuration for horse racing position prediction.
"""

# Feature configuration
FEATURE_GROUPS = {
    "basic": [
        "horse_age_num",
        "horse_sex_encoded",
        "post_position_num",
        "斤量",
        "馬体重",
    ],
    "jockey_trainer": [
        "jockey_win_rate",
        "jockey_place_rate",
        "trainer_win_rate",
        "jockey_races",
        "trainer_races",
    ],
    "race_conditions": [
        "distance_num",
        "is_turf",
        "is_dirt",
        "track_condition_num",
    ],
    "past_performance": [
        "avg_position_last_3",
        "avg_position_last_5",
        "win_rate_last_3",
        "win_rate_last_5",
        "place_rate_last_3",
        "place_rate_last_5",
        "last_position",
        "career_races",
    ],
    "odds": [
        "odds_log",
    ],
}

# All features to use for training
FEATURES = (
    FEATURE_GROUPS["basic"]
    + FEATURE_GROUPS["jockey_trainer"]
    + FEATURE_GROUPS["race_conditions"]
    + FEATURE_GROUPS["past_performance"]
    + FEATURE_GROUPS["odds"]
)

# Target column
TARGET_COL = "着順"

# ID columns
RACE_ID_COL = "レースID"
HORSE_NAME_COL = "馬名"
DATE_COL = "レース日付"

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 18,  # Positions 1-18
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# Training configuration
TRAINING_CONFIG = {
    "n_splits": 5,  # TimeSeriesSplit folds
    "early_stopping_rounds": 50,
    "num_boost_round": 1000,
    "test_size_months": 3,  # Last 3 months for final test
}

# Betting configuration
BETTING_CONFIG = {
    "ev_threshold": 1.0,  # Minimum expected value to bet
    "min_probability": 0.001,  # Minimum probability to consider
    "max_combinations": 50,  # Top N exacta combinations to evaluate
    "bet_unit": 100,  # Yen per bet
}

# Missing value defaults
MISSING_DEFAULTS = {
    "last_position": 10.0,  # Middle of pack for debut horses
    "馬体重": 480.0,  # Average horse weight
    "jockey_win_rate": 0.07,  # Average jockey win rate
    "trainer_win_rate": 0.07,  # Average trainer win rate
    "avg_position_last_3": 10.0,
    "avg_position_last_5": 10.0,
    "win_rate_last_3": 0.0,
    "win_rate_last_5": 0.0,
    "place_rate_last_3": 0.0,
    "place_rate_last_5": 0.0,
}

# Profitable segments identified from backtesting
# Filter to these for positive ROI strategy
PROFITABLE_SEGMENTS = {
    "racecourse": ["福島"],  # +58.6% ROI
    "track_condition": ["yielding", "heavy"],  # +36.3%, +2.1% ROI
    "distance_category": ["long"],  # +11.5% ROI
}

# Backtest configuration
BACKTEST_CONFIG = {
    "n_periods": 6,
    "train_months": 18,
    "test_months": 3,
    "calibration_months": 3,
    "min_prob_threshold": 0.03,  # 3% minimum probability to bet
    "max_bets_per_race": 3,
}
