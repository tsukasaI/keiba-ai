"""
Keiba AI Prediction System - Model Configuration

Model configuration for horse racing position prediction.
"""

# Feature configuration
# Note: Some features use Japanese names to match the source CSV data
FEATURE_GROUPS = {
    "basic": [
        "horse_age_num",
        "horse_sex_encoded",
        "post_position_num",
        "斤量",  # weight_carried (impost weight in kg)
        "馬体重",  # horse_weight (horse body weight in kg)
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
    "running_style": [
        "early_position",      # Avg corner 1-2 position
        "late_position",       # Avg corner 3-4 position
        "position_change",     # Late - Early (negative = moving up)
    ],
    "aptitude": [
        "aptitude_sprint",      # Win rate in sprint distances (<1400m)
        "aptitude_mile",        # Win rate in mile distances (1400-1800m)
        "aptitude_intermediate",  # Win rate in intermediate distances (1800-2200m)
        "aptitude_long",        # Win rate in long distances (>2200m)
        "aptitude_turf",        # Place rate on turf
        "aptitude_dirt",        # Place rate on dirt
        "aptitude_course",      # Place rate at specific racecourse
    ],
    "pace": [
        "last_3f_avg",          # Average final 600m time (last 5 races)
        "last_3f_best",         # Best final 600m time in career
        "last_3f_last",         # Most recent final 600m time
    ],
    "race_classification": [
        "weight_change_kg",     # Weight change from last race (kg)
        "is_graded_race",       # Binary: 1 if G1/G2/G3/Listed
        "grade_level",          # 1=G1, 2=G2, 3=G3, 4=Listed, 5=Open, 6=other
    ],
}

# All features to use for training
FEATURES = (
    FEATURE_GROUPS["basic"]
    + FEATURE_GROUPS["jockey_trainer"]
    + FEATURE_GROUPS["race_conditions"]
    + FEATURE_GROUPS["past_performance"]
    + FEATURE_GROUPS["odds"]
    + FEATURE_GROUPS["running_style"]
    + FEATURE_GROUPS["aptitude"]
    + FEATURE_GROUPS["pace"]
    + FEATURE_GROUPS["race_classification"]
)

# Target column (Japanese: finishing position)
TARGET_COL = "着順"

# ID columns (Japanese names from source CSV)
RACE_ID_COL = "レースID"  # race_id
HORSE_NAME_COL = "馬名"  # horse_name
DATE_COL = "レース日付"  # race_date

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
    "馬体重": 480.0,  # horse_weight: average horse weight in kg
    "jockey_win_rate": 0.07,  # Average jockey win rate
    "trainer_win_rate": 0.07,  # Average trainer win rate
    "avg_position_last_3": 10.0,
    "avg_position_last_5": 10.0,
    "win_rate_last_3": 0.0,
    "win_rate_last_5": 0.0,
    "place_rate_last_3": 0.0,
    "place_rate_last_5": 0.0,
    # Running style defaults (middle of pack)
    "early_position": 9.0,
    "late_position": 9.0,
    "position_change": 0.0,
    # Aptitude defaults (no prior history)
    "aptitude_sprint": 0.0,
    "aptitude_mile": 0.0,
    "aptitude_intermediate": 0.0,
    "aptitude_long": 0.0,
    "aptitude_turf": 0.0,
    "aptitude_dirt": 0.0,
    "aptitude_course": 0.0,
    # Pace defaults (average final 600m time is ~34 seconds)
    "last_3f_avg": 35.0,
    "last_3f_best": 35.0,
    "last_3f_last": 35.0,
    # Race classification defaults
    "weight_change_kg": 0.0,
    "is_graded_race": 0,
    "grade_level": 6,  # Non-graded race
}

# Profitable segments identified from backtesting
# Filter to these for positive ROI strategy
PROFITABLE_SEGMENTS = {
    "racecourse": ["福島"],  # Fukushima: +58.6% ROI
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
