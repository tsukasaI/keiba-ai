"""Odds-free fundamental win model, walk-forward (V2, Task 1.2).

Spec: docs/V2_TASKS.md sections 2.4-2.5. Primary path: LightGBM binary
classifier on the odds-free features from features_v2, whose per-horse
probabilities are then normalized within each race to give the fundamental
probability f_i. Market information enters only later, at the blend stage
(src/models/market_eval.fit_blend).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.preprocessing.features_v2 import FEATURE_COLUMNS

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


def add_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add an integer `year` column derived from race_date."""
    out = df.copy()
    out["year"] = out["race_date"].str.slice(0, 4).astype(int)
    return out


def walk_forward_splits(available_years: list[int]) -> list[tuple[int, int, list[int]]]:
    """(eval_year, blend_year, train_years) triples per the section-2.5 table.

    eval year Y needs: blend-fit year Y-1 present, and at least one train
    year <= Y-2 present.
    """
    years = sorted(set(available_years))
    splits = []
    for y in years:
        train = [t for t in years if t <= y - 2]
        if train and (y - 1) in years:
            splits.append((y, y - 1, train))
    return splits


def train_fundamental(
    df_train: pd.DataFrame, df_valid: pd.DataFrame
) -> lgb.Booster:
    """Train the binary win model; early stopping on the blend-fit year.

    Only rows with a defined label (the horse actually ran) are used.
    """
    tr = df_train[df_train["label_win"].notna()]
    va = df_valid[df_valid["label_win"].notna()]
    dtrain = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr["label_win"])
    dvalid = lgb.Dataset(va[FEATURE_COLUMNS], label=va["label_win"], reference=dtrain)
    return lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )


def predict_raw(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Raw per-horse win probabilities (not yet race-normalized)."""
    return booster.predict(df[FEATURE_COLUMNS], num_iteration=booster.best_iteration)
