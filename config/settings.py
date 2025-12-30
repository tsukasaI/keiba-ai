"""
競馬AI予測システム - 設定ファイル
"""
from pathlib import Path
from datetime import date

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# データ取得設定
DATA_CONFIG = {
    # Kaggleデータセット
    "kaggle_dataset": "takamotoki/jra-horse-racing-dataset",
    
    # 使用期間（Kaggleデータは2021年まで）
    "start_year": 2019,
    "end_year": 2021,
}

# JRA競馬場コード
RACECOURSE_CODES = {
    1: "札幌",
    2: "函館",
    3: "福島",
    4: "新潟",
    5: "東京",
    6: "中山",
    7: "中京",
    8: "京都",
    9: "阪神",
    10: "小倉",
}

# レースグレード
RACE_GRADES = ["G1", "G2", "G3", "Listed", "Open", "3勝", "2勝", "1勝", "未勝利", "新馬"]

# 馬場状態
TRACK_CONDITIONS = {
    "良": "good",
    "稍重": "yielding",
    "重": "soft",
    "不良": "heavy",
}

# コース種別
SURFACE_TYPES = {
    "芝": "turf",
    "ダート": "dirt",
    "障害": "jump",
}

# モデル設定
MODEL_CONFIG = {
    # 馬単予測
    "bet_type": "exacta",  # 馬単
    
    # 期待値閾値
    "expected_value_threshold": 1.0,
    
    # 特徴量の時間減衰（日数）
    "decay_half_life_days": 90,
    
    # 過去成績の参照レース数
    "past_races_to_consider": 5,
}

# 特徴量カテゴリ
FEATURE_CATEGORIES = {
    "basic": [
        "horse_age",
        "horse_sex",
        "horse_weight",
        "weight_carried",
        "post_position",
        "gate_number",
    ],
    "jockey_trainer": [
        "jockey_id",
        "jockey_win_rate",
        "trainer_id",
        "trainer_win_rate",
    ],
    "race_conditions": [
        "distance",
        "surface",
        "track_condition",
        "racecourse",
        "race_grade",
    ],
    "past_performance": [
        "last_finish",
        "avg_finish_last_5",
        "win_rate",
        "place_rate",
        "earnings",
    ],
    "running_style": [
        "early_position",
        "mid_position", 
        "final_position",
        "running_style_category",
    ],
    "blood": [
        "sire_id",
        "broodmare_sire_id",
        "sire_win_rate_surface",
        "sire_win_rate_distance",
    ],
}
