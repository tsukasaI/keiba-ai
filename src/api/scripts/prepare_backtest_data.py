"""
Prepare data for Rust backtest CLI.

Creates:
1. backtest_features.parquet - Features with English column names
2. Odds CSV files for each bet type:
   - exacta_odds.csv - race_id,first,second,odds
   - quinella_odds.csv - race_id,first,second,odds
   - wide_odds.csv - race_id,first,second,odds
   - trifecta_odds.csv - race_id,first,second,third,odds
   - trio_odds.csv - race_id,first,second,third,odds
"""

import pandas as pd
from pathlib import Path
from typing import Set


def prepare_features(input_path: Path, output_path: Path) -> None:
    """Prepare features parquet with English column names."""
    print(f"Loading features from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows")

    # Column mappings (Japanese -> English)
    column_map = {
        "レースID": "race_id",
        "レース日付": "race_date",
        "馬番": "horse_num",
        "着順": "position",
        "斤量": "weight_carried",
        "馬体重": "horse_weight",
    }

    # Feature columns (already English)
    feature_cols = [
        "horse_age_num",
        "horse_sex_encoded",
        "post_position_num",
        "jockey_win_rate",
        "jockey_place_rate",
        "trainer_win_rate",
        "jockey_races",
        "trainer_races",
        "distance_num",
        "is_turf",
        "is_dirt",
        "track_condition_num",
        "avg_position_last_3",
        "avg_position_last_5",
        "win_rate_last_3",
        "win_rate_last_5",
        "place_rate_last_3",
        "place_rate_last_5",
        "last_position",
        "career_races",
        "odds_log",
    ]

    # Select and rename columns
    selected_cols = list(column_map.keys()) + [c for c in feature_cols if c in df.columns]
    result = df[selected_cols].copy()
    result = result.rename(columns=column_map)

    # Convert race_id to string
    result["race_id"] = result["race_id"].astype(str)

    # Convert race_date to string format
    if result["race_date"].dtype == "datetime64[ns]":
        result["race_date"] = result["race_date"].dt.strftime("%Y-%m-%d")
    else:
        result["race_date"] = result["race_date"].astype(str)

    # Convert position to int (filter out non-finishers)
    result = result[result["position"].notna()]
    result["position"] = result["position"].astype(int)

    # Filter to 2019+ data
    result = result[result["race_date"] >= "2019-01-01"]

    print(f"  Filtered to {len(result)} rows (2019+)")
    print(f"  Columns: {list(result.columns)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")


def extract_two_horse_odds(
    df: pd.DataFrame, prefix: str, num_results: int, race_ids: Set[str]
) -> pd.DataFrame:
    """Extract 2-horse bet odds (exacta, quinella, wide).

    Args:
        df: Raw odds DataFrame
        prefix: Japanese column prefix (馬単, 馬連, ワイド)
        num_results: Number of results to extract (e.g., 2 for exacta, 7 for wide)
        race_ids: Set of valid race IDs
    """
    records = []

    for _, row in df.iterrows():
        race_id = row["race_id_str"]

        for i in range(1, num_results + 1):
            col1 = f"{prefix}{i}_組合せ1"
            col2 = f"{prefix}{i}_組合せ2"
            odds_col = f"{prefix}{i}_オッズ"

            if col1 not in df.columns or odds_col not in df.columns:
                continue

            if pd.notna(row.get(col1)) and pd.notna(row.get(odds_col)):
                first = int(row[col1])
                second = int(row[col2])
                odds = float(row[odds_col])
                records.append({"race_id": race_id, "first": first, "second": second, "odds": odds})

    result = pd.DataFrame(records)
    if len(result) > 0:
        result["race_id"] = result["race_id"].astype(str)
    return result


def extract_three_horse_odds(
    df: pd.DataFrame, prefix: str, num_results: int, race_ids: Set[str]
) -> pd.DataFrame:
    """Extract 3-horse bet odds (trifecta, trio).

    Args:
        df: Raw odds DataFrame
        prefix: Japanese column prefix (三連単, 三連複)
        num_results: Number of results to extract
        race_ids: Set of valid race IDs
    """
    records = []

    for _, row in df.iterrows():
        race_id = row["race_id_str"]

        for i in range(1, num_results + 1):
            col1 = f"{prefix}{i}_組合せ1"
            col2 = f"{prefix}{i}_組合せ2"
            col3 = f"{prefix}{i}_組合せ3"
            odds_col = f"{prefix}{i}_オッズ"

            if col1 not in df.columns or odds_col not in df.columns:
                continue

            if pd.notna(row.get(col1)) and pd.notna(row.get(odds_col)):
                first = int(row[col1])
                second = int(row[col2])
                third = int(row[col3])
                odds = float(row[odds_col])
                records.append({
                    "race_id": race_id,
                    "first": first,
                    "second": second,
                    "third": third,
                    "odds": odds,
                })

    result = pd.DataFrame(records)
    if len(result) > 0:
        result["race_id"] = result["race_id"].astype(str)
    return result


def prepare_all_odds(input_path: Path, output_dir: Path, race_ids: Set[str]) -> None:
    """Prepare odds CSV files for all bet types."""
    print(f"Loading odds from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")

    # Filter to races in our dataset (compare as strings)
    df["race_id_str"] = df["レースID"].astype(str)
    df = df[df["race_id_str"].isin(race_ids)]
    print(f"  Filtered to {len(df)} rows matching race_ids")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract each bet type
    # Exacta (馬単) - 2 results recorded
    exacta_df = extract_two_horse_odds(df, "馬単", 2, race_ids)
    exacta_path = output_dir / "exacta_odds.csv"
    exacta_df.to_csv(exacta_path, index=False)
    print(f"  Exacta: {len(exacta_df)} records -> {exacta_path}")

    # Quinella (馬連) - 2 results recorded
    quinella_df = extract_two_horse_odds(df, "馬連", 2, race_ids)
    quinella_path = output_dir / "quinella_odds.csv"
    quinella_df.to_csv(quinella_path, index=False)
    print(f"  Quinella: {len(quinella_df)} records -> {quinella_path}")

    # Wide (ワイド) - 7 results recorded
    wide_df = extract_two_horse_odds(df, "ワイド", 7, race_ids)
    wide_path = output_dir / "wide_odds.csv"
    wide_df.to_csv(wide_path, index=False)
    print(f"  Wide: {len(wide_df)} records -> {wide_path}")

    # Trifecta (三連単) - 3 results recorded
    trifecta_df = extract_three_horse_odds(df, "三連単", 3, race_ids)
    trifecta_path = output_dir / "trifecta_odds.csv"
    trifecta_df.to_csv(trifecta_path, index=False)
    print(f"  Trifecta: {len(trifecta_df)} records -> {trifecta_path}")

    # Trio (三連複) - 3 results recorded
    trio_df = extract_three_horse_odds(df, "三連複", 3, race_ids)
    trio_path = output_dir / "trio_odds.csv"
    trio_df.to_csv(trio_path, index=False)
    print(f"  Trio: {len(trio_df)} records -> {trio_path}")


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data"

    # Prepare features
    features_input = data_dir / "processed" / "features.parquet"
    features_output = data_dir / "processed" / "backtest_features.parquet"
    prepare_features(features_input, features_output)

    # Get race IDs from features
    features_df = pd.read_parquet(features_output)
    race_ids = set(features_df["race_id"].astype(str).unique())
    print(f"  Found {len(race_ids)} unique races")

    # Prepare all odds types
    odds_input = data_dir / "raw" / "19860105-20210731_odds.csv"
    prepare_all_odds(odds_input, data_dir / "processed", race_ids)

    print("\nDone! Files created:")
    print(f"  {features_output}")
    print(f"  {data_dir / 'processed' / 'exacta_odds.csv'}")
    print(f"  {data_dir / 'processed' / 'quinella_odds.csv'}")
    print(f"  {data_dir / 'processed' / 'wide_odds.csv'}")
    print(f"  {data_dir / 'processed' / 'trifecta_odds.csv'}")
    print(f"  {data_dir / 'processed' / 'trio_odds.csv'}")


if __name__ == "__main__":
    main()
