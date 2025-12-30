"""
Keiba AI Prediction System - Odds Data Loader

Load actual exacta (馬単) odds from CSV for backtesting.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OddsLoader:
    """
    Load actual exacta odds from the JRA odds dataset.

    The odds CSV contains the winning exacta combination and its payout odds
    for each race, not pre-race odds for all combinations.
    """

    def __init__(self, odds_path: Optional[Path] = None):
        if odds_path is None:
            self.odds_path = (
                Path(__file__).parent.parent.parent
                / "data"
                / "raw"
                / "19860105-20210731_odds.csv"
            )
        else:
            self.odds_path = Path(odds_path)

        self.odds_df = None
        self.exacta_results = None

    def load_odds(self, start_year: int = 2019, end_year: int = 2021) -> pd.DataFrame:
        """
        Load odds data for specified year range.

        Args:
            start_year: Start year (inclusive)
            end_year: End year (inclusive)

        Returns:
            DataFrame with odds data
        """
        logger.info(f"Loading odds from: {self.odds_path}")

        self.odds_df = pd.read_csv(self.odds_path)

        # Extract year from race ID
        self.odds_df["year"] = (
            self.odds_df["レースID"].astype(str).str[:4].astype(int)
        )

        # Filter by year range
        mask = (self.odds_df["year"] >= start_year) & (self.odds_df["year"] <= end_year)
        self.odds_df = self.odds_df[mask].copy()

        logger.info(f"Loaded {len(self.odds_df):,} races ({start_year}-{end_year})")

        return self.odds_df

    def get_exacta_results(self) -> pd.DataFrame:
        """
        Extract winning exacta combinations and payouts.

        Returns:
            DataFrame with columns:
            - レースID: Race ID
            - exacta_1st: 1st place horse number
            - exacta_2nd: 2nd place horse number
            - exacta_odds: Payout odds (e.g., 29.0 means 2900 yen for 100 yen bet)
        """
        if self.odds_df is None:
            self.load_odds()

        result = pd.DataFrame({
            "レースID": self.odds_df["レースID"],
            "exacta_1st": self.odds_df["馬単1_組合せ1"].astype(float),
            "exacta_2nd": self.odds_df["馬単1_組合せ2"].astype(float),
            "exacta_odds": self.odds_df["馬単1_オッズ"].astype(float),
        })

        # Remove rows with missing data
        result = result.dropna()

        # Convert horse numbers to int
        result["exacta_1st"] = result["exacta_1st"].astype(int)
        result["exacta_2nd"] = result["exacta_2nd"].astype(int)

        self.exacta_results = result

        logger.info(f"Extracted {len(result):,} exacta results")
        logger.info(f"Odds range: {result['exacta_odds'].min():.1f} - {result['exacta_odds'].max():.1f}")
        logger.info(f"Median odds: {result['exacta_odds'].median():.1f}")

        return result

    def get_win_odds(self) -> pd.DataFrame:
        """
        Extract win (単勝) odds for each race.

        Returns:
            DataFrame with columns:
            - レースID: Race ID
            - win_1st_horse: 1st place horse number
            - win_1st_odds: Win odds for 1st place horse
        """
        if self.odds_df is None:
            self.load_odds()

        result = pd.DataFrame({
            "レースID": self.odds_df["レースID"],
            "win_1st_horse": self.odds_df["単勝1_馬番"].astype(float),
            "win_1st_odds": self.odds_df["単勝1_オッズ"].astype(float),
        })

        result = result.dropna()
        result["win_1st_horse"] = result["win_1st_horse"].astype(int)

        return result

    def create_race_odds_lookup(self) -> Dict[int, Dict]:
        """
        Create a lookup dictionary for race odds.

        Returns:
            Dict mapping race_id to {
                'exacta': (1st_horse, 2nd_horse, odds),
                'win': (1st_horse, odds)
            }
        """
        if self.exacta_results is None:
            self.get_exacta_results()

        win_odds = self.get_win_odds()

        lookup = {}

        # Add exacta results
        for _, row in self.exacta_results.iterrows():
            race_id = row["レースID"]
            lookup[race_id] = {
                "exacta": (
                    row["exacta_1st"],
                    row["exacta_2nd"],
                    row["exacta_odds"],
                ),
            }

        # Add win odds
        for _, row in win_odds.iterrows():
            race_id = row["レースID"]
            if race_id in lookup:
                lookup[race_id]["win"] = (
                    row["win_1st_horse"],
                    row["win_1st_odds"],
                )

        logger.info(f"Created lookup for {len(lookup):,} races")

        return lookup

    def merge_with_features(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge odds data with feature data.

        Args:
            features_df: DataFrame with race features

        Returns:
            Merged DataFrame with exacta results
        """
        if self.exacta_results is None:
            self.get_exacta_results()

        merged = features_df.merge(
            self.exacta_results,
            on="レースID",
            how="left",
        )

        match_rate = merged["exacta_odds"].notna().mean() * 100
        logger.info(f"Merged with {match_rate:.1f}% match rate")

        return merged


def main():
    """Test odds loading."""
    loader = OddsLoader()
    odds_df = loader.load_odds(start_year=2019, end_year=2021)

    print("\n" + "=" * 60)
    print("ODDS DATA SUMMARY")
    print("=" * 60)

    exacta = loader.get_exacta_results()

    print(f"\nTotal races with exacta data: {len(exacta):,}")
    print(f"\nExacta odds distribution:")
    print(f"  Min:    {exacta['exacta_odds'].min():.1f}")
    print(f"  25%:    {exacta['exacta_odds'].quantile(0.25):.1f}")
    print(f"  Median: {exacta['exacta_odds'].median():.1f}")
    print(f"  75%:    {exacta['exacta_odds'].quantile(0.75):.1f}")
    print(f"  Max:    {exacta['exacta_odds'].max():.1f}")

    print("\nSample exacta results:")
    print(exacta.head(10).to_string(index=False))

    # Test lookup
    lookup = loader.create_race_odds_lookup()
    sample_race = list(lookup.keys())[0]
    print(f"\nSample lookup for race {sample_race}:")
    print(f"  {lookup[sample_race]}")


if __name__ == "__main__":
    main()
