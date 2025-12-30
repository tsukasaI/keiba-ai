"""
Keiba AI - API Formatter

Format scraped data for the Rust inference API.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from ..parsers.race_card import RaceCardData, HorseEntryData
from ..parsers.horse import HorseData
from ..parsers.jockey import JockeyData
from ..parsers.trainer import TrainerData
from .feature_builder import FeatureBuilder, HorseFeatures

logger = logging.getLogger(__name__)


@dataclass
class HorseApiEntry:
    """A single horse entry for API request."""

    horse_id: str
    horse_name: str
    features: HorseFeatures


@dataclass
class PredictRequest:
    """
    Request format for the Rust inference API.

    Matches src/api/src/types.rs:PredictRequest
    """

    race_id: str
    horses: list[HorseApiEntry]
    exacta_odds: dict[str, float]  # "horse1-horse2" -> odds

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "race_id": self.race_id,
            "horses": [
                {
                    "horse_id": h.horse_id,
                    "horse_name": h.horse_name,
                    "features": h.features.to_dict(),
                }
                for h in self.horses
            ],
            "exacta_odds": self.exacta_odds,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class ApiFormatter:
    """
    Format scraped race data for the Rust API.

    Usage:
        formatter = ApiFormatter()
        request = formatter.build_request(
            race=race_data,
            horses={horse_id: horse_data, ...},
            jockeys={jockey_id: jockey_data, ...},
            trainers={trainer_id: trainer_data, ...},
        )
        json_str = request.to_json()
    """

    def __init__(self):
        self._feature_builder = FeatureBuilder()

    def build_request(
        self,
        race: RaceCardData,
        horses: Optional[dict[str, HorseData]] = None,
        jockeys: Optional[dict[str, JockeyData]] = None,
        trainers: Optional[dict[str, TrainerData]] = None,
        exacta_odds: Optional[dict[str, float]] = None,
    ) -> PredictRequest:
        """
        Build API request from scraped data.

        Args:
            race: Race card data with horse entries
            horses: Horse profile data by horse_id
            jockeys: Jockey data by jockey_id
            trainers: Trainer data by trainer_id
            exacta_odds: Exacta odds by "horse1_id-horse2_id"

        Returns:
            PredictRequest ready for API submission
        """
        horses = horses or {}
        jockeys = jockeys or {}
        trainers = trainers or {}
        exacta_odds = exacta_odds or {}

        api_entries = []

        for entry in race.entries:
            # Get supplemental data
            horse_data = horses.get(entry.horse_id)
            jockey_data = jockeys.get(entry.jockey_id) if entry.jockey_id else None
            trainer_data = trainers.get(entry.trainer_id) if entry.trainer_id else None

            # Build features
            features = self._feature_builder.build(
                race=race,
                entry=entry,
                horse=horse_data,
                jockey=jockey_data,
                trainer=trainer_data,
            )

            api_entry = HorseApiEntry(
                horse_id=entry.horse_id,
                horse_name=entry.horse_name,
                features=features,
            )
            api_entries.append(api_entry)

        return PredictRequest(
            race_id=race.race_id,
            horses=api_entries,
            exacta_odds=exacta_odds,
        )

    def format_for_display(self, race: RaceCardData) -> str:
        """
        Format race data for human-readable display.

        Args:
            race: Race card data

        Returns:
            Formatted string
        """
        lines = [
            f"Race: {race.race_name} ({race.race_id})",
            f"Date: {race.race_date}",
            f"Course: {race.racecourse} {race.distance}m ({race.surface})",
            f"Condition: {race.track_condition}",
            f"Entries: {len(race.entries)} horses",
            "",
            "# | Horse Name          | Jockey      | Weight | Odds",
            "-" * 60,
        ]

        for entry in race.entries:
            odds_str = f"{entry.win_odds:.1f}" if entry.win_odds else "---"
            lines.append(
                f"{entry.post_position:2d} | {entry.horse_name:18s} | "
                f"{entry.jockey_name or '---':10s} | "
                f"{entry.weight_carried:5.1f} | {odds_str}"
            )

        return "\n".join(lines)
