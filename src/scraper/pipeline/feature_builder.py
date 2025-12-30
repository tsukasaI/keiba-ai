"""
Keiba AI - Feature Builder

Convert scraped data to the 23 features required by the prediction model.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from ..config import HORSE_SEX_MAP, MISSING_DEFAULTS
from ..parsers.race_card import RaceCardData, HorseEntryData
from ..parsers.horse import HorseData
from ..parsers.jockey import JockeyData
from ..parsers.trainer import TrainerData

logger = logging.getLogger(__name__)


@dataclass
class HorseFeatures:
    """
    23 features for the prediction model.

    Must match the order in src/api/src/types.rs:HorseFeatures
    and src/models/config.py:FEATURES
    """

    # Basic features (5)
    horse_age_num: float
    horse_sex_encoded: float
    post_position_num: float
    weight_carried: float  # 斤量
    horse_weight: float  # 馬体重

    # Jockey/Trainer features (5)
    jockey_win_rate: float
    jockey_place_rate: float
    trainer_win_rate: float
    jockey_races: float
    trainer_races: float

    # Race conditions (4)
    distance_num: float
    is_turf: float
    is_dirt: float
    track_condition_num: float

    # Past performance (8)
    avg_position_last_3: float
    avg_position_last_5: float
    win_rate_last_3: float
    win_rate_last_5: float
    place_rate_last_3: float
    place_rate_last_5: float
    last_position: float
    career_races: float

    # Odds (1)
    odds_log: float

    def to_array(self) -> list[float]:
        """Convert to array in model input order."""
        return [
            self.horse_age_num,
            self.horse_sex_encoded,
            self.post_position_num,
            self.weight_carried,
            self.horse_weight,
            self.jockey_win_rate,
            self.jockey_place_rate,
            self.trainer_win_rate,
            self.jockey_races,
            self.trainer_races,
            self.distance_num,
            self.is_turf,
            self.is_dirt,
            self.track_condition_num,
            self.avg_position_last_3,
            self.avg_position_last_5,
            self.win_rate_last_3,
            self.win_rate_last_5,
            self.place_rate_last_3,
            self.place_rate_last_5,
            self.last_position,
            self.career_races,
            self.odds_log,
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for API request."""
        return {
            "horse_age_num": self.horse_age_num,
            "horse_sex_encoded": self.horse_sex_encoded,
            "post_position_num": self.post_position_num,
            "weight_carried": self.weight_carried,
            "horse_weight": self.horse_weight,
            "jockey_win_rate": self.jockey_win_rate,
            "jockey_place_rate": self.jockey_place_rate,
            "trainer_win_rate": self.trainer_win_rate,
            "jockey_races": self.jockey_races,
            "trainer_races": self.trainer_races,
            "distance_num": self.distance_num,
            "is_turf": self.is_turf,
            "is_dirt": self.is_dirt,
            "track_condition_num": self.track_condition_num,
            "avg_position_last_3": self.avg_position_last_3,
            "avg_position_last_5": self.avg_position_last_5,
            "win_rate_last_3": self.win_rate_last_3,
            "win_rate_last_5": self.win_rate_last_5,
            "place_rate_last_3": self.place_rate_last_3,
            "place_rate_last_5": self.place_rate_last_5,
            "last_position": self.last_position,
            "career_races": self.career_races,
            "odds_log": self.odds_log,
        }


class FeatureBuilder:
    """
    Build model features from scraped data.

    Usage:
        builder = FeatureBuilder()
        features = builder.build(
            race=race_data,
            entry=entry_data,
            horse=horse_data,
            jockey=jockey_data,
            trainer=trainer_data,
        )
    """

    def build(
        self,
        race: RaceCardData,
        entry: HorseEntryData,
        horse: Optional[HorseData] = None,
        jockey: Optional[JockeyData] = None,
        trainer: Optional[TrainerData] = None,
    ) -> HorseFeatures:
        """
        Build features from scraped data.

        Args:
            race: Race card data
            entry: Horse entry in the race
            horse: Horse profile data (optional, uses defaults if missing)
            jockey: Jockey data (optional, uses defaults if missing)
            trainer: Trainer data (optional, uses defaults if missing)

        Returns:
            HorseFeatures with all 23 features
        """
        # Basic features
        horse_age = horse.horse_age if horse else entry.horse_age
        horse_sex_encoded = float(HORSE_SEX_MAP.get(entry.horse_sex, 0))
        post_position = float(entry.post_position)
        weight_carried = entry.weight_carried if entry.weight_carried else 55.0
        horse_weight = (
            entry.horse_weight
            if entry.horse_weight
            else MISSING_DEFAULTS["horse_weight"]
        )

        # Jockey features
        if jockey:
            jockey_win_rate = jockey.win_rate
            jockey_place_rate = jockey.place_rate
            jockey_races = float(jockey.total_races)
        else:
            jockey_win_rate = MISSING_DEFAULTS["jockey_win_rate"]
            jockey_place_rate = MISSING_DEFAULTS.get("jockey_place_rate", 0.21)
            jockey_races = float(MISSING_DEFAULTS["jockey_races"])

        # Trainer features
        if trainer:
            trainer_win_rate = trainer.win_rate
            trainer_races = float(trainer.total_races)
        else:
            trainer_win_rate = MISSING_DEFAULTS["trainer_win_rate"]
            trainer_races = float(MISSING_DEFAULTS["trainer_races"])

        # Race conditions
        distance_num = float(race.distance)
        is_turf = float(race.is_turf)
        is_dirt = float(race.is_dirt)
        track_condition_num = float(race.track_condition_num)

        # Past performance
        if horse:
            avg_position_last_3 = horse.avg_position_last_3
            avg_position_last_5 = horse.avg_position_last_5
            win_rate_last_3 = horse.win_rate_last_3
            win_rate_last_5 = horse.win_rate_last_5
            place_rate_last_3 = horse.place_rate_last_3
            place_rate_last_5 = horse.place_rate_last_5
            last_position = horse.last_position
            career_races = float(horse.career_races)
        else:
            avg_position_last_3 = MISSING_DEFAULTS["avg_position_last_3"]
            avg_position_last_5 = MISSING_DEFAULTS["avg_position_last_5"]
            win_rate_last_3 = MISSING_DEFAULTS["win_rate_last_3"]
            win_rate_last_5 = MISSING_DEFAULTS["win_rate_last_5"]
            place_rate_last_3 = MISSING_DEFAULTS["place_rate_last_3"]
            place_rate_last_5 = MISSING_DEFAULTS["place_rate_last_5"]
            last_position = MISSING_DEFAULTS["last_position"]
            career_races = float(MISSING_DEFAULTS["career_races"])

        # Odds (log-transformed)
        if entry.win_odds and entry.win_odds > 0:
            odds_log = math.log(entry.win_odds)
        else:
            odds_log = math.log(10.0)  # Default: 10x odds

        return HorseFeatures(
            horse_age_num=float(horse_age),
            horse_sex_encoded=horse_sex_encoded,
            post_position_num=post_position,
            weight_carried=weight_carried,
            horse_weight=horse_weight,
            jockey_win_rate=jockey_win_rate,
            jockey_place_rate=jockey_place_rate,
            trainer_win_rate=trainer_win_rate,
            jockey_races=jockey_races,
            trainer_races=trainer_races,
            distance_num=distance_num,
            is_turf=is_turf,
            is_dirt=is_dirt,
            track_condition_num=track_condition_num,
            avg_position_last_3=avg_position_last_3,
            avg_position_last_5=avg_position_last_5,
            win_rate_last_3=win_rate_last_3,
            win_rate_last_5=win_rate_last_5,
            place_rate_last_3=place_rate_last_3,
            place_rate_last_5=place_rate_last_5,
            last_position=last_position,
            career_races=career_races,
            odds_log=odds_log,
        )
