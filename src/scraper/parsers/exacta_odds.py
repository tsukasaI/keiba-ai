"""
Keiba AI - Exacta Odds Parser

Parse exacta (umatan/馬単) odds from netkeiba.com API.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExactaOddsData:
    """Exacta odds data for a race."""

    race_id: str
    official_datetime: str = ""

    # Odds: (first_horse_num, second_horse_num) -> odds
    odds: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Popularity rank: (first_horse_num, second_horse_num) -> rank
    popularity: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def get_odds(self, first: int, second: int) -> Optional[float]:
        """Get odds for a specific combination."""
        return self.odds.get((first, second))

    def get_top_combinations(self, n: int = 10) -> list:
        """Get top N combinations by popularity."""
        sorted_by_pop = sorted(
            self.popularity.items(),
            key=lambda x: x[1]
        )
        return [(combo, self.odds.get(combo, 0), rank) for combo, rank in sorted_by_pop[:n]]

    def get_value_combinations(
        self,
        exacta_probs: Dict[Tuple[int, int], float],
        ev_threshold: float = 1.0
    ) -> list:
        """
        Find combinations with expected value above threshold.

        Args:
            exacta_probs: Dict of (first, second) -> probability
            ev_threshold: Minimum EV to include (default: 1.0)

        Returns:
            List of (combination, probability, odds, expected_value) tuples
        """
        value_bets = []

        for combo, prob in exacta_probs.items():
            odds = self.odds.get(combo)
            if odds is None:
                continue

            ev = prob * odds
            if ev > ev_threshold:
                value_bets.append((combo, prob, odds, ev))

        # Sort by EV descending
        value_bets.sort(key=lambda x: x[3], reverse=True)
        return value_bets


class ExactaOddsParser:
    """
    Parse exacta odds from netkeiba.com API response.

    API URL: https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=6
    """

    def parse(self, json_data: dict, race_id: str) -> ExactaOddsData:
        """
        Parse exacta odds API response.

        Args:
            json_data: JSON response from API
            race_id: Race ID

        Returns:
            ExactaOddsData containing odds for all combinations
        """
        data = ExactaOddsData(race_id=race_id)

        if json_data.get("status") != "result":
            logger.warning(f"API returned non-result status: {json_data.get('status')}")
            return data

        api_data = json_data.get("data", {})
        data.official_datetime = api_data.get("official_datetime", "")

        # Parse odds (type=6 for exacta)
        odds_data = api_data.get("odds", {}).get("6", {})

        for combo_str, values in odds_data.items():
            if len(combo_str) != 4:
                continue

            try:
                first = int(combo_str[:2])
                second = int(combo_str[2:])

                # Parse odds (may have commas)
                odds_str = values[0].replace(",", "")
                odds = float(odds_str)

                # Parse popularity rank
                popularity = int(values[2]) if len(values) > 2 and values[2] else 0

                data.odds[(first, second)] = odds
                data.popularity[(first, second)] = popularity

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse exacta odds {combo_str}: {e}")
                continue

        logger.info(f"Parsed {len(data.odds)} exacta combinations")
        return data
