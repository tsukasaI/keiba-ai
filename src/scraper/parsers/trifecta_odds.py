"""
Keiba AI - Trifecta Odds Parser

Parse trifecta (3連単/sanrentan) odds from netkeiba.com API.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrifectaOddsData:
    """Trifecta odds data for a race."""

    race_id: str
    official_datetime: str = ""

    # Odds: (first, second, third) -> odds
    odds: Dict[Tuple[int, int, int], float] = field(default_factory=dict)

    # Popularity rank: (first, second, third) -> rank
    popularity: Dict[Tuple[int, int, int], int] = field(default_factory=dict)

    def get_odds(self, first: int, second: int, third: int) -> Optional[float]:
        """Get odds for a specific combination."""
        return self.odds.get((first, second, third))

    def get_top_combinations(self, n: int = 10) -> list:
        """Get top N combinations by popularity."""
        sorted_by_pop = sorted(
            self.popularity.items(),
            key=lambda x: x[1]
        )
        return [(combo, self.odds.get(combo, 0), rank) for combo, rank in sorted_by_pop[:n]]


class TrifectaOddsParser:
    """
    Parse trifecta odds from netkeiba.com API response.

    API URL: https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=8
    """

    def parse(self, json_data: dict, race_id: str) -> TrifectaOddsData:
        """
        Parse trifecta odds API response.

        Args:
            json_data: JSON response from API
            race_id: Race ID

        Returns:
            TrifectaOddsData containing odds for all combinations
        """
        data = TrifectaOddsData(race_id=race_id)

        if json_data.get("status") != "result":
            logger.warning(f"API returned non-result status: {json_data.get('status')}")
            return data

        api_data = json_data.get("data", {})
        data.official_datetime = api_data.get("official_datetime", "")

        # Parse odds (type=8 for trifecta)
        odds_data = api_data.get("odds", {}).get("8", {})

        for combo_str, values in odds_data.items():
            if len(combo_str) != 6:
                continue

            try:
                first = int(combo_str[:2])
                second = int(combo_str[2:4])
                third = int(combo_str[4:6])

                # Parse odds (may have commas)
                odds_str = values[0].replace(",", "")
                odds = float(odds_str)

                # Parse popularity rank
                popularity = int(values[2]) if len(values) > 2 and values[2] else 0

                data.odds[(first, second, third)] = odds
                data.popularity[(first, second, third)] = popularity

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse trifecta odds {combo_str}: {e}")
                continue

        logger.info(f"Parsed {len(data.odds)} trifecta combinations")
        return data
