"""
Keiba AI - Exacta Betting Pipeline

Combine race scraping + exacta odds + model predictions to find value bets.
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from ..parsers.race_card import RaceCardData
from ..parsers.exacta_odds import ExactaOddsData

logger = logging.getLogger(__name__)


@dataclass
class ValueBet:
    """A recommended value bet."""
    first_horse: str
    second_horse: str
    first_name: str
    second_name: str
    probability: float
    odds: float
    expected_value: float
    edge: float  # EV - 1.0
    kelly_fraction: float
    recommended_bet: int


@dataclass
class ExactaBettingResult:
    """Results from exacta betting analysis."""
    race_id: str
    race_name: str

    # All win probabilities
    win_probabilities: Dict[str, float] = field(default_factory=dict)

    # Horse names for display
    horse_names: Dict[str, str] = field(default_factory=dict)

    # Post position to horse ID mapping
    post_to_horse: Dict[int, str] = field(default_factory=dict)

    # Top exacta predictions (by probability)
    top_exactas: List[dict] = field(default_factory=list)

    # Value bets (EV > threshold)
    value_bets: List[ValueBet] = field(default_factory=list)

    # Odds update time
    odds_datetime: str = ""

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            f"Race: {self.race_name} ({self.race_id})",
            f"Odds updated: {self.odds_datetime}",
            "",
            "=== Win Probabilities ===",
        ]

        # Sort by probability
        sorted_probs = sorted(
            self.win_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for horse_id, prob in sorted_probs[:10]:
            name = self.horse_names.get(horse_id, horse_id)
            lines.append(f"  {name}: {prob:.1%}")

        lines.append("")
        lines.append("=== Top Exacta Predictions ===")
        for exacta in self.top_exactas[:10]:
            first_name = self.horse_names.get(exacta["first"], exacta["first"])
            second_name = self.horse_names.get(exacta["second"], exacta["second"])
            lines.append(
                f"  {first_name} -> {second_name}: "
                f"{exacta['probability']:.2%} "
                f"(odds: {exacta.get('odds', 'N/A')}, EV: {exacta.get('expected_value', 'N/A')})"
            )

        if self.value_bets:
            lines.append("")
            lines.append("=== VALUE BETS (EV > 1.0) ===")
            for bet in self.value_bets:
                # Convert odds from Japanese format (1180) to decimal (11.8)
                decimal_odds = bet.odds / 100
                lines.append(
                    f"  {bet.first_name} -> {bet.second_name}: "
                    f"Prob={bet.probability:.2%}, Odds={decimal_odds:.1f}x, "
                    f"EV={bet.expected_value:.2f}, Edge={bet.edge:+.1%}"
                )
                lines.append(f"    Kelly={bet.kelly_fraction:.1%}, Recommended=¥{bet.recommended_bet:,}")
        else:
            lines.append("")
            lines.append("No value bets found (EV > 1.0)")

        return "\n".join(lines)


class ExactaBettingPipeline:
    """
    Pipeline to find value exacta bets.

    1. Convert exacta odds to API format
    2. Call prediction API
    3. Extract value bets
    """

    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url

    def analyze(
        self,
        race_data: RaceCardData,
        exacta_odds: ExactaOddsData,
        api_request_json: str,
    ) -> ExactaBettingResult:
        """
        Analyze race for value exacta bets.

        Args:
            race_data: Scraped race card data
            exacta_odds: Scraped exacta odds
            api_request_json: JSON string for API request

        Returns:
            ExactaBettingResult with value bets
        """
        result = ExactaBettingResult(
            race_id=race_data.race_id,
            race_name=race_data.race_name,
            odds_datetime=exacta_odds.official_datetime,
        )

        # Build mappings
        for entry in race_data.entries:
            result.horse_names[entry.horse_id] = entry.horse_name
            result.post_to_horse[entry.post_position] = entry.horse_id

        # Convert exacta odds to API format: "horse1_id-horse2_id" -> odds
        api_exacta_odds = self._convert_odds_to_api_format(
            exacta_odds, result.post_to_horse
        )

        # Inject exacta odds into API request
        request_data = json.loads(api_request_json)
        request_data["exacta_odds"] = api_exacta_odds

        # Call API
        try:
            response = self._call_api(request_data)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

        # Extract results
        predictions = response.get("predictions", {})
        result.win_probabilities = predictions.get("win_probabilities", {})
        result.top_exactas = predictions.get("top_exactas", [])

        # Extract value bets from betting signals
        for signal in response.get("betting_signals", []):
            combo = signal.get("combination", [])
            if len(combo) != 2:
                continue

            first_id, second_id = combo
            result.value_bets.append(ValueBet(
                first_horse=first_id,
                second_horse=second_id,
                first_name=result.horse_names.get(first_id, first_id),
                second_name=result.horse_names.get(second_id, second_id),
                probability=signal.get("probability", 0),
                odds=signal.get("odds", 0),
                expected_value=signal.get("expected_value", 0),
                edge=signal.get("expected_value", 0) - 1.0,
                kelly_fraction=signal.get("kelly_fraction", 0),
                recommended_bet=signal.get("recommended_bet", 0),
            ))

        # Sort value bets by EV
        result.value_bets.sort(key=lambda x: x.expected_value, reverse=True)

        return result

    def _convert_odds_to_api_format(
        self,
        exacta_odds: ExactaOddsData,
        post_to_horse: Dict[int, str],
    ) -> Dict[str, float]:
        """
        Convert exacta odds from (post1, post2) -> odds
        to "horse1_id-horse2_id" -> odds format for API.

        The API expects Japanese format odds (1180 for 11.8x),
        but netkeiba returns decimal odds (11.8).
        We multiply by 100 to convert.
        """
        api_odds = {}

        for (post1, post2), odds in exacta_odds.odds.items():
            horse1 = post_to_horse.get(post1)
            horse2 = post_to_horse.get(post2)

            if horse1 and horse2:
                key = f"{horse1}-{horse2}"
                # Convert decimal odds (11.8) to Japanese format (1180)
                api_odds[key] = odds * 100

        logger.info(f"Converted {len(api_odds)} exacta odds to API format")
        return api_odds

    def _call_api(self, request_data: dict) -> dict:
        """Call prediction API."""
        url = f"{self.api_url}/predict"
        request_json = json.dumps(request_data, ensure_ascii=False)

        logger.info(f"Calling API: {url}")

        req = urllib.request.Request(
            url,
            data=request_json.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
