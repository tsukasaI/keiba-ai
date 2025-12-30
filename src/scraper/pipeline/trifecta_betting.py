"""
Keiba AI - Trifecta Betting Pipeline

Combine race scraping + trifecta odds + model predictions to find value bets.
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

from ..parsers.race_card import RaceCardData
from ..parsers.trifecta_odds import TrifectaOddsData
from ...models.trifecta_calculator import TrifectaCalculator

logger = logging.getLogger(__name__)


@dataclass
class TrifectaValueBet:
    """A recommended trifecta value bet."""
    first_horse: str
    second_horse: str
    third_horse: str
    first_name: str
    second_name: str
    third_name: str
    probability: float
    odds: float  # Japanese format (11800 = 118x)
    expected_value: float
    edge: float  # EV - 1.0
    kelly_fraction: float
    recommended_bet: int


@dataclass
class TrifectaBettingResult:
    """Results from trifecta betting analysis."""
    race_id: str
    race_name: str

    # All win probabilities
    win_probabilities: Dict[str, float] = field(default_factory=dict)

    # Horse names for display
    horse_names: Dict[str, str] = field(default_factory=dict)

    # Post position to horse ID mapping
    post_to_horse: Dict[int, str] = field(default_factory=dict)

    # Top trifecta predictions (by probability)
    top_trifectas: List[dict] = field(default_factory=list)

    # Value bets (EV > threshold)
    value_bets: List[TrifectaValueBet] = field(default_factory=list)

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
        lines.append("=== Top Trifecta Predictions ===")
        for trifecta in self.top_trifectas[:10]:
            first_name = self.horse_names.get(trifecta["first"], trifecta["first"])
            second_name = self.horse_names.get(trifecta["second"], trifecta["second"])
            third_name = self.horse_names.get(trifecta["third"], trifecta["third"])
            lines.append(
                f"  {first_name} -> {second_name} -> {third_name}: "
                f"{trifecta['probability']:.3%}"
            )

        if self.value_bets:
            lines.append("")
            lines.append(f"=== VALUE BETS ({len(self.value_bets)} bets with EV > 1.0) ===")
            for bet in self.value_bets[:20]:  # Show top 20
                # Convert odds from Japanese format (11800) to decimal (118x)
                decimal_odds = bet.odds / 100
                lines.append(
                    f"  {bet.first_name} -> {bet.second_name} -> {bet.third_name}: "
                    f"Prob={bet.probability:.3%}, Odds={decimal_odds:.1f}x, "
                    f"EV={bet.expected_value:.2f}, Edge={bet.edge:+.1%}"
                )
                lines.append(f"    Kelly={bet.kelly_fraction:.1%}, Recommended=¥{bet.recommended_bet:,}")

            if len(self.value_bets) > 20:
                lines.append(f"  ... and {len(self.value_bets) - 20} more value bets")
        else:
            lines.append("")
            lines.append("No value bets found (EV > 1.0)")

        return "\n".join(lines)


class TrifectaBettingPipeline:
    """
    Pipeline to find value trifecta bets.

    1. Get win probabilities from API
    2. Calculate trifecta probabilities using Harville formula
    3. Compare with actual odds to find value bets
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        min_ev: float = 1.0,  # Minimum expected value to bet
        bankroll: float = 100000,  # Default bankroll in Yen
        max_kelly_fraction: float = 0.05,  # Max 5% per bet
    ):
        self.api_url = api_url
        self.min_ev = min_ev
        self.bankroll = bankroll
        self.max_kelly_fraction = max_kelly_fraction
        self.trifecta_calculator = TrifectaCalculator()

    def analyze(
        self,
        race_data: RaceCardData,
        trifecta_odds: TrifectaOddsData,
        api_request_json: str,
    ) -> TrifectaBettingResult:
        """
        Analyze race for value trifecta bets.

        Args:
            race_data: Scraped race card data
            trifecta_odds: Scraped trifecta odds
            api_request_json: JSON string for API request

        Returns:
            TrifectaBettingResult with value bets
        """
        result = TrifectaBettingResult(
            race_id=race_data.race_id,
            race_name=race_data.race_name,
            odds_datetime=trifecta_odds.official_datetime,
        )

        # Build mappings
        for entry in race_data.entries:
            result.horse_names[entry.horse_id] = entry.horse_name
            result.post_to_horse[entry.post_position] = entry.horse_id

        # Call API to get win probabilities
        try:
            request_data = json.loads(api_request_json)
            response = self._call_api(request_data)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

        # Extract win probabilities
        predictions = response.get("predictions", {})
        result.win_probabilities = predictions.get("win_probabilities", {})

        # Convert to position_probs format for TrifectaCalculator
        # position_probs[horse_id] = array where [0] is P(1st)
        position_probs = {}
        for horse_id, win_prob in result.win_probabilities.items():
            # Use win prob as basis (simplified Harville assumption)
            position_probs[horse_id] = np.array([win_prob] * 18)

        # Calculate trifecta probabilities
        trifecta_probs = self.trifecta_calculator.calculate_trifecta_probs(position_probs)

        # Get top trifectas
        top_trifectas = self.trifecta_calculator.get_top_trifectas(trifecta_probs, n=100)
        for (first, second, third), prob in top_trifectas:
            result.top_trifectas.append({
                "first": first,
                "second": second,
                "third": third,
                "probability": prob,
            })

        # Create horse_id to post_position mapping (reverse)
        horse_to_post = {v: k for k, v in result.post_to_horse.items()}

        # Find value bets
        for (first, second, third), prob in trifecta_probs.items():
            # Get post positions
            post1 = horse_to_post.get(first)
            post2 = horse_to_post.get(second)
            post3 = horse_to_post.get(third)

            if not all([post1, post2, post3]):
                continue

            # Get odds (netkeiba returns decimal, e.g., 118.0)
            decimal_odds = trifecta_odds.get_odds(post1, post2, post3)
            if decimal_odds is None:
                continue

            # Convert to Japanese format (multiply by 100)
            odds = decimal_odds * 100

            # Calculate EV
            ev = prob * decimal_odds

            if ev >= self.min_ev:
                edge = ev - 1.0
                kelly = (ev - 1.0) / (decimal_odds - 1.0) if decimal_odds > 1 else 0
                kelly = min(kelly, self.max_kelly_fraction)
                recommended_bet = int(self.bankroll * kelly)

                result.value_bets.append(TrifectaValueBet(
                    first_horse=first,
                    second_horse=second,
                    third_horse=third,
                    first_name=result.horse_names.get(first, first),
                    second_name=result.horse_names.get(second, second),
                    third_name=result.horse_names.get(third, third),
                    probability=prob,
                    odds=odds,
                    expected_value=ev,
                    edge=edge,
                    kelly_fraction=kelly,
                    recommended_bet=recommended_bet,
                ))

        # Sort value bets by EV
        result.value_bets.sort(key=lambda x: x.expected_value, reverse=True)

        logger.info(f"Found {len(result.value_bets)} trifecta value bets")
        return result

    def _call_api(self, request_data: dict) -> dict:
        """Call prediction API to get win probabilities."""
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
