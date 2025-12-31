"""
Keiba AI - Result Formatter

Format prediction results for terminal display with box-drawing characters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# Racecourse code to Japanese name mapping
RACECOURSE_NAMES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


@dataclass
class FormattedResult:
    """Formatted result for display."""
    header: str
    win_probs_section: str
    value_bets_section: str
    summary_section: str

    def __str__(self) -> str:
        return f"{self.header}\n{self.win_probs_section}\n{self.value_bets_section}\n{self.summary_section}"


class ResultFormatter:
    """
    Format betting analysis results for terminal display.

    Uses Unicode box-drawing characters for clean visual output.
    """

    def __init__(self, width: int = 60):
        self.width = width

    def format_header(
        self,
        race_name: str,
        race_id: str,
        race_date: Optional[str] = None,
    ) -> str:
        """Format race header."""
        # Extract racecourse from race_id
        if len(race_id) >= 6:
            racecourse_code = race_id[4:6]
            racecourse = RACECOURSE_NAMES.get(racecourse_code, "")
            race_number = race_id[-2:].lstrip("0") + "R"
        else:
            racecourse = ""
            race_number = ""

        # Build header lines
        lines = []
        lines.append("=" * self.width)
        lines.append(f"  {race_name}")
        if race_date and racecourse:
            lines.append(f"  {race_date} {racecourse}{race_number}")
        elif racecourse:
            lines.append(f"  {racecourse}{race_number}")
        lines.append("=" * self.width)

        return "\n".join(lines)

    def format_win_probabilities(
        self,
        win_probs: Dict[str, float],
        horse_names: Dict[str, str],
        top_n: int = 5,
    ) -> str:
        """Format win probability rankings."""
        lines = []
        lines.append("")
        lines.append("[Win Probabilities]")

        # Sort by probability
        sorted_probs = sorted(
            win_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (horse_id, prob) in enumerate(sorted_probs[:top_n], 1):
            name = horse_names.get(horse_id, horse_id)
            # Truncate long names
            if len(name) > 12:
                name = name[:11] + "..."
            lines.append(f"  {i}. {name:<14} {prob:>6.1%}")

        return "\n".join(lines)

    def format_value_bets_exacta(
        self,
        value_bets: List,
        ev_threshold: float = 1.0,
    ) -> str:
        """
        Format value bets for exacta in a table.

        Args:
            value_bets: List of ValueBet objects
            ev_threshold: Minimum EV to show (for labeling)
        """
        if not value_bets:
            return f"\n[Recommended Exacta] (EV > {ev_threshold})\n  No value bets found."

        lines = []
        lines.append(f"\n[Recommended Exacta] EV > {ev_threshold}")
        lines.append("-" * self.width)
        lines.append(f"{'Combination':<24} {'Prob':>7} {'Odds':>7} {'EV':>6}")
        lines.append("-" * self.width)

        for bet in value_bets:
            # Format combination
            first = bet.first_name[:8] if len(bet.first_name) > 8 else bet.first_name
            second = bet.second_name[:8] if len(bet.second_name) > 8 else bet.second_name
            combo = f"{first} -> {second}"

            # Format odds (from Japanese format to decimal)
            odds_decimal = bet.odds / 100 if bet.odds > 100 else bet.odds
            odds_str = f"{odds_decimal:.1f}x"

            lines.append(
                f"{combo:<24} {bet.probability:>6.1%} {odds_str:>7} {bet.expected_value:>6.2f}"
            )

        lines.append("-" * self.width)

        return "\n".join(lines)

    def format_value_bets_trifecta(
        self,
        value_bets: List,
        ev_threshold: float = 1.0,
    ) -> str:
        """Format value bets for trifecta in a table."""
        if not value_bets:
            return f"\n[Recommended Trifecta] (EV > {ev_threshold})\n  No value bets found."

        lines = []
        lines.append(f"\n[Recommended Trifecta] EV > {ev_threshold}")
        lines.append("-" * self.width)
        lines.append(f"{'Combination':<32} {'Prob':>7} {'Odds':>9} {'EV':>6}")
        lines.append("-" * self.width)

        for bet in value_bets:
            # Format combination
            first = bet.first_name[:6] if len(bet.first_name) > 6 else bet.first_name
            second = bet.second_name[:6] if len(bet.second_name) > 6 else bet.second_name
            third = bet.third_name[:6] if len(bet.third_name) > 6 else bet.third_name
            combo = f"{first}->{second}->{third}"

            # Format odds
            odds_decimal = bet.odds / 100 if bet.odds > 100 else bet.odds
            odds_str = f"{odds_decimal:.0f}x"

            lines.append(
                f"{combo:<32} {bet.probability:>6.2%} {odds_str:>9} {bet.expected_value:>6.2f}"
            )

        lines.append("-" * self.width)

        return "\n".join(lines)

    def format_summary(
        self,
        value_bets: List,
        bet_type: str = "exacta",
    ) -> str:
        """Format betting summary."""
        if not value_bets:
            return ""

        lines = []
        lines.append("")

        # Calculate statistics
        total_bets = len(value_bets)
        avg_ev = sum(b.expected_value for b in value_bets) / total_bets
        total_recommended = sum(b.recommended_bet for b in value_bets)
        expected_roi = (avg_ev - 1.0) * 100

        bet_type_ja = "Exacta" if bet_type == "exacta" else "Trifecta"
        lines.append(f"[Summary]")
        lines.append(f"  Total bets: {total_bets}")
        lines.append(f"  Average EV: {avg_ev:.2f}")
        lines.append(f"  Expected ROI: {expected_roi:+.1f}%")
        if total_recommended > 0:
            lines.append(f"  Total investment: {total_recommended:,}")

        return "\n".join(lines)

    def format_betting_result(
        self,
        result,
        bet_type: str = "exacta",
        top_n: int = 5,
        ev_threshold: float = 1.0,
    ) -> str:
        """
        Format complete betting result for display.

        Args:
            result: ExactaBettingResult or TrifectaBettingResult
            bet_type: "exacta" or "trifecta"
            top_n: Number of top win probabilities to show
            ev_threshold: Minimum EV threshold for display label

        Returns:
            Formatted string ready for printing
        """
        sections = []

        # Header
        sections.append(
            self.format_header(
                race_name=result.race_name,
                race_id=result.race_id,
                race_date=result.odds_datetime.split()[0] if result.odds_datetime else None,
            )
        )

        # Win probabilities
        sections.append(
            self.format_win_probabilities(
                win_probs=result.win_probabilities,
                horse_names=result.horse_names,
                top_n=top_n,
            )
        )

        # Value bets
        if bet_type == "exacta":
            sections.append(
                self.format_value_bets_exacta(
                    value_bets=result.value_bets,
                    ev_threshold=ev_threshold,
                )
            )
        else:
            sections.append(
                self.format_value_bets_trifecta(
                    value_bets=result.value_bets,
                    ev_threshold=ev_threshold,
                )
            )

        # Summary
        sections.append(
            self.format_summary(
                value_bets=result.value_bets,
                bet_type=bet_type,
            )
        )

        return "\n".join(sections)
