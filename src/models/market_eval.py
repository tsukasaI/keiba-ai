"""Market baseline and delta-R2 evaluation harness (V2, Task 0.1).

Spec: docs/V2_TASKS.md section 2. Pure functions only — no I/O here; data
loading lives in scripts/eval_market_baseline.py.

Terminology:
- pi: market-implied win probabilities (from win odds, takeout normalized out)
- f:  fundamental model win probabilities (odds-free model; Phase 1)
- b:  recalibrated market baseline pi^beta (what "the market" means in delta-R2)
- c:  blended probabilities softmax(alpha*ln f + beta*ln pi)

All per-race probability vectors sum to 1 over the runners that pass the
drop rules in prepare_race().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

PROB_FLOOR = 1e-6
MIN_RUNNERS = 5

# Drop reasons returned by prepare_race
DROP_NO_WINNER = "no_winner"
DROP_WINNER_NO_ODDS = "winner_no_odds"
DROP_TOO_FEW_RUNNERS = "too_few_runners"


@dataclass(frozen=True)
class PreparedRace:
    """One race after the section-2.1 drop rules.

    pi and winner_mask are aligned over the surviving runners only.
    winner_mask may have >1 True (dead heat); the winner's probability is
    then the sum over winners, and the null model credit is k/n.
    """

    pi: np.ndarray
    winner_mask: np.ndarray

    @property
    def n_runners(self) -> int:
        return len(self.pi)

    @property
    def n_winners(self) -> int:
        return int(self.winner_mask.sum())


def implied_probs(odds: np.ndarray) -> np.ndarray:
    """Market-implied win probs from decimal win odds: (1/O_i) / sum(1/O_j).

    Normalization removes takeout/breakage. Caller must pass valid odds
    (finite, > 1.0) — prepare_race() enforces this for real data.
    """
    odds = np.asarray(odds, dtype=float)
    if odds.ndim != 1 or len(odds) == 0:
        raise ValueError("odds must be a non-empty 1-D array")
    if not np.all(np.isfinite(odds) & (odds > 1.0)):
        raise ValueError("all odds must be finite and > 1.0")
    inv = 1.0 / odds
    return inv / inv.sum()


def clip_normalize(probs: np.ndarray) -> np.ndarray:
    """Clip to [PROB_FLOOR, 1] then renormalize to sum 1 (spec 2.2)."""
    p = np.clip(np.asarray(probs, dtype=float), PROB_FLOOR, 1.0)
    return p / p.sum()


def prepare_race(
    odds: np.ndarray,
    finish_positions: np.ndarray,
    min_runners: int = MIN_RUNNERS,
) -> tuple[PreparedRace | None, str | None]:
    """Apply the section-2.1 drop rules to one race's raw entry rows.

    odds / finish_positions cover ALL entries (use NaN/None for missing).
    Returns (PreparedRace, None) or (None, drop_reason).

    Rules:
    - a horse is a runner iff odds is finite and > 1.0 AND finish_position present
    - race dropped if it has no finish_position == 1 (cancelled/unparsed)
    - race dropped if any winner lacks valid odds (winner prob would be undefined)
    - race dropped if fewer than min_runners runners survive
    """
    odds = np.asarray(
        [np.nan if v is None else v for v in np.asarray(odds, dtype=object)],
        dtype=float,
    )
    fp = np.asarray(
        [np.nan if v is None else v for v in np.asarray(finish_positions, dtype=object)],
        dtype=float,
    )
    if odds.shape != fp.shape:
        raise ValueError("odds and finish_positions must have the same shape")

    winner_all = fp == 1
    if not winner_all.any():
        return None, DROP_NO_WINNER

    valid_odds = np.isfinite(odds) & (odds > 1.0)
    if bool(np.any(winner_all & ~valid_odds)):
        return None, DROP_WINNER_NO_ODDS

    runners = valid_odds & np.isfinite(fp)
    if int(runners.sum()) < min_runners:
        return None, DROP_TOO_FEW_RUNNERS

    pi = implied_probs(odds[runners])
    return PreparedRace(pi=pi, winner_mask=winner_all[runners]), None


def winner_prob(probs: np.ndarray, winner_mask: np.ndarray) -> float:
    """Probability credited to the actual outcome (dead heats: sum over winners)."""
    return float(np.asarray(probs, dtype=float)[np.asarray(winner_mask, dtype=bool)].sum())


def mcfadden_r2(
    winner_probs: np.ndarray,
    field_sizes: np.ndarray,
    n_winners: np.ndarray | None = None,
) -> float:
    """McFadden pseudo-R2 = 1 - LL(model)/LL(null), winner log-likelihood (spec 2.2).

    null model credit per race = n_winners / field_size (1/n in the normal,
    non-dead-heat case), so a uniform model scores exactly 0.
    """
    p = np.clip(np.asarray(winner_probs, dtype=float), PROB_FLOOR, 1.0)
    n = np.asarray(field_sizes, dtype=float)
    k = np.ones_like(n) if n_winners is None else np.asarray(n_winners, dtype=float)
    ll = np.log(p).sum()
    ll0 = np.log(k / n).sum()
    return float(1.0 - ll / ll0)


def power_probs(pi: np.ndarray, beta: float) -> np.ndarray:
    """Recalibrated market baseline b_i(beta) = pi_i^beta / sum_j pi_j^beta (spec 2.3)."""
    logp = beta * np.log(np.clip(np.asarray(pi, dtype=float), PROB_FLOOR, 1.0))
    logp -= logp.max()
    w = np.exp(logp)
    return w / w.sum()


def blend_probs(f: np.ndarray, pi: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Benter blend c_i = softmax(alpha*ln f_i + beta*ln pi_i) (spec 2.4)."""
    lf = np.log(np.clip(np.asarray(f, dtype=float), PROB_FLOOR, 1.0))
    lp = np.log(np.clip(np.asarray(pi, dtype=float), PROB_FLOOR, 1.0))
    logc = alpha * lf + beta * lp
    logc -= logc.max()
    w = np.exp(logc)
    return w / w.sum()


def _nll_power(beta: float, races: list[PreparedRace]) -> float:
    total = 0.0
    for r in races:
        p = winner_prob(power_probs(r.pi, beta), r.winner_mask)
        total -= np.log(max(p, PROB_FLOOR))
    return total


def fit_power_recalibration(races: list[PreparedRace]) -> float:
    """MLE beta for the market-only baseline, fit on the blend-fit slice (spec 2.3)."""
    if not races:
        raise ValueError("need at least one race to fit beta")
    res = minimize_scalar(
        _nll_power, bounds=(0.05, 3.0), args=(races,), method="bounded"
    )
    return float(res.x)


def _nll_blend(
    theta: np.ndarray,
    fs: list[np.ndarray],
    pis: list[np.ndarray],
    winner_masks: list[np.ndarray],
) -> float:
    alpha, beta = theta
    total = 0.0
    for f, pi, mask in zip(fs, pis, winner_masks):
        p = winner_prob(blend_probs(f, pi, alpha, beta), mask)
        total -= np.log(max(p, PROB_FLOOR))
    return total


def fit_blend(
    fs: list[np.ndarray],
    pis: list[np.ndarray],
    winner_masks: list[np.ndarray],
) -> tuple[float, float]:
    """MLE (alpha, beta) for the blend, fit on the blend-fit slice (spec 2.4).

    alpha <= 0 means the fundamental model adds nothing over the market.
    """
    if not (len(fs) == len(pis) == len(winner_masks)) or not fs:
        raise ValueError("fs, pis, winner_masks must be equal-length and non-empty")
    res = minimize(
        _nll_blend,
        x0=np.array([0.5, 1.0]),
        args=(fs, pis, winner_masks),
        method="Nelder-Mead",
    )
    alpha, beta = res.x
    return float(alpha), float(beta)


def bootstrap_delta_r2(
    blend_winner_probs: np.ndarray,
    market_winner_probs: np.ndarray,
    field_sizes: np.ndarray,
    n_winners: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """delta-R2 = R2(blend) - R2(market) with a race-resampling bootstrap CI (spec 2.5).

    Inputs are per-eval-race winner probabilities computed with parameters
    fitted on the blend-fit slice (never on the eval races themselves).
    Returns (point_estimate, ci_lo_2.5pct, ci_hi_97.5pct).
    """
    bp = np.log(np.clip(np.asarray(blend_winner_probs, dtype=float), PROB_FLOOR, 1.0))
    mp = np.log(np.clip(np.asarray(market_winner_probs, dtype=float), PROB_FLOOR, 1.0))
    n = np.asarray(field_sizes, dtype=float)
    k = np.ones_like(n) if n_winners is None else np.asarray(n_winners, dtype=float)
    np_log0 = np.log(k / n)
    if not (len(bp) == len(mp) == len(np_log0)):
        raise ValueError("all inputs must have the same length")
    n_races = len(bp)
    if n_races == 0:
        raise ValueError("need at least one race")

    def delta(ll_b: np.ndarray, ll_m: np.ndarray, ll_0: np.ndarray) -> np.ndarray:
        # R2_b - R2_m = (1 - ll_b/ll_0) - (1 - ll_m/ll_0) = (ll_m - ll_b) / ll_0
        return (ll_m - ll_b) / ll_0

    point = float(delta(bp.sum(), mp.sum(), np_log0.sum()))

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    chunk = 1_000  # bounded memory: n_boot x n_races index matrices in chunks
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n_races, size=(size, n_races))
        deltas[start : start + size] = delta(
            bp[idx].sum(axis=1), mp[idx].sum(axis=1), np_log0[idx].sum(axis=1)
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, float(lo), float(hi)
