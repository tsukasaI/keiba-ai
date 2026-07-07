"""Tests for src/models/market_eval.py (V2 Task 0.1, spec docs/V2_TASKS.md section 2)."""

import numpy as np
import pytest

from src.models.market_eval import (
    DROP_NO_WINNER,
    DROP_TOO_FEW_RUNNERS,
    DROP_WINNER_NO_ODDS,
    PreparedRace,
    blend_probs,
    bootstrap_delta_r2,
    fit_blend,
    fit_power_recalibration,
    implied_probs,
    mcfadden_r2,
    power_probs,
    prepare_race,
    winner_prob,
)


class TestImpliedProbs:
    def test_even_two_horse(self):
        np.testing.assert_allclose(implied_probs(np.array([2.0, 2.0])), [0.5, 0.5])

    def test_sums_to_one(self):
        pi = implied_probs(np.array([1.5, 4.2, 12.0, 88.7]))
        assert pi.sum() == pytest.approx(1.0)
        assert (pi > 0).all()

    def test_takeout_invariance(self):
        # Scaling all odds by a constant (= different takeout) leaves pi unchanged.
        base = np.array([2.0, 4.0, 10.0])
        np.testing.assert_allclose(implied_probs(base), implied_probs(base / 1.25))

    def test_rejects_invalid_odds(self):
        with pytest.raises(ValueError):
            implied_probs(np.array([2.0, 1.0]))  # <= 1.0
        with pytest.raises(ValueError):
            implied_probs(np.array([2.0, np.nan]))
        with pytest.raises(ValueError):
            implied_probs(np.array([]))


class TestMcFaddenR2:
    def test_single_even_race_is_zero(self):
        # p_winner = 0.5 in a 2-horse race == null model -> R2 = 0 exactly.
        assert mcfadden_r2([0.5], [2]) == pytest.approx(0.0)

    def test_uniform_many_races_is_zero(self):
        n = 8
        assert mcfadden_r2([1.0 / n] * 50, [n] * 50) == pytest.approx(0.0)

    def test_perfect_prediction_is_one(self):
        assert mcfadden_r2([1.0] * 10, [10] * 10) == pytest.approx(1.0)

    def test_better_than_uniform_is_positive(self):
        assert mcfadden_r2([0.3] * 20, [8] * 20) > 0.0

    def test_dead_heat_null_uses_k_over_n(self):
        # Uniform probs with a 2-way dead heat: model credit 2/6 == null credit 2/6.
        assert mcfadden_r2([2.0 / 6.0], [6], n_winners=[2]) == pytest.approx(0.0)


class TestPrepareRace:
    def test_basic_race(self):
        prepared, reason = prepare_race(
            odds=[2.0, 4.0, 8.0, 8.0, 16.0, 32.0],
            finish_positions=[3, 1, 2, 4, 5, 6],
        )
        assert reason is None
        assert prepared.n_runners == 6
        assert prepared.n_winners == 1
        assert prepared.pi.sum() == pytest.approx(1.0)
        # winner (index 1) carries the mask
        assert prepared.winner_mask.tolist() == [False, True, False, False, False, False]

    def test_horse_without_odds_excluded_but_race_kept(self):
        prepared, reason = prepare_race(
            odds=[2.0, None, 8.0, 8.0, 16.0, 32.0],
            finish_positions=[1, 2, 3, 4, 5, 6],
        )
        assert reason is None
        assert prepared.n_runners == 5

    def test_scratched_horse_excluded(self):
        # odds present but no finish position (cancelled after betting opened)
        prepared, reason = prepare_race(
            odds=[2.0, 4.0, 8.0, 8.0, 16.0, 32.0],
            finish_positions=[1, None, 2, 3, 4, 5],
        )
        assert reason is None
        assert prepared.n_runners == 5

    def test_winner_without_odds_drops_race(self):
        prepared, reason = prepare_race(
            odds=[None, 4.0, 8.0, 8.0, 16.0, 32.0],
            finish_positions=[1, 2, 3, 4, 5, 6],
        )
        assert prepared is None
        assert reason == DROP_WINNER_NO_ODDS

    def test_no_winner_drops_race(self):
        prepared, reason = prepare_race(
            odds=[2.0, 4.0, 8.0, 8.0, 16.0],
            finish_positions=[None, None, None, None, None],
        )
        assert prepared is None
        assert reason == DROP_NO_WINNER

    def test_too_few_runners_drops_race(self):
        prepared, reason = prepare_race(
            odds=[2.0, 4.0, 8.0, None],
            finish_positions=[1, 2, 3, 4],
        )
        assert prepared is None
        assert reason == DROP_TOO_FEW_RUNNERS

    def test_dead_heat_two_winners(self):
        prepared, reason = prepare_race(
            odds=[2.0, 4.0, 8.0, 8.0, 16.0],
            finish_positions=[1, 1, 3, 4, 5],
        )
        assert reason is None
        assert prepared.n_winners == 2
        # winner credit is the sum of both winners' probs
        assert winner_prob(prepared.pi, prepared.winner_mask) == pytest.approx(
            prepared.pi[0] + prepared.pi[1]
        )


def _synthetic_races(n_races, n_horses, rng):
    """Market probs from a Dirichlet; returns list of pi arrays."""
    return [rng.dirichlet(np.full(n_horses, 0.8)) for _ in range(n_races)]


def _sample_winner_mask(probs, rng):
    mask = np.zeros(len(probs), dtype=bool)
    mask[rng.choice(len(probs), p=probs / probs.sum())] = True
    return mask


class TestFitPowerRecalibration:
    def test_recovers_true_beta(self):
        rng = np.random.default_rng(42)
        true_beta = 1.3
        races = []
        for pi in _synthetic_races(2000, 8, rng):
            winners = _sample_winner_mask(power_probs(pi, true_beta), rng)
            races.append(PreparedRace(pi=pi, winner_mask=winners))
        beta = fit_power_recalibration(races)
        assert beta == pytest.approx(true_beta, abs=0.15)

    def test_identity_when_market_is_truth(self):
        rng = np.random.default_rng(7)
        races = []
        for pi in _synthetic_races(2000, 8, rng):
            races.append(PreparedRace(pi=pi, winner_mask=_sample_winner_mask(pi, rng)))
        beta = fit_power_recalibration(races)
        assert beta == pytest.approx(1.0, abs=0.15)


class TestFitBlend:
    def _make(self, n_races, rng, truth):
        fs, pis, masks = [], [], []
        for pi in _synthetic_races(n_races, 8, rng):
            f = rng.dirichlet(np.full(8, 0.8))  # independent of pi
            source = pi if truth == "market" else f
            fs.append(f)
            pis.append(pi)
            masks.append(_sample_winner_mask(source, rng))
        return fs, pis, masks

    def test_alpha_near_zero_when_market_is_truth(self):
        rng = np.random.default_rng(0)
        fs, pis, masks = self._make(2000, rng, truth="market")
        alpha, beta = fit_blend(fs, pis, masks)
        assert abs(alpha) < 0.15
        assert beta == pytest.approx(1.0, abs=0.2)

    def test_beta_near_zero_when_fundamental_is_truth(self):
        rng = np.random.default_rng(1)
        fs, pis, masks = self._make(2000, rng, truth="fundamental")
        alpha, beta = fit_blend(fs, pis, masks)
        assert alpha == pytest.approx(1.0, abs=0.2)
        assert abs(beta) < 0.15

    def test_blend_probs_normalized(self):
        f = np.array([0.5, 0.3, 0.2])
        pi = np.array([0.2, 0.3, 0.5])
        c = blend_probs(f, pi, 0.7, 0.8)
        assert c.sum() == pytest.approx(1.0)


class TestBootstrapDeltaR2:
    def test_identical_models_ci_contains_zero(self):
        rng = np.random.default_rng(3)
        wp = rng.uniform(0.05, 0.6, size=500)
        ns = np.full(500, 10)
        point, lo, hi = bootstrap_delta_r2(wp, wp, ns, n_boot=500, seed=0)
        assert point == pytest.approx(0.0)
        assert lo <= 0.0 <= hi

    def test_uniformly_better_model_excludes_zero(self):
        rng = np.random.default_rng(4)
        market = rng.uniform(0.05, 0.4, size=500)
        blend = np.minimum(market * 1.3, 0.9)  # strictly better everywhere
        ns = np.full(500, 10)
        point, lo, hi = bootstrap_delta_r2(blend, market, ns, n_boot=500, seed=0)
        assert point > 0.0
        assert lo > 0.0
