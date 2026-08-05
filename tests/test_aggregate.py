"""Tests for seed aggregation and family-wise error control."""
import numpy as np
import pytest

from src.evaluation.aggregate import (
    MIN_SEEDS_FOR_A_CLAIM,
    aggregate_seeds,
    audit_table,
    degradation,
    holm_bonferroni,
)


# --------------------------------------------------------------------------
# seed aggregation
# --------------------------------------------------------------------------

def test_mean_and_sample_std():
    result = aggregate_seeds([0.40, 0.42, 0.44])
    assert result["mean"] == pytest.approx(0.42)
    assert result["std"] == pytest.approx(np.std([0.40, 0.42, 0.44], ddof=1))


def test_three_seeds_is_the_minimum_for_a_claim():
    assert aggregate_seeds([0.4, 0.5]).get("sufficient_seeds") is False
    assert aggregate_seeds([0.4, 0.5, 0.6])["sufficient_seeds"] is True
    assert MIN_SEEDS_FOR_A_CLAIM == 3


def test_single_seed_has_no_spread_estimate():
    result = aggregate_seeds([0.42])
    assert result["mean"] == pytest.approx(0.42)
    assert np.isnan(result["std"])
    assert not result["sufficient_seeds"]


def test_empty_input_returns_nan_not_zero():
    result = aggregate_seeds([])
    assert np.isnan(result["mean"]) and result["n_seeds"] == 0


def test_nan_seeds_are_dropped_and_counted():
    result = aggregate_seeds([0.4, float("nan"), 0.6, None])
    assert result["n_seeds"] == 2
    assert result["mean"] == pytest.approx(0.5)


def test_confidence_interval_brackets_the_mean():
    result = aggregate_seeds([0.30, 0.40, 0.50, 0.45])
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_wider_spread_gives_a_wider_interval():
    tight = aggregate_seeds([0.40, 0.41, 0.42])
    loose = aggregate_seeds([0.10, 0.40, 0.75])
    assert (loose["ci95_high"] - loose["ci95_low"]) > \
           (tight["ci95_high"] - tight["ci95_low"])


# --------------------------------------------------------------------------
# multiple comparisons
# --------------------------------------------------------------------------

def test_single_test_is_unchanged():
    result = holm_bonferroni({"a": 0.04})
    assert result["a"]["significant"]
    assert result["a"]["adjusted_alpha"] == pytest.approx(0.05)


def test_marginal_p_values_do_not_survive_a_large_family():
    """0.04 across 32 tests is what ~1.6 chance hits look like."""
    family = {f"cell_{i}": 0.04 for i in range(32)}
    assert not any(v["significant"] for v in holm_bonferroni(family).values())


def test_a_strong_effect_survives_correction():
    family = {"strong": 0.0001}
    family.update({f"null_{i}": 0.8 for i in range(31)})
    result = holm_bonferroni(family)
    assert result["strong"]["significant"]
    assert not result["null_0"]["significant"]


def test_holm_is_more_powerful_than_plain_bonferroni():
    family = {"a": 0.001, "b": 0.012, "c": 0.9, "d": 0.9}
    result = holm_bonferroni(family)
    # plain Bonferroni threshold would be 0.05/4 = 0.0125 for every test;
    # Holm tests 'b' at 0.05/3 = 0.0167, so it survives
    assert result["a"]["significant"] and result["b"]["significant"]


def test_stepwise_stops_at_the_first_failure():
    """Holm's validity depends on rejecting nothing after the first failure.

    Sorted: 0.001, 0.030, 0.031. Thresholds: 0.0167, 0.025, 0.05.
    'b' fails at 0.030 > 0.025. 'c' at 0.031 would clear its own 0.05
    threshold, but Holm must reject it anyway -- testing each p against its own
    threshold independently is the classic misimplementation.
    """
    result = holm_bonferroni({"a": 0.001, "b": 0.030, "c": 0.031})
    assert result["a"]["significant"]
    assert not result["b"]["significant"]
    assert result["c"]["adjusted_alpha"] == pytest.approx(0.05)
    assert not result["c"]["significant"], \
        "c clears its own threshold but ranks after a failure"


def test_ranks_are_assigned_in_ascending_p_order():
    result = holm_bonferroni({"hi": 0.5, "lo": 0.001, "mid": 0.05})
    assert result["lo"]["rank"] == 1 and result["hi"]["rank"] == 3


def test_nan_p_values_are_kept_but_never_significant():
    result = holm_bonferroni({"good": 0.001, "broken": float("nan")})
    assert result["good"]["significant"]
    assert not result["broken"]["significant"]


def test_empty_family_is_handled():
    assert holm_bonferroni({}) == {}


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------

def test_degradation_measures_warm_to_coldest():
    result = degradation({"random": 0.62, "cold_drug": 0.51,
                          "cold_target": 0.47, "cold_pair": 0.35})
    assert result["absolute_drop"] == pytest.approx(0.27)
    assert result["relative_drop"] == pytest.approx(0.27 / 0.62)
    assert result["monotonic"]


def test_non_monotonic_ladder_is_flagged():
    """A ladder that goes down then up is not 'degradation'."""
    result = degradation({"random": 0.62, "cold_drug": 0.40,
                          "cold_target": 0.55, "cold_pair": 0.35})
    assert not result["monotonic"]


def test_degradation_needs_at_least_two_levels():
    assert np.isnan(degradation({"random": 0.5})["absolute_drop"])


def test_degradation_uses_the_hardest_level_actually_present():
    result = degradation({"random": 0.6, "cold_drug": 0.4})
    assert result["coldest_level"] == "cold_drug"


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

def _cell(mean, std, n_seeds=3):
    return {"precision_at_k": {"mean": mean, "std": std, "n_seeds": n_seeds,
                               "sufficient_seeds": n_seeds >= 3}}


def test_table_has_a_row_per_model():
    grid = {
        "coldsite_dti": {l: _cell(v, 0.02) for l, v in
                         zip(("random", "cold_drug", "cold_target", "cold_pair"),
                             (0.62, 0.51, 0.47, 0.35))},
        "deepdta": {l: _cell(v, 0.03) for l, v in
                    zip(("random", "cold_drug", "cold_target", "cold_pair"),
                        (0.55, 0.48, 0.44, 0.30))},
    }
    table = audit_table(grid)
    assert "coldsite_dti" in table and "deepdta" in table
    assert "Cold-Pair" in table and "±" in table


def test_under_powered_cells_are_flagged():
    grid = {"m": {"random": _cell(0.5, float("nan"), n_seeds=1)}}
    assert "!" in audit_table(grid)


def test_missing_cells_render_as_not_available():
    grid = {"m": {"random": _cell(0.5, 0.01)}}
    assert "n/a" in audit_table(grid)
