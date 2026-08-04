"""Tests for the permutation significance test."""
import numpy as np
import pytest

from src.evaluation.significance_test import (
    evaluate_with_significance,
    permutation_test,
    permutation_test_batch,
)

SITES = {41, 42, 43, 143, 144, 145}


def perfect_attention(length=300, sites=SITES):
    attention = np.zeros(length)
    attention[sorted(sites)] = 1.0
    return attention


def test_perfect_explanation_is_significant():
    result = evaluate_with_significance(perfect_attention(), SITES, k=10)
    assert result["significant"]
    assert result["p_value"] < 0.01


def test_random_explanation_is_not_significant():
    attention = np.random.default_rng(0).random(300)
    result = evaluate_with_significance(attention, SITES, k=10)
    assert not result["significant"]


def test_p_value_is_never_exactly_zero():
    """(1 + hits) / (1 + n_trials), not a bare mean.

    A bare mean returns 0.0 when no permutation beats the observation, which
    claims infinite confidence from 1000 draws. The floor should be 1/1001.
    """
    result = permutation_test(SITES, 300, observed_precision=1.0, k=10, n_trials=1000)
    assert result["p_value"] > 0
    assert result["p_value"] == pytest.approx(1 / 1001)


def test_p_value_is_at_most_one():
    result = permutation_test(SITES, 300, observed_precision=0.0, k=10, n_trials=200)
    assert result["p_value"] <= 1.0


def test_chance_level_matches_the_analytic_expectation():
    """E[precision@k] under the null is |sites| / protein_length."""
    result = permutation_test(SITES, 300, observed_precision=0.5,
                              k=10, n_trials=4000, seed=1)
    assert result["chance_mean"] == pytest.approx(6 / 300, abs=0.01)


def test_chance_level_rises_with_more_annotated_sites():
    sparse = permutation_test({1, 2}, 500, 0.5, k=10, n_trials=2000, seed=1)
    dense = permutation_test(set(range(100)), 500, 0.5, k=10, n_trials=2000, seed=1)
    assert dense["chance_mean"] > sparse["chance_mean"]


def test_missing_observed_precision_is_a_signature_error_not_a_deep_crash():
    with pytest.raises(TypeError):
        permutation_test(SITES, 300, k=10)          # observed_precision omitted


def test_out_of_range_observed_precision_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        permutation_test(SITES, 300, observed_precision=1.4, k=10)


def test_k_larger_than_protein_is_rejected():
    with pytest.raises(ValueError, match="exceeds protein length"):
        permutation_test(SITES, 5, observed_precision=0.5, k=10)


def test_results_are_reproducible_given_a_seed():
    a = permutation_test(SITES, 300, 0.4, k=10, n_trials=500, seed=3)
    b = permutation_test(SITES, 300, 0.4, k=10, n_trials=500, seed=3)
    assert a == b


# --------------------------------------------------------------------------
# split-level test
# --------------------------------------------------------------------------

def test_batch_test_detects_a_genuinely_good_split():
    attentions = [perfect_attention() for _ in range(20)]
    sites = [SITES] * 20
    result = permutation_test_batch(attentions, sites, k=10, n_trials=300)
    assert result["significant"]
    assert result["n_proteins"] == 20


def test_batch_test_does_not_fire_on_noise():
    rng = np.random.default_rng(0)
    attentions = [rng.random(300) for _ in range(20)]
    sites = [SITES] * 20
    result = permutation_test_batch(attentions, sites, k=10, n_trials=300)
    assert not result["significant"]


def test_batch_test_skips_unusable_proteins():
    attentions = [perfect_attention(), np.random.rand(300), np.random.rand(4)]
    sites = [SITES, set(), {0, 1}]
    result = permutation_test_batch(attentions, sites, k=10, n_trials=100)
    assert result["n_proteins"] == 1


def test_batch_test_on_nothing_usable_returns_nan():
    result = permutation_test_batch([np.random.rand(300)], [set()], k=10)
    assert np.isnan(result["p_value"])
    assert result["n_proteins"] == 0
    assert not result["significant"]
