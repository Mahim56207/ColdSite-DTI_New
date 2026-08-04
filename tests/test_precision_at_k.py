"""Track C metric tests -- docs/03_GUIDE_124AD0067.md 'Common mistakes to avoid':
'Testing your evaluation code only once, on one lucky example.'"""
import numpy as np
import pytest

from src.evaluation.precision_at_k import (
    achievable_ceiling,
    batch_precision_at_k,
    normalised_precision_at_k,
    precision_at_k,
    precision_at_k_curve,
    top_k_positions,
)

SITES = {41, 42, 43, 143, 144, 145}


def perfect_attention(length=300, sites=SITES):
    attention = np.zeros(length)
    attention[sorted(sites)] = 1.0
    return attention


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------

def test_perfect_attention_hits_every_site_it_can():
    # 6 sites, k=6 -> all six retrieved
    assert precision_at_k(perfect_attention(), SITES, k=6) == 1.0


def test_attention_pointing_entirely_elsewhere_scores_zero():
    attention = np.zeros(300)
    attention[200:210] = 1.0
    assert precision_at_k(attention, SITES, k=10) == 0.0


def test_partial_overlap_counts_exactly():
    attention = np.zeros(300)
    attention[[41, 42, 250, 251]] = 1.0      # 2 of the top 4 are real sites
    assert precision_at_k(attention, SITES, k=4) == pytest.approx(0.5)


def test_score_is_a_fraction_of_k_not_of_site_count():
    # 6 sites but k=10 -> 6 hits out of 10 slots
    assert precision_at_k(perfect_attention(), SITES, k=10) == pytest.approx(0.6)


def test_single_site_protein():
    attention = np.zeros(50)
    attention[7] = 1.0
    assert precision_at_k(attention, {7}, k=1) == 1.0
    assert precision_at_k(attention, {7}, k=5) == pytest.approx(0.2)


# --------------------------------------------------------------------------
# the ceiling (k larger than the number of true sites)
# --------------------------------------------------------------------------

def test_ceiling_when_k_exceeds_site_count():
    # this is the guide's named edge case: k bigger than |true sites|
    assert achievable_ceiling(SITES, k=20) == pytest.approx(6 / 20)
    assert achievable_ceiling(SITES, k=6) == 1.0
    assert achievable_ceiling(SITES, k=3) == 1.0


def test_perfect_attention_cannot_reach_one_when_k_exceeds_sites():
    score = precision_at_k(perfect_attention(), SITES, k=20)
    assert score == pytest.approx(0.3)
    assert score < 1.0, "raw precision@20 with 6 sites can never reach 1.0"


def test_normalised_precision_reaches_one_for_perfect_attention():
    assert normalised_precision_at_k(perfect_attention(), SITES, k=20) == pytest.approx(1.0)
    assert normalised_precision_at_k(perfect_attention(), SITES, k=10) == pytest.approx(1.0)


def test_normalised_precision_is_nan_without_sites():
    assert np.isnan(normalised_precision_at_k(np.random.rand(100), set(), k=5))


def test_ceiling_of_empty_site_set_is_zero():
    assert achievable_ceiling(set(), k=10) == 0.0


# --------------------------------------------------------------------------
# input guards
# --------------------------------------------------------------------------

def test_k_larger_than_protein_raises_instead_of_silently_deflating():
    # argsort(x)[-500:] on a length-300 array returns 300 indices, and the old
    # code divided the hits by 500 anyway
    with pytest.raises(ValueError, match="exceeds protein length"):
        precision_at_k(np.random.rand(300), SITES, k=500)


def test_zero_and_negative_k_raise():
    for bad_k in (0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            precision_at_k(np.random.rand(300), SITES, k=bad_k)


def test_two_dimensional_attention_raises_with_a_useful_message():
    # a raw (1, seq_len) cross-attention map, unsqueezed
    with pytest.raises(ValueError, match="1D"):
        precision_at_k(np.random.rand(1, 300), SITES, k=10)


def test_nan_attention_raises():
    attention = np.random.rand(300)
    attention[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        precision_at_k(attention, SITES, k=10)


def test_empty_site_set_scores_zero_without_crashing():
    assert precision_at_k(np.random.rand(300), set(), k=10) == 0.0


# --------------------------------------------------------------------------
# tie-breaking
# --------------------------------------------------------------------------

def test_ties_do_not_systematically_favour_high_indices():
    """A completely flat attention map carries no information at all.

    With a stable argsort, [-k:] returns the highest indices every time, so
    sites near the C-terminus would score 1.0 on a model that learned nothing.
    """
    flat = np.ones(100)
    late_sites = set(range(90, 100))
    scores = [precision_at_k(flat, late_sites, k=10,
                             rng=np.random.default_rng(s)) for s in range(40)]
    mean = float(np.mean(scores))
    assert mean < 0.5, f"tied attention favouring late residues, mean={mean}"
    # chance level here is 10/100 = 0.1
    assert mean == pytest.approx(0.1, abs=0.12)


def test_tie_breaking_is_reproducible_given_a_seed():
    flat = np.ones(100)
    a = precision_at_k(flat, {1, 2, 3}, k=10, rng=np.random.default_rng(7))
    b = precision_at_k(flat, {1, 2, 3}, k=10, rng=np.random.default_rng(7))
    assert a == b


def test_top_k_returns_k_distinct_positions():
    top = top_k_positions(np.random.rand(300), k=25)
    assert len(top) == 25
    assert len(set(top.tolist())) == 25


def test_top_k_is_actually_the_largest_values():
    attention = np.arange(100, dtype=float)
    assert set(top_k_positions(attention, k=3).tolist()) == {97, 98, 99}


# --------------------------------------------------------------------------
# curve and batch
# --------------------------------------------------------------------------

def test_curve_reports_every_requested_k():
    curve = precision_at_k_curve(perfect_attention(), SITES, k_values=(5, 10, 20))
    assert set(curve) == {5, 10, 20}
    assert curve[5] == 1.0


def test_curve_skips_k_values_longer_than_the_protein():
    curve = precision_at_k_curve(np.random.rand(8), SITES, k_values=(5, 10, 20))
    assert set(curve) == {5}


def test_batch_skips_proteins_with_no_annotated_sites_and_says_so():
    attentions = [perfect_attention(), np.random.rand(300), perfect_attention()]
    sites = [SITES, set(), SITES]
    result = batch_precision_at_k(attentions, sites, k=6)
    assert result["n_evaluated"] == 2
    assert result["n_skipped_no_sites"] == 1
    assert result["mean_precision_at_k"] == pytest.approx(1.0)


def test_batch_skips_proteins_shorter_than_k():
    attentions = [np.random.rand(3), perfect_attention()]
    sites = [{0, 1}, SITES]
    result = batch_precision_at_k(attentions, sites, k=6)
    assert result["n_skipped_too_short"] == 1
    assert result["n_evaluated"] == 1


def test_batch_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        batch_precision_at_k([np.random.rand(300)] * 3, [SITES] * 2, k=5)


def test_batch_on_all_unusable_returns_nan_not_zero():
    result = batch_precision_at_k([np.random.rand(300)], [set()], k=5)
    assert np.isnan(result["mean_precision_at_k"])
    assert result["n_evaluated"] == 0


def test_batch_reports_mean_ceiling_so_raw_scores_are_interpretable():
    result = batch_precision_at_k([perfect_attention()] * 3, [SITES] * 3, k=20)
    assert result["mean_precision_at_k"] == pytest.approx(0.3)
    assert result["mean_ceiling"] == pytest.approx(0.3)
    assert result["mean_normalised"] == pytest.approx(1.0)
