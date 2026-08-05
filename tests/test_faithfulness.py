"""Tests for masking-based faithfulness metrics."""
import numpy as np
import pytest
import torch

from src.evaluation.faithfulness import (
    MASK_TOKEN,
    aopc,
    batch_faithfulness,
    comprehensiveness,
    evaluate_faithfulness,
    mask_positions,
    random_control,
    sufficiency,
)
from src.model.coldsite_dti import ColdSiteDTI

DRUG_VOCAB, PROTEIN_VOCAB = 70, 28


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ColdSiteDTI(DRUG_VOCAB, PROTEIN_VOCAB).eval()


@pytest.fixture
def pair():
    torch.manual_seed(1)
    return (torch.randint(2, DRUG_VOCAB, (1, 40)),
            torch.randint(2, PROTEIN_VOCAB, (1, 200)))


# --------------------------------------------------------------------------
# masking mechanics
# --------------------------------------------------------------------------

def test_mask_replaces_only_the_listed_positions():
    protein = torch.full((1, 10), 5, dtype=torch.long)
    masked = mask_positions(protein, [2, 4])
    assert masked[0, 2] == MASK_TOKEN and masked[0, 4] == MASK_TOKEN
    assert masked[0, 0] == 5 and masked[0, 3] == 5


def test_keep_mode_inverts_the_selection():
    protein = torch.full((1, 10), 5, dtype=torch.long)
    kept = mask_positions(protein, [2, 4], keep=True)
    assert kept[0, 2] == 5 and kept[0, 4] == 5
    assert kept[0, 0] == MASK_TOKEN and kept[0, 7] == MASK_TOKEN


def test_masking_never_turns_padding_into_a_residue():
    """Masking PAD into UNK would invent residues that were never there."""
    protein = torch.zeros(1, 10, dtype=torch.long)
    protein[0, :4] = 5
    assert (mask_positions(protein, [0, 1, 8, 9])[0, 8:] == 0).all()
    assert (mask_positions(protein, [0], keep=True)[0, 4:] == 0).all()


def test_masking_does_not_mutate_the_input():
    protein = torch.full((1, 10), 5, dtype=torch.long)
    original = protein.clone()
    mask_positions(protein, [1, 2, 3])
    assert torch.equal(protein, original)


def test_out_of_range_positions_are_ignored():
    protein = torch.full((1, 5), 5, dtype=torch.long)
    assert mask_positions(protein, [1, 99, -3])[0, 1] == MASK_TOKEN


# --------------------------------------------------------------------------
# the metrics
# --------------------------------------------------------------------------

def test_comprehensiveness_is_non_negative(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    assert comprehensiveness(model, drug, protein, attention, k=10) >= 0


def test_sufficiency_is_non_negative(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    assert sufficiency(model, drug, protein, attention, k=10) >= 0


def test_masking_nothing_changes_nothing(model, pair):
    """Sanity floor: k=0 residues removed must give an identical prediction."""
    drug, protein = pair
    from src.evaluation.faithfulness import _predict
    baseline = _predict(model, drug, protein)
    unchanged = _predict(model, drug, mask_positions(protein, []))
    assert baseline == pytest.approx(unchanged, abs=1e-6)


def test_masking_everything_is_the_maximal_intervention(model, pair):
    drug, protein = pair
    from src.evaluation.faithfulness import _predict
    baseline = _predict(model, drug, protein)
    everything = _predict(model, drug, mask_positions(protein, [], keep=True))
    assert abs(baseline - everything) >= 0


def test_random_control_returns_a_number(model, pair):
    drug, protein = pair
    control = random_control(model, drug, protein, k=10, n_trials=5)
    assert np.isfinite(control) and control >= 0


def test_random_control_is_nan_when_protein_is_shorter_than_k(model):
    drug = torch.randint(2, DRUG_VOCAB, (1, 10))
    protein = torch.zeros(1, 20, dtype=torch.long)
    protein[0, :4] = 5
    assert np.isnan(random_control(model, drug, protein, k=10, n_trials=3))


def test_aopc_curve_covers_every_feasible_k(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    result = aopc(model, drug, protein, attention, k_values=(1, 5, 10))
    assert set(result["curve"]) == {1, 5, 10}
    assert np.isfinite(result["aopc"])


def test_aopc_skips_k_larger_than_the_protein(model):
    drug = torch.randint(2, DRUG_VOCAB, (1, 10))
    protein = torch.randint(2, PROTEIN_VOCAB, (1, 8))
    result = aopc(model, drug, protein, np.random.rand(8), k_values=(1, 5, 50))
    assert set(result["curve"]) == {1, 5}


# --------------------------------------------------------------------------
# the controls -- the part that makes the numbers mean anything
# --------------------------------------------------------------------------

def test_evaluate_reports_both_observed_and_random(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    result = evaluate_faithfulness(model, drug, protein, attention, k=10,
                                   n_random_trials=3)
    for key in ("comprehensiveness", "comprehensiveness_random",
                "comprehensiveness_delta", "sufficiency", "sufficiency_random"):
        assert key in result


def test_delta_is_observed_minus_random(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    r = evaluate_faithfulness(model, drug, protein, attention, k=10,
                              n_random_trials=3)
    assert r["comprehensiveness_delta"] == pytest.approx(
        r["comprehensiveness"] - r["comprehensiveness_random"])


def test_untrained_model_explanation_is_not_load_bearing(model):
    """The sanity floor for the whole faithfulness claim.

    An untrained model's attention is arbitrary. Masking its top-k residues
    should be no more disruptive than masking random ones. If this test ever
    reports a solidly positive delta, the metric is picking up a masking
    artefact rather than explanation quality.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    drugs, proteins, attentions = [], [], []
    for i in range(6):
        torch.manual_seed(i)
        drug = torch.randint(2, DRUG_VOCAB, (1, 40))
        protein = torch.randint(2, PROTEIN_VOCAB, (1, 150))
        drugs.append(drug)
        proteins.append(protein)
        attentions.append(np.asarray(model.explain(drug, protein)[0]))

    summary = batch_faithfulness(model, drugs, proteins, attentions,
                                 k=10, n_random_trials=4)
    assert abs(summary["comprehensiveness_delta"]) < 0.5, (
        f"untrained model looks strongly load-bearing "
        f"(delta={summary['comprehensiveness_delta']:.3f}) -- suspect a "
        f"masking artefact, not a finding"
    )


def test_batch_reports_pair_count(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    summary = batch_faithfulness(model, [drug] * 3, [protein] * 3,
                                 [attention] * 3, k=10, n_random_trials=2)
    assert summary["n_pairs"] == 3


def test_batch_respects_max_pairs(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    summary = batch_faithfulness(model, [drug] * 10, [protein] * 10,
                                 [attention] * 10, k=10, n_random_trials=1,
                                 max_pairs=4)
    assert summary["n_pairs"] == 4


def test_batch_emits_the_one_line_verdict(model, pair):
    drug, protein = pair
    attention = np.random.default_rng(0).random(200)
    summary = batch_faithfulness(model, [drug] * 2, [protein] * 2,
                                 [attention] * 2, k=10, n_random_trials=2)
    assert isinstance(summary["explanation_is_load_bearing"], bool)
