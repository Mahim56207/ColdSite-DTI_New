"""Contract tests for model adapters.

A broken adapter does not crash the audit -- it produces a wrong precision@k.
These tests exist so contract violations fail loudly at registration time.
"""
import numpy as np
import pytest
import torch

from src.evaluation.model_registry import (
    ColdSiteDTIAdapter,
    ExplainableDTIModel,
    UniformBaseline,
    available_models,
    get_model,
    register,
    validate_adapter,
)
from src.evaluation.precision_at_k import precision_at_k

DRUG_VOCAB, PROTEIN_VOCAB = 70, 28


@pytest.fixture
def pair():
    torch.manual_seed(0)
    return (torch.randint(2, DRUG_VOCAB, (1, 40)),
            torch.randint(2, PROTEIN_VOCAB, (1, 150)))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def test_our_model_and_the_control_are_registered():
    assert "coldsite_dti" in available_models()
    assert "uniform_control" in available_models()


def test_unknown_model_names_the_alternatives():
    with pytest.raises(KeyError, match="Registered"):
        get_model("deepdta_that_nobody_wrote_yet")


def test_double_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        @register("coldsite_dti")
        class Duplicate(ExplainableDTIModel):
            def predict(self, drug, protein): return 0.0
            def explain(self, drug, protein): return np.zeros(1)


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ExplainableDTIModel()


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------

def test_coldsite_adapter_satisfies_the_contract(pair):
    drug, protein = pair
    report = validate_adapter(ColdSiteDTIAdapter(), drug, protein,
                              expected_length=150)
    assert report["valid"], report["problems"]
    assert report["n_weights"] == 150


def test_adapter_explanation_feeds_precision_at_k_directly(pair):
    drug, protein = pair
    weights = ColdSiteDTIAdapter().explain(drug, protein)
    score = precision_at_k(weights, {10, 11, 12, 90, 91}, k=10)
    assert 0.0 <= score <= 1.0


def test_adapter_accepts_an_unbatched_protein():
    torch.manual_seed(0)
    adapter = ColdSiteDTIAdapter()
    weights = adapter.explain(torch.randint(2, DRUG_VOCAB, (30,)),
                              torch.randint(2, PROTEIN_VOCAB, (120,)))
    assert weights.size == 120


def test_adapter_explanation_matches_real_residue_count_not_padded_length():
    """The misalignment that would silently corrupt every downstream number."""
    torch.manual_seed(0)
    protein = torch.zeros(1, 300, dtype=torch.long)
    protein[0, :90] = torch.randint(2, PROTEIN_VOCAB, (90,))
    weights = ColdSiteDTIAdapter().explain(torch.randint(2, DRUG_VOCAB, (1, 30)),
                                           protein)
    assert weights.size == 90, "explanation must cover real residues only"


def test_uniform_control_is_flat_and_normalised(pair):
    _drug, protein = pair
    weights = UniformBaseline().explain(None, protein)
    assert weights.size == 150
    assert np.allclose(weights, weights[0])
    assert np.isclose(weights.sum(), 1.0)


def test_uniform_control_scores_near_chance():
    """The audit's floor. A flat explainer must not look informative."""
    protein = torch.randint(2, PROTEIN_VOCAB, (1, 200))
    weights = UniformBaseline().explain(None, protein)
    sites = set(range(50, 56))
    scores = [precision_at_k(weights, sites, k=10,
                             rng=np.random.default_rng(s)) for s in range(40)]
    assert np.mean(scores) == pytest.approx(6 / 200 * 10 / 10, abs=0.12)


# --------------------------------------------------------------------------
# validate_adapter catches real mistakes
# --------------------------------------------------------------------------

class _WrongShape(ExplainableDTIModel):
    registry_name = "wrong_shape"
    def predict(self, drug, protein): return 0.5
    def explain(self, drug, protein): return np.zeros((1, 150))


class _NegativeWeights(ExplainableDTIModel):
    registry_name = "negative"
    def predict(self, drug, protein): return 0.5
    def explain(self, drug, protein): return np.full(150, -1.0)


class _NaNPrediction(ExplainableDTIModel):
    registry_name = "nan_pred"
    def predict(self, drug, protein): return float("nan")
    def explain(self, drug, protein): return np.ones(150)


class _WrongLength(ExplainableDTIModel):
    registry_name = "wrong_length"
    def predict(self, drug, protein): return 0.5
    def explain(self, drug, protein): return np.ones(300)


def test_validator_catches_two_dimensional_explanations(pair):
    report = validate_adapter(_WrongShape(), *pair)
    assert not report["valid"]
    assert any("1D" in p for p in report["problems"])


def test_validator_catches_negative_weights(pair):
    report = validate_adapter(_NegativeWeights(), *pair)
    assert any("negative" in p for p in report["problems"])


def test_validator_catches_nan_predictions(pair):
    report = validate_adapter(_NaNPrediction(), *pair)
    assert any("NaN" in p for p in report["problems"])


def test_validator_catches_the_length_mismatch(pair):
    """The single most dangerous adapter bug: it produces a number, not an error."""
    report = validate_adapter(_WrongLength(), *pair, expected_length=150)
    assert not report["valid"]
    assert any("misalign" in p for p in report["problems"])


def test_validator_reports_exceptions_rather_than_propagating(pair):
    class _Broken(ExplainableDTIModel):
        registry_name = "broken"
        def predict(self, drug, protein): raise RuntimeError("boom")
        def explain(self, drug, protein): raise ValueError("bang")

    report = validate_adapter(_Broken(), *pair)
    assert not report["valid"] and len(report["problems"]) == 2


# --------------------------------------------------------------------------
# length measurement -- the pattern STATUS.md records as a fixed bug in
# explain(), which had survived in two other places
# --------------------------------------------------------------------------

def test_every_explainer_agrees_on_length_for_an_interior_pad():
    """Counting non-pad tokens and measuring to the last non-pad position
    disagree the moment a sequence has an interior pad, and the count is
    SHORTER than the residues it covers. UniformBaseline is the audit's control
    floor; a floor line one residue short is scored against shifted ground
    truth and nothing raises.
    """
    import torch

    from src.model.coldsite_dti import ColdSiteDTI
    from src.model.protein_encoder import real_lengths

    protein = torch.zeros(1, 30, dtype=torch.long)
    protein[0, :20] = torch.randint(2, 28, (20,))
    protein[0, 7] = 0                       # interior pad
    drug = torch.randint(2, 70, (1, 20))

    expected = int(real_lengths(protein)[0])
    assert expected == 20, "fixture no longer exercises the interior-pad case"
    assert int((protein[0] != 0).sum()) == 19, "count and span must differ here"

    torch.manual_seed(0)
    model = ColdSiteDTI(70, 28).eval()
    assert len(model.explain(drug, protein)[0]) == expected
    assert get_model("uniform_control").explain(drug, protein).size == expected


def test_uniform_control_passes_its_own_contract_on_an_interior_pad():
    import torch

    from src.model.protein_encoder import real_lengths

    protein = torch.zeros(1, 30, dtype=torch.long)
    protein[0, :20] = torch.randint(2, 28, (20,))
    protein[0, 7] = 0
    drug = torch.randint(2, 70, (1, 20))

    report = validate_adapter(get_model("uniform_control"), drug, protein,
                              expected_length=int(real_lengths(protein)[0]))
    assert report["valid"], report["problems"]


def test_random_control_samples_from_the_span_not_the_count():
    """random_control must draw its positions from exactly the residues the
    explanation was scored over, or the control it provides is measured on a
    different set of positions than the observed value."""
    import torch

    from src.evaluation.faithfulness import random_control
    from src.model.coldsite_dti import ColdSiteDTI

    protein = torch.zeros(1, 20, dtype=torch.long)
    protein[0, :12] = torch.randint(2, 28, (12,))
    protein[0, 5] = 0                       # span 12, count 11
    drug = torch.randint(2, 70, (1, 10))

    torch.manual_seed(0)
    model = ColdSiteDTI(70, 28).eval()
    # k=12 is feasible against the span and infeasible against the count, so a
    # NaN here means the count is still being used.
    assert not np.isnan(random_control(model, drug, protein, k=12, n_trials=2))
