"""Tests for residue-space masking and the baseline paths through both runners.

The failure modes here are silent: a mask that lands on the wrong residue, a
mask token that means "alanine" in one model's alphabet and "unknown" in
another's, a prediction that changes between two identical calls, or a
baseline run that overwrites ColdSite-DTI's output file. None of them raise.
"""
import json

import numpy as np
import pytest
import torch

from src.evaluation.faithfulness import batch_faithfulness
from src.evaluation.residue_space import (
    MASK_CODE,
    ResidueSpaceModel,
    decode_residues,
    encode_residues,
)
from src.evaluation.run_faithfulness import collect_accuracy, output_tag
from src.model.checkpoint_naming import results_path, run_tag

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"


class FakeHAT:
    """Stands in for HyperAttentionDTIAdapter: one tensor column per residue.

    Its prediction is the number of lysines it can see, so masking a K must
    lower it by exactly one and masking anything else must not move it.
    """

    registry_name = "hyperattentiondti"

    def __init__(self, seen: int | None = None):
        self.seen = seen          # truncate like HyperAttentionDTI does at 1000
        self.predicted_on = []

    @staticmethod
    def encode(smiles, sequence):
        return smiles, sequence

    def predict(self, drug, protein):
        visible = protein[: self.seen] if self.seen else protein
        self.predicted_on.append(protein)
        return float(visible.count("K"))

    def explain(self, drug, protein):
        visible = protein[: self.seen] if self.seen else protein
        return np.array([1.0 if c == "K" else 0.01 for c in visible])


class FakeMolTrans:
    """Stands in for MolTransAdapter's calling convention, masks included."""

    registry_name = "moltrans"
    collapse = False

    @staticmethod
    def encode(smiles, sequence):
        if FakeMolTrans.collapse:
            return (torch.zeros(50), torch.zeros(50),
                    torch.zeros(545), torch.tensor([1] + [0] * 544), ["?"])
        n = len(sequence)
        return (smiles, torch.ones(50), sequence, torch.ones(n), list(sequence))

    def predict(self, drug, protein, drug_mask=None, protein_mask=None):
        assert protein_mask is not None, "MolTrans must get its real mask"
        # dropout-like noise, as the vendored forward really does in eval()
        return float(protein.count("K")) + float(torch.rand(1))

    def explain(self, drug, protein, protein_tokens=None, drug_mask=None,
                protein_mask=None):
        assert protein_tokens is not None
        return np.array([1.0 if c == "K" else 0.01 for c in protein])


# --------------------------------------------------------------------------
# the residue alphabet
# --------------------------------------------------------------------------

def test_mask_code_is_the_one_faithfulness_writes():
    from src.evaluation.faithfulness import MASK_TOKEN
    assert MASK_CODE == MASK_TOKEN == 1
    assert decode_residues(torch.tensor([[MASK_CODE]])) == "X"


def test_residue_codes_round_trip_and_never_use_pad():
    codes = encode_residues(SEQUENCE)
    assert codes.shape == (1, len(SEQUENCE))
    assert int(codes.min()) > 0, "pad (0) would be skipped by mask_positions"
    assert decode_residues(codes) == SEQUENCE


def test_non_letter_residue_is_refused():
    with pytest.raises(ValueError):
        encode_residues("MK*T")


# --------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------

def test_masked_residue_reaches_the_model_as_X():
    wrapped = ResidueSpaceModel(FakeHAT(), "hyperattentiondti")
    drug, protein, attention = wrapped.add_pair("CCO", SEQUENCE)
    assert attention.size == protein.shape[1] == len(SEQUENCE)

    k_index = SEQUENCE.index("K")
    masked = protein.clone()
    masked[0, k_index] = MASK_CODE
    assert wrapped.predict(drug, masked) == SEQUENCE.count("K") - 1
    assert wrapped.adapter.predicted_on[-1][k_index] == "X"


def test_truncated_model_gets_its_unseen_tail_back_unchanged():
    """HyperAttentionDTI sees 1000 residues; MolTrans sees what 545 tokens cover.

    The masking tensor must stop where the model's view stops (or the random
    control samples residues the explanation never scored), while the model
    must still be given the whole original sequence.
    """
    adapter = FakeHAT(seen=20)
    wrapped = ResidueSpaceModel(adapter, "hyperattentiondti")
    drug, protein, attention = wrapped.add_pair("CCO", SEQUENCE)
    assert protein.shape[1] == attention.size == 20

    wrapped.predict(drug, protein)
    assert adapter.predicted_on[-1] == SEQUENCE


def test_drug_tensor_routes_each_pair_to_its_own_sequence():
    wrapped = ResidueSpaceModel(FakeHAT(), "hyperattentiondti")
    first = wrapped.add_pair("CCO", "KKKA")
    second = wrapped.add_pair("CCN", "AAAK")
    assert wrapped.predict(first[0], first[1]) == 3
    assert wrapped.predict(second[0], second[1]) == 1


def test_a_column_count_mismatch_raises_rather_than_realigning():
    wrapped = ResidueSpaceModel(FakeHAT(), "hyperattentiondti")
    drug, protein, _ = wrapped.add_pair("CCO", SEQUENCE)
    with pytest.raises(RuntimeError, match="lost or gained"):
        wrapped.predict(drug, protein[:, :-1])


def test_coldsite_is_not_routed_through_the_wrapper():
    with pytest.raises(ValueError, match="does not need one"):
        ResidueSpaceModel(FakeHAT(), "coldsite_dti")


def test_moltrans_predictions_are_repeatable_and_leave_global_rng_alone():
    """The vendored forward keeps dropout on in eval(); see residue_space.py."""
    FakeMolTrans.collapse = False
    wrapped = ResidueSpaceModel(FakeMolTrans(), "moltrans")
    drug, protein, _ = wrapped.add_pair("CCO", SEQUENCE)

    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    first = wrapped.predict(drug, protein)
    second = wrapped.predict(drug, protein)
    assert first == second
    assert torch.equal(torch.rand(1), expected_next), "global RNG was disturbed"


def test_moltrans_collapsed_encoding_is_refused():
    FakeMolTrans.collapse = True
    try:
        wrapped = ResidueSpaceModel(FakeMolTrans(), "moltrans")
        with pytest.raises(RuntimeError, match="collapsed"):
            wrapped.add_pair("CCO", SEQUENCE)
    finally:
        FakeMolTrans.collapse = False


# --------------------------------------------------------------------------
# through the unchanged faithfulness algorithm
# --------------------------------------------------------------------------

def _faithfulness(attention_fn):
    wrapped = ResidueSpaceModel(FakeHAT(), "hyperattentiondti")
    drugs, proteins, attentions = [], [], []
    for offset in range(4):
        sequence = SEQUENCE[offset:] + SEQUENCE[:offset]
        drug, protein, attention = wrapped.add_pair("CCO", sequence)
        drugs.append(drug)
        proteins.append(protein)
        attentions.append(attention_fn(sequence, attention))
    return batch_faithfulness(wrapped, drugs, proteins, attentions, k=5,
                              n_random_trials=10, seed=0)


def test_attention_on_what_the_model_uses_is_load_bearing():
    """A small positive control: attention on exactly the residues the model
    counts must beat random masking, through the re-tokenising wrapper."""
    summary = _faithfulness(lambda seq, att: att)
    assert summary["comprehensiveness"] == pytest.approx(5.0)
    assert summary["comprehensiveness_delta"] > 0
    assert summary["explanation_is_load_bearing"]


def test_attention_away_from_what_the_model_uses_is_not():
    summary = _faithfulness(
        lambda seq, att: np.array([0.0 if c == "K" else 1.0 for c in seq]))
    assert summary["comprehensiveness"] == pytest.approx(0.0)
    assert not summary["explanation_is_load_bearing"]


# --------------------------------------------------------------------------
# the real MolTrans tokeniser, when it is installed
# --------------------------------------------------------------------------

def test_real_espf_never_collapses_an_x_masked_sequence():
    pytest.importorskip("subword_nmt")
    from src.evaluation.baseline_adapters import MolTransAdapter

    try:
        MolTransAdapter.encode("CCO", "MK")
    except FileNotFoundError:
        pytest.skip("baselines/MolTrans not vendored here")

    patterns = [SEQUENCE,
                "X" * len(SEQUENCE),
                "".join("X" if i % 3 == 0 else c for i, c in enumerate(SEQUENCE))]
    for sequence in patterns:
        _d, _dm, protein, protein_mask, tokens = MolTransAdapter.encode("CCO", sequence)
        assert int(protein_mask.sum()) == len(tokens) > 0
        assert not (int(protein_mask.sum()) == 1 and int(protein[0]) == 0)


# --------------------------------------------------------------------------
# HyperAttentionDTI's prediction is the log-odds
# --------------------------------------------------------------------------

class _TwoLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, drug, protein):
        return self.logits


def _hat_with_logits(logits):
    from src.evaluation.baseline_adapters import HyperAttentionDTIAdapter

    adapter = object.__new__(HyperAttentionDTIAdapter)
    adapter.model, adapter.device = _TwoLogits(logits), "cpu"
    return adapter


def test_hat_predict_is_the_log_odds_not_the_positive_logit():
    drug, protein = torch.zeros(1, 100).long(), torch.zeros(1, 1000).long()
    assert _hat_with_logits([0.5, 2.0]).predict(drug, protein) == pytest.approx(1.5)


def test_hat_predict_ignores_the_constant_softmax_ignores():
    """Shifting both logits leaves the prediction unchanged, so faithfulness
    must not see a change. The positive logit alone would have moved by 10."""
    drug, protein = torch.zeros(1, 100).long(), torch.zeros(1, 1000).long()
    base = _hat_with_logits([0.5, 2.0]).predict(drug, protein)
    shifted = _hat_with_logits([10.5, 12.0]).predict(drug, protein)
    assert base == pytest.approx(shifted)


# --------------------------------------------------------------------------
# the runners' model plumbing
# --------------------------------------------------------------------------

def test_coldsite_keeps_its_tag_and_baselines_are_prefixed():
    assert output_tag("coldsite_dti", "davis", 1) == "davis_seed1"
    assert output_tag("moltrans", "davis", 1) == "moltrans_davis_seed1"
    assert output_tag("hyperattentiondti", "kiba", 3) == "hyperattentiondti_kiba_seed3"


def test_ladder_and_faithfulness_share_one_naming_rule():
    import src.evaluation.run_faithfulness as faithfulness_runner
    import src.evaluation.run_ladder as ladder_runner
    assert ladder_runner.output_tag is faithfulness_runner.output_tag


def test_accuracy_is_read_from_the_named_models_own_results(tmp_path):
    """Reading ColdSite-DTI's AUROC for MolTrans's ladder would draw one
    model's attention against another model's accuracy, silently."""
    for model, auroc in (("coldsite_dti", 0.9), ("moltrans", 0.6)):
        path = results_path(str(tmp_path), run_tag("davis", "random", "binary", 1),
                            model=model)
        with open(path, "w") as f:
            json.dump({"test_metrics": {"auroc": auroc}}, f)

    moltrans = collect_accuracy(str(tmp_path), "davis", "binary", 1,
                                levels=("random",), model="moltrans")
    default = collect_accuracy(str(tmp_path), "davis", "binary", 1,
                               levels=("random",))
    assert moltrans["accuracy"]["random"] == 0.6
    assert default["accuracy"]["random"] == 0.9
