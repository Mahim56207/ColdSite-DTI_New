"""T02 (c): the two masking arms must be the same size of intervention, in each model's own
token space.

Faithfulness is comprehensiveness minus a random-masking control. That subtraction is only
meaningful if masking k attended residues and masking k random ones change the same amount of
input. For a model that reads one token per residue they do by construction; for MolTrans's
ESPF sub-words they did not (attended 43% of tokens vs random 93% on DAVIS, 2026-09-14), and
its faithfulness delta came out negative in 11 of 12 cells for that reason alone.

What is pinned here, per model class:
  * residue-level models (ColdSite-DTI, HyperAttentionDTI, DrugBAN): k residues masked is
    exactly k tokens changed, wherever the residues sit -- so any k-sized random arm matches;
  * the sub-word model (MolTrans): the control is drawn to match the explanation's token
    change, lands inside TOLERANCE when the target is reachable, is never farther from the
    target than a blind draw, and REPORTS itself as unmatched when the target is not;
  * a model cannot join the audit without a masking decision.

The last point is worth knowing about: a target the arms cannot meet (attended residues at
the very end of a sequence change 1-3% of MolTrans's tokens, and no ten random residues do
that) is a property of the tokeniser. The tests assert that it is *visible*, not that it does
not happen.
"""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.faithfulness import mask_positions
from src.evaluation.mask_comparability import (MAX_TRIES, TOLERANCE, protein_token_encoder,
                                               token_change_fraction, token_matched_control)
from src.evaluation.residue_space import (RESIDUE_LEVEL_MODELS, SUBWORD_MODELS,
                                          SUPPORTED_MODELS, ResidueSpaceModel,
                                          encode_residues)

TEST_CSV = "data/splits/davis/random/test.csv"
K = 10


def _proteins(n=4):
    if not os.path.exists(TEST_CSV):
        pytest.skip("DAVIS splits not present")
    rows = pd.read_csv(TEST_CSV).drop_duplicates("Target_ID").head(n)
    return [str(t)[:1000] for t in rows["Target"]]


def _arms(length, rng):
    """One explanation-shaped arm (a contiguous block) and one scattered arm."""
    start = int(rng.integers(0, length - K))
    return np.arange(start, start + K), rng.choice(length, size=K, replace=False)


# ---------------------------------------------------- residue-level: k residues == k tokens

def test_a_residue_level_mask_changes_exactly_k_tokens_in_either_arm():
    rng = np.random.default_rng(0)
    for sequence in _proteins():
        tensor = encode_residues(sequence)
        for arm in _arms(len(sequence), rng):
            masked = mask_positions(tensor, arm)
            assert masked.shape == tensor.shape
            assert int((masked != tensor).sum()) == K


def test_coldsite_dti_masks_to_unk_one_token_per_residue():
    """ColdSite-DTI needs no residue-space wrapper: its tensor is one token per residue and
    the mask writes the UNK token (1) directly."""
    from src.model.protein_encoder import build_protein_vocab, encode_protein
    vocab, rng = build_protein_vocab(), np.random.default_rng(0)
    for sequence in _proteins():
        tensor = torch.tensor([encode_protein(sequence, vocab)])
        for arm in _arms(min(len(sequence), tensor.shape[1]), rng):
            masked = mask_positions(tensor, arm)
            assert int((masked != tensor).sum()) == K
            assert set(masked[0, arm].tolist()) == {1}


def test_hyperattentiondti_x_replacement_is_one_token_per_residue():
    """Through the model's own tokeniser, as the audit masks it (residue -> 'X')."""
    try:
        from src.evaluation.baseline_adapters import HyperAttentionDTIAdapter as A
        A.encode("CCO", "ACDE")
    except (SystemExit, ImportError, FileNotFoundError) as exc:
        pytest.skip(f"vendored HyperAttentionDTI unavailable: {exc}")
    rng = np.random.default_rng(0)
    for sequence in _proteins():
        _d, before = A.encode("CCO", sequence)
        for arm in _arms(len(sequence), rng):
            masked = list(sequence)
            for i in arm:
                masked[int(i)] = "X"
            _d, after = A.encode("CCO", "".join(masked))
            assert before.shape == after.shape
            assert int((before != after).sum()) <= K     # a residue already X is unchanged
            assert int((before != after).sum()) >= K - sequence.count("X")


def test_drugban_x_replacement_is_one_token_per_residue():
    try:
        import dgl  # noqa: F401
        from src.evaluation.drugban_adapter import DrugBANAdapter as A
    except Exception:
        pytest.skip("needs DGL (the drugban env)")
    rng = np.random.default_rng(0)
    for sequence in _proteins():
        _g, before = A.encode("CCO", sequence)
        for arm in _arms(len(sequence), rng):
            masked = list(sequence)
            for i in arm:
                masked[int(i)] = "X"
            _g, after = A.encode("CCO", "".join(masked))
            assert before.shape == after.shape
            assert int((before != after).sum()) == K


# ------------------------------------------------------------- the sub-word model (MolTrans)

@pytest.fixture(scope="module")
def moltrans_encode():
    try:
        from src.evaluation.baseline_adapters import MolTransAdapter
        encode = protein_token_encoder(MolTransAdapter)
        encode("C", "ACDEFGHIKL")
    except (SystemExit, ImportError, FileNotFoundError) as exc:
        pytest.skip(f"vendored MolTrans unavailable: {exc}")
    return encode


def test_moltrans_arms_really_are_unequal_without_the_control(moltrans_encode):
    """The reason the control exists, measured on real proteins: a contiguous block changes a
    fraction of the tokens a scattered draw does."""
    rng = np.random.default_rng(0)
    for sequence in _proteins(3):
        block = token_change_fraction(moltrans_encode, sequence,
                                      np.arange(300, 300 + K) % len(sequence))
        scattered = np.mean([token_change_fraction(
            moltrans_encode, sequence, rng.choice(len(sequence), K, replace=False))
            for _ in range(4)])
        assert scattered > block + 0.1, (block, scattered)


def test_the_moltrans_control_lands_within_tolerance_when_the_target_is_reachable(
        moltrans_encode):
    """A target the scattered arm can hit (here: another random draw's size) is met."""
    rng = np.random.default_rng(0)
    for sequence in _proteins():
        target = token_change_fraction(moltrans_encode, sequence,
                                       rng.choice(len(sequence), K, replace=False))
        _pos, fraction, tries = token_matched_control(moltrans_encode, sequence, K,
                                                      target, rng)
        assert abs(fraction - target) <= TOLERANCE, (target, fraction, tries)


def test_the_moltrans_control_is_never_worse_than_a_blind_draw(moltrans_encode):
    rng = np.random.default_rng(1)
    for sequence in _proteins():
        attended = np.arange(300, 300 + K) % len(sequence)
        target = token_change_fraction(moltrans_encode, sequence, attended)
        _pos, fraction, _tries = token_matched_control(moltrans_encode, sequence, K,
                                                       target, rng)
        blind = token_change_fraction(moltrans_encode, sequence,
                                      rng.choice(len(sequence), K, replace=False))
        assert abs(fraction - target) <= abs(blind - target) + 1e-9


def test_an_unreachable_target_is_reported_not_hidden(moltrans_encode):
    """Attended residues at the very end of a sequence change ~1-3% of MolTrans's tokens; no
    random ten do. The control must say so: `tries` exhausts MAX_TRIES and the returned
    fraction is outside TOLERANCE. Silent 'success' here is the failure this file guards."""
    rng = np.random.default_rng(0)
    sequence = _proteins(1)[0]
    at_the_end = np.arange(len(sequence) - K, len(sequence))
    target = token_change_fraction(moltrans_encode, sequence, at_the_end)
    assert target < 0.10, "expected an end-of-sequence block to change very few tokens"
    _pos, fraction, tries = token_matched_control(moltrans_encode, sequence, K, target, rng)
    assert tries == MAX_TRIES
    assert abs(fraction - target) > TOLERANCE


# --------------------------------------------------------------- the wiring and the roster

class _StubAdapter:
    """Enough of an adapter for ResidueSpaceModel's constructor and control_positions."""
    @staticmethod
    def encode_protein(sequence):
        tokens = np.arange(len(sequence))
        return tokens, np.ones_like(tokens)


@pytest.mark.parametrize("name", RESIDUE_LEVEL_MODELS)
def test_residue_level_models_ask_for_no_matched_control(name):
    model = ResidueSpaceModel(_StubAdapter(), name)
    assert model.control_positions(None, None, K, np.random.default_rng(0), []) is None


def test_the_faithfulness_control_uses_a_models_matched_positions():
    """random_control must defer to control_positions when the model offers one, or the
    MolTrans fix is unreachable from the audit."""
    from src.evaluation import faithfulness

    class Model:
        seen = []
        def control_positions(self, drug, protein, k, rng, attended):
            Model.seen.append(list(attended))
            return list(range(k))
        def predict(self, drug, protein):
            return float((protein == 1).sum())

    protein = torch.arange(2, 62).reshape(1, -1)
    faithfulness.random_control(Model(), torch.zeros(1, 1, dtype=torch.long), protein, k=5,
                                n_trials=3, rng=np.random.default_rng(0), attended=[9, 8, 7])
    assert Model.seen == [[9, 8, 7]] * 3


def test_every_audited_model_has_a_masking_decision():
    """A model added to the audit without saying how its arms are matched would be scored
    with a uniform control by default -- exactly how MolTrans went wrong."""
    from src.evaluation.run_all import AUDITED
    decided = {"coldsite_dti"} | set(SUPPORTED_MODELS)
    assert set(AUDITED) <= decided, sorted(set(AUDITED) - decided)
    assert set(SUBWORD_MODELS) | set(RESIDUE_LEVEL_MODELS) == set(SUPPORTED_MODELS)
