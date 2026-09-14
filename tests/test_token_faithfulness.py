"""Faithfulness measured in token space, where both arms are the same intervention.

The residue-space version could not compare like with like for a sub-word model: masking
MolTrans's top-10 residues changes 48% of its tokens where 10 random residues change 95%.
Here the unit is the token, so removing k of them is k either way -- and these tests pin
that equality, since losing it is the whole failure being fixed.
"""
import numpy as np
import pytest
import torch

from src.evaluation.token_faithfulness import (pair_faithfulness, remove_tokens, report)


def test_removing_tokens_clears_exactly_those_entries():
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    out = remove_tokens(mask, [1, 3])
    assert out.tolist() == [[1, 0, 1, 0, 1, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 1, 1, 0, 0]]      # the input is not mutated


def test_keeping_tokens_clears_everything_else_but_leaves_padding_padding():
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    out = remove_tokens(mask, [0, 2], keep=True)
    assert out.tolist() == [[1, 0, 1, 0, 0, 0]]


def test_a_position_past_the_end_is_ignored_rather_than_wrapping():
    mask = torch.tensor([[1, 1, 1]])
    assert remove_tokens(mask, [99, 1]).tolist() == [[1, 0, 1]]


class _Adapter:
    """A model whose prediction is the sum of the visible tokens' weights, so the
    faithfulness of any explanation is known in advance."""

    def __init__(self, importance):
        self.importance = np.asarray(importance, dtype=float)
        self.head_reduce = "mean"
        self.attention_layer = -1
        self.device = "cpu"
        self.model = None

    def predict(self, drug, protein, drug_mask=None, protein_mask=None):
        visible = np.asarray(protein_mask).reshape(-1)[:len(self.importance)]
        return float((self.importance * visible).sum())


def _encoded(n):
    return {"drug": torch.zeros(1, 4), "drug_mask": torch.ones(1, 4),
            "protein": torch.ones(1, n), "protein_mask": torch.ones(1, n, dtype=torch.long),
            "tokens": ["A"] * n, "n_real": n}


def test_an_explanation_pointing_at_the_load_bearing_tokens_scores_positive():
    importance = np.zeros(40)
    importance[:5] = 10.0                       # only these five matter
    adapter = _Adapter(importance)
    attention = np.zeros(40)
    attention[:5] = 1.0                         # and the explanation points at them
    out = pair_faithfulness(adapter, _encoded(40), attention, k=5, n_random_trials=5)
    assert out["comprehensiveness"] == pytest.approx(50.0)
    assert out["comprehensiveness_delta"] > 0


def test_an_explanation_pointing_at_the_wrong_tokens_scores_negative():
    """The finding this measurement is allowed to make, unlike the residue-space one."""
    importance = np.zeros(40)
    importance[:5] = 10.0
    adapter = _Adapter(importance)
    attention = np.zeros(40)
    attention[35:] = 1.0                        # points at tokens that do nothing
    out = pair_faithfulness(adapter, _encoded(40), attention, k=5, n_random_trials=20)
    assert out["comprehensiveness"] == pytest.approx(0.0)
    assert out["comprehensiveness_delta"] < 0


def test_both_arms_remove_the_same_number_of_tokens():
    """The equality the residue-space test could not provide."""
    seen = []

    class Counting(_Adapter):
        def predict(self, drug, protein, drug_mask=None, protein_mask=None):
            seen.append(int((np.asarray(protein_mask).reshape(-1) == 0).sum()))
            return super().predict(drug, protein, drug_mask, protein_mask)

    adapter = Counting(np.ones(30))
    attention = np.arange(30, dtype=float)
    pair_faithfulness(adapter, _encoded(30), attention, k=6, n_random_trials=4)
    removed = [n for n in seen if n]            # the baseline pass removes nothing
    comprehensive = [n for n in removed if n == 6]
    assert len(comprehensive) == 5              # explanation + 4 random draws, all k=6


def test_k_cannot_exceed_the_tokens_the_model_saw():
    adapter = _Adapter(np.ones(4))
    out = pair_faithfulness(adapter, _encoded(4), np.arange(4.0), k=10, n_random_trials=2)
    assert out["k"] == 4


def test_the_report_marks_a_negative_delta_as_not_load_bearing():
    results = {"random": {"n": 5, "k": 10, "comprehensiveness": 0.1,
                          "comprehensiveness_random": 0.3,
                          "comprehensiveness_delta": -0.2, "sufficiency": 0.2,
                          "sufficiency_random": 0.2, "sufficiency_delta": 0.0,
                          "load_bearing": False, "median_tokens": 300}}
    text = report(results, "moltrans", "davis", 1)
    assert "**-0.2000**" in text and "| no |" in text
    assert "about the attention rather than about the intervention" in text
