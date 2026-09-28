"""New explanation methods (T08): each is checked on a planted case where the truth is known,
and against the real vendored architectures where a hook has to find the right tensor.

A method that returns the right shape but reads the wrong tensor produces a plausible
precision@k for the wrong reason, so the tests plant a model whose answer depends on
residues we choose (occlusion), check the captured attention is the tensor the model really
used and its gradient agrees with a finite difference (attention x gradient), and check
rollout against hand-computed layers.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation import explanation_methods as em
from src.evaluation.explanation_agreement import agreement, chance_iou, top_k, topk_iou
from src.model.checkpoint_naming import VARIANT_BASE, base_model_name, model_suffix

NEW = ["coldsite_dti_occlusion", "hyperattentiondti_occlusion", "moltrans_occlusion",
       "drugban_occlusion", "hyperattentiondti_attngrad", "moltrans_attngrad",
       "moltrans_rollout"]


# ---- registration ----------------------------------------------------------

def test_every_new_method_reads_its_base_models_checkpoint():
    from src.evaluation.model_registry import available_models
    for name in NEW:
        assert name in available_models(), name
        base = base_model_name(name)
        assert base != name and VARIANT_BASE[name] == base
        assert model_suffix(name) == model_suffix(base), (
            f"{name} would load a different file from {base}")


def test_the_applicability_table_names_only_registered_methods():
    from src.evaluation.model_registry import available_models
    registered = {v for v in em.APPLICABILITY.values() if v}
    assert registered == set(NEW)
    assert registered <= set(available_models())
    for key, name in em.APPLICABILITY.items():
        assert (name is None) == (key in em.NOT_APPLICABLE_REASON), key


def test_the_written_applicability_document_matches_the_table():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "method_applicability.md")
    text = open(path).read()
    for key, name in em.APPLICABILITY.items():
        if name:
            assert f"`{name}`" in text, f"{name} is applicable in code but not in the document"
        else:
            assert em.NOT_APPLICABLE_REASON[key]
    for base in {b for _m, b in em.APPLICABILITY}:
        assert base in text


# ---- occlusion, model independent -----------------------------------------

def test_windows_cover_every_residue_and_the_last_is_flush():
    for n in (3, 5, 6, 17, 100, 1000):
        windows = em.occlusion_windows(n)
        covered = np.zeros(n, dtype=int)
        for s, e in windows:
            assert 0 <= s < e <= n
            covered[s:e] += 1
        assert covered.min() >= 1, n
        assert windows[-1][1] == n
        assert all(e - s == min(em.WINDOW, n) for s, e in windows)


def test_occlusion_finds_the_one_residue_the_score_depends_on():
    planted, n = 37, 100

    def score(windows):
        return np.array([0.0 if s <= planted < e else 1.0 for s, e in windows])

    weights = em.occlusion_map(n, 1.0, score)
    assert weights.shape == (n,) and (weights >= 0).all()
    assert weights[planted] > 0
    assert np.flatnonzero(weights).min() >= planted - (em.WINDOW - 1)
    assert np.flatnonzero(weights).max() <= planted + (em.WINDOW - 1)
    assert np.allclose(np.delete(weights, np.arange(planted - 4, planted + 5)), 0.0)


def test_occlusion_of_two_planted_sites_is_supported_on_exactly_their_neighbourhoods():
    planted, n = (20, 71), 100

    def score(windows):
        return np.array([1.0 - 0.5 * sum(s <= p < e for p in planted) for s, e in windows])

    weights = em.occlusion_map(n, 1.0, score)
    near = {i for p in planted for i in range(p - 4, p + 5)}
    assert set(np.flatnonzero(weights).tolist()) <= near
    assert all(weights[p] > 0 for p in planted)


def test_signed_occlusion_keeps_the_direction_of_the_change():
    def score(windows):
        return np.array([2.0 if s <= 5 < e else 1.0 for s, e in windows])   # masking HELPS

    assert em.occlusion_map(20, 1.0, score, signed=True)[5] < 0
    assert em.occlusion_map(20, 1.0, score)[5] > 0


def test_a_window_the_model_could_not_encode_is_left_out_not_counted_as_zero():
    windows = [(0, 5), (2, 7)]
    weights = em.occlusion_from_scores(7, windows, 1.0, [0.0, float("nan")])
    assert weights[0] == 1.0 and weights[3] == 1.0       # only the valid window counts
    assert weights[6] == 0.0                              # covered by the invalid one only


# ---- occlusion, the real adapter code paths, planted models ----------------

class _PlantedColdSite(nn.Module):
    """Score = 1 if the token at `position` is untouched, 0 once it is UNK."""

    def __init__(self, position):
        super().__init__()
        self.position = position

    def forward(self, drug, protein):
        return (protein[:, self.position] != em.MASK_TOKEN).float().unsqueeze(1), None


def _bare(cls, model, **fields):
    adapter = object.__new__(cls)
    adapter.model, adapter.device = model, "cpu"
    adapter.window, adapter.stride, adapter.signed = em.WINDOW, em.STRIDE, False
    for key, value in fields.items():
        setattr(adapter, key, value)
    return adapter


def test_coldsite_occlusion_finds_the_planted_residue_through_its_real_masking_path():
    protein = torch.tensor([[5 + i % 20 for i in range(60)] + [0] * 20])
    adapter = _bare(em.ColdSiteDTIOcclusion, _PlantedColdSite(31))
    weights = adapter.explain(torch.tensor([[3, 4, 5]]), protein)
    assert weights.shape == (60,)                       # padding is not explained
    assert weights[31] > 0 and int(np.argmax(weights)) in range(27, 36)
    assert np.allclose(np.delete(weights, np.arange(27, 36)), 0.0)


class _PlantedGate(nn.Module):
    """log-odds = the code at `position`, so masking to X changes it and nothing else does."""

    def __init__(self, position):
        super().__init__()
        self.position = position

    def forward(self, drug, protein):
        v = protein[:, self.position].float()
        return torch.stack([torch.zeros_like(v), v], dim=1)


def _hat_encode_available():
    try:
        from src.evaluation.baseline_adapters import HyperAttentionDTIAdapter
        return HyperAttentionDTIAdapter.encode("CCO", "MKV")
    except Exception as exc:                            # vendored repo missing
        pytest.skip(f"HyperAttentionDTI not available: {exc}")


def test_hat_occlusion_masks_to_its_own_unknown_residue_and_finds_the_planted_one():
    _hat_encode_available()
    from src.evaluation.baseline_adapters import HyperAttentionDTIAdapter as A
    drug, protein = A.encode("CCO", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY")
    x_code = int(A.encode("C", "X")[1][0])
    assert x_code != 1, "X must be its own code, not alanine's (1) as a UNK would be"
    adapter = _bare(em.HyperAttentionDTIOcclusion, _PlantedGate(12), base=None)
    adapter.base = type("B", (), {"model": adapter.model})()
    weights = adapter.explain(drug, protein)
    n = int((protein != 0).sum())
    assert weights.shape == (n,)
    assert weights[12] > 0 and np.allclose(np.delete(weights, np.arange(8, 17)), 0.0)


# ---- attention x gradient, on the real architectures -----------------------

def _hat_model():
    _hat_encode_available()
    from src.evaluation.baseline_adapters import HyperAttentionDTIAdapter
    torch.manual_seed(0)
    return HyperAttentionDTIAdapter(checkpoint_path=None, device="cpu")


def test_hat_attngrad_captures_the_gate_the_model_really_used():
    from src.evaluation.attention_projection import hyperattentiondti_protein_attention
    adapter = _hat_model()
    drug, protein = type(adapter).encode("CC(=O)Oc1ccccc1C(=O)O", "MKVLAAGIVGLLLAQ" * 4)
    drug_b, protein_b = drug.unsqueeze(0), protein.unsqueeze(0)
    gate, grad, _ = em.hyperattentiondti_gate_and_grad(adapter.model, drug_b, protein_b)
    assert gate.shape == grad.shape == (1, 160, 979)
    plain = hyperattentiondti_protein_attention(adapter.model, drug_b, protein_b)
    assert np.allclose(gate.mean(dim=1).numpy(), plain, atol=1e-6), \
        "the captured tensor is not the gate the plain readout reads"


def test_hat_attngrad_gradient_matches_a_finite_difference():
    adapter = _hat_model()
    drug, protein = type(adapter).encode("CC(=O)Oc1ccccc1C(=O)O", "MKVLAAGIVGLLLAQ" * 4)
    drug_b, protein_b = drug.unsqueeze(0), protein.unsqueeze(0)
    gate, grad, base = em.hyperattentiondti_gate_and_grad(adapter.model, drug_b, protein_b)
    flat = grad[0].abs()
    c, j = np.unravel_index(int(flat.argmax()), flat.shape)
    eps = 1e-2

    def nudge(out):
        out = out.clone()
        out[0, c, j] = out[0, c, j] + eps
        return out

    _g, _gr, nudged = em.hyperattentiondti_gate_and_grad(adapter.model, drug_b, protein_b,
                                                         perturb=nudge)
    assert (nudged - base) / eps == pytest.approx(float(grad[0, c, j]), rel=0.05, abs=1e-5)


def test_hat_attngrad_meets_the_explanation_contract():
    adapter = _hat_model()
    sequence = "MKVLAAGIVGLLLAQ" * 4
    drug, protein = type(adapter).encode("CC(=O)Oc1ccccc1C(=O)O", sequence)
    explainer = em.HyperAttentionDTIAttnGrad.__new__(em.HyperAttentionDTIAttnGrad)
    explainer.base, explainer.model, explainer.device = adapter, adapter.model, "cpu"
    weights = explainer.explain(drug, protein)
    assert weights.shape == (len(sequence),)
    assert np.isfinite(weights).all() and (weights >= 0).all() and weights.max() > 0


SEQ = ("MSGPRAGFYRQELNKTVWEVPQRLQGLRPVGSGAYGSVCSAYDARLRQKVAVKKLSRPFQSLIHARRTYRELRLLKHLKHE"
       "NVIGLLDVFTPATSIEDFSEVYLVTTLMGADLNNIVKCQALSDEHVQFLVYQLLRGLKYIHSAGIIHRDLKPSNVAVNEDC")


def _moltrans():
    pytest.importorskip("subword_nmt")
    from src.evaluation.baseline_adapters import MolTransAdapter
    try:
        MolTransAdapter.encode("C", "MKV")
    except Exception as exc:
        pytest.skip(f"MolTrans not available: {exc}")
    torch.manual_seed(0)
    adapter = MolTransAdapter(checkpoint_path=None, device="cpu")
    d, dm, p, pm, tokens = MolTransAdapter.encode("CC(=O)Oc1ccccc1C(=O)O", SEQ)
    return adapter, d, dm, p, pm, tokens


def test_moltrans_attngrad_captures_the_probabilities_the_plain_readout_reads():
    from src.evaluation.attention_projection import capture_moltrans_attention
    adapter, d, dm, p, pm, _tokens = _moltrans()
    probs, grad, _score = em.moltrans_probs_and_grad(adapter, d, p, dm, pm)
    target = adapter.model.p_encoder.layer[-1].attention.self
    store, handle = capture_moltrans_attention(target)
    try:
        adapter.model.eval()
        adapter._fit_batch_size(1)
        with torch.no_grad():
            adapter.model(d.unsqueeze(0), p.unsqueeze(0), dm.unsqueeze(0), pm.unsqueeze(0))
    finally:
        handle.remove()
    assert probs.shape == grad.shape == store["probs"].shape
    assert torch.allclose(probs, store["probs"], atol=1e-6)


def test_moltrans_attngrad_gradient_matches_a_finite_difference():
    adapter, d, dm, p, pm, _tokens = _moltrans()
    probs, grad, base = em.moltrans_probs_and_grad(adapter, d, p, dm, pm)
    flat = grad[0].abs()
    h, i, j = np.unravel_index(int(flat.argmax()), flat.shape)
    eps = 1e-2

    def nudge(x):
        x = x.clone()
        x[0, h, i, j] = x[0, h, i, j] + eps
        return x

    _pr, _gr, nudged = em.moltrans_probs_and_grad(adapter, d, p, dm, pm, perturb=nudge)
    assert (nudged - base) / eps == pytest.approx(float(grad[0, h, i, j]), rel=0.1, abs=1e-5)


def test_moltrans_methods_meet_the_explanation_contract_and_are_repeatable():
    adapter, d, dm, p, pm, tokens = _moltrans()
    for cls in (em.MolTransAttnGrad, em.MolTransRollout, em.MolTransOcclusion):
        explainer = cls.__new__(cls)
        explainer.base, explainer.model, explainer.device = adapter, adapter.model, "cpu"
        explainer.window, explainer.stride, explainer.signed = em.WINDOW, em.STRIDE, False
        first = explainer.explain(d, p, protein_tokens=tokens, drug_mask=dm, protein_mask=pm)
        again = explainer.explain(d, p, protein_tokens=tokens, drug_mask=dm, protein_mask=pm)
        assert first.shape == (len(SEQ),), cls.__name__
        assert np.isfinite(first).all() and (first >= 0).all(), cls.__name__
        assert first.max() > 0, cls.__name__
        assert np.array_equal(first, again), \
            f"{cls.__name__} changed between calls: MolTrans's live dropout got through"


# ---- rollout, by hand --------------------------------------------------------

def test_rollout_of_one_layer_is_that_layer_with_its_residual():
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = 0.5 * a + 0.5 * np.eye(2)
    assert np.allclose(em.rollout([a]), expected)


def test_rollout_multiplies_the_layers_later_on_the_left():
    """Row = query, column = key. Layer 1: token 1 reads token 0. Layer 2: token 2 reads
    token 1. After both, token 2 must have received weight from token 0 (0 -> 1 -> 2);
    multiplying the layers the other way round would lose that path."""
    a1 = np.eye(3); a1[1, :] = [1, 0, 0]
    a2 = np.eye(3); a2[2, :] = [0, 1, 0]
    r = em.rollout([a1, a2])
    n1 = 0.5 * a1 + 0.5 * np.eye(3); n1 /= n1.sum(1, keepdims=True)
    n2 = 0.5 * a2 + 0.5 * np.eye(3); n2 /= n2.sum(1, keepdims=True)
    assert np.allclose(r, n2 @ n1)
    assert r[2, 0] > 0, "the path 0 -> 1 -> 2 was lost (layers multiplied the wrong way)"
    assert (n1 @ n2)[2, 0] == 0.0
    assert np.allclose(r.sum(axis=1), 1.0)


def test_rollout_needs_a_layer():
    with pytest.raises(ValueError):
        em.rollout([])


# ---- agreement ---------------------------------------------------------------

def test_identical_explanations_agree_completely_and_disjoint_ones_not_at_all():
    w = np.arange(50, dtype=float)
    assert topk_iou(w, w) == 1.0
    assert topk_iou(w, w[::-1]) == 0.0


def test_top_k_breaks_ties_the_same_way_every_time():
    assert top_k(np.zeros(30), 5) == {0, 1, 2, 3, 4}


def test_chance_iou_matches_a_simulation():
    rng = np.random.default_rng(0)
    n, k = 60, 10
    assert 0 < chance_iou(n, k) < 1
    draws = []
    for _ in range(20000):
        a, b = set(rng.choice(n, k, replace=False)), set(rng.choice(n, k, replace=False))
        draws.append(len(a & b) / len(a | b))
    assert chance_iou(n, k) == pytest.approx(float(np.mean(draws)), abs=0.004)


def test_agreement_reports_chance_beside_the_overlap_and_refuses_unlike_protein_lists():
    rng = np.random.default_rng(1)
    same = [rng.random(80) for _ in range(6)]
    out = agreement({"a": same, "b": same, "c": [rng.random(80) for _ in range(6)]})
    assert out["a|b"]["iou"] == 1.0
    assert out["a|c"]["iou"] < 0.3 and out["a|c"]["chance"] == pytest.approx(chance_iou(80))
    with pytest.raises(ValueError):
        agreement({"a": same, "b": same[:3]})
