"""Tests for mapping model-internal attention positions back to real residues.

Every failure guarded here is silent. An adapter that returns the padded
length, the convolution length, or the token count instead of the residue count
does not crash -- it produces a precision@k indexed against the wrong residues,
which looks like a perfectly reasonable result.
"""
import importlib.util
import sys

import numpy as np
import pytest

from src.evaluation.attention_projection import (
    HYPERATTENTION_MAX_PROTEIN_LEN,
    HYPERATTENTION_PROTEIN_KERNELS,
    MOLTRANS_MAX_PROTEIN_TOKENS,
    conv_output_length,
    conv_stack_geometry,
    espf_token_spans,
    min_length_for_conv_stack,
    moltrans_covered_residues,
    project_conv_attention,
    project_token_attention,
    valid_conv_positions,
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# convolution geometry -- the [4, 8, 12] arithmetic
# --------------------------------------------------------------------------

def test_the_kernel_stack_matches_the_vendored_hyperparameters():
    hyper = _load("_hp", "baselines/HpyerAttentionDTI/hyperparameter.py").hyperparameter()
    assert tuple(hyper.protein_kernel) == HYPERATTENTION_PROTEIN_KERNELS


def test_1000_residues_give_979_convolution_positions():
    """The number in the guide, derived rather than trusted: 1000 - (3+7+11)."""
    assert conv_output_length(1000) == 979
    assert conv_stack_geometry()["offset"] == 21
    assert conv_stack_geometry()["width"] == 22


def test_geometry_matches_the_vendored_max_pool_width():
    """The vendored model sizes its pool as MAX_LEN - k0 - k1 - k2 + 3.
    If our arithmetic disagreed with theirs, every projection would be off."""
    k = HYPERATTENTION_PROTEIN_KERNELS
    vendored = HYPERATTENTION_MAX_PROTEIN_LEN - k[0] - k[1] - k[2] + 3
    assert vendored == conv_output_length(HYPERATTENTION_MAX_PROTEIN_LEN) == 979


def test_only_padding_free_positions_count_as_valid():
    """A 300-residue protein padded to 1000 has 979 positions, 279 meaningful."""
    assert valid_conv_positions(300) == 279
    assert valid_conv_positions(1000) == 979
    assert valid_conv_positions(21) == 0


def test_minimum_usable_protein_length():
    assert min_length_for_conv_stack() == 22


# --------------------------------------------------------------------------
# the contract: exactly one weight per real residue
# --------------------------------------------------------------------------

@pytest.mark.parametrize("real_length", [22, 100, 300, 999, 1000])
@pytest.mark.parametrize("mode", ["centre", "receptive_field"])
def test_projection_returns_one_weight_per_real_residue(real_length, mode):
    conv = np.random.default_rng(0).random(979)
    out = project_conv_attention(conv, real_length, mode=mode)
    assert out.shape == (real_length,)
    assert np.all(np.isfinite(out)) and np.all(out >= 0)


def test_a_protein_longer_than_the_window_is_reported_at_the_truncated_length():
    """The model saw 1000 residues, so the explanation covers 1000 -- not the
    full sequence. Padding the array back out would misalign every
    ground-truth index past the cut."""
    conv = np.random.default_rng(0).random(979)
    assert project_conv_attention(conv, 2500).shape == (1000,)


def test_padded_positions_are_dropped_not_projected():
    """Positions whose window reaches into padding are computed over zeros.
    Here every real position has weight 0 and every padded one weight 1; if
    padding leaked, the output would be non-zero."""
    real_length = 200
    conv = np.zeros(979)
    conv[valid_conv_positions(real_length):] = 1.0
    out = project_conv_attention(conv, real_length, mode="receptive_field")
    assert out.sum() == 0.0, "attention computed over padding reached a residue"


def test_a_protein_shorter_than_the_receptive_field_is_refused():
    """21 residues cannot produce a padding-free convolution position. Silently
    returning zeros would score it as a real, uninformative protein."""
    with pytest.raises(ValueError, match="shorter than"):
        project_conv_attention(np.random.rand(979), 21)


def test_too_few_convolution_weights_is_refused():
    with pytest.raises(ValueError, match="convolution weights"):
        project_conv_attention(np.random.rand(50), 500)


def test_centre_mode_puts_the_weight_on_the_central_residue():
    conv = np.zeros(979)
    conv[0] = 1.0
    out = project_conv_attention(conv, 500, mode="centre")
    assert out[10] == 1.0                      # offset // 2
    assert out.sum() == 1.0


def test_receptive_field_mode_spreads_over_the_whole_window():
    conv = np.zeros(979)
    conv[0] = 1.0
    out = project_conv_attention(conv, 500, mode="receptive_field")
    assert np.count_nonzero(out) == 22         # the full window
    assert out[0] > 0 and out[21] > 0 and out[22] == 0


def test_the_two_modes_disagree_which_is_why_it_is_a_recorded_decision():
    rng = np.random.default_rng(1)
    conv = rng.random(979)
    centre = project_conv_attention(conv, 400, mode="centre")
    spread = project_conv_attention(conv, 400, mode="receptive_field")
    top_centre = set(np.argsort(centre)[-10:])
    top_spread = set(np.argsort(spread)[-10:])
    assert top_centre != top_spread


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be"):
        project_conv_attention(np.random.rand(979), 100, mode="average")


# --------------------------------------------------------------------------
# the vendored model, end to end
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_vendored_model_produces_the_shapes_the_projection_assumes():
    """Pins the projection against the real HyperAttentionDTI, not a mock.

    If upstream ever changes a kernel or the attention reduction, this fails
    here rather than producing a quietly misaligned precision@k.
    """
    import torch

    from src.evaluation.attention_projection import hyperattentiondti_protein_attention

    hyper = _load("_hp2", "baselines/HpyerAttentionDTI/hyperparameter.py").hyperparameter()
    model_mod = _load("_hadti", "baselines/HpyerAttentionDTI/model.py")
    model = model_mod.AttentionDTI(hyper).eval()

    drug = torch.randint(1, 60, (2, 100))
    protein = torch.randint(1, 25, (2, HYPERATTENTION_MAX_PROTEIN_LEN))

    attention = hyperattentiondti_protein_attention(model, drug, protein)
    assert attention.shape == (2, 979), attention.shape

    real_length = 640
    weights = project_conv_attention(attention[0], real_length)
    assert weights.shape == (real_length,)
    assert np.all(weights >= 0)


@pytest.mark.slow
def test_extraction_does_not_disturb_the_prediction():
    """Extracting the explanation must not change what the model predicts."""
    import torch

    from src.evaluation.attention_projection import hyperattentiondti_protein_attention

    hyper = _load("_hp3", "baselines/HpyerAttentionDTI/hyperparameter.py").hyperparameter()
    model = _load("_hadti2", "baselines/HpyerAttentionDTI/model.py").AttentionDTI(hyper).eval()

    drug = torch.randint(1, 60, (1, 100))
    protein = torch.randint(1, 25, (1, HYPERATTENTION_MAX_PROTEIN_LEN))

    with torch.no_grad():
        before = model(drug, protein).clone()
    hyperattentiondti_protein_attention(model, drug, protein)
    with torch.no_grad():
        after = model(drug, protein)

    assert torch.allclose(before, after)
    assert not model.training, "model must be left in eval mode"


@pytest.mark.slow
def test_channel_reduction_is_over_channels_not_drug_positions():
    """The vendored forward already averages the drug axis away inside
    Atten_matrix, so what is left to reduce is 160 channels. The stub docstring
    in baseline_adapters.py says 'max over the drug axis', which this pins as
    not applicable to the vendored implementation."""
    import torch

    from src.evaluation.attention_projection import hyperattentiondti_protein_attention

    hyper = _load("_hp4", "baselines/HpyerAttentionDTI/hyperparameter.py").hyperparameter()
    model = _load("_hadti3", "baselines/HpyerAttentionDTI/model.py").AttentionDTI(hyper).eval()
    drug = torch.randint(1, 60, (1, 100))
    protein = torch.randint(1, 25, (1, 1000))

    by_mean = hyperattentiondti_protein_attention(model, drug, protein, "mean")
    by_max = hyperattentiondti_protein_attention(model, drug, protein, "max")
    assert by_mean.shape == by_max.shape == (1, 979)
    assert np.all(by_max >= by_mean - 1e-9)


# --------------------------------------------------------------------------
# MolTrans / ESPF token spans
# --------------------------------------------------------------------------

def test_token_spans_tile_the_sequence_exactly():
    tokens = ["MKV", "LLA", "GG", "S"]
    spans = espf_token_spans(tokens, "MKVLLAGGS")
    assert spans == [(0, 3), (3, 6), (6, 8), (8, 9)]


def test_tokens_that_do_not_rebuild_the_sequence_are_refused():
    """A separator or a case-fold in the tokeniser shifts every span after it."""
    with pytest.raises(ValueError, match="concatenate back"):
        espf_token_spans(["MK", "VL"], "MKXVL")


def test_545_tokens_is_not_545_residues():
    """The trap: MolTrans truncates to 545 TOKENS, and a token spans several
    residues, so the covered residue count is protein-specific."""
    tokens = ["ABCDE"] * 600
    assert moltrans_covered_residues(tokens) == 545 * 5
    assert moltrans_covered_residues(tokens) != MOLTRANS_MAX_PROTEIN_TOKENS


def test_token_projection_returns_one_weight_per_covered_residue():
    tokens = ["MKV", "LL", "AGGS", "P"]
    weights = np.array([0.4, 0.1, 0.9, 0.2] + [0.0] * (545 - 4))
    out = project_token_attention(weights, tokens)
    assert out.shape == (10,)
    assert list(out[:3]) == [0.4, 0.4, 0.4]
    assert list(out[3:5]) == [0.1, 0.1]
    assert list(out[5:9]) == [0.9] * 4


def test_token_projection_respects_the_545_token_cut():
    tokens = ["AB"] * 600
    out = project_token_attention(np.ones(545), tokens)
    assert out.shape == (1090,)                # 545 kept tokens x 2 residues


def test_dividing_by_span_would_rank_short_tokens_higher():
    """Why undivided is the default: with division, a 1-residue token beats a
    4-residue token carrying the same attention, purely for being short."""
    tokens = ["A", "BCDE"]
    weights = np.array([0.5, 0.5] + [0.0] * 543)
    undivided = project_token_attention(weights, tokens)
    divided = project_token_attention(weights, tokens, divide_by_span=True)
    assert undivided[0] == undivided[1]
    assert divided[0] > divided[1]


def test_too_few_token_weights_is_refused():
    with pytest.raises(ValueError, match="token weights"):
        project_token_attention(np.ones(2), ["AB", "CD", "EF"])


def test_empty_tokens_are_refused():
    with pytest.raises(ValueError, match="no tokens"):
        project_token_attention(np.ones(545), [])


# --------------------------------------------------------------------------
# the projections satisfy the registry contract
# --------------------------------------------------------------------------

def test_projected_weights_pass_the_adapter_contract_checks():
    """What validate_adapter checks on explain(): 1D, finite, non-negative,
    and exactly expected_length values."""
    from src.evaluation.model_registry import ExplainableDTIModel, validate_adapter

    real_length = 350
    conv = np.random.default_rng(2).random(979)

    class _Projected(ExplainableDTIModel):
        registry_name = "projection_probe"

        def predict(self, drug, protein):
            return 0.5

        def explain(self, drug, protein):
            return project_conv_attention(conv, real_length)

    report = validate_adapter(_Projected(), None, None, expected_length=real_length)
    assert report["valid"], report["problems"]
    assert report["n_weights"] == real_length
