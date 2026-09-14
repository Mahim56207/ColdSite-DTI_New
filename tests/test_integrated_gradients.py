"""Integrated gradients as a second explanation, scored by the same pipeline.

The whole point of this method is to tell "attention is a poor explanation" apart from
"the model never learned the site". That only works if the attribution really is the
model's, so the tests plant a model whose answer depends on residues we choose and check
IG finds those and not others -- the kind of error that otherwise arrives as a plausible
precision@k for the wrong reason. The per-model embedding paths are checked against the
real architectures, since a path pointing at the drug side would attribute the wrong
input and still return an array of the right shape.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.integrated_gradients import (PROTEIN_EMBEDDING, attributions,
                                                 embedding_module)
from src.model.checkpoint_naming import base_model_name, model_suffix


class OneResidueModel(nn.Module):
    """Its prediction is the embedding of ONE position, so the truth is known."""

    def __init__(self, position: int, vocab: int = 8, dim: int = 4):
        super().__init__()
        self.protein_encoder = nn.Module()
        self.protein_encoder.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        with torch.no_grad():                      # distinct, non-zero embeddings
            self.protein_encoder.embedding.weight.copy_(
                torch.arange(1.0, vocab * dim + 1).reshape(vocab, dim))
            self.protein_encoder.embedding.weight[0].zero_()
        self.position = position

    def forward(self, protein):
        embedded = self.protein_encoder.embedding(protein)
        return embedded[:, self.position, :].sum(dim=-1)


def _run(model, protein, **kwargs):
    return attributions(lambda: model(protein), model,
                        "protein_encoder.embedding", **kwargs)


def test_it_finds_the_one_residue_the_prediction_depends_on():
    model = OneResidueModel(position=3)
    protein = torch.tensor([[1, 2, 3, 4, 5, 6]])
    weights = _run(model, protein)
    assert int(np.argmax(weights)) == 3
    assert weights[3] > 0
    assert np.allclose(np.delete(weights, 3), 0.0, atol=1e-6)


def test_it_finds_the_right_residue_wherever_it_is():
    for position in (0, 2, 5):
        weights = _run(OneResidueModel(position=position),
                       torch.tensor([[1, 2, 3, 4, 5, 6]]))
        assert int(np.argmax(weights)) == position, position


def test_a_padding_position_gets_no_attribution():
    """The path starts at the padding embedding, so a padded position has nowhere to
    travel: its attribution is exactly zero rather than small."""
    model = OneResidueModel(position=1)
    weights = _run(model, torch.tensor([[1, 2, 0, 0]]))
    assert weights[2] == 0.0 and weights[3] == 0.0


def test_the_attribution_is_the_prediction_it_explains():
    """IG's completeness property: the attributions sum to f(x) - f(baseline). It is the
    one check that the path, the baseline and the gradients all agree."""
    model = OneResidueModel(position=2)
    protein = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        real = float(model(protein))
        pad = float(model(torch.zeros_like(protein)))
    total = float(_run(model, protein, signed=True).sum())
    assert total == pytest.approx(real - pad, rel=1e-3)


def test_more_steps_do_not_change_a_linear_models_answer():
    model = OneResidueModel(position=1)
    protein = torch.tensor([[1, 2, 3]])
    assert _run(model, protein, steps=4) == pytest.approx(_run(model, protein, steps=64),
                                                          rel=1e-4)


def test_magnitude_by_default_and_sign_on_request():
    class Negative(OneResidueModel):
        def forward(self, protein):
            return -super().forward(protein)

    model = Negative(position=1)
    protein = torch.tensor([[1, 2, 3]])
    assert _run(model, protein)[1] > 0                        # magnitude
    assert _run(model, protein, signed=True)[1] < 0           # direction kept


def test_the_model_is_left_as_it_was_found():
    """A leftover hook would silently corrupt every later forward pass; a leftover
    gradient would leak into whatever trains next."""
    model = OneResidueModel(position=1)
    protein = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = float(model(protein))
    model.train()
    _run(model, protein)
    assert model.training                                     # state restored
    assert model.protein_encoder.embedding.weight.grad is None
    with torch.no_grad():
        assert float(model(protein)) == before                # no hook left behind
    assert not model.protein_encoder.embedding._forward_hooks


def test_a_path_that_does_not_resolve_says_which_part_is_missing():
    model = OneResidueModel(position=0)
    with pytest.raises(AttributeError, match="no 'nowhere'"):
        embedding_module(model, "nowhere.embedding")
    with pytest.raises(TypeError, match="not an nn.Embedding"):
        embedding_module(model, "protein_encoder")


def test_a_forward_pass_that_never_touches_the_embedding_is_an_error():
    """Rather than returning zeros, which would read as 'this model attends nowhere'."""
    model = OneResidueModel(position=0)
    with pytest.raises(RuntimeError, match="never reached"):
        attributions(lambda: torch.tensor(1.0, requires_grad=True), model,
                     "protein_encoder.embedding")


# ---------------------------------------------------------------------------
# the real architectures
# ---------------------------------------------------------------------------

def test_coldsite_dtis_embedding_path_resolves():
    from src.model.coldsite_dti import ColdSiteDTI
    module = embedding_module(ColdSiteDTI(70, 28), PROTEIN_EMBEDDING["coldsite_dti"])
    assert module.num_embeddings == 28            # the PROTEIN vocabulary, not the drug's


def test_the_baselines_embedding_paths_resolve_and_are_the_protein_side():
    pytest.importorskip("subword_nmt")
    from src.model.train_hyperattentiondti import _import_vendored as hat
    AttentionDTI, hyperparameter, *_ = hat()
    module = embedding_module(AttentionDTI(hyperparameter()),
                              PROTEIN_EMBEDDING["hyperattentiondti"])
    assert module.num_embeddings == 26            # 25 residues + pad, not the drug's 65

    from src.model.train_moltrans import _import_vendored as mt
    config, Flat, *_ = mt()
    module = embedding_module(Flat(**config()), PROTEIN_EMBEDDING["moltrans"])
    assert module.num_embeddings == 16693         # protein subwords, not the drug's 23532


def test_every_ig_variant_reads_its_base_models_checkpoint():
    for variant in ("coldsite_dti_ig", "hyperattentiondti_ig", "moltrans_ig"):
        base = base_model_name(variant)
        assert base in PROTEIN_EMBEDDING
        assert model_suffix(variant) == model_suffix(base)


def test_the_variants_are_registered_as_models_of_their_own():
    from src.evaluation.model_registry import available_models
    for variant in ("coldsite_dti_ig", "hyperattentiondti_ig", "moltrans_ig"):
        assert variant in available_models()


def test_faithfulness_masks_a_variant_like_its_base_model():
    """Masking is a property of the trained model, not of how it is explained."""
    from src.evaluation.residue_space import ResidueSpaceModel
    wrapped = ResidueSpaceModel(adapter=object(), model_name="moltrans_ig")
    assert wrapped.model_name == "moltrans"
    with pytest.raises(ValueError, match="no residue-space tokenisation"):
        ResidueSpaceModel(adapter=object(), model_name="deepdta_ig")


def test_the_attribution_mode_switches_every_stochastic_component_off():
    """ColdSite-DTI's protein tower is a bi-LSTM and cuDNN refuses an eval-mode RNN
    backward pass, so the attribution runs in train mode with dropout switched off by
    hand -- including the dropout *attributes* of MultiheadAttention and RNNBase, which
    are not modules and so are not touched by .eval(). Dropout live during an
    attribution would be noise wearing the shape of an explanation.
    """
    from src.evaluation.integrated_gradients import _attribution_mode

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(0.5)
            self.attention = nn.MultiheadAttention(4, 2, dropout=0.3, batch_first=True)
            self.lstm = nn.LSTM(4, 4, num_layers=2, dropout=0.4, batch_first=True)

        def forward(self, x):
            return self.drop(x).sum()

    model = Model()
    x = torch.ones(1, 3, 4)
    with _attribution_mode(model, lambda: model(x)) as how:
        assert "stochastic layers off" in how
        assert not model.drop.training              # dropout module in eval
        assert model.attention.dropout == 0.0       # attribute zeroed
        assert model.lstm.dropout == 0.0
        assert model.training                       # but the model is in train mode
    assert model.attention.dropout == 0.3           # restored afterwards
    assert model.lstm.dropout == 0.4


def test_a_model_that_stays_stochastic_falls_back_instead_of_attributing_noise():
    """If some stochastic component was missed, two forward passes disagree -- and the
    attribution must not proceed as though they had not."""
    from src.evaluation.integrated_gradients import _attribution_mode

    class Stubborn(nn.Module):
        def forward(self, x):
            return x + torch.rand(())               # random whatever the mode

    model = Stubborn()
    with _attribution_mode(model, lambda: model(torch.zeros(1))) as how:
        assert "cuDNN off" in how


def test_an_rnn_model_can_actually_be_attributed():
    """The case that failed: a recurrent protein tower, differentiated end to end."""
    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.protein_encoder = nn.Module()
            self.protein_encoder.embedding = nn.Embedding(8, 4, padding_idx=0)
            self.protein_encoder.bilstm = nn.LSTM(4, 3, batch_first=True, bidirectional=True)
            self.head = nn.Linear(6, 1)

        def forward(self, protein):
            embedded = self.protein_encoder.embedding(protein)
            out, _state = self.protein_encoder.bilstm(embedded)
            return self.head(out.mean(dim=1)).reshape(-1)

    model = Recurrent()
    weights = _run(model, torch.tensor([[1, 2, 3, 4, 5]]))
    assert weights.shape == (5,)
    assert np.all(np.isfinite(weights)) and weights.sum() > 0
