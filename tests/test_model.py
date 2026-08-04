"""Track B contract tests.

These pin the guarantees Track C's metric silently relies on. If any of them
break, precision@k keeps returning a number -- just the wrong one.
"""
import numpy as np
import pytest
import torch

from src.model.coldsite_dti import ColdSiteDTI
from src.model.dataset import collate_batch, random_dataset, make_loader
from src.model.drug_encoder import DrugEncoder, build_smiles_vocab, encode_smiles
from src.model.protein_encoder import ProteinEncoder, build_protein_vocab, encode_protein, real_lengths

DRUG_VOCAB, PROTEIN_VOCAB = 70, 28


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ColdSiteDTI(DRUG_VOCAB, PROTEIN_VOCAB).eval()


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------

def test_drug_encoder_output_shape():
    out = DrugEncoder(DRUG_VOCAB)(torch.randint(2, DRUG_VOCAB, (4, 50)))
    assert out.shape == (4, 128)


def test_protein_encoder_exposes_attention():
    """docs/02_GUIDE_124AD0015.md: 'don't let the model swallow them internally.'"""
    encoder = ProteinEncoder(PROTEIN_VOCAB)
    out, attn = encoder(torch.randint(2, PROTEIN_VOCAB, (4, 300)))
    assert out.shape == (4, 300, 128)
    assert attn is not None and attn.shape == (4, 300, 300)


def test_full_model_returns_one_weight_per_residue(model):
    pred, attn = model(torch.randint(2, DRUG_VOCAB, (4, 50)),
                       torch.randint(2, PROTEIN_VOCAB, (4, 300)))
    assert pred.shape == (4, 1)
    assert attn.shape == (4, 1, 300), "precision@k needs exactly one weight per residue"


def test_model_handles_a_batch_of_one(model):
    pred, attn = model(torch.randint(2, DRUG_VOCAB, (1, 20)),
                       torch.randint(2, PROTEIN_VOCAB, (1, 60)))
    assert pred.shape == (1, 1) and attn.shape == (1, 1, 60)


def test_model_handles_a_very_short_protein(model):
    _pred, attn = model(torch.randint(2, DRUG_VOCAB, (2, 10)),
                        torch.randint(2, PROTEIN_VOCAB, (2, 12)))
    assert attn.shape == (2, 1, 12)


# --------------------------------------------------------------------------
# padding guarantees -- these are what protect the headline number
# --------------------------------------------------------------------------

def test_no_explanation_weight_lands_on_padding(model):
    protein = torch.zeros(4, 300, dtype=torch.long)
    protein[:, :120] = torch.randint(2, PROTEIN_VOCAB, (4, 120))
    with torch.no_grad():
        _pred, attn = model(torch.randint(2, DRUG_VOCAB, (4, 50)), protein)
    assert attn[:, :, 120:].abs().max().item() == 0.0


def test_attention_sums_to_one_over_real_residues_in_eval_mode(model):
    with torch.no_grad():
        _pred, attn = model(torch.randint(2, DRUG_VOCAB, (4, 50)),
                            torch.randint(2, PROTEIN_VOCAB, (4, 300)))
    assert torch.allclose(attn.sum(dim=-1).squeeze(-1), torch.ones(4), atol=1e-5)


def test_protein_encoding_is_invariant_to_batch_padding(model):
    """A protein's explanation must not depend on how long its batch-mates are.

    Without this, the same protein scores differently depending on which batch
    it landed in, and precision@k becomes a function of DataLoader shuffling.
    """
    torch.manual_seed(1)
    seq = torch.randint(2, PROTEIN_VOCAB, (1, 80))
    padded = torch.cat([seq, torch.zeros(1, 220, dtype=torch.long)], dim=1)
    drug = torch.randint(2, DRUG_VOCAB, (1, 30))
    with torch.no_grad():
        _p1, bare = model(drug, seq)
        _p2, wide = model(drug, padded)
    assert torch.allclose(bare[0, 0], wide[0, 0, :80], atol=1e-5)


def test_prediction_is_invariant_to_batch_padding(model):
    torch.manual_seed(1)
    seq = torch.randint(2, PROTEIN_VOCAB, (1, 80))
    padded = torch.cat([seq, torch.zeros(1, 220, dtype=torch.long)], dim=1)
    drug = torch.randint(2, DRUG_VOCAB, (1, 30))
    with torch.no_grad():
        p1, _ = model(drug, seq)
        p2, _ = model(drug, padded)
    assert torch.allclose(p1, p2, atol=1e-4)


def test_drug_encoding_is_invariant_to_padding():
    encoder = DrugEncoder(DRUG_VOCAB).eval()
    seq = torch.randint(2, DRUG_VOCAB, (1, 20))
    padded = torch.cat([seq, torch.zeros(1, 30, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(encoder(seq), encoder(padded), atol=1e-6)


def test_real_lengths_measures_to_the_last_non_pad_token():
    tokens = torch.tensor([[5, 6, 7, 0, 0], [5, 0, 7, 0, 0], [0, 0, 0, 0, 0]])
    assert real_lengths(tokens).tolist() == [3, 3, 1]


# --------------------------------------------------------------------------
# the Track B -> Track C hand-off format
# --------------------------------------------------------------------------

def test_explain_returns_one_weight_per_real_residue(model):
    protein = torch.zeros(3, 300, dtype=torch.long)
    lengths = [40, 120, 300]
    for i, length in enumerate(lengths):
        protein[i, :length] = torch.randint(2, PROTEIN_VOCAB, (length,))
    weights = model.explain(torch.randint(2, DRUG_VOCAB, (3, 50)), protein)
    assert [len(w) for w in weights] == lengths


def test_explain_output_is_plain_python_floats(model):
    weights = model.explain(torch.randint(2, DRUG_VOCAB, (2, 20)),
                            torch.randint(2, PROTEIN_VOCAB, (2, 50)))
    assert isinstance(weights, list) and isinstance(weights[0], list)
    assert all(isinstance(v, float) for v in weights[0])


def test_explain_weights_are_non_negative(model):
    weights = model.explain(torch.randint(2, DRUG_VOCAB, (2, 20)),
                            torch.randint(2, PROTEIN_VOCAB, (2, 50)))
    assert min(min(w) for w in weights) >= 0.0


def test_explain_is_deterministic_because_it_forces_eval_mode(model):
    """Attention dropout is active in train mode and perturbs the weights.

    explain() calls .eval() itself, so a caller who forgot cannot accidentally
    export dropout-corrupted explanations.
    """
    model.train()
    drug = torch.randint(2, DRUG_VOCAB, (2, 20))
    protein = torch.randint(2, PROTEIN_VOCAB, (2, 50))
    assert model.explain(drug, protein) == model.explain(drug, protein)


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

def test_unseen_smiles_character_degrades_to_unk_instead_of_raising():
    """Cold-drug and cold-pair splits guarantee unseen characters at test time."""
    vocab = build_smiles_vocab(["CCO"])
    assert encode_smiles("CC[Se]", vocab) == [vocab["C"], vocab["C"], 1, 1, 1, 1]


def test_smiles_encoding_respects_max_len():
    vocab = build_smiles_vocab(["CCO"])
    assert len(encode_smiles("C" * 500, vocab, max_len=100)) == 100


def test_protein_encoding_respects_max_len_and_uppercases():
    vocab = build_protein_vocab()
    assert len(encode_protein("A" * 5000, vocab, max_len=1000)) == 1000
    assert encode_protein("acd", vocab) == encode_protein("ACD", vocab)


def test_unknown_amino_acid_becomes_unk():
    vocab = build_protein_vocab()
    assert encode_protein("A*C", vocab)[1] == 1


def test_collate_pads_to_the_longest_member_of_the_batch():
    batch = [(torch.tensor([1, 2]), torch.tensor([1, 2, 3]), torch.tensor(1.0)),
             (torch.tensor([1, 2, 3, 4]), torch.tensor([1]), torch.tensor(0.0))]
    drugs, proteins, labels = collate_batch(batch)
    assert drugs.shape == (2, 4) and proteins.shape == (2, 3) and labels.shape == (2,)
    assert drugs[0, 2:].tolist() == [0, 0]


def test_dataloader_round_trip_produces_trainable_batches():
    loader = make_loader(random_dataset(32, seed=0), batch_size=8)
    drugs, proteins, labels = next(iter(loader))
    assert drugs.shape[0] == proteins.shape[0] == labels.shape[0] == 8


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------

def test_loss_backward_reaches_every_encoder():
    torch.manual_seed(0)
    model = ColdSiteDTI(DRUG_VOCAB, PROTEIN_VOCAB)
    pred, _ = model(torch.randint(2, DRUG_VOCAB, (4, 40)),
                    torch.randint(2, PROTEIN_VOCAB, (4, 100)))
    torch.nn.functional.mse_loss(pred.squeeze(-1), torch.rand(4)).backward()

    for name in ("drug_encoder.conv1", "protein_encoder.conv", "protein_encoder.bilstm"):
        module = model.get_submodule(name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no gradient reached {name}"
        assert any(g.abs().sum() > 0 for g in grads), f"zero gradient at {name}"


def test_forward_produces_no_nans_on_a_heavily_padded_batch():
    torch.manual_seed(0)
    model = ColdSiteDTI(DRUG_VOCAB, PROTEIN_VOCAB).eval()
    protein = torch.zeros(4, 400, dtype=torch.long)
    protein[:, :5] = torch.randint(2, PROTEIN_VOCAB, (4, 5))
    with torch.no_grad():
        pred, attn = model(torch.randint(2, DRUG_VOCAB, (4, 10)), protein)
    assert torch.isfinite(pred).all() and torch.isfinite(attn).all()
