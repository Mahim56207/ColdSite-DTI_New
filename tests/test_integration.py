"""End-to-end test of the Track A -> Track B -> Track C seam.

Both guides name this hand-off as the first item of Part 2 and 'the main
integration seam'. This test runs it for real on synthetic data:

    ground-truth JSON  ->  0-indexed site sets        (Track A -> ground_truth.py)
    model.explain()    ->  per-residue weight lists   (Track B)
    precision@k        ->  a number + a p-value       (Track C)

If this passes, the only thing standing between the team and real numbers is
real data. It also pins the alignment contract: attention index i corresponds
to residue i+1 in UniProt numbering, for every protein, in every batch.
"""
import json

import numpy as np
import pytest
import torch

from src.data.ground_truth import load_site_sets
from src.evaluation.precision_at_k import batch_precision_at_k, precision_at_k
from src.evaluation.significance_test import permutation_test_batch
from src.model.coldsite_dti import ColdSiteDTI

DRUG_VOCAB, PROTEIN_VOCAB = 70, 28


@pytest.fixture
def ground_truth_file(tmp_path):
    """Three proteins with UniProt-style 1-indexed inclusive ranges."""
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "KIN_A": [{"start": 41, "end": 43, "type": "Binding site"},
                  {"start": 143, "end": 145, "type": "Active site"}],
        "KIN_B": [{"start": 10, "end": 12, "type": "Binding site"}],
        "KIN_C": [{"start": 200, "end": 204, "type": "Nucleotide binding"}],
    }))
    return str(path)


def test_full_pipeline_runs_end_to_end(ground_truth_file):
    torch.manual_seed(0)
    model = ColdSiteDTI(DRUG_VOCAB, PROTEIN_VOCAB).eval()

    site_sets = load_site_sets(ground_truth_file, max_len=1000)
    target_order = ["KIN_A", "KIN_B", "KIN_C"]

    lengths = [300, 250, 400]
    protein_batch = torch.zeros(3, max(lengths), dtype=torch.long)
    for i, length in enumerate(lengths):
        protein_batch[i, :length] = torch.randint(2, PROTEIN_VOCAB, (length,))
    drug_batch = torch.randint(2, DRUG_VOCAB, (3, 50))

    weights = model.explain(drug_batch, protein_batch)
    assert [len(w) for w in weights] == lengths

    sites = [site_sets[t].positions for t in target_order]
    result = batch_precision_at_k(weights, sites, k=10,
                                  rng=np.random.default_rng(0))

    assert result["n_evaluated"] == 3
    assert 0.0 <= result["mean_precision_at_k"] <= 1.0
    assert not np.isnan(result["mean_precision_at_k"])


def test_an_untrained_model_scores_around_chance_not_above_it(ground_truth_file):
    """Sanity floor for the whole measurement.

    An untrained model knows nothing about binding sites. If this pipeline
    reported a significant precision@k here, the metric would be measuring an
    artefact -- padding, index bias, or tie ordering -- rather than explanation
    quality, and every real number produced later would be suspect.
    """
    torch.manual_seed(0)
    model = ColdSiteDTI(DRUG_VOCAB, PROTEIN_VOCAB).eval()

    site_sets = load_site_sets(ground_truth_file, max_len=1000)
    sites_a = site_sets["KIN_A"].positions

    weights, sites = [], []
    for seed in range(24):
        torch.manual_seed(seed)
        protein = torch.randint(2, PROTEIN_VOCAB, (1, 300))
        drug = torch.randint(2, DRUG_VOCAB, (1, 50))
        weights.append(np.asarray(model.explain(drug, protein)[0]))
        sites.append(sites_a)

    result = permutation_test_batch(weights, sites, k=10, n_trials=300, seed=0)
    assert not result["significant"], (
        f"untrained model looks significant (p={result['p_value']:.4f}) -- "
        f"the metric is picking up an artefact"
    )


def test_alignment_contract_attention_index_i_is_uniprot_residue_i_plus_one(
        ground_truth_file):
    """The one thing that must never drift between tracks.

    Simulates a perfect explanation by placing all attention mass on the
    positions the adapter reports, then checks the metric returns exactly 1.0.
    """
    site_sets = load_site_sets(ground_truth_file, max_len=1000)
    sites = site_sets["KIN_A"].positions
    assert sites == {40, 41, 42, 142, 143, 144}   # UniProt 41-43 and 143-145

    attention = np.zeros(300)
    attention[sorted(sites)] = 1.0
    assert precision_at_k(attention, sites, k=6) == 1.0


def test_a_shifted_explanation_is_visibly_penalised(ground_truth_file):
    """One residue of drift must cost score, or the metric proves nothing."""
    sites = load_site_sets(ground_truth_file, max_len=1000)["KIN_A"].positions

    aligned = np.zeros(300)
    aligned[sorted(sites)] = 1.0

    shifted = np.zeros(300)
    shifted[[p + 1 for p in sorted(sites)]] = 1.0

    rng = np.random.default_rng(0)
    assert precision_at_k(aligned, sites, k=6, rng=rng) == 1.0
    assert precision_at_k(shifted, sites, k=6, rng=rng) < 1.0


def test_truncation_policy_changes_which_proteins_are_evaluated(tmp_path):
    """A protein annotated only past the model's window cannot be scored.

    Left in, it contributes a guaranteed 0.0 and drags the split mean down for
    a reason that has nothing to do with explanation quality. This is a Methods
    decision, and the test exists so it stays a deliberate one.
    """
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "IN_WINDOW": [{"start": 50, "end": 52, "type": "Binding site"}],
        "OUT_OF_WINDOW": [{"start": 1500, "end": 1502, "type": "Binding site"}],
    }))

    excluded = load_site_sets(str(path), max_len=1000)
    assert set(excluded) == {"IN_WINDOW"}

    kept = load_site_sets(str(path), max_len=1000, truncation="keep")
    assert set(kept) == {"IN_WINDOW", "OUT_OF_WINDOW"}
    assert kept["OUT_OF_WINDOW"].positions == {1499, 1500, 1501}


def test_batch_report_exposes_how_many_proteins_actually_contributed(
        ground_truth_file):
    """A mean over 1 protein and a mean over 400 must not look identical."""
    site_sets = load_site_sets(ground_truth_file, max_len=1000)
    weights = [np.random.default_rng(i).random(300) for i in range(3)]
    sites = [site_sets["KIN_A"].positions, set(), site_sets["KIN_B"].positions]

    result = batch_precision_at_k(weights, sites, k=10,
                                  rng=np.random.default_rng(0))
    assert result["n_evaluated"] == 2
    assert result["n_skipped_no_sites"] == 1
    assert len(result["per_protein"]) == 2
