"""Tests for the real collector — checkpoints in, (weights, sites, ids) out.

The failure this file guards is the one the whole project keeps re-encountering:
nothing crashes, an array of the wrong length or the wrong protein's ground
truth comes back, and the resulting precision@k looks entirely reasonable. So
these tests check alignment and length, not just that a call returns.

No training happens here. A checkpoint is a state dict on disk; a randomly
initialised model produces meaningless attention, which is exactly right for
testing plumbing — the numbers are not asserted, the shapes and the identities
are.
"""
import json
import os

import numpy as np
import pytest
import torch

from src.data.ground_truth import load_site_sets
from src.evaluation.collect import MissingCell, collect_cell, make_collect_fn
from src.model.checkpoint_naming import checkpoint_path

SEQUENCES = {
    "AAA1": "MKVLAAGIVGSPYTRQWEDNFKLMHACGVTSREDIPQWNYAFKLGHTSRVDEMPQ" * 3,
    "BBB2": "MTQERPLKDGHYFNWASVCIMLERTPQKDGHYFNWASVCIMLERTPQKDGHYFNW" * 2,
    "CCC3": "MGSHKLPQWERTYIVCADFNMGSHKLPQWERTYIVCADFNMGSHKLPQWERTYIV",
}


def write_ground_truth(path, targets=("AAA1", "BBB2", "CCC3")):
    """UniProt-shaped: 1-indexed inclusive ranges, `type` on every feature."""
    payload = {
        name: [{"start": 10, "end": 14, "type": "Binding site", "description": ""},
               {"start": 30, "end": 30, "type": "Active site", "description": ""}]
        for name in targets
    }
    path.write_text(json.dumps(payload))
    return str(path)


def write_split(split_dir, rows):
    """rows: (target_id, smiles, sequence, label). Writes train/valid/test."""
    import pandas as pd

    os.makedirs(split_dir, exist_ok=True)
    frame = pd.DataFrame(
        [{"Drug_ID": f"D{i}", "Drug": smiles, "Target_ID": tid,
          "Target": sequence, "Y": label}
         for i, (tid, smiles, sequence, label) in enumerate(rows)])
    for part in ("train", "valid", "test"):
        frame.to_csv(os.path.join(split_dir, f"{part}.csv"), index=False)
    return split_dir


def write_coldsite_checkpoint(checkpoint_dir, dataset, level, task, seed,
                              split_dir):
    """A real ColdSiteDTI state dict, untrained, at the canonical path."""
    import pandas as pd

    from src.model.coldsite_dti import ColdSiteDTI
    from src.model.drug_encoder import build_smiles_vocab
    from src.model.protein_encoder import build_protein_vocab

    train = pd.read_csv(os.path.join(split_dir, "train.csv"))
    drug_vocab = build_smiles_vocab(train["Drug"].astype(str).tolist())
    protein_vocab = build_protein_vocab()
    model = ColdSiteDTI(len(drug_vocab) + 2, len(protein_vocab) + 2)

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = checkpoint_path(checkpoint_dir, dataset, level, task, seed,
                           model="coldsite_dti")
    torch.save({"model_state": model.state_dict()}, path)
    return path


@pytest.fixture
def cell(tmp_path):
    """One collectable cell: split files, ground truth, a checkpoint."""
    rows = [("AAA1", "CCO", SEQUENCES["AAA1"], 1.0),
            ("BBB2", "CCN", SEQUENCES["BBB2"], 0.0),
            ("CCC3", "CCC", SEQUENCES["CCC3"], 1.0)]
    split_root = tmp_path / "splits"
    split_dir = write_split(str(split_root / "davis" / "random"), rows)
    gt = write_ground_truth(tmp_path / "gt.json")
    checkpoints = str(tmp_path / "results")
    write_coldsite_checkpoint(checkpoints, "davis", "random", "binary", 1, split_dir)
    return {"split_root": str(split_root), "ground_truth": gt,
            "checkpoint_dir": checkpoints,
            "site_sets": load_site_sets(gt, max_len=1000)}


def test_collect_cell_returns_aligned_weights_sites_and_ids(cell):
    weights, sites, ids = collect_cell(
        "coldsite_dti", "davis", "random", 1,
        site_sets=cell["site_sets"], task="binary",
        split_root=cell["split_root"], checkpoint_dir=cell["checkpoint_dir"],
        verbose=False)

    assert len(weights) == len(sites) == len(ids) == 3
    assert ids == ["AAA1", "BBB2", "CCC3"]
    for target_id, weight in zip(ids, weights):
        # one weight per REAL residue, not per padded position
        assert weight.ndim == 1
        assert len(weight) == len(SEQUENCES[target_id])
        assert np.all(np.asarray(weight) >= 0)


def test_each_explanation_keeps_its_own_protein_length(cell):
    """The alignment seam. If a protein's weights come back at another
    protein's length, every ground-truth index past that point is scored
    against a weight belonging to a different residue."""
    weights, _sites, ids = collect_cell(
        "coldsite_dti", "davis", "random", 1, site_sets=cell["site_sets"],
        task="binary", split_root=cell["split_root"],
        checkpoint_dir=cell["checkpoint_dir"], verbose=False)
    lengths = {i: len(w) for i, w in zip(ids, weights)}
    assert lengths == {name: len(SEQUENCES[name]) for name in ids}
    assert len(set(lengths.values())) == 3, "the three proteins differ in length"


def test_pairs_per_target_caps_repeated_proteins(tmp_path):
    """precision@k is per protein. Collecting every pair would put the same
    protein into the average once per drug it was measured against."""
    rows = [("AAA1", smiles, SEQUENCES["AAA1"], 1.0)
            for smiles in ("CCO", "CCN", "CCC", "CCF")]
    split_root = tmp_path / "splits"
    split_dir = write_split(str(split_root / "davis" / "random"), rows)
    gt = write_ground_truth(tmp_path / "gt.json")
    checkpoints = str(tmp_path / "results")
    write_coldsite_checkpoint(checkpoints, "davis", "random", "binary", 1, split_dir)
    site_sets = load_site_sets(gt, max_len=1000)

    common = dict(site_sets=site_sets, task="binary",
                  split_root=str(split_root), checkpoint_dir=checkpoints,
                  verbose=False)
    one = collect_cell("coldsite_dti", "davis", "random", 1,
                       pairs_per_target=1, **common)
    many = collect_cell("coldsite_dti", "davis", "random", 1,
                        pairs_per_target=4, **common)

    assert len(one[2]) == 1
    assert len(many[2]) == 4
    assert set(many[2]) == {"AAA1"}


def test_target_without_usable_ground_truth_is_skipped_not_scored(tmp_path):
    """Skipped, not counted as zero: an empty site set scored as a miss drags
    every mean down and the cell still looks populated."""
    rows = [("AAA1", "CCO", SEQUENCES["AAA1"], 1.0),
            ("ZZZ9", "CCN", SEQUENCES["BBB2"], 0.0)]   # ZZZ9 has no annotation
    split_root = tmp_path / "splits"
    split_dir = write_split(str(split_root / "davis" / "random"), rows)
    gt = write_ground_truth(tmp_path / "gt.json", targets=("AAA1",))
    checkpoints = str(tmp_path / "results")
    write_coldsite_checkpoint(checkpoints, "davis", "random", "binary", 1, split_dir)

    _weights, _sites, ids = collect_cell(
        "coldsite_dti", "davis", "random", 1,
        site_sets=load_site_sets(gt, max_len=1000), task="binary",
        split_root=str(split_root), checkpoint_dir=checkpoints, verbose=False)
    assert ids == ["AAA1"]


def test_uniform_control_needs_no_checkpoint(cell):
    """The floor line is flat by construction; a missing file is not a defect."""
    weights, _sites, ids = collect_cell(
        "uniform_control", "davis", "random", 1, site_sets=cell["site_sets"],
        task="binary", split_root=cell["split_root"],
        checkpoint_dir=str(cell["checkpoint_dir"]) + "_does_not_exist",
        verbose=False)
    assert len(ids) == 3
    for weight in weights:
        assert np.allclose(weight, weight[0]), "the control must be flat"


def test_deepdta_is_refused_with_its_reason(cell):
    with pytest.raises(MissingCell, match="provides_attention"):
        collect_cell("deepdta", "davis", "random", 1,
                     site_sets=cell["site_sets"], task="binary",
                     split_root=cell["split_root"],
                     checkpoint_dir=cell["checkpoint_dir"], verbose=False)


def test_missing_checkpoint_is_a_recorded_skip_not_a_crash(cell):
    """build_grid records missing cells; the reason has to survive, because
    'no checkpoint' and 'no usable ground truth' need different responses."""
    skipped = []
    collect = make_collect_fn(
        dataset="davis", ground_truth=cell["ground_truth"], task="binary",
        split_root=cell["split_root"], checkpoint_dir=cell["checkpoint_dir"],
        skipped=skipped)

    assert collect("coldsite_dti", "davis", "random", 99) is None
    assert len(skipped) == 1
    assert "no checkpoint" in skipped[0]
    assert "seed99" in skipped[0]


def test_collector_drives_the_audit_grid_end_to_end(cell):
    """The point of the module: build_grid over real checkpoints produces a
    table. Values are meaningless (untrained weights); the shape is the test."""
    from src.evaluation.run_audit import build_grid, summarise

    collect = make_collect_fn(
        dataset="davis", ground_truth=cell["ground_truth"], task="binary",
        split_root=cell["split_root"], checkpoint_dir=cell["checkpoint_dir"])

    results = build_grid(collect, ["coldsite_dti", "uniform_control"],
                         ["davis"], [1], k=5, n_trials=20)

    assert results["grid"]["coldsite_dti"]["random"]["precision_at_k"]["n_seeds"] == 1
    assert results["grid"]["uniform_control"]["random"]["precision_at_k"]["n_seeds"] == 1
    # only 'random' has a split directory in the fixture
    assert any("cold_drug" in cell_name for cell_name in results["missing_cells"])
    assert "Audit grid" in summarise(results)
