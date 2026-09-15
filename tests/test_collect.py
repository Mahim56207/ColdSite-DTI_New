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


def test_explanations_are_cut_to_the_ground_truth_window(cell, monkeypatch):
    """MolTrans's 545 tokens reach past residue 1,000 on long proteins. Its
    attention there would compete for the top k while no site can exist to hit,
    so every model's explanation is cut to the window the sites are cut to."""
    from src.evaluation import collect

    monkeypatch.setattr(collect, "_explain_row",
                        lambda *args, **kwargs: np.linspace(0, 1, 1400))
    weights, _sites, _ids = collect_cell(
        "uniform_control", "davis", "random", 1, site_sets=cell["site_sets"],
        task="binary", split_root=cell["split_root"],
        checkpoint_dir=cell["checkpoint_dir"], max_protein_len=1000, verbose=False)
    assert {w.size for w in weights} == {1000}


def test_cxsmiles_annotations_are_cut_and_unreadable_molecules_dropped(tmp_path):
    """BindingDB's panel writes `SMILES |r|`; a wildcard or a dative bond is no
    molecule any model can read. Cut the first, drop the second, for every
    model alike -- and let the protein's next readable pair stand in."""
    import pandas as pd

    from src.evaluation.collect import _read_test_rows, clean_smiles

    assert clean_smiles("CC(=O)O |r,c:3|") == "CC(=O)O"
    pd.DataFrame({
        "Target_ID": ["P1", "P1", "P2", "P3"],
        "Drug": ["*.CCO", "CCN |r|", "CC->[Re+]", "c1ccccc1"],
        # one sequence per protein: since 2026-09-13 a protein is a distinct sequence
        "Target": ["MKV", "MKV", "MKW", "MKY"],
    }).to_csv(tmp_path / "test.csv", index=False)
    rows = _read_test_rows(str(tmp_path), pairs_per_target=1)
    assert [(t, s) for t, s, _seq in rows] == [("P1", "CCN"), ("P3", "c1ccccc1")]


def test_davis_and_kiba_rows_are_untouched_by_the_cleaning():
    import json

    from src.evaluation.collect import UNREADABLE_SMILES, clean_smiles

    for dataset in ("davis", "kiba"):
        path = f"src/data/baselines/deepdta/data/{dataset}/ligands_can.txt"
        if not os.path.exists(path):
            pytest.skip("DeepDTA source files not present")
        for smiles in json.load(open(path)).values():
            assert clean_smiles(smiles) == smiles
            assert not UNREADABLE_SMILES.search(smiles)


# ----------------------------------------------------------------------------
# drug-specific ground truth (src/data/klifs_ligand_contacts.py)
# ----------------------------------------------------------------------------

def _pair_ground_truth(path, pairs):
    """Keys are "<drug>|<target>": the sites belong to the pair, not the protein."""
    payload = {key: [{"start": 10, "end": 14, "type": "Binding site", "description": ""}]
               for key in pairs}
    path.write_text(json.dumps(payload))
    return str(path)


def test_a_drug_specific_ground_truth_scores_pairs_not_proteins(cell, tmp_path):
    """Only pairs with a co-crystal are scorable, and the id says which pair it was."""
    gt = _pair_ground_truth(tmp_path / "pairs.json", ["D0|AAA1", "D2|CCC3"])
    weights, sites, ids = collect_cell(
        "coldsite_dti", "davis", "random", 1,
        site_sets=load_site_sets(gt, max_len=1000), task="binary",
        split_root=cell["split_root"], checkpoint_dir=cell["checkpoint_dir"],
        pairs_per_target=0, verbose=False)
    assert ids == ["D0|AAA1", "D2|CCC3"]            # BBB2's pair has no structure
    assert len(weights) == len(sites) == 2
    assert len(weights[0]) == len(SEQUENCES["AAA1"])


def test_the_same_protein_with_another_drug_is_another_measurement(tmp_path):
    """Two drugs against one protein are two rows of the drug-specific table, where the
    per-protein ground truth would have counted the protein once."""
    rows = [("AAA1", "CCO", SEQUENCES["AAA1"], 1.0),
            ("AAA1", "CCN", SEQUENCES["AAA1"], 0.0)]
    split_root = tmp_path / "splits"
    split_dir = write_split(str(split_root / "davis" / "random"), rows)
    checkpoints = str(tmp_path / "results")
    write_coldsite_checkpoint(checkpoints, "davis", "random", "binary", 1, split_dir)
    gt = _pair_ground_truth(tmp_path / "pairs.json", ["D0|AAA1", "D1|AAA1"])
    _w, _s, ids = collect_cell(
        "coldsite_dti", "davis", "random", 1,
        site_sets=load_site_sets(gt, max_len=1000), task="binary",
        split_root=str(split_root), checkpoint_dir=checkpoints,
        pairs_per_target=0, verbose=False)
    assert ids == ["D0|AAA1", "D1|AAA1"]


def test_a_split_with_no_crystallised_pair_says_so(cell, tmp_path):
    """A cold split may hold no co-crystal pair at all: that must read as 'no pairs
    here', not as a ground truth that fails to match the dataset's spelling."""
    gt = _pair_ground_truth(tmp_path / "pairs.json", ["D9|ZZZ9"])
    with pytest.raises(MissingCell, match="co-crystal"):
        collect_cell("coldsite_dti", "davis", "random", 1,
                     site_sets=load_site_sets(gt, max_len=1000), task="binary",
                     split_root=cell["split_root"], checkpoint_dir=cell["checkpoint_dir"],
                     pairs_per_target=0, verbose=False)


@pytest.mark.skipif(not os.path.exists("data/splits/davis/random/test.csv"),
                    reason="needs the built DAVIS splits (not in CI)")
def test_the_rows_carry_their_drug_id_only_when_asked():
    """_read_test_rows' tuple shape is the alignment seam for the pair lookup."""
    from src.evaluation.collect import _read_test_rows
    plain = _read_test_rows("data/splits/davis/random", 1, policy=False)
    withdrug = _read_test_rows("data/splits/davis/random", 1, policy=False, with_drug=True)
    assert len(plain[0]) == 3 and len(withdrug[0]) == 4
    assert [r[0] for r in plain] == [r[0] for r in withdrug]        # same rows, same order
    assert withdrug[0][2:] == plain[0][1:]                          # drug id inserted at 1
    assert all(r[1] for r in withdrug)
