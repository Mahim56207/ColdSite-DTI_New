"""Tests for the kinase confound control.

The control is the objection most likely to sink the paper, so the things
asserted here are the ones that would make it dishonest rather than broken:

  * the non-kinase arm really is scored against its own ground truth
  * an arm below the 20-target gate reports nothing rather than a number
  * a level with no checkpoint says so instead of quietly leaving one arm out
  * the report never describes the transfer arm as a stratification

The panel is monkeypatched to a small synthetic one. The real panel is 21,145
rows over 60 proteins with sequences up to 1000 residues, which is a slow way
to test bookkeeping.
"""
import json
import os

import pytest
import torch

from src.data.ground_truth import load_site_sets
from src.evaluation import run_control
from src.model.checkpoint_naming import checkpoint_path

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture(autouse=True)
def clear_registered_family_map():
    """`load_family_map` registers globally, and `panel_is_usable` calls it.

    Left registered, a synthetic panel's assignments would follow into whatever
    test ran next and classify its targets — the kind of order-dependent
    failure that appears only when someone adds a test above this file.
    """
    from src.evaluation.target_family import clear_family_map

    yield
    clear_family_map()


def make_sequence(seed, length=180):
    import random
    rng = random.Random(seed)
    return "".join(rng.choices(AA, k=length))


def build_panel(tmp_path, n_targets):
    """A stand-in for data/processed/nonkinase_panel.csv and its ground truth."""
    import pandas as pd

    rows, sites, families = [], {}, {}
    for i in range(n_targets):
        accession = f"P{10000 + i}"
        sequence = make_sequence(i)
        for drug in range(3):                      # several pairs per protein
            rows.append({"Drug_ID": f"D{drug}", "Drug": "CCO" * (drug + 1),
                         "Target_ID": accession, "Target": sequence,
                         "Y": float(drug % 2)})
        sites[accession] = [
            {"start": 20, "end": 25, "type": "Binding site", "description": ""},
            {"start": 60, "end": 60, "type": "Active site", "description": ""}]
        families[accession] = "non_kinase"

    panel_csv = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel_csv, index=False)
    panel_sites = tmp_path / "panel_sites.json"
    panel_sites.write_text(json.dumps(sites))
    panel_families = tmp_path / "panel_families.json"
    panel_families.write_text(json.dumps(families))
    return str(panel_csv), str(panel_sites), str(panel_families)


def build_kinase_cell(tmp_path, levels=("random", "cold_target"), n_targets=8):
    """Split files plus a ColdSite-DTI checkpoint per level, seed 1."""
    import pandas as pd

    from src.model.coldsite_dti import ColdSiteDTI
    from src.model.drug_encoder import build_smiles_vocab
    from src.model.protein_encoder import build_protein_vocab

    rows, sites = [], {}
    for i in range(n_targets):
        name = f"KIN{i}"
        sequence = make_sequence(1000 + i)
        rows.append({"Drug_ID": f"K{i}", "Drug": "CCN", "Target_ID": name,
                     "Target": sequence, "Y": 1.0})
        sites[name] = [{"start": 15, "end": 20, "type": "Binding site",
                        "description": ""}]

    frame = pd.DataFrame(rows)
    split_root = tmp_path / "splits"
    checkpoints = tmp_path / "results"
    os.makedirs(checkpoints, exist_ok=True)

    for level in levels:
        directory = split_root / "davis" / level
        os.makedirs(directory, exist_ok=True)
        for part in ("train", "valid", "test"):
            frame.to_csv(directory / f"{part}.csv", index=False)

        drug_vocab = build_smiles_vocab(frame["Drug"].astype(str).tolist())
        model = ColdSiteDTI(len(drug_vocab) + 2, len(build_protein_vocab()) + 2)
        torch.save({"model_state": model.state_dict()},
                   checkpoint_path(str(checkpoints), "davis", level, "binary", 1))

    ground_truth = tmp_path / "kinase_sites.json"
    ground_truth.write_text(json.dumps(sites))
    return str(split_root), str(checkpoints), str(ground_truth)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A control run with 25 panel proteins — above the gate."""
    panel_csv, panel_sites, panel_families = build_panel(tmp_path, n_targets=25)
    monkeypatch.setattr(run_control, "PANEL_ROWS", panel_csv)
    monkeypatch.setattr(run_control, "PANEL_SITES", panel_sites)
    monkeypatch.setattr(run_control, "PANEL_FAMILIES", panel_families)
    split_root, checkpoints, ground_truth = build_kinase_cell(tmp_path)
    return {"split_root": split_root, "checkpoint_dir": checkpoints,
            "ground_truth": ground_truth}


def test_both_arms_are_scored_from_the_same_checkpoint(wired):
    """The only thing that differs between the two numbers is the protein
    family. Same model, same weights, same vocabulary, same k."""
    results = run_control.run_control(
        "coldsite_dti", "davis", 1, levels=("random",), k=5, n_trials=50,
        **wired)

    entry = results["levels"]["random"]
    assert entry["kinase"] is not None
    assert entry["non_kinase"] is not None
    assert entry["kinase"]["n_proteins"] == 8
    assert entry["non_kinase"]["n_proteins"] == 25
    assert entry["gap"] == pytest.approx(
        entry["kinase"]["precision_at_k"] - entry["non_kinase"]["precision_at_k"])


def test_panel_pairs_are_capped_per_protein(wired):
    """The panel holds several pairs per protein; n must count proteins."""
    results = run_control.run_control(
        "coldsite_dti", "davis", 1, levels=("random",), k=5, n_trials=50,
        pairs_per_target=1, **wired)
    assert results["levels"]["random"]["non_kinase"]["n_proteins"] == 25


def test_an_arm_below_the_gate_reports_nothing_not_a_number(tmp_path, monkeypatch):
    """Below 20 targets the comparison has no power. A number a reviewer
    cannot believe is worse than an openly stated limitation."""
    panel_csv, panel_sites, panel_families = build_panel(tmp_path, n_targets=5)
    monkeypatch.setattr(run_control, "PANEL_ROWS", panel_csv)
    monkeypatch.setattr(run_control, "PANEL_SITES", panel_sites)
    monkeypatch.setattr(run_control, "PANEL_FAMILIES", panel_families)
    split_root, checkpoints, ground_truth = build_kinase_cell(tmp_path)

    results = run_control.run_control(
        "coldsite_dti", "davis", 1, levels=("random",), k=5, n_trials=50,
        split_root=split_root, checkpoint_dir=checkpoints,
        ground_truth=ground_truth)

    entry = results["levels"]["random"]
    assert entry["non_kinase"] is None
    assert "gate" in entry["non_kinase_reason"]
    assert "gap" not in entry


def test_a_level_without_a_checkpoint_says_so(wired):
    """cold_pair has no checkpoint in the fixture. Both arms need the same
    weights, so the level drops out with a reason rather than half-reported."""
    results = run_control.run_control(
        "coldsite_dti", "davis", 1, levels=("random", "cold_pair"), k=5,
        n_trials=50, **wired)

    entry = results["levels"]["cold_pair"]
    assert entry["kinase"] is None and entry["non_kinase"] is None
    assert "no checkpoint" in entry["kinase_reason"]
    assert "cold_pair" in run_control.report(results)


def test_the_report_calls_the_control_a_transfer_not_a_stratification(wired):
    """DAVIS has zero non-kinase targets, so nothing is being stratified. The
    panel is out-of-distribution, which makes it harder than cold_target --
    describing it as stratification would overstate what was measured."""
    results = run_control.run_control(
        "coldsite_dti", "davis", 1, levels=("random",), k=5, n_trials=50,
        **wired)
    text = run_control.report(results)

    assert "transfer" in text.lower()
    assert "not a stratification" in text.lower()
    assert "harder than" in text.lower()
    assert "transfer" in results["design"]


def test_the_committed_panel_clears_its_own_gate():
    """Not a fixture: the real 60-target panel in the repository.

    `python -m src.evaluation.target_family --panel` reports the same thing.
    If this fails, the control arm has stopped existing and the ladder must
    not be presented as if the confound were controlled.
    """
    if not os.path.exists(run_control.PANEL_SITES):
        pytest.skip("panel ground truth not present")
    gate = run_control.panel_is_usable(
        load_site_sets(run_control.PANEL_SITES, max_len=1000))
    assert gate["control_is_usable"]
    assert gate["distinct_non_kinase"] >= run_control.MIN_TARGETS
