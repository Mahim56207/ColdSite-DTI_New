"""The Kaggle analysis notebook: its input check, and the names it expects back.

This notebook analyses trained cells rather than training any, so its one job before it
burns GPU hours is to refuse anything that is not the grid this project verified. The
check is executed here against synthetic inputs, including the two failures that have
actually happened: MolTrans's seeds that trained as seed 1, and a checkpoint whose cell
never finished.
"""
import json
import os

import pytest

NB = "notebooks/kaggle_analysis_davis.ipynb"
MODELS = ("deepdta", "coldsite_dti", "hyperattentiondti", "moltrans")
LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")


@pytest.fixture(scope="module")
def cells():
    nb = json.load(open(NB))
    return ["".join(c["source"]) for c in nb["cells"]]


@pytest.fixture(scope="module")
def verify_cell(cells):
    found = [c for c in cells if "EXPECTED_AUROC = {" in c]
    assert len(found) == 1, "the input check should live in exactly one cell"
    return found[0]


def _expected(verify_cell):
    scope = {"DATASET": "davis"}
    exec(verify_cell.split("def names(")[0], scope)      # just the table
    return scope["EXPECTED_AUROC"]


def test_the_table_covers_every_cell_of_the_grid(verify_cell):
    table = _expected(verify_cell)
    assert len(table) == 48
    assert {k[0] for k in table} == set(MODELS)
    assert {k[1] for k in table} == set(LEVELS)
    assert {k[2] for k in table} == {1, 2, 3}
    assert all(0.5 <= v <= 1.0 for v in table.values())


def _fixture_inputs(tmp_path, table, tweak=None):
    """A fake attached dataset holding every cell, with the verified AUROCs."""
    inp = tmp_path / "input" / "cells"
    inp.mkdir(parents=True)
    for (model, level, seed), auroc in table.items():
        suffix = "" if model == "coldsite_dti" else f"_{model}"
        with open(inp / f"davis_{level}_binary_seed{seed}{suffix}_results.json", "w") as f:
            json.dump({"test_metrics": {"auroc": auroc}}, f)
        open(inp / f"coldsite_dti_davis_{level}_binary_seed{seed}{suffix}.pt", "wb").close()
    if tweak:
        tweak(inp)
    return inp


def _run(verify_cell, tmp_path, inp):
    results = tmp_path / "results"
    results.mkdir()
    scope = {"DATASET": "davis", "INPUT_ROOT": str(inp.parent), "RESULTS": str(results),
             "os": os, "MODELS": ["hyperattentiondti", "moltrans"], "RUN_STAGE3": True,
             "AUDIT_MODELS": ["coldsite_dti", "hyperattentiondti", "moltrans"],
             "ALL_MODELS": ["deepdta", "coldsite_dti", "hyperattentiondti", "moltrans"]}
    exec(verify_cell, scope)
    return results, scope


def test_the_verified_grid_is_accepted_and_linked(verify_cell, tmp_path):
    table = _expected(verify_cell)
    results, scope = _run(verify_cell, tmp_path, _fixture_inputs(tmp_path, table))
    assert len(os.listdir(results)) == 96                 # 48 results files + 48 checkpoints
    assert scope["PRESENT"] == ["deepdta", "coldsite_dti", "hyperattentiondti", "moltrans"]


def test_nothing_attached_says_so_instead_of_listing_every_cell(verify_cell, tmp_path):
    """What a first run looks like when Add Input was forgotten: the old message buried
    the cause under 48 'no results file found' lines."""
    (tmp_path / "input").mkdir()
    with pytest.raises(AssertionError, match="nothing to analyse"):
        _run(verify_cell, tmp_path, tmp_path / "input" / "none")


def test_deepdta_is_optional_because_it_is_only_the_accuracy_anchor(verify_cell, tmp_path):
    table = {k: v for k, v in _expected(verify_cell).items() if k[0] != "deepdta"}
    results, scope = _run(verify_cell, tmp_path, _fixture_inputs(tmp_path, table))
    assert len(os.listdir(results)) == 72                 # the other three models
    assert "deepdta" not in scope["PRESENT"]
    assert scope["AUDIT_MODELS"] == ["coldsite_dti", "hyperattentiondti", "moltrans"]


def test_a_seed_that_trained_as_another_seed_is_refused(verify_cell, tmp_path):
    """MolTrans's first grid: every seed gave seed 1's metrics (vendored reseed)."""
    table = _expected(verify_cell)

    def as_seed_one(inp):
        one = json.load(open(inp / "davis_cold_pair_binary_seed1_moltrans_results.json"))
        with open(inp / "davis_cold_pair_binary_seed2_moltrans_results.json", "w") as f:
            json.dump(one, f)

    with pytest.raises(AssertionError, match="not the verified ones"):
        _run(verify_cell, tmp_path, _fixture_inputs(tmp_path, table, as_seed_one))


def test_a_cell_with_no_results_file_is_refused(verify_cell, tmp_path):
    table = _expected(verify_cell)

    def drop(inp):
        os.remove(inp / "davis_random_binary_seed3_hyperattentiondti_results.json")

    with pytest.raises(AssertionError, match="not the verified ones"):
        _run(verify_cell, tmp_path, _fixture_inputs(tmp_path, table, drop))


def test_a_results_file_without_its_checkpoint_is_refused(verify_cell, tmp_path):
    table = _expected(verify_cell)

    def drop(inp):
        os.remove(inp / "coldsite_dti_davis_cold_drug_binary_seed1_hyperattentiondti.pt")

    with pytest.raises(AssertionError, match="not the verified ones"):
        _run(verify_cell, tmp_path, _fixture_inputs(tmp_path, table, drop))


def test_the_positional_control_tags_carry_their_own_underscore(cells):
    """--tag is appended to the file stem with no separator: 'policyA' would name the
    output positional_control_..._davispolicyA.json and never match the project's."""
    stage2 = [c for c in cells if "positional + residue nulls" in c]
    assert len(stage2) == 1
    assert "('_policyA', GT), ('_klifs_policyA', GT_KLIFS)" in stage2[0]
    assert "{DATASET}{tag}" in stage2[0]


def test_every_module_it_calls_exists(cells):
    import re
    modules = set()
    for c in cells:
        modules |= set(re.findall(r"py\('([\w.]+)'", c))
    assert modules, "no commands found in the notebook"
    for module in modules:
        assert os.path.exists(module.replace(".", "/") + ".py"), module


def test_it_trains_nothing_and_pins_the_audit_family(cells):
    joined = "\n".join(cells)
    for trainer in ("train_deepdta", "train_moltrans", "train_hyperattentiondti", "run_grid"):
        assert trainer not in joined, f"an analysis notebook must not call {trainer}"
    assert "AUDIT_MODELS = ['coldsite_dti', 'hyperattentiondti', 'moltrans']" in joined
    assert "'--steps', 'audit,positive,summary'" in joined     # Holm runs once, over all
