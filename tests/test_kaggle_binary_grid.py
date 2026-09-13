"""The Kaggle notebook's identical-seed warning (notebooks/kaggle_binary_grid.ipynb)."""
import glob
import json
import os
import re

NOTEBOOK = "notebooks/kaggle_binary_grid.ipynb"


def _warning_function():
    cells = json.load(open(NOTEBOOK))["cells"]
    source = next("".join(c["source"]) for c in cells
                  if "def same_as_other_seed" in "".join(c["source"]))
    start = source.index("def same_as_other_seed")
    end = source.index("\ndef ", start + 1)
    namespace = {"json": json, "glob": glob, "re": re, "os": os}
    exec(source[start:end], namespace)
    return namespace["same_as_other_seed"]


def _result(folder, seed, auroc, model_suffix="_moltrans"):
    path = folder / f"davis_cold_pair_binary_seed{seed}{model_suffix}_results.json"
    path.write_text(json.dumps({"test_metrics": {"auroc": auroc, "auprc": 0.1}}))
    return str(path)


def test_identical_seeds_are_flagged(tmp_path):
    """What MolTrans's DAVIS grid produced: seed 2 identical to seed 1."""
    check = _warning_function()
    _result(tmp_path, 1, 0.5899)
    assert "SEEDS IDENTICAL" in check(_result(tmp_path, 2, 0.5899))


def test_different_seeds_pass(tmp_path):
    check = _warning_function()
    _result(tmp_path, 1, 0.5899)
    assert check(_result(tmp_path, 2, 0.6012)) is None


def test_another_models_identical_numbers_are_not_confused_for_a_seed(tmp_path):
    """ColdSite-DTI's files have no model suffix; a MolTrans file must not match them."""
    check = _warning_function()
    _result(tmp_path, 1, 0.5899, model_suffix="_moltrans")
    assert check(_result(tmp_path, 2, 0.5899, model_suffix="")) is None


def test_a_lone_seed_passes(tmp_path):
    assert _warning_function()(_result(tmp_path, 1, 0.5899)) is None
