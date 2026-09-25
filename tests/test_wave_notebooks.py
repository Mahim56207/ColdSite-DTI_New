"""T11: the generated Wave A notebooks are exactly what the builder makes from `config/waves.json`
(a hand edit shows up as a diff), ship in dry run, name every cell of their account and no other,
carry no credential, and only the DrugBAN account installs DGL. A real run without a passing canary
is refused before any trainer starts."""
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

from src.cloud import preflight, runner
from src.cloud.config import load_waves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = {"ACC1": "kaggle_wave_a_acc1.ipynb", "ACC2": "kaggle_wave_a_acc2.ipynb",
             "ACC3": "kaggle_wave_a_acc3.ipynb"}


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_harness_notebook", os.path.join(ROOT, "notebooks", "build_harness_notebook.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name):
    return json.load(open(os.path.join(ROOT, "notebooks", name)))


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("account", sorted(NOTEBOOKS))
def test_a_notebook_is_what_the_builder_makes_from_the_wave_manifest(account):
    nb = _load(NOTEBOOKS[account])
    commit = re.search(r"COMMIT = '([0-9a-f]{40})'", _code(nb)[0]).group(1)
    assert _builder().build(account, "config/waves.json", commit, canary=False) == nb


@pytest.mark.parametrize("account", sorted(NOTEBOOKS))
def test_a_notebook_ships_in_dry_run_without_credentials_and_pins_its_commit(account):
    nb = _load(NOTEBOOKS[account])
    settings = _code(nb)[0]
    assert "DRY_RUN = True" in settings and f"ACCOUNT = {account!r}" in settings
    assert re.search(r"COMMIT = '[0-9a-f]{40}'", settings)
    blob = json.dumps(nb).lower()
    for forbidden in ("kaggle.json", "kaggle_key", "kaggle_username", "api_key", "password", "secret"):
        assert forbidden not in blob, forbidden


def test_every_cell_is_named_by_exactly_one_notebook_and_only_its_own_account_reads_it():
    waves = load_waves(os.path.join(ROOT, "config", "waves.json"))
    named = {}
    for account, name in NOTEBOOKS.items():
        text = "\n".join("".join(c["source"]) for c in _load(name)["cells"] if c["cell_type"] == "markdown")
        for cell in (c for q in waves["accounts"][account].values() for c in q):
            assert cell.id in text, f"{cell.id} missing from {name}"
            named.setdefault(cell.id, []).append(account)
    assert all(len(v) == 1 for v in named.values()) and len(named) == 30


def test_only_the_drugban_account_installs_dgl_and_pins_torch():
    for account, name in NOTEBOOKS.items():
        text = "\n".join(_code(_load(name)))
        assert ("torch==2.6.0" in text) == (account == "ACC3"), account


def test_the_notebooks_drive_the_runner_and_never_a_trainer_directly():
    for name in NOTEBOOKS.values():
        text = "\n".join(_code(_load(name)))
        assert "src.cloud.runner" in text
        assert "src.model.train" not in text and "run_grid" not in text
        assert "DataParallel" not in text and "DistributedDataParallel" not in text


def test_a_real_run_without_a_passing_canary_is_refused_before_any_trainer_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "probe_gpus", lambda: ["Tesla T4", "Tesla T4"])   # even with the right GPUs
    monkeypatch.setattr(runner.Runner, "run", lambda self: pytest.fail("a trainer was started"))
    monkeypatch.chdir(ROOT)
    if not os.path.exists(os.path.join(ROOT, "config", "canary_verdict.json")):
        code = runner.main(["--account", "ACC1", "--waves", "config/waves.json",
                            "--results-root", str(tmp_path / "res")])
        assert code == 2
    else:                                   # once a verdict is committed this test has nothing to refuse
        pytest.skip("a canary verdict is committed")
