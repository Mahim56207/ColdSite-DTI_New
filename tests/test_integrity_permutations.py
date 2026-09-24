"""T02 (d): a permutation test's resolution must sit well below the bar it is tested against.

The DAVIS audit ran at 500 permutations. Its smallest reportable p was 1/501 = 0.001996; the
strictest Holm threshold over its 20 cells was 0.05/20 = 0.0025. The test could resolve a
verdict only 25% wider than its own floor. The 10,000-permutation rerun changed no verdict, but
the primary file was the 500 one, and nothing in the code stopped anyone producing another.
"""
import importlib
import json
import os
import sys

import pytest

from src.evaluation import integrity
from src.evaluation.integrity import (MIN_PERMUTATIONS, check_permutations, min_achievable_p,
                                      permutations_needed, smallest_holm_threshold)

RUNNERS = ["run_audit", "run_ladder", "run_control", "positional_control", "positive_control"]

# Arguments a runner needs beyond --n-trials just to get past argparse.
REQUIRED = {"positional_control": ["--checkpoint-dir", "x"]}


def _argv(runner, *extra):
    return [runner, *REQUIRED.get(runner, []), *extra]


# The pre-specified families (docs/inventory.md §8, read from the committed audit files).
DAVIS_AUDIT_CELLS = 20
KIBA_AUDIT_CELLS = 8


class _Stop(Exception):
    pass


# ------------------------------------------------------------------ the arithmetic

def test_the_floor_is_one_over_n_plus_one():
    assert min_achievable_p(500) == pytest.approx(0.001996007984031936)
    assert min_achievable_p(10_000) == pytest.approx(9.999000099990002e-05)


def test_500_permutations_barely_clear_the_davis_family_and_10000_clear_it_easily():
    threshold = smallest_holm_threshold(DAVIS_AUDIT_CELLS)
    assert threshold == pytest.approx(0.0025)
    assert min_achievable_p(500) / threshold > 0.79        # the floor was 80% of the bar
    assert min_achievable_p(MIN_PERMUTATIONS) / threshold < 0.05


@pytest.mark.parametrize("cells", [DAVIS_AUDIT_CELLS, KIBA_AUDIT_CELLS])
def test_the_floor_is_below_the_strictest_holm_threshold_of_each_primary_family(cells):
    assert min_achievable_p(MIN_PERMUTATIONS) < smallest_holm_threshold(cells)


@pytest.mark.parametrize("cells", [1, 8, 20, 100, 400])
def test_permutations_needed_is_the_smallest_sufficient_count(cells):
    n = permutations_needed(cells)
    assert min_achievable_p(n) < smallest_holm_threshold(cells)
    assert min_achievable_p(n - 1) >= smallest_holm_threshold(cells)


def test_the_default_floor_supports_families_up_to_the_stated_limit():
    """10,000 permutations resolve any family below 500 cells. An extension that grows the
    family past that must raise the count, and the guard says so."""
    assert permutations_needed(499) <= MIN_PERMUTATIONS
    assert permutations_needed(501) > MIN_PERMUTATIONS
    with pytest.raises(SystemExit, match="cannot resolve this family"):
        check_permutations(MIN_PERMUTATIONS, family_size=600)


# ------------------------------------------------------------------------ the guard

def test_a_low_count_is_refused_and_says_why():
    with pytest.raises(SystemExit) as exc:
        check_permutations(500, context="audit grid")
    message = str(exc.value)
    assert "500" in message and "10000" in message and "0.001996" in message
    assert "audit grid" in message


def test_the_floor_itself_and_above_pass():
    check_permutations(MIN_PERMUTATIONS)
    check_permutations(50_000, family_size=DAVIS_AUDIT_CELLS)


def test_the_family_check_bites_even_above_the_flat_floor():
    """A flat 10,000 is not enough for every family: the bar moves with the family size."""
    with pytest.raises(SystemExit, match="Holm's strictest threshold"):
        check_permutations(MIN_PERMUTATIONS, family_size=1000)


def test_the_opt_out_exists_only_to_reproduce_old_outputs():
    check_permutations(500, allow_low=True)
    check_permutations(500, family_size=DAVIS_AUDIT_CELLS, allow_low=True)


# ------------------------------------------------------- every runner is actually gated

@pytest.mark.parametrize("runner", RUNNERS)
def test_every_runner_defaults_to_the_floor_and_reaches_the_guard(runner, monkeypatch):
    """No --n-trials given: the value the runner hands to the guard is the floor."""
    module = importlib.import_module(f"src.evaluation.{runner}")
    seen = {}

    def capture(n_trials, **kwargs):
        seen["n"], seen["kwargs"] = n_trials, kwargs
        raise _Stop

    monkeypatch.setattr(module, "check_permutations", capture)
    monkeypatch.setattr(sys, "argv", _argv(runner))
    with pytest.raises(_Stop):
        module.main()
    assert seen["n"] == MIN_PERMUTATIONS
    assert seen["kwargs"].get("allow_low") is False


@pytest.mark.parametrize("runner", RUNNERS)
def test_every_runner_refuses_a_low_count_on_the_command_line(runner, monkeypatch):
    module = importlib.import_module(f"src.evaluation.{runner}")
    monkeypatch.setattr(sys, "argv", _argv(runner, "--n-trials", "500"))
    with pytest.raises(SystemExit, match="below the floor"):
        module.main()


@pytest.mark.parametrize("runner", RUNNERS)
def test_every_runner_lets_a_low_count_through_only_when_told_to(runner, monkeypatch):
    module = importlib.import_module(f"src.evaluation.{runner}")
    seen = {}

    def capture(n_trials, **kwargs):
        seen.update(n=n_trials, **kwargs)
        raise _Stop

    monkeypatch.setattr(module, "check_permutations", capture)
    monkeypatch.setattr(sys, "argv", _argv(runner, "--n-trials", "500", "--allow-low-permutations"))
    with pytest.raises(_Stop):
        module.main()
    assert seen["n"] == 500 and seen["allow_low"] is True


def test_the_audit_checks_its_own_family_size(monkeypatch, tmp_path):
    """run_audit knows how many cells it will Holm-correct, so it asks the sharper question."""
    from src.evaluation import run_audit
    seen = {}

    def capture(n_trials, **kwargs):
        if "family_size" in kwargs:
            seen.update(kwargs)
            raise _Stop

    monkeypatch.setattr(run_audit, "check_permutations", capture)
    monkeypatch.setattr(run_audit, "build_grid",
                        lambda *a, **k: {"p_values_raw": {f"c{i}": 0.5 for i in range(20)}})
    monkeypatch.setattr(sys, "argv", ["run_audit", "--dummy", "--out-dir", str(tmp_path)])
    with pytest.raises(_Stop):
        run_audit.main()
    assert seen["family_size"] == 20


def test_the_audit_records_what_it_ran_at():
    from src.evaluation.run_audit import build_grid
    result = build_grid(lambda *a: ([[0.0] * 30] * 3, [{1, 2}] * 3, ["A", "B", "C"]),
                        ["coldsite_dti"], ["davis"], [1], k=5, n_trials=50)
    assert result["n_trials"] == 50
    assert result["min_achievable_p"] == pytest.approx(min_achievable_p(50))


# -------------------------------------------------- the committed audits, read as data

@pytest.mark.parametrize("path,cells", [
    ("results/analysis_davis_policyA/audit_davis_binary_10k_permutations.json", 20),
    ("results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.json", 8),
])
def test_the_committed_10k_audits_have_the_expected_family_and_resolve_it(path, cells):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    with open(path) as handle:
        audit = json.load(handle)
    p_values = audit["p_values_raw"]
    assert len(p_values) == cells
    assert min(p_values.values()) >= min_achievable_p(MIN_PERMUTATIONS) - 1e-12
    assert min_achievable_p(MIN_PERMUTATIONS) < smallest_holm_threshold(len(p_values))
