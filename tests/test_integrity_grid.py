"""T02 (e): run_all must not quietly freeze a partial grid, and an extension must not write
over the primary analysis.

run_all skips a step whose output file exists, and one ladder or faithfulness file covers every
level of a (model, seed). Run it while only `random` is trained and the file is written with
`random` alone; when `cold_drug` lands the file is skipped and the level is never scored.
Nothing complains. It was caught on 2026-09-15 with KIBA at 8 of 18 cells, before it bit.
"""
import json
import os
import sys

import pytest

from src.evaluation import integrity, run_all
from src.evaluation.integrity import (GRID_STATE_FILE, check_extension_dir,
                                      check_grid_unchanged, grid_fingerprint,
                                      is_scratch_dir, read_grid_state, write_grid_state)
from tests.test_run_all import _cell


@pytest.fixture
def not_scratch(monkeypatch):
    """pytest's tmp_path lives under the system temp dir, which is scratch by design.
    Make nothing count as scratch so the guard's refusal can be observed."""
    monkeypatch.setattr(integrity, "is_scratch_dir", lambda path: False)


def _state(*complete):
    return {c: "complete" for c in complete}


A = ("moltrans", "random", 1)
B = ("moltrans", "cold_drug", 1)


# --------------------------------------------------------------------- the fingerprint

def test_the_fingerprint_lists_only_complete_cells_and_ignores_order():
    state = {A: "complete", B: "interrupted", ("moltrans", "cold_pair", 1): "missing"}
    assert grid_fingerprint(state) == ["moltrans|random|1"]
    assert grid_fingerprint({B: "complete", A: "complete"}) == \
        grid_fingerprint({A: "complete", B: "complete"})


def test_the_state_round_trips_and_is_absent_before_a_first_run(tmp_path):
    assert read_grid_state(str(tmp_path)) is None
    write_grid_state(str(tmp_path), _state(A), dataset="davis")
    assert read_grid_state(str(tmp_path)) == ["moltrans|random|1"]


# ----------------------------------------------------------------------------- the guard

def test_a_first_run_is_always_allowed(tmp_path, not_scratch):
    check_grid_unchanged(str(tmp_path), _state(A), skip_existing=True)


def test_the_same_grid_may_be_resumed(tmp_path, not_scratch):
    """A Colab session that dropped halfway is re-run on the same grid: that is what
    skip-existing is for, and it must stay possible."""
    write_grid_state(str(tmp_path), _state(A, B))
    check_grid_unchanged(str(tmp_path), _state(A, B), skip_existing=True)


def test_a_grown_grid_is_refused_because_old_files_would_be_skipped(tmp_path, not_scratch):
    write_grid_state(str(tmp_path), _state(A))
    with pytest.raises(SystemExit) as exc:
        check_grid_unchanged(str(tmp_path), _state(A, B), skip_existing=True)
    message = str(exc.value)
    assert "moltrans|cold_drug|1" in message and "never scored" in message
    assert "--no-skip-existing" in message


def test_a_shrunk_grid_is_refused_too(tmp_path, not_scratch):
    write_grid_state(str(tmp_path), _state(A, B))
    with pytest.raises(SystemExit):
        check_grid_unchanged(str(tmp_path), _state(A), skip_existing=True)


def test_recomputing_everything_is_the_sanctioned_way_through(tmp_path, not_scratch):
    write_grid_state(str(tmp_path), _state(A))
    check_grid_unchanged(str(tmp_path), _state(A, B), skip_existing=False)


def test_an_explicit_override_is_honoured(tmp_path, not_scratch):
    write_grid_state(str(tmp_path), _state(A))
    check_grid_unchanged(str(tmp_path), _state(A, B), skip_existing=True, allow_partial=True)


def test_a_scratch_directory_is_exempt(tmp_path):
    """Probing a half-trained grid is legitimate when the answer cannot be mistaken for the
    paper's. tmp_path is under the system temp dir, which counts as scratch."""
    write_grid_state(str(tmp_path), _state(A))
    check_grid_unchanged(str(tmp_path), _state(A, B), skip_existing=True)


def test_what_counts_as_scratch():
    """Relative paths resolve against the working directory, which for the suite is the repo
    root (not under the system temp dir)."""
    assert is_scratch_dir("/tmp/anything")
    assert is_scratch_dir("results/scratch_kiba_probe")
    assert is_scratch_dir("results/tmp")
    assert not is_scratch_dir("results/analysis_davis_policyA")
    assert not is_scratch_dir("results/analysis_kiba_v2")


# -------------------------------------------------------------- through run_all.main itself

def _run_all_argv(checkpoints, out_dir, *extra):
    return ["run_all", "--dataset", "davis", "--models", "moltrans", "--seeds", "1",
            "--checkpoint-dir", str(checkpoints), "--out-dir", str(out_dir),
            "--steps", "inventory", "--device", "cpu", *extra]


def test_run_all_refuses_to_top_up_a_partial_grid_in_a_real_output_dir(
        tmp_path, monkeypatch, not_scratch):
    checkpoints, out = tmp_path / "ckpt", tmp_path / "results" / "analysis_davis"
    checkpoints.mkdir()
    _cell(checkpoints, "moltrans", "random", 1, auroc=0.9)
    monkeypatch.setattr(sys, "argv", _run_all_argv(checkpoints, out))
    run_all.main()
    assert read_grid_state(str(out)) == ["moltrans|random|1"]

    _cell(checkpoints, "moltrans", "cold_drug", 1, auroc=0.7)       # the grid grew
    monkeypatch.setattr(sys, "argv", _run_all_argv(checkpoints, out))
    with pytest.raises(SystemExit, match="never scored"):
        run_all.main()
    assert read_grid_state(str(out)) == ["moltrans|random|1"], \
        "a refused run must not overwrite the record it was refused against"

    monkeypatch.setattr(sys, "argv", _run_all_argv(checkpoints, out, "--no-skip-existing"))
    run_all.main()                                                   # the sanctioned route
    assert read_grid_state(str(out)) == ["moltrans|cold_drug|1", "moltrans|random|1"]


def test_a_dry_run_neither_refuses_nor_records(tmp_path, monkeypatch, not_scratch):
    checkpoints, out = tmp_path / "ckpt", tmp_path / "out"
    checkpoints.mkdir()
    _cell(checkpoints, "moltrans", "random", 1, auroc=0.9)
    monkeypatch.setattr(sys, "argv", _run_all_argv(checkpoints, out, "--dry-run"))
    run_all.main()
    assert not os.path.exists(os.path.join(out, GRID_STATE_FILE))


# ----------------------------------------------------------------------------- extensions

def test_an_extension_may_not_write_into_a_directory_that_holds_results(tmp_path):
    out = tmp_path / "analysis_davis_policyA"
    out.mkdir()
    (out / "audit_davis_binary.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        check_extension_dir(str(out))
    assert "byte-for-byte" in str(exc.value) and "audit_davis_binary.json" in str(exc.value)


def test_an_extension_may_use_a_new_or_empty_directory(tmp_path):
    check_extension_dir(str(tmp_path / "does_not_exist_yet"))
    (tmp_path / "empty").mkdir()
    check_extension_dir(str(tmp_path / "empty"))


def test_only_the_grid_record_does_not_count_as_an_output(tmp_path):
    (tmp_path / GRID_STATE_FILE).write_text(json.dumps({"complete_cells": []}))
    check_extension_dir(str(tmp_path))


@pytest.mark.parametrize("directory", [
    "results/analysis_davis_policyA", "results/analysis_kiba_policyA",
    "results/analysis_davis_policyA_klifs", "results/analysis_kiba_policyA_klifs",
    "results/analysis_davis_klifs"])
def test_every_primary_result_directory_is_refused_as_an_extension_target(directory):
    """The directories the paper's numbers come from, as they exist on disk."""
    if not os.path.isdir(directory):
        pytest.skip(f"{directory} not present")
    with pytest.raises(SystemExit):
        check_extension_dir(directory)


def test_run_all_wires_the_extension_flag(tmp_path, monkeypatch):
    checkpoints, out = tmp_path / "ckpt", tmp_path / "primary"
    checkpoints.mkdir(); out.mkdir()
    _cell(checkpoints, "moltrans", "random", 1, auroc=0.9)
    (out / "faithfulness_davis_seed1.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", _run_all_argv(checkpoints, out, "--extension"))
    with pytest.raises(SystemExit, match="own directory"):
        run_all.main()
    monkeypatch.setattr(sys, "argv", _run_all_argv(checkpoints, tmp_path / "v2", "--extension"))
    run_all.main()
    assert os.path.exists(tmp_path / "v2" / GRID_STATE_FILE)
    assert (out / "faithfulness_davis_seed1.json").read_text() == "{}", \
        "the primary directory was touched"
