"""Tests for the run-tag / checkpoint-filename contract.

This is a cross-track contract: Track B writes these names, Track C reads them.
The failure mode it guards is silent in both directions -- a writer that omits
the seed overwrites its own runs, and a reader that guesses the wrong spelling
reports "no checkpoint" and drops the level from the ladder.
"""
import pytest

from src.model.checkpoint_naming import (
    checkpoint_name,
    checkpoint_path,
    discover_checkpoints,
    history_path,
    parse_run_tag,
    results_path,
    run_tag,
)


# --------------------------------------------------------------------------
# the bug this module exists to fix
# --------------------------------------------------------------------------

def test_different_seeds_produce_different_checkpoint_paths():
    """The regression. Three seeds per cell must be three files, not one."""
    paths = {checkpoint_path("results", "davis", "cold_target", "regression", s)
             for s in (1, 2, 3)}
    assert len(paths) == 3, paths


def test_different_seeds_produce_different_results_files():
    """The accuracy hand-off to Track C collides the same way if it is unseeded."""
    paths = {results_path("results", run_tag("davis", "cold_target", "regression", s))
             for s in (1, 2, 3)}
    assert len(paths) == 3, paths


def test_the_seed_is_mandatory():
    with pytest.raises(ValueError, match="seed is required"):
        run_tag("davis", "cold_target", "regression", seed=None)


def test_splits_do_not_collide_with_each_other():
    """The most expensive mistake in the project: a cold-target number from a
    model trained on cold-drug."""
    paths = {checkpoint_path("results", "davis", split, "regression", 1)
             for split in ("random", "cold_drug", "cold_target", "cold_pair")}
    assert len(paths) == 4


def test_datasets_and_tasks_do_not_collide():
    assert (checkpoint_name("davis", "random", "regression", 1)
            != checkpoint_name("kiba", "random", "regression", 1))
    assert (checkpoint_name("davis", "random", "regression", 1)
            != checkpoint_name("davis", "random", "binary", 1))


# --------------------------------------------------------------------------
# round trip -- what lets Track C read a directory instead of being told
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", ["davis", "kiba", "antiviral"])
@pytest.mark.parametrize("split", ["random", "cold_drug", "cold_target", "cold_pair"])
@pytest.mark.parametrize("task", ["regression", "binary"])
def test_run_tag_round_trips(dataset, split, task):
    parsed = parse_run_tag(run_tag(dataset, split, task, seed=7))
    assert parsed == {"dataset": dataset, "split": split, "task": task,
                      "seed": 7, "split_seed": None}


def test_round_trip_survives_an_optional_split_seed():
    tag = run_tag("davis", "cold_pair", "regression", seed=3, split_seed=11)
    assert parse_run_tag(tag) == {"dataset": "davis", "split": "cold_pair",
                                  "task": "regression", "seed": 3,
                                  "split_seed": 11}


def test_an_unseeded_tag_is_rejected_rather_than_assumed_to_be_seed_one():
    """A pre-seed artefact cannot be attributed to a run. Guessing turns one
    run into an 'n_seeds = 3' claim."""
    with pytest.raises(ValueError, match="not a run tag"):
        parse_run_tag("davis_cold_target_regression")


def test_underscores_in_dataset_or_task_are_refused():
    """They would make the tag ambiguous to parse back."""
    with pytest.raises(ValueError, match="must not contain"):
        run_tag("davis_v2", "random", "regression", 1)
    with pytest.raises(ValueError, match="must not contain"):
        run_tag("davis", "random", "multi_task", 1)


# --------------------------------------------------------------------------
# writer / reader agreement, and the helpers
# --------------------------------------------------------------------------

def test_history_sits_beside_its_checkpoint():
    path = checkpoint_path("results", "davis", "random", "regression", 2)
    assert history_path(path) == "results/coldsite_dti_davis_random_regression_seed2_history.json"


def test_history_path_refuses_a_non_checkpoint():
    with pytest.raises(ValueError):
        history_path("results/coldsite_dti_davis_random_regression_seed2.json")


def test_discover_checkpoints_finds_every_seed(tmp_path):
    for seed in (3, 1, 2):
        (tmp_path / checkpoint_name("davis", "cold_target", "regression", seed)).touch()
    (tmp_path / checkpoint_name("kiba", "cold_target", "regression", 1)).touch()

    found = discover_checkpoints(str(tmp_path), dataset="davis")
    assert [entry["seed"] for entry in found] == [1, 2, 3]
    assert all(entry["dataset"] == "davis" for entry in found)


def test_discover_checkpoints_filters_by_split(tmp_path):
    (tmp_path / checkpoint_name("davis", "cold_drug", "regression", 1)).touch()
    (tmp_path / checkpoint_name("davis", "cold_target", "regression", 1)).touch()
    found = discover_checkpoints(str(tmp_path), dataset="davis", split="cold_target")
    assert len(found) == 1 and found[0]["split"] == "cold_target"


def test_discover_checkpoints_ignores_unseeded_legacy_files(tmp_path):
    """Left out, not guessed at: aggregate.py then flags the cell as
    under-powered, which is the honest outcome."""
    (tmp_path / "coldsite_dti_davis_cold_target_regression.pt").touch()
    (tmp_path / "coldsite_dti_best.pt").touch()
    assert discover_checkpoints(str(tmp_path)) == []


def test_the_trainer_and_the_ladder_build_the_same_path():
    """The whole point of the module. If these two ever disagree the ladder
    silently skips a level that has in fact been trained."""
    from src.evaluation import run_ladder
    from src.model import train

    assert train.build_checkpoint_path is run_ladder.build_checkpoint_path
