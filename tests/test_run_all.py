"""The one-command analysis must wire each runner to the right files.

The expensive mistake here is quiet: MolTrans's ladder drawn against
ColdSite-DTI's accuracy, a seed that was never trained read as a result, or a
step skipped because a stale output was lying around. These pin the wiring
and the summary arithmetic; the runners themselves are tested elsewhere.
"""
import json
import os

import pytest

from src.evaluation import run_all
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag


def _cell(tmp, model, level, seed, auroc=None, checkpoint=True):
    if checkpoint:
        open(checkpoint_path(str(tmp), "davis", level, "binary", seed, model=model), "wb").close()
    if auroc is not None:
        with open(results_path(str(tmp), run_tag("davis", level, "binary", seed), model=model), "w") as f:
            json.dump({"test_metrics": {"auroc": auroc}}, f)


def _cfg(tmp, **over):
    cfg = {"dataset": "davis", "models": ["deepdta", "coldsite_dti", "hyperattentiondti", "moltrans"],
           "seeds": [1, 2, 3], "checkpoint_dir": str(tmp), "results_dir": str(tmp),
           "split_root": "data/splits", "ground_truth": "gt.json",
           "out_dir": str(tmp / "out"), "max_pairs": 200, "device": "cpu",
           "volume_control_dir": None}
    cfg.update(over)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    return cfg


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------

def test_a_checkpoint_without_results_is_interrupted_not_complete(tmp_path):
    _cell(tmp_path, "moltrans", "random", 1, auroc=0.8)
    _cell(tmp_path, "moltrans", "random", 2)                    # cut off
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", ["moltrans"], [1, 2, 3])
    assert state[("moltrans", "random", 1)] == "complete"
    assert state[("moltrans", "random", 2)] == "interrupted"
    assert state[("moltrans", "random", 3)] == "missing"


# --------------------------------------------------------------------------
# the commands
# --------------------------------------------------------------------------

def test_every_runner_is_told_the_task_is_binary(tmp_path):
    """run_faithfulness and run_ladder default to regression; forgetting the
    flag looks for checkpoints named ..._regression_... and finds none."""
    _cell(tmp_path, "coldsite_dti", "random", 1, auroc=0.9)
    cfg = _cfg(tmp_path)
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    for step in ("faithfulness", "ladder", "audit", "control"):
        for _label, cmd, _out in run_all.commands(step, cfg, state):
            assert cmd[cmd.index("--task") + 1] == "binary", (step, cmd)


def test_each_ladder_reads_its_own_models_accuracy(tmp_path):
    for model in ("coldsite_dti", "moltrans"):
        _cell(tmp_path, model, "random", 1, auroc=0.8)
    cfg = _cfg(tmp_path)
    for model in ("coldsite_dti", "moltrans"):
        tag = run_all.output_tag(model, "davis", 1)
        open(os.path.join(cfg["out_dir"], f"accuracy_{tag}.json"), "w").write("{}")
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    for label, cmd, _out in run_all.commands("ladder", cfg, state):
        model = cmd[cmd.index("--model") + 1]
        accuracy = cmd[cmd.index("--accuracy-json") + 1]
        assert os.path.basename(accuracy) == f"accuracy_{run_all.output_tag(model, 'davis', 1)}.json"


def test_untrained_models_and_seeds_get_no_commands(tmp_path):
    _cell(tmp_path, "coldsite_dti", "random", 2, auroc=0.9)
    cfg = _cfg(tmp_path)
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    labels = [label for label, _c, _o in run_all.commands("faithfulness", cfg, state)]
    assert labels == ["faithfulness coldsite_dti seed 2"]


def test_deepdta_is_never_audited(tmp_path):
    _cell(tmp_path, "deepdta", "random", 1, auroc=0.93)
    _cell(tmp_path, "coldsite_dti", "random", 1, auroc=0.9)
    cfg = _cfg(tmp_path)
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    for step in ("faithfulness", "ladder", "control"):
        for _label, cmd, _out in run_all.commands(step, cfg, state):
            assert cmd[cmd.index("--model") + 1] != "deepdta", step
    (_l, audit_cmd, _o), = run_all.commands("audit", cfg, state)
    assert audit_cmd[audit_cmd.index("--models") + 1] == "coldsite_dti,uniform_control"


def test_control_runs_both_ion_settings(tmp_path):
    _cell(tmp_path, "hyperattentiondti", "random", 1, auroc=0.8)
    cfg = _cfg(tmp_path)
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    outputs = sorted(os.path.basename(o) for _l, _c, o in run_all.commands("control", cfg, state))
    assert outputs == ["control_hyperattentiondti_davis_seed1.json",
                       "control_hyperattentiondti_davis_seed1_noions.json"]


def test_an_existing_output_is_not_recomputed(tmp_path):
    done = tmp_path / "out.json"
    done.write_text("{}")
    log = str(tmp_path / "log")
    assert run_all.run_job("x", ["false"], str(done), True, log) == "skipped"
    assert run_all.run_job("x", ["false"], str(done), False, log) == "failed"


def test_a_command_that_writes_nothing_counts_as_failed(tmp_path):
    assert run_all.run_job("x", ["true"], str(tmp_path / "never.json"), True,
                           str(tmp_path / "log")) == "failed"


# --------------------------------------------------------------------------
# the summary page
# --------------------------------------------------------------------------

def test_summary_aggregates_seeds_and_carries_the_holm_verdict(tmp_path):
    for seed, auroc in ((1, 0.80), (2, 0.84), (3, 0.82)):
        _cell(tmp_path, "coldsite_dti", "random", seed, auroc=auroc)
    cfg = _cfg(tmp_path)
    out = cfg["out_dir"]
    for seed, p in ((1, 0.04), (2, 0.05), (3, 0.03)):
        tag = run_all.output_tag("coldsite_dti", "davis", seed)
        json.dump({"random": {"by_k": {"10": {"precision_at_k": p, "normalised": p,
                                              "chance": 0.02, "ceiling": 0.99,
                                              "n_evaluated": 402}}}},
                  open(os.path.join(out, f"ladder_{tag}.json"), "w"))
        json.dump({"levels": {"random": {"comprehensiveness_delta": 0.07,
                                         "explanation_is_load_bearing": True}}},
                  open(os.path.join(out, f"faithfulness_{tag}.json"), "w"))
    json.dump({"p_values_corrected": {"coldsite_dti|davis|random":
                                      {"p_value": 0.001, "significant": True}}},
              open(os.path.join(out, "audit_davis_binary.json"), "w"))

    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    page = run_all.summary(cfg, state)
    assert "0.820 ± 0.020" in page                       # AUROC over three seeds
    assert "0.040 ± 0.010" in page                       # precision@10
    assert "yes (p = 0.001)" in page                     # Holm verdict
    assert "3/3 > 0" in page                             # load-bearing seeds


def test_summary_flags_a_cell_with_too_few_seeds(tmp_path):
    _cell(tmp_path, "moltrans", "cold_pair", 1, auroc=0.7)
    cfg = _cfg(tmp_path)
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    assert "0.700 (1 seed)" in run_all.summary(cfg, state)


def test_summary_reads_the_volume_control_against_the_grid(tmp_path):
    for seed in (1, 2, 3):
        _cell(tmp_path, "coldsite_dti", "random", seed, auroc=0.93)
        _cell(tmp_path, "coldsite_dti", "cold_pair", seed, auroc=0.73)
    vc = tmp_path / "vc"
    vc.mkdir()
    for seed, auroc in ((1, 0.88), (2, 0.90), (3, 0.89)):
        json.dump({"test_metrics": {"auroc": auroc}},
                  open(vc / f"davis_random_binary_seed{seed}_trainsub15190_results.json", "w"))
    cfg = _cfg(tmp_path, volume_control_dir=str(vc))
    state = run_all.inventory(str(tmp_path), str(tmp_path), "davis", cfg["models"], cfg["seeds"])
    page = run_all.summary(cfg, state)
    assert "Cost of fewer rows alone: +0.040" in page
    assert "Genuine cold-pair difficulty: +0.160" in page


def test_a_failed_positive_control_is_shouted_in_the_summary(tmp_path):
    cfg = _cfg(tmp_path)
    json.dump({"verdicts": [[False, True, "random: 3 site(s) outside"]]},
              open(os.path.join(cfg["out_dir"], "positive_control_davis.json"), "w"))
    page = run_all.summary(cfg, run_all.inventory(str(tmp_path), str(tmp_path), "davis",
                                                  cfg["models"], cfg["seeds"]))
    assert "HARD CHECK FAILED" in page


def test_summary_never_calls_an_unreadable_curve_chance(tmp_path):
    cfg = _cfg(tmp_path)
    json.dump({"verdicts": [], "compare": {"moltrans_seed1": {"cold_pair": {
        "precision_at_k": 0.067, "equivalent_dose": None,
        "reason": "curve not monotone (too few proteins to read)"}}}},
        open(os.path.join(cfg["out_dir"], "positive_control_davis.json"), "w"))
    page = run_all.summary(cfg, run_all.inventory(str(tmp_path), str(tmp_path), "davis",
                                                  cfg["models"], cfg["seeds"]))
    assert "curve not monotone" in page
    assert "at or below chance" not in page
