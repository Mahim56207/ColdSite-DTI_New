"""The amended epoch-1 gate rule (src/cloud/gate_rule_v2.py): calibrated against a healthy process, still catches what the gate is for,
and re-evaluates the recorded runs without changing which harness the verdicts are bound to.
"""
import json
import math
import os
import statistics

import numpy as np
import pytest
from scipy import stats

from src.cloud import canary, epoch_gate, gate_rule_v2, markers
from src.cloud.config import Cell
from src.model import resume

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = epoch_gate.load_reference(os.path.join(ROOT, "config", "epoch1_reference.json"))["by_seed"]


def _new(**runs):
    return {int(s.lstrip("s")): {**m, "initial_weight_hash": f"hash-{s}"} for s, m in runs.items()}


REPLAY = {"train_loss": 0.2902, "val_loss": 0.2199, "auroc": 0.7926}                 # the recorded gate run, seed 1
SEED4 = {"train_loss": 0.2855, "val_loss": 0.2328, "auroc": 0.7959}                  # recorded: original rule FAILED on val_loss
SEED5 = {"train_loss": 0.2895, "val_loss": 0.2142, "auroc": 0.8059}


def test_the_prediction_interval_is_the_textbook_formula():
    values = [REF[s]["val_loss"] for s in (1, 2, 3)]
    low, high, q, sd = gate_rule_v2.prediction_interval(values, m_checks=1)
    mean = statistics.fmean(values)
    half = stats.t.ppf(0.975, 2) * statistics.stdev(values) * math.sqrt(1 + 1 / 3)
    assert (low, high) == pytest.approx((mean - half, mean + half)) and q == pytest.approx(4.3027, abs=1e-3)
    assert sd == pytest.approx(statistics.stdev(values))


def test_more_checks_widen_the_interval_family_wise():
    values = [REF[s]["auroc"] for s in (1, 2, 3)]
    widths = [gate_rule_v2.prediction_interval(values, m)[1] - gate_rule_v2.prediction_interval(values, m)[0] for m in (1, 3, 6, 12)]
    assert widths == sorted(widths) and widths[0] < widths[-1]
    low6, high6, q6, _ = gate_rule_v2.prediction_interval(values, 6)
    assert q6 == pytest.approx(stats.t.ppf(1 - 0.05 / 12, 2))


def test_a_healthy_process_almost_never_fails_the_amended_gate_but_often_failed_the_original():
    """The calibration the original rule never had. Healthy = every epoch-1 value drawn from one normal process."""
    rng = np.random.default_rng(1)
    N, n_new, n_metrics = 20_000, 2, 3
    q = stats.t.ppf(1 - 0.05 / (2 * n_new * n_metrics), 2)
    amended_fail = original_fail = 0
    committed = rng.standard_normal((N, n_metrics, 3))
    new = rng.standard_normal((N, n_metrics, n_new))
    mean, sd = committed.mean(2), committed.std(2, ddof=1)
    half = q * sd * math.sqrt(1 + 1 / 3)
    out_amended = (np.abs(new - mean[:, :, None]) > half[:, :, None]).any(axis=(1, 2))
    lo, hi = committed.min(2) - sd, committed.max(2) + sd
    out_original = ((new < lo[:, :, None]) | (new > hi[:, :, None])).any(axis=(1, 2))
    assert out_amended.mean() <= 0.06                       # designed for 5 % family-wise
    assert out_original.mean() >= 0.5                       # the flaw: the original rule failed a healthy harness most of the time


def test_the_recorded_runs_pass_the_amended_rule_and_failed_the_original():
    out = gate_rule_v2.evaluate_v2(_new(s1=REPLAY, s4=SEED4, s5=SEED5), REF, 1, [4, 5])
    assert out["verdict"] == "pass" and out["original_rule"]["verdict"] == "fail"
    assert out["original_rule"]["gate_b"] == {"4": "fail", "5": "pass"}
    assert out["n_envelope_checks"] == 6 and out["gate_a"]["verdict"] == "pass"


def test_the_sensitivity_to_the_family_wise_adjustment_is_recorded_not_hidden():
    out = gate_rule_v2.evaluate_v2(_new(s1=REPLAY, s4=SEED4, s5=SEED5), REF, 1, [4, 5])
    row = out["gate_b"]["4"]["val_loss"]
    assert row["verdict"] == "pass" and row["inside_unadjusted_95"] is False          # only the adjustment brings it inside
    assert row["unadjusted_95_interval"][1] < row["new"] < row["interval"][1]


def test_gross_failures_still_fail_the_amended_rule():
    ten_times = {**SEED4, "val_loss": 2.328}
    assert gate_rule_v2.evaluate_v2(_new(s1=REPLAY, s4=ten_times, s5=SEED5), REF, 1, [4, 5])["gate_b"]["4"]["val_loss"]["verdict"] == "fail"
    nan = {**SEED4, "train_loss": float("nan")}
    out = gate_rule_v2.evaluate_v2(_new(s1=REPLAY, s4=nan, s5=SEED5), REF, 1, [4, 5])
    assert out["gate_b"]["4"]["train_loss"]["verdict"] == "fail" and out["verdict"] == "fail"
    collapsed = {**SEED4, "auroc": 0.5}
    assert gate_rule_v2.evaluate_v2(_new(s1=REPLAY, s4=collapsed, s5=SEED5), REF, 1, [4, 5])["gate_b"]["4"]["auroc"]["verdict"] == "fail"


def test_seeds_that_are_one_run_still_fail_distinctness_under_the_amended_rule():
    same_metrics = _new(s1=REPLAY, s4=REPLAY, s5=SEED5)
    out = gate_rule_v2.evaluate_v2(same_metrics, REF, 1, [4, 5])
    assert out["gate_b"]["4"]["distinct"]["verdict"] == "fail" and out["verdict"] == "fail"
    same_init = _new(s1=REPLAY, s4=SEED4, s5=SEED5)
    same_init[4]["initial_weight_hash"] = same_init[1]["initial_weight_hash"]
    assert gate_rule_v2.evaluate_v2(same_init, REF, 1, [4, 5])["verdict"] == "fail"


def test_gate_a_still_rejects_a_gross_deviation():
    off = {**REPLAY, "train_loss": REPLAY["train_loss"] + 0.02}
    out = gate_rule_v2.evaluate_v2(_new(s1=off, s4=SEED4, s5=SEED5), REF, 1, [4, 5])
    assert out["gate_a"]["train_loss"]["verdict"] == "fail" and out["verdict"] == "fail"


def test_the_family_size_follows_the_number_of_new_seeds():
    one = gate_rule_v2.evaluate_v2(_new(s1=REPLAY, s4=SEED4), REF, 1, [4])
    two = gate_rule_v2.evaluate_v2(_new(s1=REPLAY, s4=SEED4, s5=SEED5), REF, 1, [4, 5])
    assert one["n_envelope_checks"] == 3 and two["n_envelope_checks"] == 6
    assert one["gate_b"]["4"]["val_loss"]["interval"][1] < two["gate_b"]["4"]["val_loss"]["interval"][1]


def test_the_cli_re_evaluates_recorded_runs_and_binds_the_verdict_to_the_unchanged_harness(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    root = str(tmp_path / "epoch_gate")
    for seed, row in ((1, REPLAY), (4, SEED4), (5, SEED5)):
        cell = Cell("davis", "coldsite_dti", "cold_target", seed)
        p = markers.paths(root, cell)
        os.makedirs(p["dir"], exist_ok=True)
        resume.save(p["resume"], {"epoch": 1, "extra": {"history": [{"epoch": 1, **row, "auprc": 0.3, "accuracy": 0.93}]}})
        with open(p["start"], "w") as handle:
            json.dump({"seed": seed, "initial_weight_hash": f"w{seed}"}, handle)
    out_path = str(tmp_path / "verdict.json")
    assert gate_rule_v2.main(["--evaluate", "--seeds", "1", "4", "5", "--root", root, "--out", out_path]) == 0
    verdict = json.load(open(out_path))
    assert verdict["verdict"] == "pass" and verdict["seeds_checked"] == [4, 5] and verdict["original_rule"]["verdict"] == "fail"
    assert verdict["harness_sha256"] == canary.harness_hash() and verdict["rule"].startswith("v2")


def test_the_amended_module_is_not_part_of_the_hash_the_canary_is_bound_to():
    """Editing it must not stale the canary's verdict; the hashed files are exactly the ones that were canaried."""
    assert "src/cloud/gate_rule_v2.py" not in canary.HARNESS_FILES
    path = os.path.join(ROOT, "config", "canary_verdict.json")
    if not os.path.exists(path):
        pytest.skip("no committed canary verdict yet")
    assert json.load(open(path))["harness_sha256"] == canary.harness_hash(), "a hashed harness file changed after the canary passed"


# ---- Gate A, widened to 4 committed-seed SDs before the re-run (2026-10-17) ---------------------------------------------------------

def _sd(metric):
    return statistics.stdev([REF[s][metric] for s in (1, 2, 3)])


def _replay(**shift):
    base = {m: REF[1][m] for m in ("train_loss", "val_loss", "auroc")}
    return {m: base[m] + shift.get(m, 0.0) for m in base}


def test_gate_a_tolerance_is_four_committed_seed_sds_to_the_edge():
    assert gate_rule_v2.GATE_A_SDS == 4
    for metric in ("train_loss", "val_loss", "auroc"):
        inside = gate_rule_v2.evaluate_v2(_new(s1=_replay(**{metric: 3.9 * _sd(metric)}), s4=SEED4, s5=SEED5), REF, 1, [4, 5])
        outside = gate_rule_v2.evaluate_v2(_new(s1=_replay(**{metric: 4.1 * _sd(metric)}), s4=SEED4, s5=SEED5), REF, 1, [4, 5])
        assert inside["gate_a"][metric]["verdict"] == "pass", metric
        assert outside["gate_a"][metric]["verdict"] == "fail" and outside["verdict"] == "fail", metric
        row = inside["gate_a"][metric]
        assert row["tolerance_multiplier"] == 4 and row["tolerance"] == pytest.approx(4 * _sd(metric))


def test_gate_a_passes_the_three_recorded_replays_that_a_one_sd_gate_could_fail():
    """Validation-loss deviations of the three recorded replays of seed 1: -0.0023, -0.0044, +0.0023; the tolerance was one SD (0.0029)."""
    for dev in (-0.0023, -0.0044, 0.0023):
        out = gate_rule_v2.evaluate_v2(_new(s1=_replay(val_loss=dev), s4=SEED4, s5=SEED5), REF, 1, [4, 5])
        assert out["gate_a"]["val_loss"]["verdict"] == "pass", dev
        assert out["original_rule"]["gate_a"] == ("pass" if abs(dev) <= _sd("val_loss") else "fail")       # the old rule is kept beside it
    assert gate_rule_v2.evaluate_v2(_new(s1=_replay(val_loss=-0.0044), s4=SEED4, s5=SEED5), REF, 1, [4, 5])["original_rule"]["gate_a"] == "fail"


def test_gate_a_still_fails_nan_and_a_collapsed_replay():
    nan = _replay(); nan["val_loss"] = float("nan")
    assert gate_rule_v2.evaluate_v2(_new(s1=nan, s4=SEED4, s5=SEED5), REF, 1, [4, 5])["gate_a"]["val_loss"]["verdict"] == "fail"
    collapsed = _replay(auroc=-0.30)                                   # validation AUROC near chance
    assert gate_rule_v2.evaluate_v2(_new(s1=collapsed, s4=SEED4, s5=SEED5), REF, 1, [4, 5])["gate_a"]["auroc"]["verdict"] == "fail"


def test_a_healthy_replay_almost_never_fails_gate_a_but_often_failed_the_original():
    """Replay noise as large as the recorded one (about 1.2 committed-seed SDs, from the three replays): the old one-SD rule fails a healthy harness most of the time."""
    rng = np.random.default_rng(2)
    noise = rng.standard_normal((20_000, 3)) * 1.2
    assert (np.abs(noise) > 4).any(axis=1).mean() <= 0.01
    assert (np.abs(noise) > 1).any(axis=1).mean() >= 0.5

