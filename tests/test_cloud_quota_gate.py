"""The weekly-quota manager and the epoch-1 gate (user request 2026-09-29).

Quota: the ledger counts what trained, closes what a dead session left open, merges across sessions and never
across accounts; the start/stop rules refuse or pause as the usable hours (quota x (1 - reserve)) fill; and a real
runner over the tiny trainer pauses a cell BETWEEN EPOCHS at the quota and resumes it later.
Gate: the two rules are exercised on the committed epoch-1 numbers, including the same-seed defect (three seeds,
one run) that this gate exists to catch.
"""
import json
import os
import shutil
import time
import types

import pytest

from src.cloud import canary, epoch_gate, markers, preflight, quota, recipes
from src.cloud.config import Cell, load_harness
from src.cloud.runner import Runner
from src.model import resume
from tests.test_cloud_harness import CFG, CELL, builder, ok_predictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "config", "harness.json")
T0 = 10_000_000.0
DAY = 86400.0


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def _qcfg(**kw):
    base = dict(weekly_hours=30.0, reserve_fraction=0.15, soft_fraction=0.95, window_days=7.0,
                unit="gpu_hours", min_start_hours=0.5, workers=2)
    return quota.QuotaCfg(**{**base, **kw})


def _ledger(path, clock, account="ACC1"):
    return quota.Ledger(str(path), account, clock)


def _train(ledger, start, end, gpu=0, cell="c"):
    ledger.entries.append({"id": f"{cell}@{gpu}@{start}", "kind": "train", "cell": cell, "gpu": gpu,
                           "start": start, "end": end, "heartbeat": end if end is not None else start})


# ---- the ledger ---------------------------------------------------------------------------

def test_usage_counts_the_rolling_window_including_partial_overlap_and_open_intervals(tmp_path):
    clock = Clock()
    ledger = _ledger(tmp_path / "l.json", clock)
    _train(ledger, T0 - 8 * DAY, T0 - 8 * DAY + 3600)                    # before the window: not counted
    _train(ledger, T0 - 7 * DAY - 1800, T0 - 7 * DAY + 1800)             # half inside: 0.5 h
    _train(ledger, T0 - 3600, T0 - 1800)                                 # inside: 0.5 h
    _train(ledger, T0 - 1800, None)                                      # still running: 0.5 h up to now
    assert ledger.used_hours("gpu_hours", 7, T0) == pytest.approx(1.5)


def test_gpu_hours_counts_every_gpu_and_session_hours_counts_overlap_once(tmp_path):
    ledger = _ledger(tmp_path / "l.json", Clock())
    _train(ledger, T0 - 3600, T0, gpu=0)
    _train(ledger, T0 - 3600, T0, gpu=1)
    assert ledger.used_hours("gpu_hours", 7, T0) == pytest.approx(2.0)
    assert ledger.used_hours("session_hours", 7, T0) == pytest.approx(1.0)


def test_a_session_that_died_has_its_interval_closed_at_its_last_heartbeat_never_later(tmp_path):
    clock = Clock(T0)
    path = tmp_path / "l.json"
    ledger = _ledger(path, clock)
    eid = ledger.open("c", 0)
    clock.t = T0 + 60.0
    ledger.beat(eid)
    clock.t = T0 + 3600.0                                                 # the process died at +60 s; an hour later:
    reopened = _ledger(path, clock)
    (entry,) = reopened.entries
    assert entry["end"] == T0 + 60.0 and reopened.closed_after_crash == [eid]
    assert reopened.used_hours("gpu_hours", 7, clock.t) == pytest.approx(60 / 3600)


def test_ledgers_merge_by_id_and_a_ledger_of_another_account_is_refused(tmp_path):
    clock = Clock()
    a, b = _ledger(tmp_path / "a.json", clock), _ledger(tmp_path / "b.json", clock)
    a.add_external("shared", 1.0)
    b.add_external("shared", 1.0)
    b.add_external("only-in-b", 2.0)
    assert a.merge_file(str(tmp_path / "b.json")) == 1
    assert a.used_hours("gpu_hours", 7, T0) == pytest.approx(3.0)
    other = _ledger(tmp_path / "x.json", clock, account="ACC2")
    other.add_external("z", 1.0)
    with pytest.raises(ValueError, match="belongs to account"):
        a.merge_file(str(tmp_path / "x.json"))
    with pytest.raises(ValueError, match="each account has its own quota"):
        _ledger(tmp_path / "x.json", clock, account="ACC1")


def test_external_usage_is_recorded_once_per_id(tmp_path):
    ledger = _ledger(tmp_path / "l.json", Clock())
    assert ledger.add_external("canary", 2.2, "canary + gate") is True
    assert ledger.add_external("canary", 9.9) is False
    assert ledger.used_hours("gpu_hours", 7, T0) == pytest.approx(2.2)


def test_restore_ledger_merges_every_previous_output_and_skips_its_own(tmp_path):
    clock = Clock()
    for i, name in enumerate(("v1", "v2")):
        led = _ledger(tmp_path / name / "results" / quota.LEDGER_NAME, clock)
        led.add_external(f"prev-{i}", float(i + 1))
    report = quota.restore_ledger("ACC1", str(tmp_path / "now"), str(tmp_path))
    assert report["merged_files"] == 2 and report["new_entries"] == 2 and not report["problems"]
    merged = _ledger(tmp_path / "now" / quota.LEDGER_NAME, clock)
    assert merged.used_hours("gpu_hours", 7, T0) == pytest.approx(3.0)


# ---- the rules ----------------------------------------------------------------------------

def _quota_with(tmp_path, used, unit="gpu_hours", **kw):
    clock = Clock()
    led = _ledger(tmp_path / f"q{used}{unit}.json", clock)
    led.add_external("prior", used)
    return quota.Quota(_qcfg(unit=unit, **kw), led, clock)


def test_usable_hours_are_the_quota_less_its_reserve_and_the_plan_fits_them():
    cfg = quota.QuotaCfg.from_harness(load_harness(HARNESS))
    assert cfg.usable_hours == pytest.approx(25.5) and cfg.unit == "gpu_hours" and cfg.workers == 2
    budget = json.load(open(os.path.join(ROOT, "config", "wave_budget.json")))["per_account"]
    for acct in ("ACC1", "ACC2"):
        assert budget[acct]["gpu_hours_high"] <= cfg.usable_hours, acct     # the plan's worst case still fits


def test_no_cell_starts_once_the_usable_hours_are_spent(tmp_path):
    ok, why = _quota_with(tmp_path, 25.6).can_start(0.0, resume=False, now=T0)
    assert not ok and "weekly quota spent" in why
    ok, why = _quota_with(tmp_path, 25.5).can_start(0.0, resume=True, now=T0)
    assert not ok and "weekly quota spent" in why                          # not even a resume


def test_a_fresh_cell_is_refused_near_the_quota_but_a_partial_cell_may_resume(tmp_path):
    q = _quota_with(tmp_path, 24.4)                                        # 95 % of 25.5 is 24.225
    ok, why = q.can_start(0.0, resume=False, now=T0)
    assert not ok and "approaching" in why and "may still resume" in why
    assert q.can_start(0.0, resume=True, now=T0)[0]


def test_a_cell_needs_enough_hours_left_for_one_epoch(tmp_path):
    q = _quota_with(tmp_path, 25.3)                                        # 0.2 usable hours left
    assert not q.can_start(0.0, resume=True, now=T0)[0]                    # min_start_hours 0.5
    q2 = _quota_with(tmp_path, 24.0)                                       # 1.5 h left
    assert not q2.can_start(2 * 3600.0, resume=True, now=T0)[0]            # a two-hour epoch does not fit
    assert q2.can_start(3600.0, resume=True, now=T0)[0]
    assert _quota_with(tmp_path, 10.0).can_start(0.0, resume=False, now=T0)[0]


def test_the_stop_time_assumes_every_gpu_keeps_training(tmp_path):
    assert _quota_with(tmp_path, 24.5).stop_at(T0) == pytest.approx(T0 + 1800.0)                       # 1 h left, 2 GPUs burn it in 30 min
    assert _quota_with(tmp_path, 24.5, unit="session_hours").stop_at(T0) == pytest.approx(T0 + 3600.0)
    assert _quota_with(tmp_path, 30.0).stop_at(T0) == T0                                               # nothing left: stop now


def test_the_quota_config_is_validated_and_absent_means_off():
    assert quota.QuotaCfg.from_harness(CFG) is None
    for bad in ({"quota_unit": "days"}, {"quota_reserve_fraction": 1.0}, {"weekly_quota_gpu_hours": 0},
                {"quota_soft_fraction": 0}, {"quota_window_days": 0}):
        with pytest.raises(ValueError):
            quota.QuotaCfg.from_harness({**CFG, "weekly_quota_gpu_hours": 30, **bad})


def test_a_negative_amount_of_external_usage_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _ledger(tmp_path / "l.json", Clock()).add_external("x", -1.0)


def test_the_cli_reports_status_and_adds_external_usage_once(tmp_path, capsys):
    harness = os.path.join(ROOT, "config", "harness.json")
    args = ["--root", str(tmp_path), "--account", "ACC1", "--harness", harness]
    assert quota.main([*args, "--add-external", "2.5", "--external-id", "canary", "--note", "x"]) == 0
    assert "added: canary" in capsys.readouterr().out
    assert quota.main([*args, "--add-external", "2.5", "--external-id", "canary"]) == 0
    out = capsys.readouterr().out
    assert "already recorded" in out and '"used": 2.5' in out
    with pytest.raises(SystemExit):
        quota.main([*args, "--add-external", "1"])                        # no id


# ---- the pre-flight -----------------------------------------------------------------------

def test_preflight_refuses_a_real_run_once_the_quota_is_spent_and_only_reports_in_a_dry_run(tmp_path):
    cfg = load_harness(HARNESS)
    quota.Ledger(str(tmp_path / quota.LEDGER_NAME), "ACC1").add_external("x", 26.0)
    real = preflight.check_quota(cfg, str(tmp_path), "ACC1", dry_run=False)
    assert real.status == "fail" and "weekly quota spent" in real.detail
    dry = preflight.check_quota(cfg, str(tmp_path), "ACC1", dry_run=True)
    assert dry.status == "skipped" and dry.ok and "would be REFUSED" in dry.detail
    fresh = preflight.check_quota(cfg, str(tmp_path / "empty"), "ACC1", dry_run=False)
    assert fresh.status == "pass" and "0.00 of 25.50" in fresh.detail
    assert preflight.check_quota(CFG, str(tmp_path), "ACC1", False).status == "skipped"     # no quota configured


# ---- the runner, on the tiny trainer -------------------------------------------------------

def _q(root, weekly_hours, reserve=0.0, soft=1.0, min_start=0.0):
    cfg = _qcfg(weekly_hours=weekly_hours, reserve_fraction=reserve, soft_fraction=soft, min_start_hours=min_start)
    return quota.Quota(cfg, quota.Ledger(os.path.join(root, quota.LEDGER_NAME), "ACC1"))


def _runner(tmp_path, cells=(CELL,), **kw):
    root = str(tmp_path / "results")
    return Runner("ACC1", {0: list(cells), 1: []}, root, CFG, time.time(), ["Tesla T4", "Tesla T4"],
                  predictor=ok_predictor, log=lambda *_: None, **kw), root


def test_a_fresh_cell_is_not_started_when_the_quota_is_spent(tmp_path):
    root = str(tmp_path / "results")
    q = _q(root, 30.0, reserve=0.15)
    q.ledger.add_external("prior", 26.0)
    runner, _ = _runner(tmp_path, quota=q, command_builder=builder())
    (row,) = runner.run()
    assert row["outcome"] == "not started (quota)" and "weekly quota spent" in row["detail"]
    assert markers.state(root, CELL)[0] == "fresh"                       # no trainer ran


def test_a_cell_pauses_between_epochs_at_the_quota_and_a_later_session_finishes_it(tmp_path):
    root = str(tmp_path / "results")
    q = _q(root, 0.0008)                                                 # 2.9 GPU-seconds usable, two GPUs: stop in ~1.4 s
    runner, _ = _runner(tmp_path, quota=q, command_builder=builder(epochs=30, sleep=0.3))
    row = runner.run()[-1]                                               # rows: "starting", then the outcome
    assert row["outcome"] == "paused (quota)", row
    state = resume.load(markers.paths(root, CELL)["resume"], "cpu")
    assert state is not None and 1 <= state["epoch"] < 30                # stopped BETWEEN epochs, state saved
    assert markers.state(root, CELL)[0] == "partial"
    assert [e["kind"] for e in q.ledger.entries] == ["train"] and q.ledger.entries[0]["end"] is not None
    assert q.used() > 0
    # next week: a fresh ledger, the same results folder; the cell resumes and completes
    later, _ = _runner(tmp_path, quota=None, command_builder=builder(epochs=30, sleep=0.0))
    done = later.run()[-1]
    assert done["outcome"] == "complete", done
    assert markers.state(root, CELL)[0] == "complete"


def test_the_time_a_cell_trains_is_counted_in_the_ledger(tmp_path):
    root = str(tmp_path / "results")
    q = _q(root, 1000.0)
    runner, _ = _runner(tmp_path, quota=q, command_builder=builder(epochs=3, sleep=0.05))
    row = runner.run()[-1]
    assert row["outcome"] == "complete"
    (entry,) = q.ledger.entries
    assert entry["cell"] == CELL.id and entry["end"] > entry["start"] and "closed_by" not in entry
    assert 0 < q.used() < 0.1
    saved = json.load(open(os.path.join(root, quota.LEDGER_NAME)))
    assert saved["account"] == "ACC1" and len(saved["entries"]) == 1


def test_a_runner_without_quota_keys_behaves_as_before(tmp_path):
    runner, root = _runner(tmp_path, command_builder=builder(epochs=2))   # quota="auto" -> CFG has no keys
    assert runner.quota is None
    row = runner.run()[-1]
    assert row["outcome"] == "complete" and not os.path.exists(os.path.join(root, quota.LEDGER_NAME))


# ---- the epoch-1 gate ----------------------------------------------------------------------

REF = {1: {"train_loss": 0.2900, "val_loss": 0.2176, "auroc": 0.7925},
       2: {"train_loss": 0.2843, "val_loss": 0.2127, "auroc": 0.8010},
       3: {"train_loss": 0.2835, "val_loss": 0.2178, "auroc": 0.7687}}


def _new(seed_metrics):
    return {s: {**m, "initial_weight_hash": h} for s, (m, h) in seed_metrics.items()}


def _run(m4=None, h4="h4", m1=None, h1="h1", h5="h5", m5=None):
    m1 = m1 or {"train_loss": 0.2910, "val_loss": 0.2170, "auroc": 0.7900}
    m4 = m4 or {"train_loss": 0.2870, "val_loss": 0.2150, "auroc": 0.7800}
    m5 = m5 or {"train_loss": 0.2860, "val_loss": 0.2190, "auroc": 0.7750}
    return epoch_gate.evaluate(_new({1: (m1, h1), 4: (m4, h4), 5: (m5, h5)}), REF, 1, [4, 5])


def test_the_replayed_seed_passes_within_one_committed_sd_and_fails_beyond_it():
    assert _run()["gate_a"]["verdict"] == "pass"
    bad = _run(m1={"train_loss": 0.2900 + 0.02, "val_loss": 0.2176, "auroc": 0.7925})
    assert bad["gate_a"]["train_loss"]["verdict"] == "fail" and bad["verdict"] == "fail"


def test_a_zero_committed_sd_is_inconclusive_with_no_invented_floor():
    flat = {s: {"train_loss": 0.29, "val_loss": 0.21, "auroc": 0.79} for s in (1, 2, 3)}
    out = epoch_gate.evaluate(_new({1: (flat[1], "a"), 4: (flat[1], "b")}), flat, 1, [4])
    assert out["gate_a"]["verdict"] == "inconclusive"


def test_the_gate_uses_the_sample_sd_over_the_three_committed_seeds():
    import statistics
    out = _run()
    assert out["reference_spread"]["train_loss"]["sd"] == pytest.approx(
        statistics.stdev([REF[s]["train_loss"] for s in (1, 2, 3)]))


def test_a_new_seed_inside_the_committed_envelope_passes_and_one_outside_it_fails():
    assert _run()["gate_b"]["4"]["verdict"] == "pass"
    wild = _run(m4={"train_loss": 0.2870, "val_loss": 0.2150, "auroc": 0.95})
    assert wild["gate_b"]["4"]["auroc"]["verdict"] == "fail" and wild["verdict"] == "fail"
    nan = _run(m4={"train_loss": float("nan"), "val_loss": 0.2150, "auroc": 0.78})
    assert nan["gate_b"]["4"]["train_loss"]["verdict"] == "fail"


def test_three_seeds_that_are_one_run_fail_the_distinct_check_the_moltrans_defect():
    same_metrics = {"train_loss": 0.2910, "val_loss": 0.2170, "auroc": 0.7900}          # identical to the replay
    bug = _run(m4=same_metrics, h4="h4")
    assert bug["gate_b"]["4"]["distinct"]["verdict"] == "fail" and bug["gate_b"]["4"]["distinct"]["identical_metrics_to"] == [1]
    assert bug["verdict"] == "fail"
    same_init = _run(h4="h1")                                                           # same initial weights, different metrics
    assert same_init["gate_b"]["4"]["distinct"]["same_initial_weights_as"] == [1] and same_init["verdict"] == "fail"


def test_a_missing_initial_weight_hash_is_inconclusive_not_a_pass():
    out = _run(h4=None)
    assert out["gate_b"]["4"]["distinct"]["verdict"] == "inconclusive" and out["verdict"] == "inconclusive"


def test_the_reference_needs_the_three_committed_seeds():
    with pytest.raises(ValueError):
        epoch_gate.evaluate(_new({1: (REF[1], "a")}), {1: REF[1], 2: REF[2]}, 1, [])
    with pytest.raises(ValueError, match="no run for seed"):
        epoch_gate.evaluate(_new({1: (REF[1], "a")}), REF, 1, [4])


def _fake_trainer(root, ignore_seed=False):
    """Stands in for `--stop-after-epoch 1`: writes the resume file and start record a real one leaves."""
    def fake(command, env=None, stdout=None, stderr=None):
        ctx = json.loads(env[resume.ENV_CONTEXT])
        cell = Cell(ctx["dataset"], ctx["model"], ctx["level"], ctx["seed"])
        seed = 1 if ignore_seed else cell.seed
        p = markers.paths(root, cell)
        os.makedirs(p["dir"], exist_ok=True)
        resume.save(p["resume"], {"epoch": 1, "extra": {"history": [
            {"epoch": 1, "train_loss": 0.2900 - 0.001 * (seed - 1), "val_loss": 0.2170 - 0.0005 * (seed - 1),
             "auroc": 0.7900 - 0.004 * (seed - 1), "auprc": 0.3, "accuracy": 0.93}]}})
        with open(p["start"], "w") as handle:
            json.dump({"seed": seed, "initial_weight_hash": f"weights-of-seed-{seed}"}, handle)
        return types.SimpleNamespace(returncode=0)
    return fake


def _gate_cells(seeds=(1, 4, 5)):
    return [Cell("davis", "coldsite_dti", "cold_target", s) for s in seeds]


def test_the_gate_runs_each_seed_for_one_epoch_and_passes_when_the_seeds_are_real(tmp_path):
    root = str(tmp_path / "gate")
    epoch_gate.run_gate(_gate_cells(), root, run=_fake_trainer(root))
    new = {c.seed: epoch_gate.read_epoch1(root, c) for c in _gate_cells()}
    assert new[4]["initial_weight_hash"] == "weights-of-seed-4"
    assert epoch_gate.evaluate(new, REF, 1, [4, 5])["verdict"] == "pass"


def test_the_gate_catches_a_launcher_that_trains_every_seed_as_seed_1(tmp_path):
    root = str(tmp_path / "gate")
    epoch_gate.run_gate(_gate_cells(), root, run=_fake_trainer(root, ignore_seed=True))
    new = {c.seed: epoch_gate.read_epoch1(root, c) for c in _gate_cells()}
    out = epoch_gate.evaluate(new, REF, 1, [4, 5])
    assert out["verdict"] == "fail" and out["gate_b"]["4"]["distinct"]["verdict"] == "fail"


def test_the_gate_refuses_to_run_over_an_earlier_gates_files(tmp_path):
    root = str(tmp_path / "gate")
    epoch_gate.run_gate(_gate_cells((1,)), root, run=_fake_trainer(root))
    with pytest.raises(SystemExit, match="already exist"):
        epoch_gate.run_gate(_gate_cells((1,)), root, run=_fake_trainer(root))


def test_a_failed_trainer_stops_the_gate_loudly(tmp_path):
    root = str(tmp_path / "gate")
    with pytest.raises(SystemExit, match="trainer exited 3"):
        epoch_gate.run_gate(_gate_cells((1,)), root, run=lambda *a, **k: types.SimpleNamespace(returncode=3))


def test_the_one_epoch_command_is_the_real_cells_command_plus_the_stop_flag():
    command = epoch_gate.epoch1_command(Cell("davis", "coldsite_dti", "cold_target", 4), "results/epoch_gate", "py")
    assert command[0] == "py" and command[1:3] == ["-m", "src.model.train"]
    assert command[command.index("--seed") + 1] == "4" and command[command.index("--split") + 1] == "cold_target"
    assert command[command.index("--batch-size") + 1] == str(recipes.COLDSITE_BATCH)
    assert command[command.index("--min-epochs") + 1] == "10"
    assert command[-2:] == ["--stop-after-epoch", "1"] and "--amp" not in command       # DAVIS: full precision
    other = epoch_gate.epoch1_command(Cell("davis", "hyperattentiondti", "random", 4), "r", "py")
    assert other[-2:] == ["--stop-after-epoch", "1"] and "--seed" in other


def test_the_reference_is_frozen_from_committed_histories_with_their_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    committed = tmp_path / "committed"
    for s, row in REF.items():
        path = markers.paths(str(committed), Cell("davis", "coldsite_dti", "cold_target", s))["history"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump([{"epoch": 1, **row, "auprc": 0.3, "accuracy": 0.93}, {"epoch": 2, "train_loss": 0.1, "val_loss": 0.1, "auroc": 0.9}],
                  open(path, "w"))
    ref = epoch_gate.write_reference(str(committed), str(tmp_path / "ref.json"))
    assert sorted(ref["seeds"]) == ["1", "2", "3"] and len(ref["seeds"]["1"]["source_sha256"]) == 64
    assert epoch_gate.load_reference(str(tmp_path / "ref.json"))["by_seed"][2]["auroc"] == REF[2]["auroc"]
    with pytest.raises(SystemExit, match="exists"):
        epoch_gate.write_reference(str(committed), str(tmp_path / "ref.json"))


def test_the_committed_reference_is_for_the_canary_cell_and_has_a_nonzero_spread():
    ref = epoch_gate.load_reference(os.path.join(ROOT, "config", "epoch1_reference.json"))
    cref = canary.load_reference(os.path.join(ROOT, "config", "canary_reference.json"))
    assert (ref["dataset"], ref["level"], ref["model"]) == (cref["dataset"], cref["level"], cref["model"])
    assert sorted(ref["by_seed"]) == [1, 2, 3]
    import statistics
    for m in epoch_gate.METRICS:
        assert statistics.stdev([ref["by_seed"][s][m] for s in (1, 2, 3)]) > 0


# ---- the pre-flight gate ---------------------------------------------------------------------

def _root(tmp_path, verdict=None):
    root = tmp_path / "r"
    (root / "config").mkdir(parents=True)
    for relative in canary.HARNESS_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, relative), root / relative)
    shutil.copy(os.path.join(ROOT, "config", "epoch1_reference.json"), root / "config")
    if verdict is not None:
        v = {"verdict": "pass", "dataset": "davis", "level": "cold_target", "model": "coldsite_dti",
             "harness_sha256": canary.harness_hash(str(root)), "seeds_checked": [4, 5], **verdict}
        (root / "config" / "epoch_gate_verdict.json").write_text(json.dumps(v))
    return str(root)


WAVE_CELLS = [Cell("davis", "hyperattentiondti", "random", 4), Cell("davis", "moltrans", "cold_pair", 5)]


def test_a_wave_is_refused_until_the_epoch_gate_has_passed_on_this_harness_for_its_seeds(tmp_path):
    assert preflight.check_epoch_gate(_root(tmp_path / "a"), False, "A", WAVE_CELLS).status == "fail"
    ok = preflight.check_epoch_gate(_root(tmp_path / "b", {}), False, "A", WAVE_CELLS)
    assert ok.status == "pass", ok.detail
    for verdict in ("fail", "inconclusive"):
        bad = preflight.check_epoch_gate(_root(tmp_path / verdict, {"verdict": verdict}), False, "A", WAVE_CELLS)
        assert bad.status == "fail" and "not 'pass'" in bad.detail
    other = preflight.check_epoch_gate(_root(tmp_path / "x", {"level": "random"}), False, "A", WAVE_CELLS)
    assert other.status == "fail" and "different cell" in other.detail


def test_the_verdict_must_cover_every_new_seed_the_account_trains(tmp_path):
    check = preflight.check_epoch_gate(_root(tmp_path / "s", {"seeds_checked": [4]}), False, "A", WAVE_CELLS)
    assert check.status == "fail" and "does not cover seed(s) [5]" in check.detail
    drugban = [Cell("kiba", "drugban", "random", 1), Cell("kiba", "drugban", "cold_drug", 3)]   # ACC3: committed seeds only
    assert preflight.check_epoch_gate(_root(tmp_path / "t", {"seeds_checked": []}), False, "A", drugban).status == "pass"


def test_a_changed_harness_stales_the_epoch_gate_verdict(tmp_path):
    root = _root(tmp_path / "h", {})
    with open(os.path.join(root, "src/cloud/quota.py"), "a") as handle:
        handle.write("\n# changed after the gate ran\n")
    stale = preflight.check_epoch_gate(root, False, "A", WAVE_CELLS)
    assert stale.status == "fail" and "harness files changed" in stale.detail


def test_the_canary_wave_is_exempt_and_a_dry_run_reports_without_passing(tmp_path):
    assert preflight.check_epoch_gate(_root(tmp_path / "c"), False, "canary", []).status == "pass"
    dry = preflight.check_epoch_gate(_root(tmp_path / "d"), True, "A", WAVE_CELLS)
    assert dry.status == "skipped" and dry.ok and "would be REFUSED" in dry.detail


def test_the_new_modules_are_part_of_the_hash_the_verdicts_are_bound_to():
    for name in ("src/cloud/quota.py", "src/cloud/epoch_gate.py"):
        assert name in canary.HARNESS_FILES and os.path.exists(os.path.join(ROOT, name))
