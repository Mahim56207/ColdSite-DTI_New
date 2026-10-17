"""The runner's live monitor (time checker + progress saver): what the notebook shows while the GPUs work, and progress.json.
The monitor is read-only: it must never change what is trained, and its own failure must never stop a session."""
import json
import os
import sys
import time

from src.cloud import markers, quota as quota_mod
from src.cloud.config import Cell
from src.cloud.runner import Runner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cloud_harness import CFG, CELL, builder, ok_predictor  # noqa: E402  (same tiny-trainer fixtures as the harness tests)


def _runner(tmp_path, cells=(CELL,), start=1000.0, now=1000.0, lines=None, **kw):
    root = str(tmp_path / "results")
    log = (lambda m: lines.append(m)) if lines is not None else (lambda *_: None)
    r = Runner("ACC1", {0: list(cells), 1: []}, root, CFG, start, ["Tesla T4", "Tesla T4"], predictor=ok_predictor,
               clock=lambda: now, log=log, **kw)
    return r, root


def _write_status(root, cell, **fields):
    path = markers.paths(root, cell)["status"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(fields, open(path, "w"))


def test_the_block_shows_elapsed_time_time_to_the_self_stop_and_the_cell_in_progress(tmp_path):
    r, root = _runner(tmp_path, start=0.0, now=2 * 3600 + 14 * 60)
    r.current[0] = CELL
    _write_status(root, CELL, epoch=7, mean_epoch_s=361.2, projected_finish_unix_upper_bound=2 * 3600 + 14 * 60 + 5400)
    text = r.status_block()
    assert "elapsed 2h14m" in text and "session limit 11 h" in text
    deadline = (11 * 3600 - 45 * 60) - (2 * 3600 + 14 * 60)             # 11 h session, 45 min margin
    assert f"self-stop in {deadline // 3600}h{deadline % 3600 // 60:02d}m" in text
    assert "cells complete 0/1" in text
    assert f"{CELL.id} epoch 7, 361.2 s/epoch" in text and "within 1h30m" in text and "upper bound" in text
    assert "gpu1: idle" in text


def test_a_cell_that_has_not_finished_an_epoch_says_so_instead_of_inventing_a_number(tmp_path):
    r, root = _runner(tmp_path)
    r.current[0] = CELL
    assert "starting, no epoch finished yet" in r.status_block()


def test_cells_complete_counts_markers_cheaply_and_ignores_a_cell_with_a_resume_file(tmp_path):
    other = Cell("davis", "coldsite_dti", "cold_drug", 4)
    r, root = _runner(tmp_path, cells=(CELL, other))
    for cell in (CELL, other):
        os.makedirs(markers.paths(root, cell)["dir"], exist_ok=True)
    open(markers.paths(root, CELL)["marker"], "w").write("{}")            # present marker: counted without hashing anything
    open(markers.paths(root, other)["marker"], "w").write("{}")
    open(markers.paths(root, other)["resume"], "w").write("x")            # a resume file beside it: not complete
    assert r.progress()["cells_complete"] == 1 and "cells complete 1/2" in r.status_block()


def test_the_quota_line_appears_only_when_a_quota_is_configured(tmp_path):
    r, root = _runner(tmp_path)
    assert "quota" not in r.status_block()
    qcfg = quota_mod.QuotaCfg(weekly_hours=30, reserve_fraction=0.15, soft_fraction=0.95, window_days=7, unit="gpu_hours", min_start_hours=0.5, workers=2)
    q = quota_mod.Quota(qcfg, quota_mod.Ledger(os.path.join(root, quota_mod.LEDGER_NAME), "ACC1", lambda: 1000.0), lambda: 1000.0)
    r2, _ = _runner(tmp_path, quota=q)
    assert "quota used 0.0 of 25.5 gpu-hours" in r2.status_block()


def test_progress_json_is_written_atomically_and_holds_the_numbers_the_block_shows(tmp_path):
    r, root = _runner(tmp_path, start=0.0, now=600.0)
    r.save_progress()
    data = json.load(open(os.path.join(root, "progress.json")))
    assert data["account"] == "ACC1" and data["elapsed_s"] == 600.0 and data["cells_total"] == 1 and "gpu0" in data["gpus"]
    assert not os.path.exists(os.path.join(root, "progress.json.tmp"))


def test_a_real_run_prints_monitor_lines_and_leaves_progress_json_and_the_same_result(tmp_path):
    lines = []
    r, root = _runner(tmp_path, start=time.time(), now=None, lines=lines, command_builder=builder(epochs=4, sleep=0.15), monitor_s=0.05)
    r.clock = time.time
    summary = r.run()
    assert [row["outcome"] for row in summary][-1] == "complete"
    assert any(line.startswith("[monitor]") for line in lines)
    data = json.load(open(os.path.join(root, "progress.json")))
    assert data["cells_complete"] == 1                                    # the final write happens after the cells end
    # the monitor is read-only: the same cell trained without it ends in the same state
    r2, root2 = _runner(tmp_path / "again", start=time.time(), now=None, command_builder=builder(epochs=4, sleep=0.0), monitor_s=0)
    r2.clock = time.time
    assert [row["outcome"] for row in r2.run()][-1] == "complete"


def test_a_failing_monitor_never_stops_the_training(tmp_path):
    lines = []
    r, root = _runner(tmp_path, start=time.time(), now=None, lines=lines, command_builder=builder(epochs=3, sleep=0.1), monitor_s=0.05)
    r.clock = time.time
    r.status_block = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    assert [row["outcome"] for row in r.run()][-1] == "complete"
    assert any("could not read progress" in line for line in lines)


def test_a_zero_interval_switches_the_monitor_off(tmp_path):
    lines = []
    r, root = _runner(tmp_path, start=time.time(), now=None, lines=lines, command_builder=builder(epochs=2, sleep=0.0), monitor_s=0)
    r.clock = time.time
    r.run()
    assert not any(line.startswith("[monitor]") for line in lines)
