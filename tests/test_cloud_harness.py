"""T09: the cloud harness. Pre-flight refuses each failure it lists, the trainers stop between
epochs at the deadline or on a signal and the next session finishes them bit for bit, and the
files a session leaves (start record, status, complete-marker) say what they should.

The trainers here are `src/cloud/tiny_trainer.py`, which uses the same `resume.Resumable` as the
four real ones, so the whole path runs on a CPU in seconds.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from src.cloud import canary, markers, preflight, recipes, restore
from src.cloud.config import Cell, load_harness, load_waves, stop_at
from src.cloud.runner import Runner, child_env, finalize
from src.model import resume

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {"session_limit_hours": 11, "safety_margin_minutes": 45, "required_gpus": 2,
       "gpu_name_contains": "T4", "disk_reserve_gb": 5,
       "checkpoint_bytes": {"coldsite_dti": 10, "deepdta": 10, "hyperattentiondti": 10,
                            "moltrans": 250_000_000, "drugban": 10},
       "planned_bytes_per_cell_factor": 4}
CELL = Cell("davis", "coldsite_dti", "random", 4)


def tiny_command(cell, results_root, python=None, epochs=6, sleep=0.0, extra=()):
    p = markers.paths(results_root, cell)
    return [python or sys.executable, "-m", "src.cloud.tiny_trainer",
            "--checkpoint-dir", p["dir"], "--results-dir", p["dir"], "--seed", str(cell.seed),
            "--epochs", str(epochs), "--epoch-sleep", str(sleep),
            "--stem", os.path.basename(p["checkpoint"])[:-len(".pt")],
            "--results-file", p["results"], *extra]


def builder(**kw):
    return lambda cell, root, python=None: tiny_command(cell, root, python, **kw)


def ok_predictor(cell, root, env):
    return "f" * 64, "stub predictions"


def _runner(tmp_path, start, clock=time.time, cells=(CELL,), **kw):
    root = str(tmp_path / "results")
    pred = kw.pop("predictor", ok_predictor)
    return Runner("ACC1", {0: list(cells), 1: []}, root, CFG, start, ["Tesla T4", "Tesla T4"],
                  predictor=pred, clock=clock, log=lambda *_: None, **kw), root


def _params(root, cell=CELL):
    with open(markers.paths(root, cell)["results"]) as handle:
        return json.load(handle)["final_params"]


# ---- config ---------------------------------------------------------------

def test_the_repo_config_loads_and_its_margin_is_the_plans_default():
    cfg = load_harness(os.path.join(ROOT, "config", "harness.json"))
    assert cfg["safety_margin_minutes"] == 45 and cfg["required_gpus"] == 2
    assert stop_at(1000.0, cfg) == 1000.0 + cfg["session_limit_hours"] * 3600 - 45 * 60


def test_a_margin_that_leaves_no_time_is_refused(tmp_path):
    path = tmp_path / "h.json"
    path.write_text(json.dumps({**CFG, "safety_margin_minutes": 11 * 60}))
    with pytest.raises(ValueError):
        load_harness(str(path))


def test_a_cell_listed_for_two_accounts_is_refused(tmp_path):
    cell = {"dataset": "davis", "model": "moltrans", "level": "random", "seed": 4}
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"wave": "A", "accounts": {"ACC1": {"gpu0": [cell]},
                                                          "ACC2": {"gpu1": [cell]}}}))
    with pytest.raises(ValueError, match="trained once"):
        load_waves(str(path))


# ---- pre-flight: each refusal ---------------------------------------------

def test_two_t4s_pass_and_anything_else_is_refused():
    assert preflight.check_gpus(CFG, False, lambda: ["Tesla T4", "Tesla T4"]).ok
    assert not preflight.check_gpus(CFG, False, lambda: ["Tesla T4"]).ok
    assert not preflight.check_gpus(CFG, False, lambda: []).ok
    bad = preflight.check_gpus(CFG, False, lambda: ["Tesla T4", "Tesla P100-PCIE-16GB"])
    assert bad.status == "fail" and "P100" in bad.detail


def test_a_dry_run_reports_the_gpu_check_as_skipped_never_as_passed():
    check = preflight.check_gpus(CFG, True, lambda: [])
    assert check.status == "skipped" and check.ok


def _mini_repo(tmp_path):
    """A repo-shaped folder with a real manifest over three tiny files."""
    root = tmp_path / "repo"
    for rel, text in {"data/splits/davis/random/train.csv": "a,b\n1,2\n",
                      "data/davis_ground_truth_sites.json": "{}",
                      "data/processed/nonkinase_panel.csv": "x\n1\n"}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    from src.data import manifest
    (root / "data" / "splits").mkdir(parents=True, exist_ok=True)
    with open(root / manifest.MANIFEST_PATH, "w") as handle:
        json.dump(manifest.build(str(root)), handle)
    return root


def test_a_changed_split_file_is_refused(tmp_path):
    root = _mini_repo(tmp_path)
    assert preflight.check_splits(str(root)).ok
    (root / "data/splits/davis/random/train.csv").write_text("a,b\n1,3\n")
    check = preflight.check_splits(str(root))
    assert check.status == "fail" and "changed" in check.detail


def test_derived_splits_are_checked_only_when_a_planned_cell_uses_them(tmp_path):
    root = _mini_repo(tmp_path)
    from src.data import manifest
    extra = root / "data/splits/davis/cold_target_seqclean/train.csv"
    extra.parent.mkdir(parents=True)
    extra.write_text("a\n")
    with open(root / manifest.MANIFEST_PATH, "w") as handle:
        json.dump(manifest.build(str(root)), handle)
    extra.unlink()                                       # built by a separate step, absent now
    assert not preflight.check_splits(str(root)).ok       # checked in full: refused
    assert preflight.check_splits(str(root), [Cell("davis", "coldsite_dti", "random", 4)]).ok


def test_a_missing_or_extra_split_file_is_refused(tmp_path):
    root = _mini_repo(tmp_path)
    (root / "data/splits/davis/random/valid.csv").write_text("a\n")
    assert not preflight.check_splits(str(root)).ok


def _cloud_manifest(root, vendored=None, raw=None):
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    with open(os.path.join(root, "config", "cloud_manifest.json"), "w") as handle:
        json.dump({"vendored": vendored or {}, "raw_dataset": raw or {}}, handle)


def test_a_changed_raw_dataset_file_is_refused(tmp_path):
    root = tmp_path / "r"
    for rel in preflight.RAW_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x")
    _cloud_manifest(str(root), raw=preflight.raw_hashes(str(root)))
    assert preflight.check_raw_dataset(str(root)).ok
    (root / preflight.RAW_FILES[0]).write_text("edited")
    assert not preflight.check_raw_dataset(str(root)).ok


def test_a_changed_vendored_file_is_refused(tmp_path):
    root = tmp_path / "r"
    for name, rel in preflight.VENDORED.items():
        (root / rel).mkdir(parents=True, exist_ok=True)
        (root / rel / "models.py").write_text(f"# {name}")
    _cloud_manifest(str(root), vendored=preflight.vendored_hashes(str(root)))
    assert preflight.check_vendored({"moltrans"}, str(root)).ok
    (root / "baselines/MolTrans/models.py").write_text("# edited")
    assert not preflight.check_vendored({"moltrans"}, str(root)).ok
    assert preflight.check_vendored({"coldsite_dti"}, str(root)).ok       # no vendored code used
    # a model whose vendored tree is untouched is unaffected by another's edit
    assert preflight.check_vendored({"hyperattentiondti"}, str(root)).ok


def test_the_repo_manifest_matches_the_vendored_trees_and_raw_files_on_disk():
    check = preflight.check_vendored({"moltrans", "hyperattentiondti", "drugban", "deepdta"})
    assert check.ok, check.detail
    raw = preflight.check_raw_dataset()
    if not raw.ok and "missing" in raw.detail and "changed: []" in raw.detail:
        pytest.skip("raw DeepDTA files not present on this machine")
    assert raw.ok, raw.detail


def test_a_vendored_module_that_moves_the_rng_on_import_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(preflight.RNG_IMPORTS, "moltrans",
                        "import torch; torch.manual_seed(1)")
    check = preflight.check_rng_import({"moltrans"}, False, ROOT)
    assert check.status == "fail" and "moved" in check.detail


def test_the_real_vendored_imports_leave_the_rng_alone():
    if not (os.path.isdir(os.path.join(ROOT, "baselines/MolTrans"))):
        pytest.skip("vendored baselines not present")
    check = preflight.check_rng_import({"moltrans", "hyperattentiondti"}, False, ROOT)
    assert check.ok, check.detail


def test_too_little_disk_is_refused_and_the_estimate_counts_moltrans_heavily():
    big = Cell("davis", "moltrans", "random", 4)
    small = Cell("davis", "coldsite_dti", "random", 4)
    assert preflight.planned_bytes([big], CFG) == 4 * 250_000_000
    tight = preflight.check_disk([big], "/tmp", CFG, free_bytes=lambda: 5 * 10**9)
    assert tight.status == "fail"
    assert preflight.check_disk([small], "/tmp", CFG, free_bytes=lambda: 6 * 10**9).ok


def test_an_unexplained_results_file_is_never_skipped_or_overwritten(tmp_path):
    root = str(tmp_path)
    p = markers.paths(root, CELL)
    os.makedirs(p["dir"])
    open(p["results"], "w").write("{}")
    check, plan = preflight.check_cells([CELL], root)
    assert check.status == "fail" and plan[CELL.id] == "stop"
    assert "without a complete-marker" in check.detail


def test_an_orphan_checkpoint_is_refused_but_a_resume_file_means_resume(tmp_path):
    root = str(tmp_path)
    p = markers.paths(root, CELL)
    os.makedirs(p["dir"])
    open(p["checkpoint"], "w").write("x")
    assert not preflight.check_cells([CELL], root)[0].ok
    open(p["resume"], "w").write("x")
    check, plan = preflight.check_cells([CELL], root)
    assert check.ok and plan[CELL.id] == "resume"


def test_the_wave_manifest_must_list_the_account(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"wave": "A", "accounts": {"ACC1": {"gpu0": []}}}))
    assert preflight.check_wave(str(path), "ACC1")[0].ok
    check, waves = preflight.check_wave(str(path), "ACC9")
    assert check.status == "fail" and waves is None
    assert preflight.check_wave(str(tmp_path / "missing.json"), "ACC1")[0].status == "fail"


def test_a_failed_preflight_starts_no_trainer(tmp_path, monkeypatch):
    from src.cloud import runner
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"wave": "A", "accounts": {"ACC1": {"gpu0": [
        {"dataset": "davis", "model": "coldsite_dti", "level": "random", "seed": 4}]}}}))
    monkeypatch.setattr(preflight, "probe_gpus", lambda: [])       # no GPU: must refuse
    monkeypatch.setattr(runner.Runner, "run", lambda self: pytest.fail("a trainer was started"))
    code = runner.main(["--account", "ACC1", "--waves", str(path),
                        "--results-root", str(tmp_path / "res")])
    assert code == 2


def test_the_dry_run_passes_on_this_repo_and_exits_before_any_gpu_work(tmp_path):
    if not os.path.isdir(os.path.join(ROOT, "data", "splits", "davis")):
        pytest.skip("splits not built on this machine")
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"wave": "A", "accounts": {"ACC1": {"gpu0": [
        {"dataset": "davis", "model": "hyperattentiondti", "level": "random", "seed": 4}]}}}))
    out = subprocess.run([sys.executable, "-m", "src.cloud.runner", "--account", "ACC1",
                          "--waves", str(path), "--results-root", str(tmp_path / "res"),
                          "--dry-run"], capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": ROOT})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "SKIPPED  gpus" in out.stdout and "dry run: exiting before any GPU work" in out.stdout
    assert not os.path.exists(tmp_path / "res" / "davis_binary" /
                              "coldsite_dti_davis_random_binary_seed4.pt")


# ---- the trainers' side: deadline, signals, start record, status ------------

def _clock(t=0.0):
    box = [t]
    return (lambda: box[0]), box


def _resumable(tmp_path, **env):
    import numpy as np
    import torch
    for key, value in env.items():
        os.environ[key] = value
    try:
        ckpt = str(tmp_path / "c" / "x.pt")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        run = resume.Resumable(ckpt, "cpu", {"seed": 1, "epochs": 10}, ("seed",), None)
    finally:
        for key in env:
            del os.environ[key]
    return run


def test_without_the_harness_variables_nothing_new_happens(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    run = _resumable(tmp_path)
    import torch
    from src.model.early_stopping import CheckpointSelector
    model = torch.nn.Linear(2, 1)
    run.begin(model, torch.optim.SGD(model.parameters(), lr=0.1), CheckpointSelector(5, 1, 10))
    assert signal.getsignal(signal.SIGTERM) is before
    assert not os.path.exists(str(tmp_path / "c" / "x_start.json"))
    assert run.harness is False and run.stop_at is None


def test_the_next_epoch_is_not_started_when_it_would_end_after_the_deadline(tmp_path):
    run = _resumable(tmp_path, COLDSITE_HARNESS="1", COLDSITE_STOP_AT="1000")
    clock, box = _clock(0.0)
    run.clock = clock
    run.epoch_seconds = [100.0, 120.0]
    box[0] = 800.0
    assert run.interrupt_now(3) is False          # 800 + 120 = 920 <= 1000
    box[0] = 890.0
    assert run.interrupt_now(4) is True           # 890 + 120 = 1010 > 1000


def test_the_projection_uses_the_longest_measured_epoch_and_needs_a_measurement(tmp_path):
    run = _resumable(tmp_path, COLDSITE_HARNESS="1", COLDSITE_STOP_AT="1000")
    clock, box = _clock(999.0)
    run.clock = clock
    assert run.interrupt_now(1) is False          # nothing measured yet: the launcher's margin covers it


def test_a_signal_makes_the_trainer_stop_after_the_epoch_in_progress(tmp_path):
    run = _resumable(tmp_path, COLDSITE_HARNESS="1")
    assert run.interrupt_now(2) is False
    run.signalled = "SIGTERM"
    assert run.interrupt_now(2) is True


def test_every_epoch_writes_a_status_file_with_the_checkpoint_hash(tmp_path):
    status = tmp_path / "status.json"
    root = str(tmp_path / "results")
    cmd = tiny_command(CELL, root, epochs=3)
    p = markers.paths(root, CELL)
    os.makedirs(p["dir"])
    env = child_env(0, CELL, "ACC1", "Tesla T4", time.time() + 3600, str(status))
    assert subprocess.run(cmd, env={**env, "PYTHONPATH": ROOT}, cwd=ROOT).returncode == 0
    data = json.loads(status.read_text())
    assert data["cell"] == CELL.id and data["account"] == "ACC1" and data["gpu"] == "Tesla T4"
    assert data["epoch"] == 3 and data["checkpoint"]["sha256"] and data["resume_file"]["sha256"]
    assert data["checkpoint"]["path"].endswith(".pt")


def test_the_start_record_holds_the_seed_and_distinct_seeds_give_distinct_weight_hashes(tmp_path):
    hashes = {}
    for seed in (1, 2, 3):
        cell = Cell("davis", "coldsite_dti", "random", seed)
        root = str(tmp_path / "results")
        os.makedirs(markers.paths(root, cell)["dir"], exist_ok=True)
        env = child_env(0, cell, "ACC1", "Tesla T4", time.time() + 3600,
                        markers.paths(root, cell)["status"])
        assert subprocess.run(tiny_command(cell, root, epochs=1), env={**env, "PYTHONPATH": ROOT},
                              cwd=ROOT).returncode == 0
        start = json.load(open(markers.paths(root, cell)["start"]))
        assert start["seed"] == seed
        assert set(start["rng_state_sha256"]) == {"python", "numpy", "torch"}
        hashes[seed] = start["initial_weight_hash"]
    assert len(set(hashes.values())) == 3


def test_a_resumed_cell_keeps_its_original_start_record(tmp_path):
    root = str(tmp_path / "results")
    p = markers.paths(root, CELL)
    os.makedirs(p["dir"])
    env = {**child_env(0, CELL, "ACC1", "T4", time.time() + 3600, p["status"]),
           "PYTHONPATH": ROOT}
    subprocess.run(tiny_command(CELL, root, epochs=4, extra=("--stop-after-epoch", "2")),
                   env=env, cwd=ROOT, check=True)
    first = open(p["start"]).read()
    subprocess.run(tiny_command(CELL, root, epochs=4), env=env, cwd=ROOT, check=True)
    assert open(p["start"]).read() == first


def test_sigterm_finishes_the_epoch_saves_and_exits_cleanly(tmp_path):
    root = str(tmp_path / "results")
    p = markers.paths(root, CELL)
    os.makedirs(p["dir"])
    env = {**child_env(0, CELL, "ACC1", "T4", time.time() + 3600, p["status"]),
           "PYTHONPATH": ROOT}
    proc = subprocess.Popen(tiny_command(CELL, root, epochs=50, sleep=0.3), env=env, cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):                                    # wait for the first epoch's save
        if os.path.exists(p["resume"]):
            break
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out
    assert "SIGTERM" in out and os.path.exists(p["resume"]) and not os.path.exists(p["results"])
    assert markers.state(root, CELL)[0] == "partial"


# ---- the whole path: deadline, resume, bit-for-bit, marker ------------------

def test_a_cell_cut_by_the_deadline_resumes_next_session_and_ends_where_an_uninterrupted_one_does(tmp_path):
    # session 1: a deadline two epochs in (each epoch sleeps 0.4 s)
    t0 = time.time()
    r1, root1 = _runner(tmp_path / "cut", t0, command_builder=builder(epochs=6, sleep=0.4))
    r1.deadline = t0 + 1.3
    r1.hard_limit = t0 + 600
    r1.run()
    state, _ = markers.state(root1, CELL)
    assert state == "partial", r1.summary
    saved = resume.load(markers.paths(root1, CELL)["resume"], "cpu")
    assert 0 < saved["epoch"] < 6, "the deadline should have cut the cell part-way"
    # session 2: plenty of time; the cell continues and finishes
    r2 = Runner("ACC1", {0: [CELL], 1: []}, root1, CFG, time.time(), ["Tesla T4"] * 2,
                predictor=ok_predictor, log=lambda *_: None,
                command_builder=builder(epochs=6, sleep=0.0))
    r2.run()
    assert markers.state(root1, CELL)[0] == "complete", r2.summary
    # an uninterrupted control
    r3, root3 = _runner(tmp_path / "straight", time.time(),
                        command_builder=builder(epochs=6, sleep=0.0))
    r3.run()
    assert _params(root1) == _params(root3), "resume through the harness changed the result"


def test_a_finished_cell_gets_a_marker_that_names_its_hashes_and_environment(tmp_path):
    r, root = _runner(tmp_path, time.time(), command_builder=builder(epochs=2))
    r.run()
    p = markers.paths(root, CELL)
    marker = json.load(open(p["marker"]))
    assert marker["checkpoint"]["sha256"] == markers.sha256_file(p["checkpoint"])
    assert marker["results"]["sha256"] == markers.sha256_file(p["results"])
    assert marker["predictions_sha256"] == "f" * 64
    assert marker["start_record"]["sha256"] == markers.sha256_file(p["start"])
    env = marker["environment"]
    assert {"python", "torch", "cuda", "cudnn", "vendored_sha256"} <= set(env)
    assert marker["account"] == "ACC1" and marker["gpu"] == "Tesla T4"
    assert not os.path.exists(p["resume"])


def test_no_marker_is_written_without_a_predictions_hash(tmp_path):
    failing = lambda cell, root, env: (None, "prediction step failed: no DGL")
    r, root = _runner(tmp_path, time.time(), predictor=failing,
                      command_builder=builder(epochs=2))
    r.run()
    assert not os.path.exists(markers.paths(root, CELL)["marker"])
    assert markers.state(root, CELL)[0] == "needs_finalize"
    assert r.summary[-1]["outcome"] == "TRAINED, NOT FINALIZED"
    # a later session finalises it without retraining
    r2 = Runner("ACC1", {0: [CELL], 1: []}, root, CFG, time.time(), ["Tesla T4"] * 2,
                predictor=ok_predictor, log=lambda *_: None,
                command_builder=lambda *a, **k: pytest.fail("retrained a finished cell"))
    r2.run()
    assert markers.state(root, CELL)[0] == "complete"


def test_a_complete_cell_is_skipped_and_a_tampered_one_is_refused(tmp_path):
    r, root = _runner(tmp_path, time.time(), command_builder=builder(epochs=2))
    r.run()
    r2 = Runner("ACC1", {0: [CELL], 1: []}, root, CFG, time.time(), ["T4", "T4"],
                predictor=ok_predictor, log=lambda *_: None,
                command_builder=lambda *a, **k: pytest.fail("trained a complete cell"))
    r2.run()
    assert r2.summary[0]["outcome"] == "skipped"
    p = markers.paths(root, CELL)
    open(p["checkpoint"], "ab").write(b"tamper")
    assert markers.state(root, CELL)[0] == "inconsistent"
    r2.summary.clear()
    r2.run()
    assert r2.summary[0]["outcome"] == "REFUSED"


def test_no_cell_is_started_once_the_deadline_has_passed(tmp_path):
    t0 = time.time()
    r, root = _runner(tmp_path, t0 - 11 * 3600, command_builder=lambda *a, **k: pytest.fail("started"))
    r.run()
    assert r.summary[0]["outcome"] == "not started"


def test_each_gpu_gets_its_own_process_and_device(tmp_path):
    env0 = child_env(0, CELL, "ACC1", "Tesla T4", 1.0, "s.json")
    env1 = child_env(1, CELL, "ACC1", "Tesla T4", 1.0, "s.json")
    assert env0["CUDA_VISIBLE_DEVICES"] == "0" and env1["CUDA_VISIBLE_DEVICES"] == "1"
    assert env0[resume.ENV_HARNESS] == "1"
    cells = [Cell("davis", "coldsite_dti", "random", 4), Cell("davis", "coldsite_dti", "cold_drug", 4)]
    launched = []

    class FakeProc:
        def __init__(self, cmd, env, **kw):
            launched.append((env["CUDA_VISIBLE_DEVICES"], cmd))
        def poll(self): return 0
        def wait(self): return 0
        def kill(self): pass

    root = str(tmp_path / "results")
    r = Runner("ACC1", {0: [cells[0]], 1: [cells[1]]}, root, CFG, time.time(), ["T4", "T4"],
               popen=FakeProc, predictor=ok_predictor, log=lambda *_: None)
    r.run()
    assert sorted(g for g, _ in launched) == ["0", "1"]


def test_a_stop_request_reaches_the_children_and_no_new_cell_starts(tmp_path):
    r, root = _runner(tmp_path, time.time(), cells=[CELL, Cell("davis", "coldsite_dti", "cold_drug", 4)],
                      command_builder=builder(epochs=40, sleep=0.3))
    import threading
    thread = threading.Thread(target=r.run)
    thread.start()
    p = markers.paths(root, CELL)
    for _ in range(100):
        if os.path.exists(p["resume"]):
            break
        time.sleep(0.1)
    r.request_stop()
    thread.join(timeout=60)
    assert not thread.is_alive()
    outcomes = [s["outcome"] for s in r.summary]
    assert "partial" in outcomes and outcomes.count("starting") == 1


# ---- restore ----------------------------------------------------------------

def _trained(tmp_path, name, **kw):
    r, root = _runner(tmp_path / name, time.time(), command_builder=builder(epochs=2), **kw)
    r.run()
    return root


def test_restore_copies_only_the_planned_cells_and_never_overwrites(tmp_path):
    src_root = _trained(tmp_path, "prev")
    other = Cell("davis", "coldsite_dti", "cold_drug", 5)
    stray = markers.paths(src_root, other)
    os.makedirs(stray["dir"], exist_ok=True)
    open(stray["checkpoint"], "w").write("stray")
    dest = str(tmp_path / "new" / "results")
    rep = restore.restore([CELL], dest, src_root)
    p = markers.paths(dest, CELL)
    assert os.path.exists(p["checkpoint"]) and os.path.exists(p["marker"])
    assert not os.path.exists(markers.paths(dest, other)["checkpoint"])
    assert markers.state(dest, CELL)[0] == "complete"
    open(p["checkpoint"], "w").write("local edit")
    rep2 = restore.restore([CELL], dest, src_root)
    assert open(p["checkpoint"]).read() == "local edit" and p["checkpoint"] in rep2["present"]


def test_restore_refuses_two_different_files_of_one_name(tmp_path):
    src_root = _trained(tmp_path, "prev")
    dup = tmp_path / "input2"
    shutil.copytree(src_root, dup)
    p = markers.paths(str(dup), CELL)
    open(p["checkpoint"], "ab").write(b"different")
    rep = restore.restore([CELL], str(tmp_path / "new"), str(tmp_path))
    assert any("different files" in x for x in rep["problems"])


def test_restore_removes_a_cell_whose_marker_disagrees_with_its_files(tmp_path):
    src_root = _trained(tmp_path, "prev")
    p = markers.paths(src_root, CELL)
    open(p["checkpoint"], "ab").write(b"corrupt")          # the marker still names the old hash
    dest = str(tmp_path / "new")
    rep = restore.restore([CELL], dest, src_root)
    assert rep["problems"] and rep["removed"]
    assert not os.path.exists(markers.paths(dest, CELL)["checkpoint"])


def test_restore_from_a_missing_folder_is_a_problem_not_a_silent_fresh_start(tmp_path):
    assert restore.restore([CELL], str(tmp_path), str(tmp_path / "nope"))["problems"]


# ---- marker ------------------------------------------------------------------

def test_a_marker_cannot_be_written_while_a_resume_file_exists(tmp_path):
    root = str(tmp_path)
    p = markers.paths(root, CELL)
    os.makedirs(p["dir"])
    for key in ("checkpoint", "results", "resume"):
        open(p[key], "w").write("x")
    with pytest.raises(RuntimeError, match="resume file"):
        markers.write_marker(root, CELL, "a" * 64, "n", "ACC1", "T4")


# ---- canary ------------------------------------------------------------------

def _committed(values_auroc, values_auprc):
    return {s: {"auroc": a, "auprc": b} for s, a, b in zip((1, 2, 3), values_auroc, values_auprc)}


def test_the_canary_passes_within_one_committed_seed_sd_on_both_metrics():
    committed = _committed([0.85, 0.86, 0.87], [0.40, 0.45, 0.50])           # SD 0.01, 0.05
    ok = canary.evaluate({"auroc": 0.855, "auprc": 0.42}, committed, 1)
    assert ok["verdict"] == "pass" and ok["metrics"]["auroc"]["tolerance_sd"] == pytest.approx(0.01)
    late = canary.evaluate({"auroc": 0.85 + 0.0101, "auprc": 0.40}, committed, 1)
    assert late["verdict"] == "fail"
    second = canary.evaluate({"auroc": 0.85, "auprc": 0.40 + 0.0501}, committed, 1)
    assert second["verdict"] == "fail", "both metrics must be within tolerance"


def test_a_zero_sd_makes_the_canary_inconclusive_with_no_invented_floor():
    committed = _committed([0.9, 0.9, 0.9], [0.4, 0.45, 0.5])
    assert canary.evaluate({"auroc": 0.9, "auprc": 0.4}, committed, 1)["verdict"] == "inconclusive"


def test_the_canary_needs_all_three_committed_seeds():
    with pytest.raises(ValueError):
        canary.evaluate({"auroc": 0.9, "auprc": 0.4}, {1: {"auroc": 0.9, "auprc": 0.4}}, 1)


def test_the_canary_uses_the_sample_sd_not_the_population_sd():
    committed = _committed([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])                 # sample SD = 1
    assert canary.evaluate({"auroc": 1.0, "auprc": 1.0}, committed, 2)["metrics"]["auroc"][
        "tolerance_sd"] == pytest.approx(1.0)


# ---- recipes -----------------------------------------------------------------

def _notebook_text():
    nb = json.load(open(os.path.join(ROOT, "notebooks", "kaggle_binary_grid.ipynb")))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("model", ["deepdta", "hyperattentiondti", "moltrans", "coldsite_dti"])
def test_a_harness_command_carries_the_flags_the_notebook_builds(model):
    text = _notebook_text()
    command = recipes.train_command(Cell("davis", model, "random", 4), "res", python="python")
    joined = " ".join(command)
    def notebook_has(flag, value):
        import re
        return re.search(rf"'{re.escape(flag)}',\s*(?:str\()?'?{re.escape(str(value))}", text) is not None
    for flag in ("--min-epochs", "--epochs"):
        assert notebook_has(flag, command[command.index(flag) + 1]), flag
    if "--patience" in command:
        assert notebook_has("--patience", command[command.index("--patience") + 1])
    assert "--amp" not in joined                                            # DAVIS: full precision
    assert "res/davis_binary" in joined


def test_amp_is_on_for_kiba_except_moltrans_and_off_for_davis():
    for model, expected in (("deepdta", True), ("coldsite_dti", True), ("hyperattentiondti", True),
                            ("drugban", True), ("moltrans", False)):
        cmd = recipes.train_command(Cell("kiba", model, "random", 1), "res", "python")
        assert ("--amp" in cmd) is expected, model
    assert "--amp" not in recipes.train_command(Cell("davis", "drugban", "random", 4), "res", "python")


def test_the_batch_sizes_are_the_notebooks_for_a_16_gb_card():
    text = _notebook_text()
    assert "COLDSITE_BATCH = 64 if big" in text and recipes.COLDSITE_BATCH == 64
    assert "HAT_BATCH, HAT_ACCUM = (32, 1) if big" in text and (recipes.HAT_BATCH, recipes.HAT_ACCUM) == (32, 1)
    assert "DEEPDTA_BATCH = 256" in text and "MOLTRANS_BATCH = 16" in text


# ---- the generated notebooks -------------------------------------------------

def _load_builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_harness_notebook", os.path.join(ROOT, "notebooks", "build_harness_notebook.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(nb, kind="code"):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def test_a_generated_notebook_ships_in_dry_run_pinned_to_a_commit_and_without_credentials():
    builder_module = _load_builder()
    nb = builder_module.build("CANARY", "config/canary_wave.json", "a" * 40, canary=True)
    code_cells = "\n".join(_cells(nb))
    assert "DRY_RUN = True" in code_cells
    assert f"COMMIT = '{'a' * 40}'" in code_cells and "assert head == COMMIT" in code_cells
    for forbidden in ("kaggle.json", "KAGGLE_KEY", "KAGGLE_USERNAME", "api_key", "token", "password"):
        assert forbidden.lower() not in json.dumps(nb).lower(), forbidden
    assert "src.cloud.runner" in code_cells and "src.cloud.canary" in code_cells
    assert "DDP" not in code_cells and "DataParallel" not in code_cells


def test_only_a_drugban_account_installs_dgl(tmp_path):
    builder_module = _load_builder()
    waves = tmp_path / "w.json"
    waves.write_text(json.dumps({"wave": "A", "accounts": {
        "ACC1": {"gpu0": [{"dataset": "kiba", "model": "drugban", "level": "random", "seed": 1}]},
        "ACC2": {"gpu0": [{"dataset": "davis", "model": "moltrans", "level": "random", "seed": 4}]}}}))
    rel = os.path.relpath(str(waves), ROOT)
    with_dgl = "\n".join(_cells(builder_module.build("ACC1", rel, "b" * 40, False)))
    without = "\n".join(_cells(builder_module.build("ACC2", rel, "b" * 40, False)))
    assert "torch==2.6.0" in with_dgl and "dgl" in with_dgl
    assert "torch==2.6.0" not in without


def test_the_runner_command_a_notebook_builds_runs_a_dry_run_and_stops_before_gpu_work(tmp_path):
    if not os.path.isdir(os.path.join(ROOT, "data", "splits", "davis")):
        pytest.skip("splits not built on this machine")
    builder_module = _load_builder()
    nb = builder_module.build("CANARY", "config/canary_wave.json", "c" * 40, canary=True)
    settings = _cells(nb)[0]
    runner_cell = [c for c in _cells(nb) if "src.cloud.runner" in c][0]
    captured = {}

    class FakeSubprocess:
        @staticmethod
        def run(cmd, *a, **k):
            captured["cmd"] = cmd
            class R: returncode = 0
            return R()

    scope = {"os": os, "sys": sys, "time": time, "subprocess": FakeSubprocess,
             "WORK": str(tmp_path), "START": 1.0}
    exec(settings, scope)
    exec(runner_cell, scope)
    cmd = captured["cmd"]
    assert "--dry-run" in cmd and "--restore-from" not in cmd
    out = subprocess.run([sys.executable if c == scope["sys"].executable else c for c in cmd],
                         capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": ROOT})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "dry run: exiting before any GPU work" in out.stdout
    assert "SKIPPED  gpus" in out.stdout


def test_the_canary_wave_is_one_committed_dav_cell_of_the_cheapest_attention_model():
    waves = load_waves(os.path.join(ROOT, "config", "canary_wave.json"))
    cells = [c for q in waves["accounts"]["CANARY"].values() for c in q]
    assert cells == [Cell("davis", "coldsite_dti", "cold_target", 1)]
    ref = canary.load_reference(os.path.join(ROOT, "config", "canary_reference.json"))
    assert (ref["dataset"], ref["level"], ref["model"]) == ("davis", "cold_target", "coldsite_dti")
    assert sorted(ref["by_seed"]) == [1, 2, 3]
