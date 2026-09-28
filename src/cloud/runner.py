"""One process per GPU, cells from the wave manifest, pre-flight first, self-stop, markers.

    python -m src.cloud.runner --account ACC1 --waves config/waves.json \
        --results-root /kaggle/working/results --restore-from /kaggle/input --session-start <unix>
    python -m src.cloud.runner ... --dry-run        # pre-flight and the plan, then exit: no GPU work

What it does, in order (docs/REMEDIATION_PLAN.md, cloud_rules):
  1. restore this account's cells from the previous version's output, if asked;
  2. pre-flight (src/cloud/preflight.py): any failure exits 2 before a trainer is started;
  3. finalise cells a previous session trained but did not mark (predictions hash, marker);
  4. one worker per GPU, each a separate process with CUDA_VISIBLE_DEVICES set to its own
     device (no DDP, no DataParallel: that would change the recipes and the 84 cells' comparability),
     running the queue the manifest gives that GPU, cheapest-state first: complete cells are
     skipped, partial ones resume (the trainers' own epoch-level resume), fresh ones start;
  5. the trainers stop themselves between epochs when the next one would end after
     `session_start + limit - margin` (src/model/resume.py), and on SIGTERM/SIGINT finish the
     epoch in progress and save; this process forwards SIGTERM, and kills only at the hard limit;
  6. a finished cell gets its test predictions hashed and a complete-marker written;
  7. the weekly quota (src/cloud/quota.py) is counted in a ledger that travels with the results: no cell
     starts once the usable hours are spent, a fresh one not near them, and the trainers' stop time is
     the earlier of the session deadline and the moment the usable hours run out, so a cell pauses
     between epochs and resumes next week.

It never touches the recipes: the command for a cell is `src/cloud/recipes.py`, equal to the
notebooks'. It has no Kaggle credentials and needs none.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

from src.cloud import markers, preflight
from src.cloud import quota as quota_mod
from src.cloud.config import Cell, load_harness, stop_at
from src.cloud.recipes import train_command
from src.cloud.restore import restore
from src.model import resume


def child_env(gpu_index: int, cell: Cell, account: str, gpu_name: str, deadline: float,
              status_path: str) -> dict:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index), "PYTHONUNBUFFERED": "1",
            resume.ENV_HARNESS: "1", resume.ENV_STOP_AT: f"{deadline:.3f}",
            resume.ENV_STATUS: status_path,
            resume.ENV_CONTEXT: json.dumps({"cell": cell.id, "account": account, "gpu": gpu_name,
                                                "dataset": cell.dataset, "model": cell.model,
                                                "level": cell.level, "seed": cell.seed})}


def known_epoch_seconds(results_root: str, cell: Cell) -> float:
    """The longest epoch a previous session measured for this cell, from its status file."""
    try:
        with open(markers.paths(results_root, cell)["status"]) as handle:
            return float(json.load(handle).get("mean_epoch_s") or 0.0)
    except (OSError, ValueError):
        return 0.0


def predictions_step(cell: Cell, results_root: str, env: dict, python: str | None = None) -> tuple:
    """Hash the cell's test-set predictions (T04's `predict_cell`, run in a child process on
    the worker's own GPU). Returns (sha256 | None, note)."""
    out_dir = os.path.join(results_root, "predictions_v2")
    code = ("import sys, json; from src.evaluation.accuracy_table import predict_cell; "
            f"cid = predict_cell({cell.dataset!r}, {cell.model!r}, {cell.level!r}, {cell.seed}, "
            f"{results_root!r}, {out_dir!r}, 'cuda' if __import__('torch').cuda.is_available() "
            f"else 'cpu'); print('CID', cid)")
    out = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True,
                         env=env)
    meta_path = os.path.join(out_dir, "predictions", f"{cell.id}.meta.json")
    if out.returncode != 0 or not os.path.exists(meta_path):
        tail = ((out.stderr or "").strip().splitlines() or ["?"])[-1]
        return None, f"prediction step failed: {tail}"
    with open(meta_path) as handle:
        return json.load(handle)["predictions_sha256"], f"predictions_v2/predictions/{cell.id}.csv.gz"


def finalize(cell: Cell, results_root: str, account: str, gpu_name: str, env: dict,
             predictor=predictions_step) -> tuple:
    """Write the complete-marker for a trained cell. No marker without a predictions hash: a
    cell whose predictions cannot be produced stays 'needs_finalize' and is reported."""
    sha, note = predictor(cell, results_root, env)
    if sha is None:
        return False, note
    markers.write_marker(results_root, cell, sha, note, account, gpu_name)
    return True, note


class Runner:
    def __init__(self, account: str, cells_by_gpu: dict, results_root: str, cfg: dict,
                 session_start: float, gpu_names: list, python: str | None = None,
                 predictor=predictions_step, popen=subprocess.Popen, clock=time.time,
                 log=print, command_builder=train_command, quota="auto",
                 heartbeat_s: float = quota_mod.HEARTBEAT_S):
        self.account, self.queues, self.root, self.cfg = account, cells_by_gpu, results_root, cfg
        self.start, self.gpu_names, self.python = session_start, gpu_names, python
        self.predictor, self.popen, self.clock, self.log = predictor, popen, clock, log
        self.command_builder = command_builder
        self.deadline = stop_at(session_start, cfg)
        self.hard_limit = session_start + cfg["session_limit_hours"] * 3600
        # quota="auto": built from the harness config (None when it declares no weekly quota);
        # pass None to switch it off or a Quota to inject one.
        self.quota = (quota_mod.Quota.from_harness(cfg, results_root, account, clock)
                      if quota == "auto" else quota)
        self.heartbeat_s = heartbeat_s
        self.children: dict = {}
        self.stop_requested = False
        self.summary: list = []
        self._lock = threading.Lock()

    def request_stop(self, *_):
        """SIGTERM/SIGINT to this process: pass it on; each trainer finishes its epoch and saves."""
        self.stop_requested = True
        for proc in list(self.children.values()):
            if proc.poll() is None:
                proc.terminate()

    def _record(self, cell, outcome, detail=""):
        with self._lock:
            self.summary.append({"cell": cell.id, "outcome": outcome, "detail": detail})
        self.log(f"[{cell.id}] {outcome} {detail}".rstrip())

    def run_cell(self, gpu: int, cell: Cell) -> None:
        st, why = markers.state(self.root, cell)
        gpu_name = self.gpu_names[gpu] if gpu < len(self.gpu_names) else f"gpu{gpu}"
        p = markers.paths(self.root, cell)
        env = child_env(gpu, cell, self.account, gpu_name, self.deadline, p["status"])
        if st == "complete":
            return self._record(cell, "skipped", "complete-marker present")
        if st == "inconsistent":
            return self._record(cell, "REFUSED", why)
        if st == "needs_finalize":
            ok, note = finalize(cell, self.root, self.account, gpu_name, env, self.predictor)
            return self._record(cell, "finalized" if ok else "NOT FINALIZED", note)
        if self.stop_requested or self.clock() >= self.deadline - known_epoch_seconds(self.root, cell):
            return self._record(cell, "not started", "no time left before the self-stop")
        deadline = self.deadline
        if self.quota is not None:
            allowed, why_not = self.quota.can_start(known_epoch_seconds(self.root, cell),
                                                    resume=st == "partial", now=self.clock())
            if not allowed:
                return self._record(cell, "not started (quota)", why_not)
            deadline = min(self.deadline, self.quota.stop_at(self.clock()))
            env = child_env(gpu, cell, self.account, gpu_name, deadline, p["status"])
        os.makedirs(p["dir"], exist_ok=True)
        command = self.command_builder(cell, self.root, self.python)
        self._record(cell, "resuming" if st == "partial" else "starting", " ".join(command[3:7]))
        interval = self.quota.ledger.open(cell.id, gpu) if self.quota is not None else None
        last_beat, asked_to_stop = self.clock(), False
        try:
            with open(os.path.join(self.root, f"runner_gpu{gpu}.log"), "a") as log_file:
                proc = self.popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
                self.children[gpu] = proc
                while proc.poll() is None:
                    now = self.clock()
                    if now >= self.hard_limit:      # the trainer's own stop failed
                        proc.kill()
                        self._record(cell, "KILLED", "hard session limit reached")
                        break
                    if self.quota is not None and now - last_beat >= self.heartbeat_s:
                        self.quota.ledger.beat(interval)
                        last_beat = now
                        if self.quota.over_nominal(now):      # past the whole weekly quota: no epoch is worth it
                            proc.kill()
                            self._record(cell, "KILLED", "weekly quota reached while training")
                            break
                        if not asked_to_stop and self.quota.exhausted(now):
                            asked_to_stop = True              # finish the epoch, save, stop (SIGTERM)
                            self.log(f"[{cell.id}] usable quota spent: asking the trainer to stop after this epoch")
                            proc.terminate()
                    time.sleep(0.2)
            code = proc.wait()
        finally:
            if interval is not None:
                self.quota.ledger.close(interval)
        st_after, _ = markers.state(self.root, cell)
        if code != 0:
            return self._record(cell, "FAILED", f"trainer exited {code}; see runner_gpu{gpu}.log")
        if st_after == "needs_finalize":
            ok, note = finalize(cell, self.root, self.account, gpu_name, env, self.predictor)
            return self._record(cell, "complete" if ok else "TRAINED, NOT FINALIZED", note)
        if st_after == "partial":
            if self.quota is not None and deadline < self.deadline:
                return self._record(cell, "paused (quota)",
                                    "stopped between epochs at the weekly quota; resumes when the quota allows")
            return self._record(cell, "partial", "stopped between epochs; resumes next session")
        self._record(cell, "unexpected state", f"{st_after}")

    def work(self, gpu: int) -> None:
        for cell in self.queues.get(gpu, []):
            if self.stop_requested:
                break
            self.run_cell(gpu, cell)

    def run(self) -> list:
        threads = [threading.Thread(target=self.work, args=(g,), daemon=True)
                   for g in sorted(self.queues) if self.queues[g]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return self.summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--account", required=True)
    ap.add_argument("--waves", default="config/waves.json")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--restore-from", default=None,
                    help="e.g. /kaggle/input: the previous version's output, attached as a dataset")
    ap.add_argument("--session-start", type=float, default=None,
                    help="unix time this session began (default: now)")
    ap.add_argument("--dry-run", action="store_true",
                    help="pre-flight and the plan, then exit before any GPU work")
    args = ap.parse_args(argv)

    cfg = load_harness()
    start = args.session_start or time.time()
    waves = preflight.load_waves(args.waves) if os.path.exists(args.waves) else None
    if waves and args.account in waves["accounts"] and args.restore_from and not args.dry_run:
        mine = [c for q in waves["accounts"][args.account].values() for c in q]
        rep = restore(mine, args.results_root, args.restore_from)
        print(f"restore: copied {len(rep['copied'])}, already present {len(rep['present'])}, "
              f"removed {len(rep['removed'])}, problems {rep['problems']}")
        if rep["problems"]:
            return 2

    if args.restore_from and not args.dry_run:
        led = quota_mod.restore_ledger(args.account, args.results_root, args.restore_from)
        print(f"quota ledger: merged {led['merged_files']} file(s), {led['new_entries']} new entr(ies), "
              f"problems {led['problems']}")
        if led["problems"]:
            return 2

    report = preflight.run_preflight(args.account, args.waves, args.results_root, args.dry_run, cfg)
    print(preflight.format_report(report))
    if not report["ok"]:
        return 2
    if args.dry_run:
        print("dry run: exiting before any GPU work.")
        return 0

    names = preflight.probe_gpus()
    queues = {i: report_queue for i, report_queue in
              enumerate([waves["accounts"][args.account]["gpu0"],
                         waves["accounts"][args.account]["gpu1"]])}
    runner = Runner(args.account, queues, args.results_root, cfg, start, names)
    signal.signal(signal.SIGTERM, runner.request_stop)
    signal.signal(signal.SIGINT, runner.request_stop)
    summary = runner.run()
    path = os.path.join(args.results_root, f"runner_summary_{args.account}.json")
    with open(path, "w") as handle:
        json.dump({"account": args.account, "session_start": start, "cells": summary,
                   "quota": runner.quota.report() if runner.quota is not None else None}, handle, indent=1)
    bad = [s for s in summary if s["outcome"] in ("FAILED", "REFUSED", "KILLED", "NOT FINALIZED",
                                                    "TRAINED, NOT FINALIZED", "unexpected state")]
    print(f"summary -> {path}; {len(summary)} cell(s), {len(bad)} needing attention")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
