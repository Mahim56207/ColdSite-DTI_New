"""The epoch-1 gate: does the harness start training the way the committed cells started, and do the
new seeds really differ from each other? (user request 2026-09-29; complements the canary, amendment section 6)

The canary retrains one committed cell to the end (about an hour on a T4). This gate is the cheap check that
runs beside it: ONE epoch each of the canary cell at the committed seed and at every new wave seed, all
under the harness, then two rules, fixed here before any number exists.

  Gate A (replay, seed 1). The replayed seed's epoch-1 train loss, validation loss and validation AUROC
      must each lie within ONE sample SD (ddof = 1, over the three committed seeds' epoch-1 values) of the
      committed seed-1 value. SD = 0 makes the gate inconclusive; no floor is invented.
      This is what "matches the expected trajectory of seed 1" can honestly mean: the same seed replays the
      same start. It cannot be asked of a DIFFERENT seed, whose start is supposed to differ.
  Gate B (each new seed, e.g. 4 and 5). (i) every metric finite; (ii) every metric inside the envelope of
      the three committed seeds' epoch-1 values, widened by one SD on each side (a new seed is a fourth
      draw from the same process, so it must look like one); (iii) DISTINCT: its initial-weight hash differs
      from every other run in the gate, and its three metrics are not identical to another run's. (iii) is
      the signature of the MolTrans defect (three seeds, one training run), which passed every accuracy check
      until the three AUROCs were compared.

A pass, copied to `config/epoch_gate_verdict.json` and committed, is required (with the canary's) by the
pre-flight for every wave: the verdict must be for this cell, carry this harness's hash, and cover every new
seed (seed > 3) the account trains. Editing the harness files stales it, exactly like the canary.

    python -m src.cloud.epoch_gate --write-reference --committed ~/ColdSite-results
    python -m src.cloud.epoch_gate --run --seeds 1 4 5 --root results/epoch_gate --out results/epoch_gate_verdict.json
    python -m src.cloud.epoch_gate --evaluate --seeds 1 4 5 --root results/epoch_gate --out ...   # runs already done
    exit 0 pass, 1 fail, 3 inconclusive
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

from src.cloud import markers, recipes
from src.cloud.config import Cell
from src.model import resume

METRICS = ("train_loss", "val_loss", "auroc")
COMMITTED_SEEDS = (1, 2, 3)
REPLAY_SEED = 1
DEFAULT_SEEDS = (1, 4, 5)
IDENTICAL_TOL = 1e-9
REFERENCE_PATH = os.path.join("config", "epoch1_reference.json")
VERDICT_PATH = os.path.join("config", "epoch_gate_verdict.json")


def _first_epoch(history: list) -> dict:
    for entry in history:
        if int(entry.get("epoch", 0)) == 1:
            return {m: float(entry[m]) for m in METRICS}
    raise ValueError("history has no epoch 1")


def committed_epoch1(committed_dir: str, dataset: str, level: str, model: str, seed: int) -> tuple:
    """(epoch-1 metrics, history path) of a committed cell, from its `_history.json`."""
    path = markers.paths(committed_dir, Cell(dataset, model, level, seed))["history"]
    with open(path) as handle:
        return _first_epoch(json.load(handle)), path


def write_reference(committed_dir: str, path: str = REFERENCE_PATH) -> dict:
    """Freeze the committed seeds' epoch-1 values, for the SAME cell the canary uses."""
    from src.cloud import canary
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; delete it deliberately to re-freeze the reference")
    cref = canary.load_reference()
    seeds = {}
    for seed in COMMITTED_SEEDS:
        metrics, source = committed_epoch1(committed_dir, cref["dataset"], cref["level"], cref["model"], seed)
        seeds[str(seed)] = {**metrics, "source_file": os.path.basename(source),
                            "source_sha256": markers.sha256_file(source)}
    reference = {"dataset": cref["dataset"], "level": cref["level"], "model": cref["model"],
                 "metrics": list(METRICS), "seeds": seeds,
                 "rule": "src/cloud/epoch_gate.py docstring (Gate A: one SD; Gate B: envelope +/- one SD, distinct)"}
    with open(path, "w") as handle:
        json.dump(reference, handle, indent=1)
        handle.write("\n")
    return reference


def load_reference(path: str = REFERENCE_PATH) -> dict:
    with open(path) as handle:
        ref = json.load(handle)
    ref["by_seed"] = {int(s): {m: float(v[m]) for m in METRICS} for s, v in ref["seeds"].items()}
    return ref


def read_epoch1(results_root: str, cell: Cell) -> dict:
    """Epoch-1 metrics and initial-weight hash of a run stopped after its first epoch: the metrics are in
    the resume file's history (`--stop-after-epoch` leaves it), the hash in the harness's start record."""
    p = markers.paths(results_root, cell)
    state = resume.load(p["resume"], "cpu")
    if state is None:
        raise FileNotFoundError(f"no resume file for {cell.id}: {p['resume']}")
    metrics = _first_epoch(state.get("extra", {}).get("history", []))
    with open(p["start"]) as handle:
        start = json.load(handle)
    return {**metrics, "initial_weight_hash": start.get("initial_weight_hash"), "seed": start.get("seed", cell.seed)}


def _overall(verdicts) -> str:
    verdicts = list(verdicts)
    return ("fail" if "fail" in verdicts else "inconclusive" if "inconclusive" in verdicts else "pass")


def evaluate(new: dict, ref_by_seed: dict, replay_seed: int = REPLAY_SEED, wave_seeds=()) -> dict:
    """new: {seed: {metric: value, 'initial_weight_hash': str}} for the replay seed and the wave seeds."""
    if set(ref_by_seed) != set(COMMITTED_SEEDS):
        raise ValueError(f"need the three committed seeds {COMMITTED_SEEDS}, got {sorted(ref_by_seed)}")
    missing = [s for s in (replay_seed, *wave_seeds) if s not in new]
    if missing:
        raise ValueError(f"no run for seed(s) {missing}")
    spread = {}
    for m in METRICS:
        values = [ref_by_seed[s][m] for s in COMMITTED_SEEDS]
        spread[m] = {"sd": statistics.stdev(values), "min": min(values), "max": max(values)}
    result = {"replay_seed": replay_seed, "reference_spread": spread, "gate_a": {}, "gate_b": {}}
    for m in METRICS:
        got, reference, sd = new[replay_seed][m], ref_by_seed[replay_seed][m], spread[m]["sd"]
        verdict = ("fail" if not math.isfinite(got) else "inconclusive" if sd == 0
                   else "pass" if abs(got - reference) <= sd else "fail")
        result["gate_a"][m] = {"committed": reference, "new": got, "difference": got - reference,
                               "tolerance_sd": sd, "verdict": verdict}
    result["gate_a"]["verdict"] = _overall(v["verdict"] for m, v in result["gate_a"].items() if m in METRICS)
    for seed in wave_seeds:
        rows = {}
        for m in METRICS:
            got = new[seed][m]
            low, high = spread[m]["min"] - spread[m]["sd"], spread[m]["max"] + spread[m]["sd"]
            ok = math.isfinite(got) and low <= got <= high
            rows[m] = {"new": got, "envelope": [low, high], "verdict": "pass" if ok else "fail"}
        others = [t for t in new if t != seed]
        digest = new[seed].get("initial_weight_hash")
        same_hash = [t for t in others if digest and new[t].get("initial_weight_hash") == digest]
        same_metrics = [t for t in others
                        if all(abs(new[t][m] - new[seed][m]) <= IDENTICAL_TOL for m in METRICS)]
        distinct = ("fail" if (same_hash or same_metrics) else "inconclusive" if not digest else "pass")
        rows["distinct"] = {"same_initial_weights_as": same_hash, "identical_metrics_to": same_metrics,
                            "initial_weight_hash": digest, "verdict": distinct}
        rows["verdict"] = _overall(v["verdict"] for v in rows.values())
        result["gate_b"][str(seed)] = rows
    result["verdict"] = _overall([result["gate_a"]["verdict"], *(v["verdict"] for v in result["gate_b"].values())])
    return result


def epoch1_command(cell: Cell, results_root: str, python: str | None = None) -> list:
    """The cell's training command with `--stop-after-epoch 1`. For ColdSite-DTI it is built by
    `run_grid.train_command`, the function the real cell's launcher uses, so no training code is touched."""
    if cell.model == "coldsite_dti":
        from src.model import run_grid
        out = markers.paths(results_root, cell)["dir"]
        extra = ("--batch-size", str(recipes.COLDSITE_BATCH), "--min-epochs", "10", "--stop-after-epoch", "1")
        if recipes.uses_amp(cell):
            extra += ("--amp",)
        command = run_grid.train_command({"dataset": cell.dataset, "split": cell.level, "seed": cell.seed},
                                         "data/splits", out, "binary", 100, extra)
        if python:
            command[0] = python
        return command
    return [*recipes.train_command(cell, results_root, python), "--stop-after-epoch", "1"]


def run_gate(cells: list, results_root: str, python: str | None = None, run=subprocess.run,
             n_gpus: int = 2, clock=time.time) -> None:
    """Train each cell for one epoch, one process per GPU, sequentially within a GPU. Refuses to run over
    an earlier gate's files: a stale resume file would resume instead of starting."""
    from src.cloud.runner import child_env
    for cell in cells:
        p = markers.paths(results_root, cell)
        stale = [k for k in ("resume", "start", "checkpoint") if os.path.exists(p[k])]
        if stale:
            raise SystemExit(f"{cell.id}: {stale} already exist under {results_root}; delete the folder "
                             f"deliberately to run the gate again")
    import threading
    failures = []

    def worker(gpu: int, mine: list) -> None:
        for cell in mine:
            p = markers.paths(results_root, cell)
            os.makedirs(p["dir"], exist_ok=True)
            env = child_env(gpu, cell, "EPOCH_GATE", f"gpu{gpu}", clock() + 86400.0, p["status"])
            with open(os.path.join(results_root, f"epoch_gate_gpu{gpu}.log"), "a") as log:
                done = run(epoch1_command(cell, results_root, python), env=env, stdout=log,
                           stderr=subprocess.STDOUT)
            if done.returncode != 0:
                failures.append(f"{cell.id}: trainer exited {done.returncode} (see epoch_gate_gpu{gpu}.log)")

    queues = {g: [c for i, c in enumerate(cells) if i % n_gpus == g] for g in range(n_gpus)}
    threads = [threading.Thread(target=worker, args=(g, q)) for g, q in queues.items() if q]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise SystemExit("epoch gate training failed: " + "; ".join(failures))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-reference", action="store_true")
    ap.add_argument("--committed", help="~/ColdSite-results (only to write the reference)")
    ap.add_argument("--run", action="store_true", help="train one epoch per seed, then evaluate")
    ap.add_argument("--evaluate", action="store_true", help="evaluate runs that already exist")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                    help="the replay seed (1) and the new wave seeds")
    ap.add_argument("--root", default=os.path.join("results", "epoch_gate"))
    ap.add_argument("--reference", default=REFERENCE_PATH)
    ap.add_argument("--out", help="write the verdict here (JSON)")
    args = ap.parse_args(argv)
    if args.write_reference:
        if not args.committed:
            ap.error("--write-reference needs --committed")
        ref = write_reference(os.path.expanduser(args.committed), args.reference)
        print(f"wrote {args.reference}: {ref['model']} {ref['dataset']} {ref['level']}, seeds {sorted(ref['seeds'])}")
        return 0
    if not (args.run or args.evaluate):
        ap.error("one of --write-reference, --run, --evaluate is required")
    from src.cloud import canary
    ref = load_reference(args.reference)
    seeds = list(dict.fromkeys(args.seeds))
    if REPLAY_SEED not in seeds:
        ap.error(f"--seeds must include the replay seed {REPLAY_SEED}")
    wave_seeds = [s for s in seeds if s not in COMMITTED_SEEDS]
    cells = [Cell(ref["dataset"], ref["model"], ref["level"], s) for s in seeds]
    if args.run:
        run_gate(cells, args.root)
    new = {c.seed: read_epoch1(args.root, c) for c in cells}
    result = evaluate(new, ref["by_seed"], REPLAY_SEED, wave_seeds)
    result.update(dataset=ref["dataset"], level=ref["level"], model=ref["model"], seeds_checked=wave_seeds,
                  runs={str(s): v for s, v in new.items()}, harness_sha256=canary.harness_hash(),
                  reference_sha256=markers.sha256_file(args.reference))
    print(json.dumps(result, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return {"pass": 0, "fail": 1, "inconclusive": 3}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
