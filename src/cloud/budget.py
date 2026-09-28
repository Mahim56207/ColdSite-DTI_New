"""Wave A budget and partition (remediation task T10).

    python -m src.cloud.budget --write            # writes config/waves.json and config/wave_budget.json
    python -m src.cloud.budget --check            # exits 1 if either file differs from what this computes

Every hour below is derived, in this order, from files in the repository and from no other number:

  seconds per training batch   `results/speed_test_kiba_t4.md`, the `fp32` row of each model (measured on a
                               Tesla T4, `python -m src.model.benchmark_speed`, the grid's batch sizes)
  batches per epoch            training rows / batch size (`recipes.py`; the trainers' own batch sizes)
  overhead of validation       the ratio of the epoch time the Kaggle logs of 2026-09-12 gave (CLAUDE.md section 4:
                               ColdSite-DTI ~2 min 15 s, HyperAttentionDTI ~6 min, MolTrans ~5 min on DAVIS `random`)
                               to batches x s/batch, floored at 1
  epochs a cell will run       `best_epoch` of the model's three committed seeds at the same level, plus the patience
                               after which early stopping fires (`early_stopping.py`: `epoch - best_epoch >= patience`;
                               ColdSite-DTI's own histories show +16 in 18 of 18 cells). Mean, min and max of the three.
  rows                         `n_train_rows` recorded in the committed `_results.json` of the same level

No number for DrugBAN exists in the repository (nothing recorded its training wall clock; `docs/inventory.md` row
"101 GPU-hours ... NOT FOUND"), so its cells carry `hours: null` and the plan schedules a timed smoke run before
anything is queued against them. Nothing here assumes a weekly quota: the plan states the quota each account needs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

from src.cloud.config import Cell, load_harness

SPEED_MD = os.path.join("results", "speed_test_kiba_t4.md")
COMMITTED = os.path.expanduser("~/ColdSite-results")
WAVES_PATH = os.path.join("config", "waves.json")
BUDGET_PATH = os.path.join("config", "wave_budget.json")

BATCH = {"coldsite_dti": 64, "hyperattentiondti": 32, "moltrans": 16}
DROP_LAST = {"coldsite_dti": False, "hyperattentiondti": False, "moltrans": True}
PATIENCE_OFFSET = {"coldsite_dti": 16, "hyperattentiondti": 15, "moltrans": 15}
RESULT_SUFFIX = {"coldsite_dti": "", "hyperattentiondti": "_hyperattentiondti", "moltrans": "_moltrans"}
# Kaggle-log epoch seconds on DAVIS `random` (21,039 rows), CLAUDE.md section 4
KAGGLE_EPOCH_SECONDS = {"coldsite_dti": 135.0, "hyperattentiondti": 360.0, "moltrans": 300.0}
LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
RESERVE = 0.15                    # docs/REMEDIATION_PLAN.md T10: >= 15 % reserve
SEEDS_DAVIS = (4, 5)
DRUGBAN_KIBA = [(level, seed) for level in ("random", "cold_drug") for seed in (1, 2, 3)]


def parse_speed(path: str = SPEED_MD) -> dict:
    """model -> seconds per training batch, from the `fp32` rows of the speed table."""
    speeds = {}
    section = None
    for line in open(path):
        if line.startswith("#"):
            section = line.strip()
        # only the first table (s/batch); the projection table further down also has fp32 rows
        if section and section.startswith("# Speed test"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3 and cells[1] == "fp32" and cells[0] in BATCH:
                speeds[cells[0]] = float(cells[2])
    missing = set(BATCH) - set(speeds)
    if missing:
        raise ValueError(f"{path} has no fp32 row for {sorted(missing)}")
    return speeds


def batches(model: str, rows: int) -> int:
    return rows // BATCH[model] if DROP_LAST[model] else math.ceil(rows / BATCH[model])


def overhead(model: str, speeds: dict, random_rows: int) -> float:
    computed = batches(model, random_rows) * speeds[model]
    return max(1.0, KAGGLE_EPOCH_SECONDS[model] / computed)


def committed(committed_dir: str, level: str, model: str, seed: int) -> dict:
    name = f"davis_{level}_binary_seed{seed}{RESULT_SUFFIX[model]}_results.json"
    with open(os.path.join(committed_dir, "davis_binary", name)) as handle:
        return json.load(handle)


def estimate(model: str, level: str, speeds: dict, committed_dir: str = COMMITTED) -> dict:
    """Hours for one DAVIS cell of a measured model."""
    runs = [committed(committed_dir, level, model, s) for s in (1, 2, 3)]
    rows = {r["n_train_rows"] for r in runs}
    if len(rows) != 1:
        raise ValueError(f"{model}/{level}: committed seeds disagree on n_train_rows {rows}")
    rows = rows.pop()
    random_rows = committed(committed_dir, "random", model, 1)["n_train_rows"]
    epoch_s = batches(model, rows) * speeds[model] * overhead(model, speeds, random_rows)
    epochs = [r["best_epoch"] + PATIENCE_OFFSET[model] for r in runs]
    out = {"rows": rows, "epoch_seconds": round(epoch_s, 1),
           "epochs_mean": round(sum(epochs) / 3, 1), "epochs_min": min(epochs),
           "epochs_max": max(epochs)}
    out.update(hours_mean=round(epoch_s * sum(epochs) / 3 / 3600, 2),
               hours_low=round(epoch_s * min(epochs) / 3600, 2),
               hours_high=round(epoch_s * max(epochs) / 3600, 2))
    return out


KIBA_LEVELS = ("random", "cold_drug")
KIBA_BATCH = {"deepdta": 256, **BATCH}
KIBA_PATIENCE_OFFSET = {"deepdta": 10, **PATIENCE_OFFSET}
KIBA_SUFFIX = {"deepdta": "_deepdta", **RESULT_SUFFIX}
REPORT_KIBA_GPU_HOURS = 101.0     # paper/PROJECT_REPORT.md:88, a claim the repo cannot verify (docs/inventory.md)


def crosscheck_kiba(committed_dir: str = COMMITTED, speed_md: str = SPEED_MD) -> dict:
    """The same method applied to the 24 KIBA cells that were actually trained, to be read against the
    report's ~101 GPU-hour aggregate: a check of the method, never an input to the plan. Settings follow the
    KIBA protocol (mixed precision except MolTrans, `recipes.uses_amp`)."""
    speeds, section = {}, None
    for line in open(speed_md):
        if line.startswith("#"):
            section = line.strip()
        if section and section.startswith("# Speed test"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 3 and c[0] in KIBA_BATCH and c[1] in ("fp32", "amp+benchmark"):
                speeds[(c[0], c[1])] = float(c[2])
    total, per_model = 0.0, {}
    for model in KIBA_BATCH:
        setting = "fp32" if model == "moltrans" else "amp+benchmark"
        hours = 0.0
        for level in KIBA_LEVELS:
            for seed in (1, 2, 3):
                path = os.path.join(committed_dir, "kiba_binary",
                                    f"kiba_{level}_binary_seed{seed}{KIBA_SUFFIX[model]}_results.json")
                r = json.load(open(path))
                rows = r["n_train_rows"]
                n = rows // KIBA_BATCH[model] if model == "moltrans" else math.ceil(rows / KIBA_BATCH[model])
                hours += n * speeds[(model, setting)] * (r["best_epoch"] + KIBA_PATIENCE_OFFSET[model]) / 3600
        per_model[model] = round(hours, 1)
        total += hours
    return {"method_gpu_hours": round(total, 1), "per_model": per_model,
            "report_gpu_hours": REPORT_KIBA_GPU_HOURS,
            "difference_percent": round(100 * (total - REPORT_KIBA_GPU_HOURS) / REPORT_KIBA_GPU_HOURS, 1),
            "note": ("training batches x fp32/amp s-per-batch x (best_epoch + patience), no validation-overhead "
                     "factor; the 101 figure is the report's, unverified in the repository")}


def build(committed_dir: str = COMMITTED, speed_md: str = SPEED_MD) -> tuple:
    """(waves manifest, budget). Deterministic: the same files give the same output."""
    speeds = parse_speed(speed_md)
    cells, estimates = {}, {}
    for seed in SEEDS_DAVIS:
        for model in ("coldsite_dti", "hyperattentiondti", "moltrans"):
            for level in LEVELS:
                cell = Cell("davis", model, level, seed)
                cells.setdefault(f"ACC{seed - 3}", []).append(cell)
                estimates[cell.id] = estimate(model, level, speeds, committed_dir)
    for level, seed in DRUGBAN_KIBA:
        cell = Cell("kiba", "drugban", level, seed)
        cells.setdefault("ACC3", []).append(cell)
        estimates[cell.id] = {"hours_mean": None, "hours_low": None, "hours_high": None,
                              "note": "unmeasured: no DrugBAN training time exists in the repository"}

    accounts, per_account = {}, {}
    for account, planned in sorted(cells.items()):
        # longest first onto the lighter GPU (LPT); unmeasured cells count equal, by row count
        weight = lambda c: (estimates[c.id]["hours_mean"] or 0.0, c.id)   # noqa: E731
        queues = {"gpu0": [], "gpu1": []}
        load = {"gpu0": 0.0, "gpu1": 0.0}
        for cell in sorted(planned, key=weight, reverse=True):
            gpu = "gpu0" if load["gpu0"] <= load["gpu1"] else "gpu1"
            queues[gpu].append(cell)
            load[gpu] += estimates[cell.id]["hours_mean"] or 1.0          # equal weights if unmeasured
        accounts[account] = queues
        measured = [estimates[c.id] for c in planned if estimates[c.id]["hours_mean"] is not None]
        per_account[account] = {
            "cells": len(planned), "measured_cells": len(measured),
            "unmeasured_cells": len(planned) - len(measured),
            "gpu_hours_mean": round(sum(e["hours_mean"] for e in measured), 1),
            "gpu_hours_low": round(sum(e["hours_low"] for e in measured), 1),
            "gpu_hours_high": round(sum(e["hours_high"] for e in measured), 1),
            "gpu0_cells": len(queues["gpu0"]), "gpu1_cells": len(queues["gpu1"])}
        if measured:
            gpu_h = {g: round(sum(estimates[c.id]["hours_mean"] for c in q), 1)
                     for g, q in queues.items()}
            per_account[account]["gpu_hours_mean_by_gpu"] = gpu_h
            for key in ("gpu_hours_mean", "gpu_hours_high"):
                per_account[account]["required_weekly_quota_gpu_h_for_" + key.split("_")[-1]] = round(
                    per_account[account][key] / (1 - RESERVE), 1)

    waves = {"schema": 1, "wave": "A",
             "note": ("Generated by `python -m src.cloud.budget --write`; every cell appears once. ACC1 = DAVIS seed 4, "
                      "ACC2 = DAVIS seed 5 (HyperAttentionDTI, MolTrans, ColdSite-DTI x 4 levels), ACC3 = DrugBAN on "
                      "KIBA {random, cold_drug} x seeds 1-3 (its own account: DGL pins torch)."),
             "accounts": {a: {g: [{"dataset": c.dataset, "model": c.model, "level": c.level, "seed": c.seed}
                                  for c in q] for g, q in queues.items()} for a, queues in accounts.items()}}
    cfg = load_harness()
    budget = {"schema": 1, "reserve": RESERVE, "inputs": {
                  "seconds_per_batch_fp32": speeds, "batch": BATCH,
                  "overhead": {m: round(overhead(m, speeds, committed("%s" % committed_dir, "random", m, 1)["n_train_rows"]), 3)
                               for m in BATCH},
                  "patience_offset": PATIENCE_OFFSET, "kaggle_epoch_seconds": KAGGLE_EPOCH_SECONDS,
                  "session_limit_hours": cfg["session_limit_hours"],
                  "safety_margin_minutes": cfg["safety_margin_minutes"]},
              "gpu_hours_per_commit_per_account": round(
                  2 * (cfg["session_limit_hours"] - cfg["safety_margin_minutes"] / 60), 2),
              "crosscheck_kiba": crosscheck_kiba(committed_dir, speed_md),
              "per_account": per_account, "cells": estimates}
    return waves, budget


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--committed", default=COMMITTED)
    args = ap.parse_args(argv)
    waves, budget = build(args.committed)
    if args.write:
        for path, data in ((WAVES_PATH, waves), (BUDGET_PATH, budget)):
            with open(path, "w") as handle:
                json.dump(data, handle, indent=1)
                handle.write("\n")
        print(f"wrote {WAVES_PATH} and {BUDGET_PATH}")
        return 0
    bad = [p for p, d in ((WAVES_PATH, waves), (BUDGET_PATH, budget))
           if json.load(open(p)) != json.loads(json.dumps(d))]
    print("plan files match" if not bad else f"DIFFER: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
