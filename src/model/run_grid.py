"""
Track B (124AD0015) — the 24-run training grid.

    2 datasets x 4 splits x 3 training seeds = 24 runs

The three seeds are TRAINING seeds (weight init, batch order) on one fixed
split per cell, not three regenerated splits. That reading is what the
repository supports: the Part 2 guide's loop varies only `--seed` against a
seed-independent `--split-dir`, `build_all_splits()` takes no seed argument and
writes one split per cell, `run_audit.build_grid` has a single seed axis, and
both the guide and STATUS.md state the count as 24. It is 24 runs, never 72.

Usage
-----
    python -m src.model.run_grid --preflight     # check, launch nothing
    python -m src.model.run_grid --plan          # print the 24 cells
    python -m src.model.run_grid                 # validate one cell, then run

Why one cell runs first
-----------------------
The guide's warning is the design here: "a shape error discovered on run 23
costs a week." The first cell is trained, then verified end to end -- checkpoint
written, uniquely named, loadable, results JSON present, accuracy recorded,
discoverable by the downstream evaluation -- before anything else is launched.
Any failure aborts the grid rather than producing 23 more of the same.

This module deliberately shells out to `python -m src.model.train` rather than
importing its loop. The trainer's argument handling and file naming are what
the grid has to exercise; calling the functions underneath would test a path
nobody runs.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from src.model.checkpoint_naming import (
    checkpoint_path as build_checkpoint_path,
    discover_checkpoints,
    results_path,
    run_tag,
)
from src.model.train import accuracy_metric_for

DATASETS = ("davis", "kiba")
SPLITS = ("random", "cold_drug", "cold_target", "cold_pair")
SEEDS = (1, 2, 3)
TASK = "regression"
EPOCHS = 100

# The status table's accuracy column follows --task, it is not a constant. A
# hardcoded "ci" made every cell of a --task binary grid fail verification with
# "no 'ci' in test_metrics" -- which reads like a training failure rather than a
# metric-name mismatch. Imported from train.py so it cannot drift from the keys
# compute_metrics actually emits.

SPLIT_FILES = ("train.csv", "valid.csv", "test.csv")


def grid_cells(datasets=DATASETS, splits=SPLITS, seeds=SEEDS) -> list:
    """Every (dataset, split, seed) cell, in launch order."""
    return [{"dataset": d, "split": s, "seed": seed}
            for d in datasets for s in splits for seed in seeds]


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(cells, split_root="data/splits", results_dir="results",
              task=TASK) -> dict:
    """Everything checkable before a single epoch is spent.

    The expensive failure is not a crash on run 1; it is 24 runs that complete
    and then turn out to have overwritten each other, or to have been trained
    against a split directory that was not the one the name claims.
    """
    problems, warnings, missing_splits, tags = [], [], [], {}

    for cell in cells:
        split_dir = os.path.join(split_root, cell["dataset"], cell["split"])
        for filename in SPLIT_FILES:
            path = os.path.join(split_dir, filename)
            if not os.path.exists(path):
                missing_splits.append(path)

        tag = run_tag(cell["dataset"], cell["split"], task, cell["seed"])
        if tag in tags:
            problems.append(f"tag collision: {tag} produced by two cells")
        tags[tag] = cell

        checkpoint = build_checkpoint_path(
            results_dir, cell["dataset"], cell["split"], task, cell["seed"])
        if os.path.exists(checkpoint):
            warnings.append(f"checkpoint already exists; the cell will be "
                            f"skipped if it is complete, retrained if it was "
                            f"interrupted: {checkpoint}")

    # the property the whole naming module exists to guarantee
    n_expected = len(cells)
    if len(tags) != n_expected:
        problems.append(
            f"{len(tags)} unique tags for {n_expected} cells — runs would "
            f"overwrite each other")

    if len(SEEDS) < 3:
        problems.append("fewer than three seeds: no cell could be quoted as an "
                        "estimate (aggregate.MIN_SEEDS_FOR_A_CLAIM)")
    if task != "regression":
        warnings.append(f"task is {task!r}, not 'regression'")

    return {
        "n_cells": n_expected,
        "unique_tags": len(tags),
        "missing_split_files": sorted(set(missing_splits)),
        "splits_ready": not missing_splits,
        "problems": problems,
        "warnings": warnings,
        "ready_to_launch": not problems and not missing_splits,
    }


def format_preflight(report: dict) -> str:
    lines = [f"Grid: {report['n_cells']} cells, "
             f"{report['unique_tags']} unique run tags"]
    if report["missing_split_files"]:
        missing = report["missing_split_files"]
        lines.append(f"\nBLOCKED — {len(missing)} split file(s) missing:")
        lines += [f"  {p}" for p in missing[:12]]
        if len(missing) > 12:
            lines.append(f"  ... and {len(missing) - 12} more")
        lines.append(
            "\nSplit generation is Track A (124AD0008). Build them with:\n"
            "  python -m src.data.build_splits\n"
            "which needs the DeepDTA data files under "
            "src/data/baselines/deepdta/data/.")
    for problem in report["problems"]:
        lines.append(f"PROBLEM: {problem}")
    for warning in report["warnings"]:
        lines.append(f"warning: {warning}")
    lines.append(f"\nready_to_launch: {report['ready_to_launch']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# running and verifying one cell
# ---------------------------------------------------------------------------

def train_command(cell, split_root="data/splits", results_dir="results",
                  task=TASK, epochs=EPOCHS, extra=()) -> list:
    return [
        sys.executable, "-m", "src.model.train",
        "--split-dir", os.path.join(split_root, cell["dataset"], cell["split"]),
        "--dataset", cell["dataset"],
        "--split", cell["split"],
        "--task", task,
        "--seed", str(cell["seed"]),
        "--epochs", str(epochs),
        "--results-dir", results_dir,
        *extra,
    ]


def verify_cell(cell, results_dir="results", task=TASK,
                accuracy_metric=None) -> dict:
    """Everything the guide asks be confirmed before the grid is trusted.

    Checks the artefacts, not the training curve: whether a model learned well
    is not this function's business, but whether the run produced a uniquely
    named, loadable checkpoint with a recorded accuracy that the downstream
    evaluation can find, is.
    """
    problems = []
    accuracy_metric = accuracy_metric or accuracy_metric_for(task)
    tag = run_tag(cell["dataset"], cell["split"], task, cell["seed"])
    checkpoint = build_checkpoint_path(
        results_dir, cell["dataset"], cell["split"], task, cell["seed"])
    metrics_file = results_path(results_dir, tag)

    if not os.path.exists(checkpoint):
        problems.append(f"no checkpoint at {checkpoint}")

    accuracy = None
    if not os.path.exists(metrics_file):
        problems.append(f"no results JSON at {metrics_file}")
    else:
        with open(metrics_file) as f:
            payload = json.load(f)
        metrics = payload.get("test_metrics", {})
        if accuracy_metric not in metrics:
            problems.append(
                f"no '{accuracy_metric}' in test_metrics (have {sorted(metrics)}) "
                f"— the Track C accuracy hand-off would be empty")
        else:
            accuracy = float(metrics[accuracy_metric])
        for field, expected in (("dataset", cell["dataset"]),
                                ("split", cell["split"]),
                                ("seed", cell["seed"])):
            if payload.get(field) != expected:
                problems.append(
                    f"results JSON says {field}={payload.get(field)!r}, cell is "
                    f"{expected!r} — a number attributed to the wrong run")

    # loadable, and actually a ColdSite-DTI state dict
    if os.path.exists(checkpoint):
        try:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "model_state" not in state:
                problems.append("checkpoint has no 'model_state' key")
            elif not state["model_state"]:
                problems.append("checkpoint 'model_state' is empty")
        except Exception as exc:
            problems.append(f"checkpoint will not load: {type(exc).__name__}: {exc}")

    # the downstream seam: can run_ladder / run_faithfulness find this run?
    found = discover_checkpoints(results_dir, dataset=cell["dataset"],
                                 split=cell["split"], task=task)
    if cell["seed"] not in [entry["seed"] for entry in found]:
        problems.append(
            f"discover_checkpoints cannot see seed {cell['seed']} for "
            f"{cell['dataset']}/{cell['split']} — downstream evaluation would "
            f"silently skip this cell")

    return {"tag": tag, "checkpoint": checkpoint, "results_json": metrics_file,
            "accuracy": accuracy, "valid": not problems, "problems": problems}


def run_cell(cell, split_root="data/splits", results_dir="results", task=TASK,
             epochs=EPOCHS, extra=(), dry_run=False) -> dict:
    """Train one cell, then verify its artefacts."""
    command = train_command(cell, split_root, results_dir, task, epochs, extra)
    label = f"{cell['dataset']}/{cell['split']}/seed{cell['seed']}"

    if dry_run:
        return {"cell": cell, "status": "dry-run", "command": " ".join(command)}

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    started = time.time()
    completed = subprocess.run(command)
    elapsed = time.time() - started

    if completed.returncode != 0:
        return {"cell": cell, "status": "train-failed", "seconds": elapsed,
                "problems": [f"trainer exited {completed.returncode}"],
                "command": " ".join(command)}

    verification = verify_cell(cell, results_dir, task)
    return {"cell": cell, "seconds": elapsed,
            "status": "ok" if verification["valid"] else "verify-failed",
            **verification}


# ---------------------------------------------------------------------------
# the status table
# ---------------------------------------------------------------------------

def status_table(results: list) -> str:
    """dataset | split | seed | checkpoint | accuracy | status — as asked for."""
    lines = ["| dataset | split | seed | checkpoint | accuracy | status |",
             "|---|---|---|---|---|---|"]
    for entry in results:
        cell = entry["cell"]
        accuracy = entry.get("accuracy")
        lines.append(
            f"| {cell['dataset']} | {cell['split']} | {cell['seed']} | "
            f"`{os.path.basename(entry.get('checkpoint', '—'))}` | "
            f"{'—' if accuracy is None else f'{accuracy:.4f}'} | "
            f"{entry['status']} |")

    done = sum(1 for e in results if e["status"] == "ok")
    lines += ["", f"{done} of {len(results)} cells complete "
                  f"(accuracy = the task's headline metric on the test split; "
                  f"concordance index for regression, AUROC for binary)."]
    failures = [e for e in results if e["status"] not in ("ok", "skipped", "dry-run")]
    if failures:
        lines += ["", "## Failures", ""]
        for entry in failures:
            cell = entry["cell"]
            for problem in entry.get("problems", []):
                lines.append(f"- {cell['dataset']}/{cell['split']}/"
                             f"seed{cell['seed']}: {problem}")
    return "\n".join(lines)


def write_status(results: list, out_dir="results") -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "grid_status.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    table_path = os.path.join(out_dir, "grid_status.md")
    with open(table_path, "w") as f:
        f.write("# Training grid — 2 datasets x 4 splits x 3 training seeds\n\n")
        f.write(status_table(results) + "\n")
    return json_path, table_path


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the 24-cell training grid")
    parser.add_argument("--preflight", action="store_true",
                        help="check readiness and exit without training")
    parser.add_argument("--plan", action="store_true",
                        help="print the cells and the commands, train nothing")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--task", default=TASK, choices=["regression", "binary"])
    parser.add_argument("--overwrite", action="store_true",
                        help="retrain cells that already have a checkpoint")
    parser.add_argument("--skip-validation-cell", action="store_true",
                        help="do NOT stop to verify the first cell. Only for a "
                             "grid that has already been validated once.")
    args = parser.parse_args()

    cells = grid_cells(
        [d.strip() for d in args.datasets.split(",") if d.strip()],
        [s.strip() for s in args.splits.split(",") if s.strip()],
        [int(s) for s in args.seeds.split(",") if s.strip()])

    report = preflight(cells, args.split_root, args.results_dir, args.task)
    print(format_preflight(report))

    if args.plan:
        print("\n# Cells\n")
        for cell in cells:
            print("  " + " ".join(train_command(
                cell, args.split_root, args.results_dir, args.task, args.epochs)))
        return

    if args.preflight:
        return

    if not report["ready_to_launch"]:
        raise SystemExit(
            "\nRefusing to launch. Fix the problems above first — a grid "
            "started against missing or colliding cells wastes the compute and "
            "produces results nobody can attribute to a run.")

    results, remaining = [], list(cells)

    if not args.skip_validation_cell:
        first = remaining.pop(0)
        print(f"\nValidating ONE cell end to end before launching the other "
              f"{len(remaining)}.")
        outcome = run_cell(first, args.split_root, args.results_dir, args.task,
                           args.epochs)
        results.append(outcome)
        if outcome["status"] != "ok":
            write_status(results, args.results_dir)
            raise SystemExit(
                f"\nValidation cell failed: {outcome.get('problems')}\n"
                f"The remaining {len(remaining)} cells were NOT launched. "
                f"A shape error found on run 23 costs a week.")
        print(f"\nValidation cell OK "
              f"({accuracy_metric_for(args.task)}={outcome['accuracy']:.4f}). "
              f"Launching the remaining {len(remaining)}.")

    for cell in remaining:
        checkpoint = build_checkpoint_path(
            args.results_dir, cell["dataset"], cell["split"], args.task,
            cell["seed"])
        # A checkpoint on disk does NOT mean the cell finished. train() writes
        # one on the first improving epoch -- usually epoch 0 -- while the
        # results JSON is only written after the test pass. An interrupted run
        # (a dropped Colab session) therefore leaves a checkpoint without a
        # results JSON. Skipping on the checkpoint alone banks a half-trained
        # model as a finished cell, with no accuracy, and reports success.
        # Skip only what verify_cell says is actually complete.
        if os.path.exists(checkpoint) and not args.overwrite:
            verdict = verify_cell(cell, args.results_dir, args.task)
            if verdict["valid"]:
                print(f"[skip] {cell['dataset']}/{cell['split']}/seed{cell['seed']} "
                      f"— already complete")
                results.append({"cell": cell, "status": "skipped",
                                "checkpoint": checkpoint,
                                "accuracy": verdict["accuracy"]})
                continue
            print(f"[retrain] {cell['dataset']}/{cell['split']}/seed{cell['seed']} "
                  f"— checkpoint exists but the cell is incomplete, so it was "
                  f"interrupted rather than finished:")
            for problem in verdict["problems"]:
                print(f"    - {problem}")
        results.append(run_cell(cell, args.split_root, args.results_dir,
                                args.task, args.epochs))

    json_path, table_path = write_status(results, args.results_dir)
    print("\n" + status_table(results))
    print(f"\nSaved -> {json_path}\nSaved -> {table_path}")

    failed = [e for e in results if e["status"] not in ("ok", "skipped")]
    if failed:
        raise SystemExit(f"{len(failed)} cell(s) did not complete.")


if __name__ == "__main__":
    main()
