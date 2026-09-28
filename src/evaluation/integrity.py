"""
The guards that keep a past bug from coming back quietly.

Every rule here was a real failure first. None of them changes a number: each one
refuses to let a run produce a number that would be read as something it is not.

    permutation resolution   a permutation test cannot report a p smaller than
                             1/(n+1). The DAVIS audit ran at n = 500, so its floor
                             was 0.001996 against a smallest Holm threshold of
                             0.05/20 = 0.0025 -- the test's resolution was 80% of the
                             bar it had to clear. `check_permutations` refuses that.

    partial grids            `run_all` skips a step whose output file exists, and one
                             ladder file covers every level of a (model, seed). Run it
                             while only `random` is trained and the file is written with
                             `random` alone; when `cold_drug` lands, the file is skipped
                             and that level is never scored. Nothing complains. The grid
                             fingerprint written beside the outputs makes the second run
                             notice that the grid moved.

    extension directories    the original analyses are the primary result and stay
                             byte-for-byte. An extension (new seeds, new models, new
                             methods) writes to its own directory or not at all.

Scratch directories are exempt from the grid guard on purpose: probing a half-trained
grid is a legitimate thing to do, as long as the answer cannot be mistaken for the
paper's.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

# The floor every permutation/null test runs at (docs/REMEDIATION_PLAN.md,
# statistical_integrity_rules). Raising a default costs time and nothing else; a
# p-value whose floor sits near the threshold it is tested against costs a claim.
MIN_PERMUTATIONS = 10_000
ALPHA = 0.05

# The run_all outputs that already exist are only safe to skip if the grid behind them
# has not changed. This records what it was.
GRID_STATE_FILE = "grid_state.json"

# A directory whose results are understood to be throwaway.
SCRATCH_MARKERS = ("scratch", "tmp", "temp")


# ---------------------------------------------------------------- permutations

def min_achievable_p(n_trials: int) -> float:
    """The smallest p an `n_trials` permutation test can report.

    The add-one estimator (1 + hits) / (1 + n) never reaches 0, which is the point:
    a finite number of draws cannot express infinite confidence.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    return 1.0 / (n_trials + 1)


def smallest_holm_threshold(family_size: int, alpha: float = ALPHA) -> float:
    """Holm's strictest step: alpha / m, the bar the most significant cell must clear."""
    if family_size < 1:
        raise ValueError(f"family_size must be >= 1, got {family_size}")
    return alpha / family_size


def permutations_needed(family_size: int, alpha: float = ALPHA) -> int:
    """The fewest permutations whose floor sits strictly below alpha / family_size."""
    n = int(math.ceil(family_size / alpha))
    while min_achievable_p(n) >= smallest_holm_threshold(family_size, alpha):
        n += 1
    return n


def check_permutations(n_trials: int, *, family_size: int | None = None,
                       alpha: float = ALPHA, allow_low: bool = False,
                       context: str = "") -> None:
    """Refuse a permutation count that cannot support the conclusion drawn from it.

    `allow_low` exists for one purpose: reproducing an output that was computed before
    this floor existed. It is never how a new result is produced.
    """
    where = f" ({context})" if context else ""
    if allow_low:
        return
    if n_trials < MIN_PERMUTATIONS:
        raise SystemExit(
            f"--n-trials {n_trials} is below the floor of {MIN_PERMUTATIONS}{where}. "
            f"A permutation test cannot report a p below {min_achievable_p(n_trials):.6f}, "
            f"and the protocol requires the floor to be far under every threshold it is "
            f"tested against. Pass --allow-low-permutations only to reproduce an output "
            f"computed before this rule.")
    if family_size is not None:
        threshold = smallest_holm_threshold(family_size, alpha)
        if min_achievable_p(n_trials) >= threshold:
            raise SystemExit(
                f"{n_trials} permutations cannot resolve this family{where}: the smallest "
                f"p it can report is {min_achievable_p(n_trials):.6f}, and Holm's strictest "
                f"threshold over {family_size} cells is alpha/{family_size} = "
                f"{threshold:.6f}. Use at least {permutations_needed(family_size, alpha)}.")


# ----------------------------------------------------------- output directories

def is_scratch_dir(path: str) -> bool:
    """Is this a directory whose contents nobody will mistake for the paper's numbers?

    True under the system temp directory, or when a component of the path *below the
    working directory* is or begins with `scratch`, `tmp` or `temp` (`results/scratch_x`).
    Components above the working directory do not count: a project that happens to live in
    `~/tmp/` has not made its `results/` a scratch area.
    """
    absolute = os.path.realpath(os.path.abspath(path))
    for root in {tempfile.gettempdir(), "/tmp", "/var/tmp"}:
        root = os.path.realpath(root)
        if absolute == root or absolute.startswith(root + os.sep):
            return True
    cwd = os.path.realpath(os.getcwd())
    if not (absolute == cwd or absolute.startswith(cwd + os.sep)):
        return False
    for part in os.path.relpath(absolute, cwd).split(os.sep):
        lowered = part.lower()
        if any(lowered == marker or lowered.startswith((marker + "_", marker + "-"))
               for marker in SCRATCH_MARKERS):
            return True
    return False


def check_extension_dir(out_dir: str) -> None:
    """An extension run writes somewhere new. Refuse a directory holding results.

    The primary analyses stay byte-for-byte; a run declared as an extension that points
    at a directory with outputs in it is about to overwrite one of them.
    """
    if not os.path.isdir(out_dir):
        return
    existing = [name for name in sorted(os.listdir(out_dir))
                if name.endswith((".json", ".md", ".csv"))
                and name != GRID_STATE_FILE]
    if existing:
        raise SystemExit(
            f"--extension was given but {out_dir} already holds "
            f"{len(existing)} output file(s) (first: {existing[0]}). An extension family "
            f"writes to its own directory so the primary analysis stays byte-for-byte. "
            f"Choose a new --out-dir.")


# ------------------------------------------------------------------- the grid

def grid_fingerprint(state: dict) -> list:
    """The sorted 'model|level|seed' of every cell that was complete."""
    return sorted(f"{model}|{level}|{seed}"
                  for (model, level, seed), status in state.items()
                  if status == "complete")


def read_grid_state(out_dir: str):
    """The fingerprint recorded beside these outputs, or None if there is none."""
    path = os.path.join(out_dir, GRID_STATE_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle).get("complete_cells")


def write_grid_state(out_dir: str, state: dict, dataset: str = "") -> str:
    path = os.path.join(out_dir, GRID_STATE_FILE)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"dataset": dataset, "complete_cells": grid_fingerprint(state)},
                  handle, indent=2)
    return path


def check_grid_unchanged(out_dir: str, state: dict, *, skip_existing: bool,
                         allow_partial: bool = False) -> None:
    """Refuse to top up outputs that were written against a different grid.

    Only bites when outputs are being skipped: with `--no-skip-existing` everything is
    recomputed, so a changed grid is not a hazard. Scratch directories are exempt.
    """
    if not skip_existing or allow_partial or is_scratch_dir(out_dir):
        return
    recorded = read_grid_state(out_dir)
    if recorded is None:
        return
    now = grid_fingerprint(state)
    if now == recorded:
        return
    added = sorted(set(now) - set(recorded))
    removed = sorted(set(recorded) - set(now))
    raise SystemExit(
        f"{out_dir} holds outputs written when {len(recorded)} cells were complete; "
        f"{len(now)} are complete now "
        f"(+{len(added)} {added[:3]}, -{len(removed)} {removed[:3]}). "
        f"A ladder or faithfulness file covers every level of a (model, seed), so the "
        f"existing files would be skipped and the new cells never scored. Re-run with "
        f"--no-skip-existing, or into a new --out-dir, or into a scratch directory if "
        f"this is only a probe.")
