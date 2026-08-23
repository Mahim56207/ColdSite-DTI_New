"""
The kinase confound control — the objection most likely to sink the paper.

DAVIS and KIBA are kinase panels. A "cold-target" kinase is not an unfamiliar
protein: it shares the ATP pocket of the several hundred kinases already in
training. A model can score well on cold-target precision@k by learning "ATP
pockets look like this", which is exactly the generic pattern-matching the
ladder claims to be testing against.

This is a transfer test, not a stratification
---------------------------------------------
`target_family.stratified_indices` splits one dataset's targets into kinase and
non-kinase. On this data that comparison is impossible, and the measured
numbers say so plainly:

    DAVIS  442 targets  429 kinase   0 non-kinase  13 unknown
    KIBA   229 targets  227 kinase   0 non-kinase   2 unknown

There is no non-kinase arm to stratify *into*. So the control has to come from
outside the training distribution entirely: 60 BindingDB proteins with real
UniProt binding-site annotation, none of which any DAVIS- or KIBA-trained model
has ever seen.

That makes this a strictly harder condition than the dataset's own cold_target
level -- the protein is unseen *and* from a different family, and its drugs are
unseen too. Write it up that way. Reported as if it were within-dataset
stratification, it would overstate what was measured.

What the three outcomes mean
----------------------------
| result                                   | reading                          |
|------------------------------------------|----------------------------------|
| holds on kinases, collapses off them     | the ladder measured family
                                             similarity -- this IS the finding |
| degrades similarly on both               | degradation is real; lead with it |
| holds on both                            | honest negative result; report it |

What is not publishable is the unstratified ladder presented as though the
confound were absent.

Usage
-----
    python -m src.evaluation.run_control --dataset davis --seed 1 --task binary
    python -m src.evaluation.run_control --dataset davis --seed 1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.data.ground_truth import load_site_sets
from src.evaluation.collect import MissingCell, collect_cell
from src.evaluation.precision_at_k import batch_precision_at_k
from src.evaluation.significance_test import permutation_test_batch
from src.evaluation.target_family import (NON_KINASE, confound_report,
                                          load_family_map)

PANEL_ROWS = "data/processed/nonkinase_panel.csv"
PANEL_SITES = "data/nonkinase_ground_truth_sites.json"
PANEL_FAMILIES = "data/nonkinase_panel_families.json"

# target_family's own gate. Below it the comparison has no power and the honest
# move is to state the limitation rather than report a number nobody believes.
MIN_TARGETS = 20

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")


def score(weights, sites, k=10, n_trials=1000, seed=0) -> dict:
    """precision@k plus its split-level permutation test, for one arm."""
    batch = batch_precision_at_k(weights, sites, k=k,
                                 rng=np.random.default_rng(seed))
    significance = permutation_test_batch(weights, sites, k=k,
                                          n_trials=n_trials, seed=seed)
    return {
        "precision_at_k": batch["mean_precision_at_k"],
        "normalised": batch["mean_normalised"],
        "ceiling": batch["mean_ceiling"],
        "chance": significance.get("chance_mean"),
        "p_value": significance.get("p_value"),
        "significant": significance.get("significant"),
        "n_proteins": batch["n_evaluated"],
    }


def panel_is_usable(panel_sites: dict) -> dict:
    """Confirm the control arm clears its own gate before anything is run."""
    load_family_map(PANEL_FAMILIES)
    return confound_report(list(panel_sites))


def run_control(model_name: str, dataset: str, seed: int, task: str = "binary",
                levels=LEVELS, split_root: str = "data/splits",
                checkpoint_dir: str = "results", ground_truth: str = None,
                k: int = 10, n_trials: int = 1000, max_protein_len: int = 1000,
                pairs_per_target: int = 1, device: str = "cpu") -> dict:
    """Both arms of the control, per level, for one model and seed.

    The kinase arm is the model's own test split -- the same proteins the
    ladder scores. The non-kinase arm is the panel, evaluated with the *same*
    checkpoint and the same vocabulary, so the only thing that changes between
    the two numbers is the protein family.
    """
    ground_truth = ground_truth or f"data/{dataset}_ground_truth_sites.json"
    kinase_sites = load_site_sets(ground_truth, max_len=max_protein_len)
    panel_sites = load_site_sets(PANEL_SITES, max_len=max_protein_len)

    gate = panel_is_usable(panel_sites)
    results = {
        "model": model_name, "dataset": dataset, "seed": seed, "task": task,
        "k": k, "panel_gate": gate, "levels": {},
        "design": ("transfer, not stratification: the non-kinase arm is 60 "
                   "BindingDB proteins outside the training distribution, so "
                   "it is a strictly harder condition than this dataset's own "
                   "cold_target level"),
    }

    common = dict(task=task, split_root=split_root,
                  checkpoint_dir=checkpoint_dir,
                  max_protein_len=max_protein_len,
                  pairs_per_target=pairs_per_target, device=device,
                  verbose=False)

    for level in levels:
        entry = {}
        try:
            weights, sites, ids = collect_cell(
                model_name, dataset, level, seed,
                site_sets=kinase_sites, **common)
            entry["kinase"] = score(weights, sites, k=k, n_trials=n_trials,
                                    seed=seed)
        except MissingCell as reason:
            entry["kinase"] = None
            entry["kinase_reason"] = str(reason)

        try:
            weights, sites, ids = collect_cell(
                model_name, dataset, level, seed,
                site_sets=panel_sites, rows_csv=PANEL_ROWS, **common)
            arm = score(weights, sites, k=k, n_trials=n_trials, seed=seed)
            if arm["n_proteins"] < MIN_TARGETS:
                entry["non_kinase"] = None
                entry["non_kinase_reason"] = (
                    f"only {arm['n_proteins']} proteins scored, below the "
                    f"{MIN_TARGETS}-target gate")
            else:
                entry["non_kinase"] = arm
        except MissingCell as reason:
            entry["non_kinase"] = None
            entry["non_kinase_reason"] = str(reason)

        if entry.get("kinase") and entry.get("non_kinase"):
            entry["gap"] = (entry["kinase"]["precision_at_k"]
                            - entry["non_kinase"]["precision_at_k"])
        results["levels"][level] = entry

    return results


def report(results: dict) -> str:
    """Markdown for the paper's confound-control subsection."""
    k = results["k"]
    lines = [
        f"# Kinase confound control — {results['model']}, "
        f"{results['dataset']}, seed {results['seed']}",
        "",
        "The non-kinase arm is a **transfer** condition, not a stratification: "
        "60 BindingDB proteins that no model trained on this dataset has seen, "
        "with their own UniProt binding-site annotation. It is therefore "
        "harder than this dataset's own cold_target level, where the protein "
        "is unseen but still a kinase.",
        "",
        f"| Level | kinase p@{k} | non-kinase p@{k} | gap | kinase n | "
        f"non-kinase n |",
        "|---|---|---|---|---|---|",
    ]

    def cell(arm):
        if arm is None:
            return "n/a"
        star = "*" if arm.get("significant") else ""
        return f"{arm['precision_at_k']:.3f}{star}"

    for level, entry in results["levels"].items():
        kin, non = entry.get("kinase"), entry.get("non_kinase")
        gap = entry.get("gap")
        lines.append(
            f"| {level} | {cell(kin)} | {cell(non)} | "
            f"{f'{gap:+.3f}' if gap is not None else 'n/a'} | "
            f"{kin['n_proteins'] if kin else '—'} | "
            f"{non['n_proteins'] if non else '—'} |")

    lines += ["", "`*` = significant before correction. Correct across the "
                  "whole grid with `run_audit`, not per cell.", ""]

    gate = results["panel_gate"]
    lines.append(
        f"Control arm: **{gate['distinct_non_kinase']} distinct non-kinase "
        f"targets**, `control_is_usable: {gate['control_is_usable']}`.")

    reasons = {level: entry.get("non_kinase_reason") or entry.get("kinase_reason")
               for level, entry in results["levels"].items()
               if entry.get("non_kinase_reason") or entry.get("kinase_reason")}
    if reasons:
        lines += ["", "## Levels that could not be scored", ""]
        lines += [f"- `{level}` — {reason}" for level, reason in reasons.items()]

    scored = [e for e in results["levels"].values() if e.get("gap") is not None]
    if not scored:
        lines += ["", "**No level produced both arms.** Train the checkpoints "
                      "first; until then the ladder must not be presented as "
                      "if the kinase confound had been controlled."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Kinase vs non-kinase control, by transfer to the panel")
    parser.add_argument("--model", default="coldsite_dti")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task", default="binary",
                        choices=["regression", "binary"])
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--ground-truth")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--pairs-per-target", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the panel and its gate, train nothing, "
                             "read no checkpoint")
    args = parser.parse_args()

    for path in (PANEL_ROWS, PANEL_SITES, PANEL_FAMILIES):
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} is missing. Build the control panel first:\n"
                f"    python -m src.data.build_nonkinase_panel "
                f"--scan/--select/--build")

    if args.dry_run:
        gate = panel_is_usable(load_site_sets(PANEL_SITES, max_len=1000))
        for key, value in gate.items():
            print(f"  {key:24s} {value}")
        print(f"\nPanel pairs: {PANEL_ROWS}")
        print("Gate cleared — run without --dry-run once checkpoints exist."
              if gate["control_is_usable"] else
              "Gate NOT cleared: state the confound as a limitation instead.")
        return

    results = run_control(
        args.model, args.dataset, args.seed, task=args.task,
        levels=[level.strip() for level in args.levels.split(",")],
        split_root=args.split_root, checkpoint_dir=args.checkpoint_dir,
        ground_truth=args.ground_truth, k=args.k, n_trials=args.n_trials,
        pairs_per_target=args.pairs_per_target, device=args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"control_{args.model}_{args.dataset}_seed{args.seed}"
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    text = report(results)
    with open(os.path.join(args.out_dir, f"{tag}.md"), "w") as handle:
        handle.write(text)
    print(text)
    print(f"\nSaved -> {os.path.join(args.out_dir, tag)}.{{json,md}}")


if __name__ == "__main__":
    main()
