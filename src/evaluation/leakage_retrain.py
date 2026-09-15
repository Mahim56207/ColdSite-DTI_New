"""
What DAVIS's sequence leakage is worth, measured by retraining without it.

`clean_accuracy` re-scores a trained cell on the test rows whose target is unseen by
sequence. That removes the leak from the *measurement* but not from *training*: the model
still learned from proteins it would later be tested on. This module reads the other
experiment -- the same model retrained on splits with the leak removed
(`src/data/seqclean_splits.py`) and on a control with the leak kept at the same row count
and class balance -- and scores all three arms on the one test set they share:

    original      the published split: 21,080 (cold-target) / 15,190 (cold-pair) train rows
    seqmatched    17,748 / 12,936 rows, leak KEPT   -- fewer rows, so this isolates volume
    seqclean      17,748 / 12,936 rows, leak GONE   -- seqmatched minus seqclean = the leak

Because the test file is identical in all three, every difference is a training-set
difference. And because the test file contains both leaked and unleaked targets, each arm
is scored three times: on every row, on the rows whose target was seen by sequence in the
*original* training set, and on the rest. The leak can only have helped on the first
group, so that is where removing it must hurt; if `seqclean` drops on the unleaked rows
too, we are looking at something other than leakage.

Scored by the trainer's own test pass (`clean_accuracy._scores`), on CPU, so the number
must reproduce each cell's recorded test AUROC.

    python -m src.evaluation.leakage_retrain \\
        --grid-dir ~/ColdSite-results/davis_binary \\
        --derived-dir ~/ColdSite-results/leakage_davis/results
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st

import numpy as np
import pandas as pd

from src.evaluation.clean_accuracy import (REPRODUCE_TOLERANCE, _metrics, _scores,
                                           seen_by_sequence)
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag

LEVELS = ("cold_target", "cold_pair")
ARMS = ("original", "seqmatched", "seqclean")
TASK = "binary"
SUBSETS = ("all", "leaked", "unleaked")


def split_name(level: str, arm: str) -> str:
    return level if arm == "original" else f"{level}_{arm}"


def subset_masks(test_csv: str, leaked: frozenset) -> dict:
    """{'all' | 'leaked' | 'unleaked': boolean mask over the test rows, in file order}."""
    targets = pd.read_csv(test_csv).Target_ID.astype(str)
    is_leaked = targets.isin({str(t) for t in leaked}).to_numpy()
    return {"all": np.ones(len(targets), bool), "leaked": is_leaked,
            "unleaked": ~is_leaked}


def subset_metrics(labels, scores, masks: dict) -> dict:
    """Metrics per subset. A subset with one class only has no AUROC and is left out."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    out = {}
    for name, mask in masks.items():
        if mask.sum() == 0 or len(set(labels[mask].tolist())) < 2:
            continue
        out[name] = dict(_metrics(labels[mask], scores[mask]), n=int(mask.sum()),
                         positives=int(labels[mask].sum()))
    return out


def evaluate_cell(model: str, dataset: str, level: str, arm: str, seed: int,
                  checkpoint_dir: str, split_root: str, device: str = "cpu") -> dict | None:
    """One (arm, seed): the trainer's own test pass, split into the three subsets."""
    split = split_name(level, arm)
    ckpt = checkpoint_path(checkpoint_dir, dataset, split, TASK, seed, model=model)
    res = results_path(checkpoint_dir, run_tag(dataset, split, TASK, seed), model=model)
    if not (os.path.exists(ckpt) and os.path.exists(res)):
        return None
    recorded = json.load(open(res))
    split_dir = os.path.join(split_root, dataset, split)
    labels, scores = _scores(model, split_dir, dataset, ckpt, recorded, seed, device)
    masks = subset_masks(os.path.join(split_dir, "test.csv"),
                         seen_by_sequence(dataset, level))
    subsets = subset_metrics(labels, scores, masks)
    recorded_auroc = recorded["test_metrics"]["auroc"]
    return {"model": model, "dataset": dataset, "level": level, "arm": arm, "seed": seed,
            "n_train_rows": recorded.get("n_train_rows"),
            "recorded_auroc": recorded_auroc,
            "rescored_auroc": subsets["all"]["auroc"],
            "reproduces_recorded":
                abs(subsets["all"]["auroc"] - recorded_auroc) <= REPRODUCE_TOLERANCE,
            "subsets": subsets}


def _agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return (st.mean(values), st.stdev(values) if len(values) > 1 else 0.0, len(values))


def _fmt(agg) -> str:
    return "—" if agg is None else f"{agg[0]:.3f} ± {agg[1]:.3f}"


def _by(cells, level, arm, subset, key="auroc"):
    return [c["subsets"][subset][key] for c in cells
            if c["level"] == level and c["arm"] == arm and subset in c["subsets"]]


def report(cells: list, model: str) -> str:
    """The markdown table, and the two differences that are the result."""
    out = [f"# What DAVIS's sequence leakage is worth — {model}, retrained without it", "",
           "Three training sets, one test set each level (`src/data/seqclean_splits.py`, "
           "`src/evaluation/leakage_retrain.py`). `seqmatched` keeps the leak at "
           "`seqclean`'s row count and positive count, so **seqmatched − seqclean is the "
           "leak** and **original − seqmatched is the smaller training set**. Test AUROC, "
           "mean ± sd over seeds, scored by the trainer's own test pass on CPU.", ""]
    for level in LEVELS:
        if not any(c["level"] == level for c in cells):
            continue
        leaked_n = _by(cells, level, "original", "leaked", "n")
        unleaked_n = _by(cells, level, "original", "unleaked", "n")
        out += [f"## {level.replace('_', '-')}", "",
                f"Test rows: {leaked_n[0] if leaked_n else '?'} on targets seen by sequence "
                f"in the original training set, {unleaked_n[0] if unleaked_n else '?'} on the rest.",
                "", "| arm | train rows | all rows | leaked targets | unleaked targets |",
                "|---|---|---|---|---|"]
        for arm in ARMS:
            mine = [c for c in cells if c["level"] == level and c["arm"] == arm]
            if not mine:
                continue
            rows = {c["n_train_rows"] for c in mine}
            out.append(f"| {arm} | {rows.pop() if len(rows) == 1 else sorted(rows)} | "
                       + " | ".join(_fmt(_agg(_by(cells, level, arm, s))) for s in SUBSETS)
                       + " |")
        out.append("")
        for label, a, b in (("the leak", "seqmatched", "seqclean"),
                            ("fewer training rows", "original", "seqmatched")):
            paired = []
            for s in sorted({c["seed"] for c in cells if c["level"] == level}):
                x = [c for c in cells if c["level"] == level and c["arm"] == a and c["seed"] == s]
                y = [c for c in cells if c["level"] == level and c["arm"] == b and c["seed"] == s]
                if x and y:
                    paired.append((s, {sub: x[0]["subsets"][sub]["auroc"] - y[0]["subsets"][sub]["auroc"]
                                       for sub in SUBSETS
                                       if sub in x[0]["subsets"] and sub in y[0]["subsets"]}))
            if not paired:
                continue
            out.append(f"**{a} − {b} = {label}**, per seed (same seed both sides):")
            out.append("")
            out.append("| seed | " + " | ".join(SUBSETS) + " |")
            out.append("|---|" + "---|" * len(SUBSETS))
            for s, diffs in paired:
                out.append(f"| {s} | " + " | ".join(f"{diffs.get(sub, float('nan')):+.3f}"
                                                    for sub in SUBSETS) + " |")
            means = {sub: st.mean([d[sub] for _s, d in paired if sub in d]) for sub in SUBSETS
                     if any(sub in d for _s, d in paired)}
            out.append("| **mean** | " + " | ".join(f"**{means.get(sub, float('nan')):+.3f}**"
                                                    for sub in SUBSETS) + " |")
            out.append("")
    bad = [c for c in cells if not c["reproduces_recorded"]]
    out += ["## Checks", "",
            f"- {'PASS' if not bad else 'FAIL'}: all {len(cells)} cells' re-scored AUROC "
            f"reproduce their recorded value within {REPRODUCE_TOLERANCE}"
            + ("" if not bad else ": " + ", ".join(
                f"{c['level']} {c['arm']} s{c['seed']} {c['recorded_auroc']:.4f} vs "
                f"{c['rescored_auroc']:.4f}" for c in bad))]
    same_rows = all(len({c["n_train_rows"] for c in cells
                         if c["level"] == lv and c["arm"] == arm}) <= 1
                    for lv in LEVELS for arm in ARMS)
    matched = all(len({c["n_train_rows"] for c in cells
                       if c["level"] == lv and c["arm"] in ("seqclean", "seqmatched")}) <= 1
                  for lv in LEVELS)
    out += [f"- {'PASS' if same_rows else 'FAIL'}: every seed of an arm trained on the "
            "same number of rows",
            f"- {'PASS' if matched else 'FAIL'}: seqclean and seqmatched trained on the "
            "same number of rows (the control is volume-matched)", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", default="davis", choices=("davis",))
    parser.add_argument("--model", default="deepdta")
    parser.add_argument("--grid-dir", required=True,
                        help="checkpoints of the original splits")
    parser.add_argument("--derived-dir", required=True,
                        help="checkpoints of the seqclean / seqmatched splits")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cells = []
    for level in LEVELS:
        for arm in ARMS:
            for seed in (int(s) for s in args.seeds.split(",")):
                where = args.grid_dir if arm == "original" else args.derived_dir
                cell = evaluate_cell(args.model, args.dataset, level, arm, seed,
                                     os.path.expanduser(where), args.split_root, args.device)
                if cell is None:
                    print(f"   missing: {level} {arm} seed {seed}")
                    continue
                cells.append(cell)
                flag = "ok" if cell["reproduces_recorded"] else "DIFFERS FROM RECORDED"
                print(f"{level:12s} {arm:11s} s{seed}  recorded {cell['recorded_auroc']:.4f}  "
                      f"re-scored {cell['rescored_auroc']:.4f} ({flag})  leaked "
                      f"{cell['subsets'].get('leaked', {}).get('auroc', float('nan')):.4f}  "
                      f"unleaked {cell['subsets'].get('unleaked', {}).get('auroc', float('nan')):.4f}")
    if not cells:
        raise SystemExit("no cells found -- check --grid-dir and --derived-dir")
    out = args.out or f"results/leakage_retrain_{args.dataset}.md"
    with open(out, "w") as f:
        f.write(report(cells, args.model))
    with open(out.replace(".md", ".json"), "w") as f:
        json.dump(cells, f, indent=1)
    print(f"\nSaved -> {out} and {out.replace('.md', '.json')}")


if __name__ == "__main__":
    main()
