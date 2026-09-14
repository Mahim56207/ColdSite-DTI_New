"""
Confidence intervals for every number in the audit, by resampling proteins.

Why
---
The tables so far carry a spread over three seeds. That answers "how much does retraining
move this?" and not "how precisely is this measured?" -- and at the cold levels, where a
cell rests on 68 proteins (or, for the drug-specific ground truth, 6 pairs), the second
question is the one a reviewer asks. A precision@10 of 0.133 on 6 pairs is one lucky
protein; the same number on 349 is a result. Without an interval the two look identical.

What is resampled
-----------------
**Proteins, not pairs and not seeds.** The measurement averages over proteins, so the
protein is the unit that varies: draw n proteins with replacement from the n that were
scored, recompute the mean, repeat. Where a protein was scored under several seeds, the
resample takes that protein *in every seed at once* and averages -- so the interval
carries both the protein sampling and the seed noise, and a protein cannot enter the
sample under one seed and not another.

A percentile interval is used. The statistic is a bounded mean of values in [0, 1] with
no transformation, so the refinements of BCa buy little, and the honest limit here is the
number of proteins rather than the interval's method.

    python -m src.evaluation.bootstrap_ci --dirs results/analysis_davis_policyA \\
        --models coldsite_dti,hyperattentiondti,moltrans --out results/ci_davis.md

Reads the `per_protein` scores `run_ladder` records. A ladder written before those were
kept has none, and is reported as such rather than silently skipped.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.evaluation.run_faithfulness import output_tag

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
DEFAULT_K = "10"
N_RESAMPLES = 10000
CONFIDENCE = 0.95


def percentile_ci(samples, confidence: float = CONFIDENCE) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)))


def bootstrap_mean(per_protein: dict, n_resamples: int = N_RESAMPLES, seed: int = 0,
                   confidence: float = CONFIDENCE) -> dict:
    """CI for the mean over proteins, averaging each protein across the seeds it has.

    `per_protein` is {protein id: [value per seed]}. Proteins are drawn with replacement;
    a protein enters the sample with all of its seeds, which is what keeps the interval
    from treating one protein measured three times as three proteins.
    """
    ids = sorted(per_protein)
    if not ids:
        return {"n": 0, "mean": float("nan"), "low": float("nan"), "high": float("nan"),
                "n_resamples": 0}
    values = np.array([float(np.mean(per_protein[i])) for i in ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = percentile_ci(means, confidence)
    return {"n": len(values), "mean": float(values.mean()), "low": low, "high": high,
            "n_resamples": n_resamples,
            "seeds": int(np.median([len(per_protein[i]) for i in ids]))}


def collect_per_protein(folder: str, model: str, dataset: str, seeds, level: str,
                        k: str = DEFAULT_K) -> tuple[dict, list]:
    """{protein id: [score per seed]} for one cell, plus the seeds that were missing it."""
    out: dict = {}
    missing = []
    for seed in seeds:
        path = os.path.join(folder, f"ladder_{output_tag(model, dataset, seed)}.json")
        if not os.path.exists(path):
            missing.append(f"seed {seed}: no ladder")
            continue
        payload = json.load(open(path))
        entry = payload.get(level)
        if not entry:
            missing.append(f"seed {seed}: no {level}")
            continue
        scores = entry.get("by_k", {}).get(k, {}).get("per_protein")
        ids = entry.get("ids")
        if scores is None or ids is None:
            missing.append(f"seed {seed}: ladder predates per-protein scores")
            continue
        if len(scores) != len(ids):
            missing.append(f"seed {seed}: {len(scores)} scores for {len(ids)} ids")
            continue
        for protein, score in zip(ids, scores):
            out.setdefault(protein, []).append(float(score))
    return out, missing


def cells(folders: dict, models, dataset: str, seeds, k: str = DEFAULT_K,
          n_resamples: int = N_RESAMPLES, confidence: float = CONFIDENCE) -> list:
    rows = []
    for label, folder in folders.items():
        for model in models:
            for level in LEVELS:
                per_protein, missing = collect_per_protein(folder, model, dataset,
                                                           seeds, level, k)
                row = {"ground_truth": label, "model": model, "level": level,
                       "missing": missing}
                row.update(bootstrap_mean(per_protein, n_resamples, confidence=confidence))
                rows.append(row)
    return rows


def report(rows: list, k: str = DEFAULT_K, confidence: float = CONFIDENCE) -> str:
    out = [f"# Confidence intervals — precision@{k}, resampling proteins", "",
           f"{int(confidence * 100)}% percentile intervals from {N_RESAMPLES:,} resamples "
           "of the proteins a cell scored, each protein carried in with all of its seeds "
           "(`src/evaluation/bootstrap_ci.py`). The mean is the same number the ladder "
           "reports; the interval says how precisely a cell of this size measures it.", "",
           "| ground truth | model | level | n | precision@" + k + " | 95% CI | width |",
           "|---|---|---|---|---|---|---|"]
    for row in rows:
        if not row["n"]:
            out.append(f"| {row['ground_truth']} | {row['model']} | "
                       f"{row['level'].replace('_', '-')} | — | — | — | — |")
            continue
        out.append(f"| {row['ground_truth']} | {row['model']} | "
                   f"{row['level'].replace('_', '-')} | {row['n']} | {row['mean']:.3f} | "
                   f"{row['low']:.3f}–{row['high']:.3f} | {row['high'] - row['low']:.3f} |")
    problems = [(r, m) for r in rows for m in r["missing"]]
    if problems:
        out += ["", "## Cells that could not be given an interval", ""]
        for row, message in problems[:20]:
            out.append(f"- {row['ground_truth']} {row['model']} {row['level']}: {message}")
        if len(problems) > 20:
            out.append(f"- ... and {len(problems) - 20} more")
    widest = max((r for r in rows if r["n"]), key=lambda r: r["high"] - r["low"], default=None)
    if widest is not None:
        out += ["", f"The widest interval is {widest['model']} at "
                f"{widest['level'].replace('_', '-')} against {widest['ground_truth']} "
                f"({widest['n']} proteins, {widest['high'] - widest['low']:.3f} wide): "
                "a cell that small cannot separate anything from chance, and the audit "
                "should say so rather than quote its point estimate.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dirs", required=True,
                        help="comma-separated, each optionally 'label=path' "
                             "(default labels: the folder name)")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--models", default="coldsite_dti,hyperattentiondti,moltrans")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--k", default=DEFAULT_K)
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    folders = {}
    for item in args.dirs.split(","):
        label, _, path = item.partition("=")
        if not path:
            label, path = os.path.basename(label.rstrip("/")), label
        folders[label] = os.path.expanduser(path)
    rows = cells(folders, args.models.split(","), args.dataset,
                 [int(s) for s in args.seeds.split(",")], args.k,
                 args.n_resamples, args.confidence)
    text = report(rows, args.k, args.confidence)
    print(text)
    out = args.out or f"results/ci_{args.dataset}.md"
    with open(out, "w") as handle:
        handle.write(text)
    with open(out.replace(".md", ".json"), "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nSaved -> {out} and {out.replace('.md', '.json')}")


if __name__ == "__main__":
    main()
