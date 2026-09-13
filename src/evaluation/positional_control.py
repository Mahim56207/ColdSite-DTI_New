"""
Is a precision@k above chance explained by position alone?

Why this exists
---------------
`permutation_test_batch` draws its null by placing the k "attended" positions
uniformly at random along each protein. That is the right null for "does the
explanation know where the sites are?" only if attention has no positional habit.
An attention map that always favours, say, the first 30 residues will beat a
uniform null on any panel whose sites cluster near the N-terminus, without
knowing anything about the protein in front of it.

On 2026-09-13 ColdSite-DTI's attention was at chance on its own kinase test sets
but above chance on the 60-protein non-kinase panel in 7 of 12 cells. Before that
is read as a finding, this module asks the question a reviewer will ask.

The test
--------
Give each protein another protein's attention map and score it against the
first protein's sites. A borrowed map keeps the model's positional habit and
loses everything specific to the protein. If borrowed maps score as well as the
real ones, the "signal" is position.

    absolute   the other map laid on residue by residue from the N-terminus
               (cut, or padded with its minimum, to this protein's length)
    relative   the other map stretched to this protein's length, so a habit
               defined as "the first 10% of the chain" is kept as well

The p-value compares the real mean precision@k with the same mean over random
reassignments of maps to proteins (each protein given a map not its own).

Usage
-----
    python -m src.evaluation.positional_control --model coldsite_dti --seeds 1,2,3 \\
        --checkpoint-dir <folder with the binary checkpoints>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.data.ground_truth import COTRANSPORT_IONS, load_site_sets
from src.evaluation.collect import MissingCell, collect_cell
from src.evaluation.precision_at_k import precision_at_k
from src.evaluation.run_control import LEVELS, PANEL_ROWS, PANEL_SITES

EDGE = 0.10          # "near an end" = within the first or last 10% of the chain


def borrow(weights, length: int, mode: str) -> np.ndarray:
    """One protein's attention map, laid onto a protein of `length` residues."""
    weights = np.asarray(weights, dtype=float)
    if mode == "absolute":
        if weights.size >= length:
            return weights[:length].copy()
        pad = np.full(length - weights.size, weights.min() - 1.0)
        return np.concatenate([weights, pad])
    if mode == "relative":
        old = np.linspace(0.0, 1.0, weights.size)
        return np.interp(np.linspace(0.0, 1.0, length), old, weights)
    raise ValueError(f"mode must be 'absolute' or 'relative', got {mode!r}")


def _derangement(n: int, rng) -> np.ndarray:
    while True:
        order = rng.permutation(n)
        if not np.any(order == np.arange(n)):
            return order


def mean_precision(weights, sites, k: int, rng) -> float:
    return float(np.mean([precision_at_k(w, s, k, rng=rng) for w, s in zip(weights, sites)]))


def borrowed_null(weights, sites, k: int = 10, mode: str = "absolute",
                  n_trials: int = 1000, seed: int = 0) -> dict:
    """Mean precision@k when every protein wears another protein's map."""
    rng = np.random.default_rng(seed)
    usable = [(np.asarray(w, float), s) for w, s in zip(weights, sites)
              if len(s) > 0 and np.asarray(w).size >= k]
    maps = [w for w, _ in usable]
    site_sets = [s for _, s in usable]
    observed = mean_precision(maps, site_sets, k, rng)
    null = []
    for _ in range(n_trials):
        order = _derangement(len(maps), rng)
        null.append(mean_precision(
            [borrow(maps[j], maps[i].size, mode) for i, j in enumerate(order)],
            site_sets, k, rng))
    null = np.asarray(null)
    return {"observed": observed, "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "p_value": float((1 + np.sum(null >= observed)) / (1 + n_trials)),
            "n_proteins": len(maps), "n_trials": n_trials}


def residue_type_null(weights, sites, sequences, k: int = 10, n_trials: int = 1000,
                      seed: int = 0) -> dict:
    """Mean precision@k when each protein's attention is shuffled only among residues
    of the same amino acid. Keeps any preference for residue types (histidines and
    cysteines line many metal and ligand pockets), removes where along the chain it
    falls. Real maps that do no better are explained by amino-acid preference."""
    rng = np.random.default_rng(seed)
    usable = []
    for w, s, seq in zip(weights, sites, sequences):
        w = np.asarray(w, float)
        if len(s) == 0 or w.size < k or seq is None or len(seq) < w.size:
            continue
        letters = np.frombuffer(seq[:w.size].encode(), dtype=np.uint8)
        groups = [np.flatnonzero(letters == a) for a in np.unique(letters)]
        usable.append((w, s, groups))
    observed = mean_precision([w for w, _, _ in usable], [s for _, s, _ in usable], k, rng)
    null = []
    for _ in range(n_trials):
        shuffled = []
        for w, _s, groups in usable:
            out = w.copy()
            for g in groups:
                out[g] = w[rng.permutation(g)]
            shuffled.append(out)
        null.append(mean_precision(shuffled, [s for _, s, _ in usable], k, rng))
    null = np.asarray(null)
    return {"observed": observed, "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "p_value": float((1 + np.sum(null >= observed)) / (1 + n_trials)),
            "n_proteins": len(usable), "n_trials": n_trials}


def span_null(weights, sites, k: int = 10, n_trials: int = 1000, seed: int = 0) -> dict:
    """Mean precision@k when attention is shuffled only within the stretch the sites
    span (first to last site) and, separately, outside it. Keeps how much attention
    falls in that stretch -- for the KLIFS pocket, the kinase domain -- and removes
    where inside it. Real maps that do no better know the domain, not the pocket."""
    rng = np.random.default_rng(seed)
    usable = []
    for w, s in zip(weights, sites):
        w = np.asarray(w, float)
        if len(s) == 0 or w.size < k:
            continue
        lo, hi = min(s), min(max(s), w.size - 1)
        inside = np.arange(lo, hi + 1)
        outside = np.setdiff1d(np.arange(w.size), inside)
        usable.append((w, s, [inside, outside], (hi - lo + 1) / w.size))
    maps, site_sets = [u[0] for u in usable], [u[1] for u in usable]
    observed = mean_precision(maps, site_sets, k, rng)
    null = []
    for _ in range(n_trials):
        shuffled = []
        for w, _s, groups, _f in usable:
            out = w.copy()
            for g in groups:
                out[g] = w[rng.permutation(g)]
            shuffled.append(out)
        null.append(mean_precision(shuffled, site_sets, k, rng))
    null = np.asarray(null)
    from src.evaluation.precision_at_k import top_k_positions
    in_span = [np.mean([min(s) <= p <= max(s) for p in top_k_positions(w, k, rng=rng)])
               for w, s, _g, _f in usable]
    return {"observed": observed, "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "p_value": float((1 + np.sum(null >= observed)) / (1 + n_trials)),
            "top_k_in_span": float(np.mean(in_span)),
            "span_fraction_of_chain": float(np.mean([u[3] for u in usable])),
            "n_proteins": len(usable), "n_trials": n_trials}


def fixed_window(weights, sites, k: int, where: str) -> float:
    """precision@k of an 'explanation' that is just the first or last k residues."""
    scores = []
    for w, s in zip(weights, sites):
        n = np.asarray(w).size
        if len(s) == 0 or n < k:
            continue
        window = range(k) if where == "first" else range(n - k, n)
        scores.append(sum(p in s for p in window) / k)
    return float(np.mean(scores))


def describe(weights, sites, k: int = 10, seed: int = 0) -> dict:
    """Where the top-k attention and the annotated sites sit along the chain."""
    from src.evaluation.precision_at_k import top_k_positions
    rng = np.random.default_rng(seed)
    top, site_pos = [], []
    for w, s in zip(weights, sites):
        n = np.asarray(w).size
        if len(s) == 0 or n < k:
            continue
        top += [p / n for p in top_k_positions(w, k, rng=rng)]
        site_pos += [p / n for p in s]
    top, site_pos = np.asarray(top), np.asarray(site_pos)
    near_end = lambda x: float(np.mean((x < EDGE) | (x > 1 - EDGE)))
    return {"top_k_mean_relative_position": float(top.mean()),
            "top_k_fraction_near_an_end": near_end(top),
            "sites_mean_relative_position": float(site_pos.mean()),
            "sites_fraction_near_an_end": near_end(site_pos),
            "top_k_fraction_first_10pct": float(np.mean(top < EDGE)),
            "sites_fraction_first_10pct": float(np.mean(site_pos < EDGE))}


_SEQUENCES: dict = {}


def sequences(dataset: str, level: str, rows_csv: str | None) -> dict:
    """Target_ID -> the sequence the model read, from the rows the cell was scored on."""
    import pandas as pd
    path = rows_csv or os.path.join("data/splits", dataset, level, "test.csv")
    if path not in _SEQUENCES:
        frame = pd.read_csv(path, usecols=["Target_ID", "Target"]).drop_duplicates("Target_ID")
        _SEQUENCES[path] = dict(zip(frame["Target_ID"].astype(str), frame["Target"]))
    return _SEQUENCES[path]


def run(model: str, dataset: str, seeds, checkpoint_dir: str, k: int = 10,
        n_trials: int = 1000, exclude_ligands=COTRANSPORT_IONS, device: str = "cpu",
        ground_truth: str | None = None, arms=("kinase", "non_kinase")) -> dict:
    ground_truth = ground_truth or f"data/{dataset}_ground_truth_sites.json"
    kinase_sites = load_site_sets(ground_truth, max_len=1000)
    panel_sites = load_site_sets(PANEL_SITES, max_len=1000, exclude_ligands=exclude_ligands)
    out = {"model": model, "dataset": dataset, "k": k, "n_trials": n_trials,
           "ground_truth": ground_truth,
           "exclude_ligands": list(exclude_ligands), "cells": []}
    for seed in seeds:
        for level in LEVELS:
            for arm, site_sets, rows in (("kinase", kinase_sites, None),
                                         ("non_kinase", panel_sites, PANEL_ROWS)):
                if arm not in arms:
                    continue
                try:
                    weights, sites, ids = collect_cell(
                        model, dataset, level, seed, site_sets=site_sets, rows_csv=rows,
                        checkpoint_dir=checkpoint_dir, device=device, verbose=False)
                except MissingCell as reason:
                    print(f"[skip] seed {seed} {level} {arm}: {reason}")
                    continue
                cell = {"seed": seed, "level": level, "arm": arm,
                        "absolute": borrowed_null(weights, sites, k, "absolute", n_trials, seed),
                        "relative": borrowed_null(weights, sites, k, "relative", n_trials, seed),
                        "residue_type": residue_type_null(
                            weights, sites, [sequences(dataset, level, rows).get(t) for t in ids],
                            k, n_trials, seed),
                        "site_span": span_null(weights, sites, k, n_trials, seed),
                        "first_k": fixed_window(weights, sites, k, "first"),
                        "last_k": fixed_window(weights, sites, k, "last"),
                        "positions": describe(weights, sites, k, seed)}
                out["cells"].append(cell)
                a = cell["absolute"]
                print(f"seed {seed} {level:11s} {arm:10s} p@{k} {a['observed']:.3f} | "
                      f"borrowed abs {a['null_mean']:.3f} (p {a['p_value']:.3f}) "
                      f"rel {cell['relative']['null_mean']:.3f} "
                      f"(p {cell['relative']['p_value']:.3f}) | same-residue "
                      f"{cell['residue_type']['null_mean']:.3f} (p {cell['residue_type']['p_value']:.3f}) "
                      f"| in-span shuffle {cell['site_span']['null_mean']:.3f} "
                      f"(p {cell['site_span']['p_value']:.3f}, top-{k} in span "
                      f"{cell['site_span']['top_k_in_span']:.2f} vs span "
                      f"{cell['site_span']['span_fraction_of_chain']:.2f} of chain)",
                      flush=True)
    return out


def report(results: dict) -> str:
    k = results["k"]
    lines = [f"# Positional control — {results['model']}, {results['dataset']}\n",
             "Does precision@k survive when each protein is given **another protein's** "
             "attention map? A borrowed map keeps the model's positional habit and loses "
             "everything specific to the protein. See `src/evaluation/positional_control.py`. "
             f"k = {k}; {results['n_trials']} reassignments per cell; non-kinase sites exclude "
             "cotransport ions (the primary setting).\n",
             f"| seed | level | arm | p@{k} | borrowed, absolute (p) | borrowed, relative (p) "
             f"| same-residue shuffle (p) | in-span shuffle (p) | top-{k} in site span "
             f"| span / chain | first {k} residues | last {k} |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in results["cells"]:
        a, r, pos = c["absolute"], c["relative"], c["positions"]
        star = lambda x: "*" if x["p_value"] < 0.05 else ""
        lines.append(
            f"| {c['seed']} | {c['level']} | {c['arm']} | {a['observed']:.3f} | "
            f"{a['null_mean']:.3f} ({a['p_value']:.3f}{star(a)}) | "
            f"{r['null_mean']:.3f} ({r['p_value']:.3f}{star(r)}) | "
            f"{c['residue_type']['null_mean']:.3f} ({c['residue_type']['p_value']:.3f}"
            f"{star(c['residue_type'])}) | {c['site_span']['null_mean']:.3f} "
            f"({c['site_span']['p_value']:.3f}{star(c['site_span'])}) | "
            f"{c['site_span']['top_k_in_span']:.2f} | "
            f"{c['site_span']['span_fraction_of_chain']:.2f} | {c['first_k']:.3f} | "
            f"{c['last_k']:.3f} |")
    lines.append(f"\n`*` = the real maps beat the null (p < 0.05, before correction). "
                 f"*Same-residue shuffle* = attention permuted only among residues of the same "
                 f"amino acid (keeps residue-type preference, removes location). "
                 f"*In-span shuffle* = attention permuted within the stretch from the first to "
                 f"the last site, and separately outside it (keeps how much attention reaches "
                 f"that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). "
                 f"*Near an end* = within the first or last {int(EDGE * 100)}% of the chain.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="coldsite_dti")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ground-truth", help="default: data/<dataset>_ground_truth_sites.json")
    parser.add_argument("--arms", default="kinase,non_kinase")
    parser.add_argument("--tag", default="", help="appended to the output file names")
    args = parser.parse_args()
    results = run(args.model, args.dataset, [int(s) for s in args.seeds.split(",")],
                  args.checkpoint_dir, args.k, args.n_trials, device=args.device,
                  ground_truth=args.ground_truth, arms=tuple(args.arms.split(",")))
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir,
                        f"positional_control_{args.model}_{args.dataset}{args.tag}")
    with open(stem + ".json", "w") as f:
        json.dump(results, f, indent=2)
    with open(stem + ".md", "w") as f:
        f.write(report(results))
    print(f"Saved -> {stem}.json\nSaved -> {stem}.md")


if __name__ == "__main__":
    main()
