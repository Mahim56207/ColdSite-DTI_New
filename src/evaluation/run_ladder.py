"""
Track C (124AD0067) — the experiment ladder.

Produces the paper's headline result: explanation fidelity (precision@k) and
prediction accuracy across the four difficulty levels, with significance.

docs/03_GUIDE_124AD0067.md Step 3 asks for the figure to be designed *before*
real results exist, so October is a matter of dropping numbers in rather than
building the plumbing under time pressure. This module is that plumbing: it
runs today on dummy data (`--dummy`) and unchanged on real checkpoints
(`--checkpoint`).

Usage
-----
    # today, no data needed -- proves the pipeline end to end
    python -m src.evaluation.run_ladder --dummy

    # October, once Track B has checkpoints and Track A has splits
    python -m src.evaluation.run_ladder \
        --dataset davis \
        --ground-truth data/davis_ground_truth_sites.json \
        --checkpoint-dir results

    # the audited baselines; pass the accuracy file run_faithfulness wrote for
    # the SAME model (accuracy_moltrans_davis_seed1.json), never ColdSite-DTI's
    python -m src.evaluation.run_ladder --model moltrans --task binary \
        --dataset davis --ground-truth data/davis_ground_truth_sites.json \
        --accuracy-json results/accuracy_moltrans_davis_seed1.json
"""
import argparse
import json
import os

import numpy as np
import torch

from src.data.ground_truth import load_site_sets
from src.model.checkpoint_naming import checkpoint_path as build_checkpoint_path
from src.model.checkpoint_naming import DEFAULT_MODEL, discover_checkpoints
from src.evaluation.precision_at_k import batch_precision_at_k
from src.evaluation.significance_test import permutation_test_batch

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
LEVEL_LABELS = {
    "random": "Warm",
    "cold_drug": "Cold-Drug",
    "cold_target": "Cold-Target",
    "cold_pair": "Cold-Pair",
}
# One naming rule for both runners: the accuracy file run_faithfulness writes
# is the one this runner reads, so the two tags must never be spelled twice.
from src.evaluation.run_faithfulness import MODELS, output_tag  # noqa: E402


def collect_explanations(model, dataloader, target_ids, site_sets, device="cpu",
                         max_proteins=None, pairs_per_target: int = 1):
    """Run a split through the model and pair each explanation with its sites.

    `target_ids` must be aligned to the dataloader's row order. This is the
    alignment the two guides call the main integration seam; if it drifts,
    every protein gets scored against another protein's ground truth and the
    result still looks like a plausible number.

    `pairs_per_target` keeps the first N test rows of each protein, in file
    order -- the same rows `collect._read_test_rows` keeps for the audit table,
    so the ladder and the audit score the same pairs. precision@k is a
    per-protein quantity: averaging every pair enters a protein once per drug
    it was measured against and makes n a count of correlated pairs. 0 keeps
    every pair (the behaviour before 2026-09-12).
    """
    model.eval().to(device)
    weights, sites, used_ids = [], [], []
    seen: dict = {}
    cursor = 0

    with torch.no_grad():
        for drug_batch, protein_batch, _labels in dataloader:
            batch_ids = target_ids[cursor:cursor + len(drug_batch)]
            cursor += len(drug_batch)
            explanations = model.explain(drug_batch.to(device), protein_batch.to(device))

            for target_id, explanation in zip(batch_ids, explanations):
                count = seen.get(target_id, 0)
                if pairs_per_target and count >= pairs_per_target:
                    continue
                seen[target_id] = count + 1
                site_set = site_sets.get(target_id)
                if site_set is None or not site_set.usable:
                    continue
                weights.append(np.asarray(explanation, dtype=float))
                sites.append(site_set.positions)
                used_ids.append(target_id)
                if max_proteins and len(weights) >= max_proteins:
                    return weights, sites, used_ids

    return weights, sites, used_ids


def evaluate_level(weights, sites, k_values=(5, 10, 20), n_trials=1000, seed=0,
                   pairs_per_target=None):
    """One difficulty level -> fidelity at each k, plus a split-level p-value.

    `pairs_per_target` is recorded, not used: it says whether `n` counts
    proteins (1) or test pairs (0), which a reader of the JSON cannot tell.
    """
    result = {"n_proteins": len(weights), "by_k": {}}
    if pairs_per_target is not None:
        result["pairs_per_target"] = pairs_per_target
    for k in k_values:
        batch = batch_precision_at_k(weights, sites, k=k,
                                     rng=np.random.default_rng(seed))
        significance = permutation_test_batch(weights, sites, k=k,
                                              n_trials=n_trials, seed=seed)
        result["by_k"][k] = {
            "precision_at_k": batch["mean_precision_at_k"],
            "normalised": batch["mean_normalised"],
            "ceiling": batch["mean_ceiling"],
            "chance": significance.get("chance_mean"),
            "p_value": significance.get("p_value"),
            "significant": significance.get("significant"),
            "n_evaluated": batch["n_evaluated"],
            "n_skipped_no_sites": batch["n_skipped_no_sites"],
            "n_skipped_too_short": batch["n_skipped_too_short"],
        }
    return result


def ladder_table(results: dict, k: int = 10) -> str:
    """Markdown table of the ladder at one k -- goes straight into the paper."""
    header = (
        f"| Level | precision@{k} | normalised | ceiling | chance | p | n |\n"
        f"|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for level in LEVELS:
        if level not in results:
            continue
        entry = results[level]["by_k"].get(k)
        if entry is None:
            continue

        def fmt(value):
            return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) \
                else f"{value:.3f}"

        marker = "*" if entry.get("significant") else ""
        rows.append(
            f"| {LEVEL_LABELS[level]} | {fmt(entry['precision_at_k'])}{marker} | "
            f"{fmt(entry['normalised'])} | {fmt(entry['ceiling'])} | "
            f"{fmt(entry['chance'])} | {fmt(entry['p_value'])} | "
            f"{entry['n_evaluated']} |"
        )
    units = {results[l].get("pairs_per_target") for l in LEVELS if l in results}
    if units == {1}:
        unit = "\n`n` = proteins, one test pair each."
    elif units == {0}:
        unit = "\n`n` = test pairs (every pair of every protein)."
    else:
        unit = ""
    return (header + "\n".join(rows)
            + "\n\n`*` = significantly above chance (p < 0.05)." + unit + "\n")


def write_results(results: dict, out_dir: str, tag: str, k: int = 10,
                  accuracy: dict | None = None):
    """Write the JSON, the markdown table, and the headline figure."""
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"ladder_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    table_path = os.path.join(out_dir, f"ladder_{tag}.md")
    with open(table_path, "w") as f:
        f.write(f"# Explanation-fidelity ladder — {tag}\n\n")
        f.write(ladder_table(results, k=k))
    print(f"\n{ladder_table(results, k=k)}")

    figure_path = None
    fidelity = [results[l]["by_k"][k]["precision_at_k"]
                for l in LEVELS if l in results and k in results[l]["by_k"]]
    if len(fidelity) == 4 and accuracy is not None:
        from src.evaluation.plots import plot_degradation_curve

        # plots.py now takes the audit grid shape rather than two flat lists, so
        # this single-model run is wrapped as a one-model grid. Same figure,
        # same code path the full audit uses -- no second plotting route to
        # drift out of sync.
        model_name = tag
        grid = {model_name: {
            level: {"precision_at_k": {
                "mean": results[level]["by_k"][k]["precision_at_k"],
                "std": float("nan"), "n_seeds": 1, "sufficient_seeds": False}}
            for level in LEVELS if level in results and k in results[level]["by_k"]
        }}
        figure_path = os.path.join(out_dir, f"headline_{tag}.png")
        at_k = {l: results[l]["by_k"][k] for l in LEVELS
                if l in results and k in results[l]["by_k"]}
        plot_degradation_curve(
            grid, {model_name: {l: accuracy.get(l, float("nan")) for l in LEVELS}},
            title=f"Explanation fidelity vs accuracy — {tag}",
            save_path=figure_path,
            chance={l: e["chance"] for l, e in at_k.items() if e.get("chance") is not None},
            ceiling={l: e["ceiling"] for l, e in at_k.items() if e.get("ceiling") is not None})
    elif len(fidelity) == 4:
        print("\nAccuracy values not supplied, so the headline figure was not "
              "drawn. Pass --accuracy-json with one AUROC/CI per level; the "
              "fidelity curve alone is only half the paper's claim.")

    print(f"Saved -> {json_path}\nSaved -> {table_path}")
    return json_path, table_path, figure_path


# --------------------------------------------------------------------------
# dummy mode
# --------------------------------------------------------------------------

def run_dummy(out_dir="results", n_proteins=24, protein_length=300, seed=0):
    """Exercise the whole ladder on synthetic data, no checkpoints required.

    The numbers this produces are meaningless by construction -- the attention
    comes from an untrained model. That is the point: it proves the plumbing
    while making it impossible to mistake the output for a result.
    """
    from src.model.coldsite_dti import ColdSiteDTI

    torch.manual_seed(seed)
    model = ColdSiteDTI(70, 28).eval()
    rng = np.random.default_rng(seed)

    sites = {p for start in (41, 143, 260) for p in range(start, start + 3)}
    results = {}

    for level in LEVELS:
        weights, site_list = [], []
        for i in range(n_proteins):
            torch.manual_seed(seed + i)
            protein = torch.randint(2, 28, (1, protein_length))
            drug = torch.randint(2, 70, (1, 50))
            weights.append(np.asarray(model.explain(drug, protein)[0]))
            site_list.append(sites)
        results[level] = evaluate_level(weights, site_list, n_trials=300, seed=seed)
        print(f"{LEVEL_LABELS[level]:12s} "
              f"precision@10={results[level]['by_k'][10]['precision_at_k']:.3f} "
              f"(chance {results[level]['by_k'][10]['chance']:.3f}, "
              f"p={results[level]['by_k'][10]['p_value']:.3f})")

    fake_accuracy = {l: v for l, v in zip(LEVELS, [0.89, 0.81, 0.78, 0.69])}
    write_results(results, out_dir, "DUMMY_PLACEHOLDER", accuracy=fake_accuracy)
    print("\n" + "=" * 70)
    print("DUMMY RUN — untrained model, synthetic sites, placeholder accuracy.")
    print("These are NOT results. Outputs are tagged DUMMY_PLACEHOLDER so they")
    print("cannot be mistaken for the real ladder later.")
    print("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the explanation-fidelity ladder")
    parser.add_argument("--dummy", action="store_true",
                        help="synthetic run, no data or checkpoints needed")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS,
                        help="whose checkpoints to read. Outputs for any model "
                             "but coldsite_dti are prefixed with its name")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--ground-truth", help="path to *_ground_truth_sites.json")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--seed", type=int, default=1,
                        help="training seed of the checkpoints to read; must "
                             "match a seed Track B actually trained")
    parser.add_argument("--task", default="regression",
                        choices=["regression", "binary"],
                        help="must match the task the checkpoints were trained on")
    parser.add_argument("--accuracy-json",
                        help="{level: accuracy} produced by the training runs")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-protein-len", type=int, default=1000)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--pairs-per-target", type=int, default=1,
                        help="test pairs scored per protein, first in file order "
                             "(default 1, as run_audit). 0 scores every pair, "
                             "which reproduces ladders run before 2026-09-12")
    parser.add_argument("--device", default="cpu",
                        help="for --model hyperattentiondti / moltrans; "
                             "ColdSite-DTI runs on CPU as before")
    args = parser.parse_args()

    if args.dummy:
        run_dummy(out_dir=args.out_dir)
        return

    if not args.ground_truth:
        parser.error("--ground-truth is required unless --dummy is set")
    if args.model != DEFAULT_MODEL and args.task != "binary":
        parser.error(f"--model {args.model} only exists as a binary classifier; "
                     f"pass --task binary")

    from src.model.coldsite_dti import ColdSiteDTI
    from src.model.dataset import load_split

    site_sets = load_site_sets(args.ground_truth, max_len=args.max_protein_len)
    print(f"Loaded ground truth for {len(site_sets)} usable targets")

    accuracy = json.load(open(args.accuracy_json)) if args.accuracy_json else None
    results = {}

    for level in LEVELS:
        split_dir = os.path.join(args.split_root, args.dataset, level)
        # Built by the same module the trainer writes with, so the two ends
        # cannot drift apart. See src/model/checkpoint_naming.py.
        checkpoint = build_checkpoint_path(
            args.checkpoint_dir, args.dataset, level, args.task, args.seed,
            model=args.model)
        if not (os.path.isdir(split_dir) and os.path.exists(checkpoint)):
            print(f"[skip] {level}: missing {split_dir} or {checkpoint}")
            available = [c["seed"] for c in discover_checkpoints(
                args.checkpoint_dir, dataset=args.dataset, split=level,
                task=args.task, model=args.model)]
            if available:
                print(f"        (seeds present for this cell: {available})")
            continue

        if args.model == DEFAULT_MODEL:
            import pandas as pd
            test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
            target_ids = test_df["Target_ID"].tolist()

            _train, _valid, test_loader, drug_vocab, protein_vocab = load_split(
                split_dir, args.max_protein_len)
            model = ColdSiteDTI(len(drug_vocab) + 2, len(protein_vocab) + 2)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state"])

            weights, sites, used = collect_explanations(
                model, test_loader, target_ids, site_sets,
                pairs_per_target=args.pairs_per_target)
        else:
            from src.evaluation.collect import MissingCell, collect_cell

            # Same rows as the ColdSite-DTI branch above and as run_audit:
            # the first `pairs_per_target` test pairs of each protein.
            try:
                weights, sites, used = collect_cell(
                    args.model, args.dataset, level, args.seed,
                    site_sets=site_sets, task=args.task,
                    split_root=args.split_root,
                    checkpoint_dir=args.checkpoint_dir,
                    max_protein_len=args.max_protein_len,
                    pairs_per_target=args.pairs_per_target,
                    device=args.device, verbose=False)
            except MissingCell as reason:
                print(f"[skip] {level}: {reason}")
                continue
        unit = "proteins" if args.pairs_per_target == 1 else "pairs"
        print(f"{level}: {len(used)} {unit} with usable ground truth")
        results[level] = evaluate_level(weights, sites, n_trials=args.n_trials,
                                        pairs_per_target=args.pairs_per_target)

    if not results:
        raise SystemExit(
            "No levels could be evaluated. Train the models first:\n"
            "  python -m src.model.train --split-dir data/splits/davis/random "
            "--dataset davis --split random"
        )
    # Seed in the output tag as well as the input path: a ladder run per seed
    # writing to one `ladder_davis.json` reintroduces at the reading end
    # exactly the overwrite the checkpoint naming just fixed.
    write_results(results, args.out_dir,
                  output_tag(args.model, args.dataset, args.seed),
                  k=args.k, accuracy=accuracy)


if __name__ == "__main__":
    main()
