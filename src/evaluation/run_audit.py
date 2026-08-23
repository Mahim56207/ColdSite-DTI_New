"""
The audit grid — the top-level runner for the reframed project.

`run_ladder.py` measures one model across four splits. This runs the actual
paper: every registered model, across four splits, two datasets and three
seeds, with faithfulness, family stratification and family-wise error control
applied over the whole grid at once.

    python -m src.evaluation.run_audit --dummy
    python -m src.evaluation.run_audit --models coldsite_dti,uniform_control \
        --datasets davis --seeds 1,2,3 \
        --ground-truth data/davis_ground_truth_sites.json

Why the correction happens here and not per cell
------------------------------------------------
Holm-Bonferroni is only valid over a family fixed in advance. Correcting inside
each model's run, then pooling, controls nothing -- the family would be defined
after seeing the results. So every cell's raw p-value is collected first, the
grid is completed, and correction runs once at the end over the whole set.
"""
import argparse
import json
import os

import numpy as np

from src.evaluation.aggregate import aggregate_seeds, audit_table, degradation, holm_bonferroni

# Imported for its registration side-effect. @register runs at import time, so
# without this line `--models deepdta` fails with "unknown model" and an error
# telling you to write an adapter that exists, is implemented, and already
# passes validate_adapter. The audit could only ever see its own two models.
from src.evaluation import baseline_adapters  # noqa: F401
from src.evaluation.model_registry import available_models, get_model, model_class
from src.evaluation.precision_at_k import batch_precision_at_k
from src.evaluation.significance_test import permutation_test_batch
from src.evaluation.target_family import KINASE, NON_KINASE, confound_report, stratified_indices

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")


def evaluate_cell(weights, sites, k=10, n_trials=1000, seed=0) -> dict:
    """One (model, dataset, split, seed) cell of the grid."""
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
        "n_evaluated": batch["n_evaluated"],
    }


def stratified_cell(weights, sites, target_ids, k=10, seed=0) -> dict:
    """The confound control: the same cell, split by target family.

    Returns None for a family with too few targets rather than a number built
    on three proteins. A stratified figure that a reviewer cannot believe is
    worse than an openly stated limitation.
    """
    indices = stratified_indices(target_ids)
    out = {}
    for family in (KINASE, NON_KINASE):
        rows = indices[family]
        if len(rows) < 20:
            out[family] = None
            continue
        out[family] = batch_precision_at_k(
            [weights[i] for i in rows], [sites[i] for i in rows],
            k=k, rng=np.random.default_rng(seed))["mean_precision_at_k"]
    return out


def build_grid(collect_fn, models, datasets, seeds, k=10, n_trials=1000) -> dict:
    """Run every cell.

    collect_fn(model_name, dataset, level, seed) must return
    (weights, sites, target_ids) or None when that cell has no checkpoint yet.
    Missing cells are skipped and reported, not silently treated as zero.
    """
    raw, p_values, stratified, missing = {}, {}, {}, []

    for model_name in models:
        raw[model_name] = {}
        stratified[model_name] = {}
        for dataset in datasets:
            for level in LEVELS:
                per_seed, per_seed_p, per_seed_strat = [], [], []

                for seed in seeds:
                    collected = collect_fn(model_name, dataset, level, seed)
                    if collected is None:
                        missing.append(f"{model_name}/{dataset}/{level}/seed{seed}")
                        continue
                    weights, sites, target_ids = collected
                    cell = evaluate_cell(weights, sites, k=k,
                                         n_trials=n_trials, seed=seed)
                    per_seed.append(cell["precision_at_k"])
                    per_seed_p.append(cell["p_value"])
                    per_seed_strat.append(
                        stratified_cell(weights, sites, target_ids, k=k, seed=seed))

                if not per_seed:
                    continue

                key = f"{model_name}|{dataset}|{level}"
                raw[model_name].setdefault(level, {})["precision_at_k"] = \
                    aggregate_seeds(per_seed, label=key)
                # median across seeds, not min: taking the smallest p of three
                # runs is cherry-picking dressed as aggregation
                p_values[key] = float(np.median(
                    [p for p in per_seed_p if p is not None and np.isfinite(p)]
                )) if any(p is not None for p in per_seed_p) else float("nan")

                for family in (KINASE, NON_KINASE):
                    values = [s[family] for s in per_seed_strat
                              if s.get(family) is not None]
                    if values:
                        stratified[model_name].setdefault(level, {})[family] = \
                            aggregate_seeds(values, label=f"{key}|{family}")

    return {
        "grid": raw,
        "stratified": stratified,
        "p_values_raw": p_values,
        "p_values_corrected": holm_bonferroni(p_values),
        "missing_cells": missing,
        "k": k,
        "seeds": list(seeds),
    }


def summarise(results: dict) -> str:
    """Human-readable audit summary -- the shape of the paper's results section."""
    lines = ["# Audit grid", "",
             audit_table(results["grid"]), "",
             "## Significance (Holm-Bonferroni over the whole grid)", ""]

    corrected = results["p_values_corrected"]
    n_significant = sum(1 for v in corrected.values() if v["significant"])
    lines.append(f"{n_significant} of {len(corrected)} cells survive correction "
                 f"at alpha = 0.05.")
    lines.append("")

    for key in sorted(corrected, key=lambda k: corrected[k]["p_value"]):
        entry = corrected[key]
        mark = "yes" if entry["significant"] else "no"
        lines.append(f"- `{key}` p={entry['p_value']:.4g} "
                     f"(threshold {entry['adjusted_alpha']:.4g}) -> {mark}")

    lines += ["", "## Confound control (kinase vs non-kinase)", ""]
    any_stratified = False
    for model_name, levels in results["stratified"].items():
        for level, families in levels.items():
            kinase = families.get(KINASE)
            non_kinase = families.get(NON_KINASE)
            if kinase and non_kinase:
                any_stratified = True
                gap = kinase["mean"] - non_kinase["mean"]
                lines.append(
                    f"- {model_name} / {level}: kinase {kinase['mean']:.3f}, "
                    f"non-kinase {non_kinase['mean']:.3f} (gap {gap:+.3f})")
    if not any_stratified:
        lines.append(
            "**No stratified comparison was possible.** Fewer than 20 "
            "non-kinase targets were available in every cell. The unstratified "
            "ladder must NOT be presented as if the kinase confound were "
            "absent -- state it as an explicit limitation in the Discussion, "
            "or enlarge the antiviral subset (Track A, Priority 1).")

    if results["missing_cells"]:
        lines += ["", f"## Missing cells ({len(results['missing_cells'])})", ""]
        # Reasons where they were recorded: "no checkpoint" and "no usable
        # ground truth" call for completely different responses, and a bare
        # list of cell names cannot tell them apart.
        reasons = {entry.split(":")[0]: entry.split(":", 1)[1].strip()
                   for entry in results.get("skipped_reasons", [])}
        for name in results["missing_cells"][:20]:
            reason = reasons.get(name)
            lines.append(f"- {name}" + (f" — {reason}" if reason else ""))

    return "\n".join(lines)


# --------------------------------------------------------------------------
# dummy mode
# --------------------------------------------------------------------------

def dummy_collect_fn(n_proteins=24, protein_length=250, seed_offset=0):
    """Synthetic collector: untrained models, arbitrary sites, no data needed."""
    import torch

    sites = {p for start in (40, 120, 200) for p in range(start, start + 4)}
    target_ids = ([f"ABL{i}" for i in range(n_proteins // 2)]
                  + [f"HIV-{i} protease" for i in range(n_proteins - n_proteins // 2)])

    def collect(model_name, dataset, level, seed):
        model = get_model(model_name) if model_name != "uniform_control" \
            else get_model("uniform_control")
        weights, site_list = [], []
        for i in range(n_proteins):
            torch.manual_seed(seed * 1000 + i + seed_offset)
            protein = torch.randint(2, 28, (1, protein_length))
            drug = torch.randint(2, 70, (1, 40))
            weights.append(np.asarray(model.explain(drug, protein), dtype=float))
            site_list.append(sites)
        return weights, site_list, target_ids

    return collect


def main():
    parser = argparse.ArgumentParser(description="Run the full audit grid")
    parser.add_argument("--dummy", action="store_true",
                        help="synthetic run, no data or checkpoints needed")
    parser.add_argument("--models", default="coldsite_dti,uniform_control")
    parser.add_argument("--datasets", default="davis")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--ground-truth")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--task", default="binary",
                        choices=["regression", "binary"],
                        help="must match the task the checkpoints were trained "
                             "on -- it is part of their filename")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--pairs-per-target", type=int, default=1,
                        help="test pairs collected per protein. precision@k is "
                             "a per-protein quantity, so >1 makes n a count of "
                             "pairs rather than proteins")
    parser.add_argument("--max-protein-len", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in models:
        if name not in available_models():
            raise SystemExit(
                f"unknown model '{name}'. Registered: {available_models()}\n"
                f"Add an adapter in src/evaluation/model_registry.py "
                f"(see docs/PART2_GUIDE_124AD0008.md Priority 5).")

    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    if len(seeds) < 3:
        print(f"WARNING: {len(seeds)} seed(s). Cells will be flagged '!' and "
              f"cannot be quoted as estimates. See aggregate.MIN_SEEDS_FOR_A_CLAIM.")

    skipped = []
    if args.dummy:
        collect = dummy_collect_fn()
        tag = "DUMMY_PLACEHOLDER"
    else:
        if not args.ground_truth:
            parser.error("--ground-truth is required unless --dummy is set")
        if len(datasets) != 1:
            parser.error(
                "real mode takes one --datasets at a time: the ground-truth "
                "file is per dataset, and DAVIS sites scored against KIBA "
                "proteins would return a number rather than an error")

        from src.evaluation.collect import make_collect_fn

        # DeepDTA anchors the accuracy axis and has no attention to collect.
        # Dropped here with a note rather than inside the grid, so the report
        # does not list its four levels as "missing" as though a checkpoint
        # were merely absent.
        auditable = [m for m in models
                     if getattr(model_class(m), "provides_attention", True)]
        for name in sorted(set(models) - set(auditable)):
            print(f"[skip] {name}: provides_attention = False — accuracy anchor "
                  f"only, nothing to explain")
        models = auditable
        if not models:
            raise SystemExit(
                "No model in --models can produce explanations. The audit's "
                "explanation axis needs at least one model with attention.")

        collect = make_collect_fn(
            dataset=datasets[0], ground_truth=args.ground_truth,
            task=args.task, split_root=args.split_root,
            checkpoint_dir=args.checkpoint_dir,
            max_protein_len=args.max_protein_len,
            pairs_per_target=args.pairs_per_target,
            device=args.device, skipped=skipped)
        tag = f"{datasets[0]}_{args.task}"

    results = build_grid(collect, models, datasets, seeds,
                         k=args.k, n_trials=args.n_trials)
    if skipped:
        results["skipped_reasons"] = skipped

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"audit_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    report = summarise(results)
    with open(os.path.join(args.out_dir, f"audit_{tag}.md"), "w") as f:
        f.write(report)

    print(report)
    if args.dummy:
        print("\n" + "=" * 70)
        print("DUMMY RUN — untrained models, synthetic sites. NOT results.")
        print("=" * 70)


if __name__ == "__main__":
    main()
