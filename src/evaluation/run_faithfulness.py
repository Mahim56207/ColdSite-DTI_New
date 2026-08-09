"""
Track B (124AD0015) — the faithfulness runner and the accuracy hand-off.

`faithfulness.py` is the algorithm and is not touched here. This is the
plumbing around it: load a trained checkpoint per difficulty level, collect the
pairs and their attention, run `batch_faithfulness` with its random-masking
control, and write both outputs the rest of the project needs.

    # today, no data and no checkpoints needed -- proves the plumbing
    python -m src.evaluation.run_faithfulness --dummy

    # once the grid has run
    python -m src.evaluation.run_faithfulness \
        --dataset davis --seed 1 --checkpoint-dir results

Two artefacts come out, and they are for different readers.

`faithfulness_{dataset}_seed{N}.json`
    The measurement. Per level: comprehensiveness, its random control, and the
    delta between them. Only the delta is a result -- masking anything at all
    moves the prediction (the ROAR critique), so a comprehensiveness of 0.4 is
    meaningless until you see the random-masking 0.05 beside it.

`accuracy_{dataset}_seed{N}.json`
    The hand-off to Track C: `{level: accuracy}`, exactly the shape
    `run_ladder.py --accuracy-json` and `plots.plot_degradation_curve` expect.
    It is read out of the `*_results.json` files the trainer already writes, so
    the accuracy axis of the headline figure can never disagree with the
    numbers the training runs actually reported.

A negative delta is not a failed run
------------------------------------
If `comprehensiveness_delta <= 0` on the cold splits, masking the attended
residues perturbs the prediction no more than masking arbitrary ones: the
attention is decoration there. That is the "plausible but not faithful" cell of
the master plan's table and the most interesting result available to this
project. It is reported plainly, not retried until it looks better.
"""
import argparse
import json
import os

import numpy as np
import torch

from src.evaluation.faithfulness import batch_faithfulness
from src.model.checkpoint_naming import checkpoint_path as build_checkpoint_path
from src.model.checkpoint_naming import discover_checkpoints, results_path, run_tag

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
LEVEL_LABELS = {"random": "Warm", "cold_drug": "Cold-Drug",
                "cold_target": "Cold-Target", "cold_pair": "Cold-Pair"}

# Re-exported, not redefined: the canonical mapping lives beside
# train.compute_metrics, which is what decides these keys exist. Kept importable
# from here because that is where consumers already look for it.
from src.model.train import DEFAULT_ACCURACY_METRIC, accuracy_metric_for  # noqa: E402


# --------------------------------------------------------------------------
# collecting the inputs batch_faithfulness needs
# --------------------------------------------------------------------------

def collect_pairs(model, dataloader, max_pairs: int = 200, device: str = "cpu"):
    """(drugs, proteins, attentions) as one-row tensors plus real-length weights.

    `batch_faithfulness` masks positions of the protein tensor using indices
    taken from the attention array, so the two must be indexed the same way.
    `ColdSiteDTI.explain` returns exactly one weight per REAL residue, measured
    with `real_lengths()`, and padding is trailing -- so attention index j is
    protein column j. Returning padded-length attention here would not crash;
    it would silently shift every masked position.
    """
    model.eval().to(device)
    drugs, proteins, attentions = [], [], []

    with torch.no_grad():
        for drug_batch, protein_batch, _labels in dataloader:
            drug_batch = drug_batch.to(device)
            protein_batch = protein_batch.to(device)
            explanations = model.explain(drug_batch, protein_batch)

            for i, explanation in enumerate(explanations):
                drugs.append(drug_batch[i:i + 1].cpu())
                proteins.append(protein_batch[i:i + 1].cpu())
                attentions.append(np.asarray(explanation, dtype=float))
                if len(attentions) >= max_pairs:
                    return drugs, proteins, attentions

    return drugs, proteins, attentions


def faithfulness_for_level(model, dataloader, k: int = 10,
                           n_random_trials: int = 5, max_pairs: int = 200,
                           seed: int = 0, device: str = "cpu") -> dict:
    """One difficulty level: collect pairs, then measure with the control.

    Cost is `(2 + 2*n_random_trials + len(k_values))` forward passes per pair,
    so `max_pairs` is the budget knob. The mean over a few hundred pairs is the
    number; the mean over all of them is the same number and a much longer wait.
    """
    drugs, proteins, attentions = collect_pairs(model, dataloader, max_pairs, device)
    if not attentions:
        return {"n_pairs": 0, "comprehensiveness_delta": float("nan"),
                "explanation_is_load_bearing": False}
    return batch_faithfulness(model, drugs, proteins, attentions, k=k,
                              n_random_trials=n_random_trials, seed=seed,
                              max_pairs=max_pairs)


# --------------------------------------------------------------------------
# the accuracy hand-off to Track C
# --------------------------------------------------------------------------

def collect_accuracy(results_dir: str, dataset: str, task: str, seed: int,
                     metric: str = None, levels=LEVELS) -> dict:
    """Read `{level: accuracy}` out of the trainer's per-run results files.

    This is the Track C hand-off. It is derived rather than re-computed on
    purpose: `run_ladder` refuses to draw the headline figure without accuracy
    values, and an accuracy axis recomputed here could disagree with the
    numbers the training runs reported without anyone noticing.

    A level with no results file is reported as NaN and named in
    `missing_levels`, never defaulted to zero -- a zero would draw as a real
    point on the figure.
    """
    metric = metric or accuracy_metric_for(task)
    accuracy, missing, found_metrics = {}, [], {}

    for level in levels:
        tag = run_tag(dataset, level, task, seed)
        path = results_path(results_dir, tag)
        if not os.path.exists(path):
            accuracy[level] = float("nan")
            missing.append(level)
            continue
        with open(path) as f:
            payload = json.load(f)
        metrics = payload.get("test_metrics", {})
        if metric not in metrics:
            accuracy[level] = float("nan")
            missing.append(f"{level} (no '{metric}' in {sorted(metrics)})")
            continue
        accuracy[level] = float(metrics[metric])
        found_metrics[level] = metrics

    return {"accuracy": accuracy, "metric": metric, "missing_levels": missing,
            "all_metrics": found_metrics}


def write_accuracy_json(accuracy: dict, out_dir: str, tag: str) -> str:
    """Flat `{level: value}` — the exact shape `--accuracy-json` reads.

    Kept flat and separate from the richer faithfulness file so Track C can
    pass it straight to `run_ladder` without unwrapping anything.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"accuracy_{tag}.json")
    with open(path, "w") as f:
        json.dump(accuracy, f, indent=2)
    return path


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def faithfulness_table(summaries: dict) -> str:
    """The table that goes in the paper. Delta gets the verdict column."""
    lines = [
        "| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value):
        return "n/a" if value is None or not np.isfinite(value) else f"{value:.4f}"

    for level in LEVELS:
        entry = summaries.get(level)
        if entry is None:
            continue
        delta = entry.get("comprehensiveness_delta", float("nan"))
        verdict = "yes" if entry.get("explanation_is_load_bearing") else "**no**"
        lines.append(
            f"| {LEVEL_LABELS[level]} | {fmt(entry.get('comprehensiveness'))} | "
            f"{fmt(entry.get('comprehensiveness_random'))} | **{fmt(delta)}** | "
            f"{fmt(entry.get('sufficiency'))} | "
            f"{fmt(entry.get('sufficiency_random'))} | "
            f"{fmt(entry.get('aopc'))} | {entry.get('n_pairs', 0)} | {verdict} |")

    lines += [
        "",
        "`delta` = comprehensiveness minus its random-masking control, and it is "
        "the only column that is a result. Masking anything moves the "
        "prediction, so the raw comprehensiveness means nothing on its own.",
        "",
        "`load-bearing? no` means masking the attended residues perturbed the "
        "prediction no more than masking arbitrary ones — the explanation is "
        "decoration at that level. That is a finding, not a failed run.",
    ]
    return "\n".join(lines)


def write_results(summaries: dict, out_dir: str, tag: str, accuracy: dict = None,
                  metadata: dict = None, draw_figure: bool = True) -> dict:
    """JSON, markdown table, the bar figure, and the flat accuracy hand-off."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {**(metadata or {}), "levels": summaries}
    if accuracy is not None:
        payload["accuracy"] = accuracy

    json_path = os.path.join(out_dir, f"faithfulness_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    table = faithfulness_table(summaries)
    table_path = os.path.join(out_dir, f"faithfulness_{tag}.md")
    with open(table_path, "w") as f:
        f.write(f"# Faithfulness — {tag}\n\n{table}\n")

    figure_path = None
    if draw_figure and summaries:
        from src.evaluation.plots import plot_faithfulness

        figure_path = os.path.join(out_dir, f"faithfulness_{tag}.png")
        plot_faithfulness(summaries,
                          title=f"Are the explanations load-bearing? — {tag}",
                          save_path=figure_path)

    accuracy_path = None
    if accuracy is not None:
        accuracy_path = write_accuracy_json(accuracy, out_dir, tag)

    print(f"\n{table}\n")
    print(f"Saved -> {json_path}")
    print(f"Saved -> {table_path}")
    if accuracy_path:
        print(f"Saved -> {accuracy_path}   (Track C: pass to run_ladder --accuracy-json)")
    return {"json": json_path, "table": table_path, "figure": figure_path,
            "accuracy": accuracy_path}


# --------------------------------------------------------------------------
# dummy mode
# --------------------------------------------------------------------------

def run_dummy(out_dir="results", n_pairs=6, protein_length=120, k=10,
              n_random_trials=2, seed=0) -> dict:
    """The whole workflow on an untrained model and synthetic pairs.

    Deliberately meaningless numbers: an untrained model's attention is
    arbitrary, so the delta should sit near zero. That is the point -- it
    exercises every step while making the output impossible to mistake for a
    result, and it doubles as the sanity floor. A strongly positive delta here
    would mean the metric is reading a masking artefact.
    """
    from src.model.coldsite_dti import ColdSiteDTI

    torch.manual_seed(seed)
    model = ColdSiteDTI(70, 28).eval()
    summaries = {}

    for level_index, level in enumerate(LEVELS):
        drugs, proteins, attentions = [], [], []
        for i in range(n_pairs):
            torch.manual_seed(seed + level_index * 100 + i)
            drug = torch.randint(2, 70, (1, 40))
            protein = torch.randint(2, 28, (1, protein_length))
            drugs.append(drug)
            proteins.append(protein)
            attentions.append(np.asarray(model.explain(drug, protein)[0]))

        summaries[level] = batch_faithfulness(
            model, drugs, proteins, attentions, k=k,
            n_random_trials=n_random_trials, seed=seed)
        print(f"{LEVEL_LABELS[level]:12s} "
              f"comp={summaries[level]['comprehensiveness']:.4f}  "
              f"random={summaries[level]['comprehensiveness_random']:.4f}  "
              f"delta={summaries[level]['comprehensiveness_delta']:+.4f}")

    fake_accuracy = dict(zip(LEVELS, [0.89, 0.81, 0.78, 0.69]))
    paths = write_results(
        summaries, out_dir, "DUMMY_PLACEHOLDER", accuracy=fake_accuracy,
        metadata={"model": "coldsite_dti", "dataset": "DUMMY_PLACEHOLDER",
                  "task": "regression", "seed": seed, "k": k,
                  "n_random_trials": n_random_trials,
                  "accuracy_metric": "DUMMY_PLACEHOLDER",
                  "checkpoints": "none — untrained model"})

    print("\n" + "=" * 70)
    print("DUMMY RUN — untrained model, synthetic pairs, placeholder accuracy.")
    print("These are NOT results. Every filename carries DUMMY_PLACEHOLDER so")
    print("they cannot be mistaken for the real measurement later.")
    print("=" * 70)
    return {"summaries": summaries, "paths": paths}


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run faithfulness on trained checkpoints and emit the "
                    "accuracy hand-off for Track C")
    parser.add_argument("--dummy", action="store_true",
                        help="synthetic run, no data or checkpoints needed")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--seed", type=int, default=1,
                        help="training seed of the checkpoints to read")
    parser.add_argument("--task", default="regression",
                        choices=["regression", "binary"])
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--results-dir", default="results",
                        help="where the trainer wrote its *_results.json")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-random-trials", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=200,
                        help="pairs per level; the budget knob")
    parser.add_argument("--max-protein-len", type=int, default=1000)
    parser.add_argument("--accuracy-metric",
                        help="field of test_metrics to use as accuracy "
                             "(default: ci for regression, auroc for binary)")
    args = parser.parse_args()

    if args.dummy:
        run_dummy(out_dir=args.out_dir, k=args.k,
                  n_random_trials=args.n_random_trials)
        return

    from src.model.coldsite_dti import ColdSiteDTI
    from src.model.dataset import load_split

    tag = f"{args.dataset}_seed{args.seed}"
    summaries, evaluated = {}, []

    for level in LEVELS:
        split_dir = os.path.join(args.split_root, args.dataset, level)
        checkpoint = build_checkpoint_path(
            args.checkpoint_dir, args.dataset, level, args.task, args.seed)

        if not (os.path.isdir(split_dir) and os.path.exists(checkpoint)):
            print(f"[skip] {level}: missing {split_dir} or {checkpoint}")
            available = [c["seed"] for c in discover_checkpoints(
                args.checkpoint_dir, dataset=args.dataset, split=level,
                task=args.task)]
            if available:
                print(f"        (seeds present for this cell: {available})")
            continue

        _train, _valid, test_loader, drug_vocab, protein_vocab = load_split(
            split_dir, args.max_protein_len)
        model = ColdSiteDTI(len(drug_vocab) + 2, len(protein_vocab) + 2)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])

        print(f"\n{level}: measuring up to {args.max_pairs} pairs "
              f"({2 + 2 * args.n_random_trials + 5} forward passes each)")
        summaries[level] = faithfulness_for_level(
            model, test_loader, k=args.k, n_random_trials=args.n_random_trials,
            max_pairs=args.max_pairs, seed=args.seed)
        evaluated.append(level)

    if not summaries:
        raise SystemExit(
            "No level could be evaluated — no checkpoints found.\n"
            "Train one cell first:\n"
            f"  python -m src.model.train --split-dir "
            f"{args.split_root}/{args.dataset}/random "
            f"--dataset {args.dataset} --split random --seed {args.seed}\n"
            "Or prove the workflow with no data at all:\n"
            "  python -m src.evaluation.run_faithfulness --dummy")

    collected = collect_accuracy(args.results_dir, args.dataset, args.task,
                                 args.seed, metric=args.accuracy_metric)
    if collected["missing_levels"]:
        print(f"\nWARNING: no accuracy for {collected['missing_levels']}. "
              f"run_ladder will refuse to draw the headline figure until every "
              f"level has one — fidelity alone is half the claim.")

    write_results(summaries, args.out_dir, tag,
                  accuracy=collected["accuracy"],
                  metadata={"model": "coldsite_dti", "dataset": args.dataset,
                            "task": args.task, "seed": args.seed, "k": args.k,
                            "n_random_trials": args.n_random_trials,
                            "max_pairs": args.max_pairs,
                            "accuracy_metric": collected["metric"],
                            "levels_evaluated": evaluated,
                            "accuracy_missing_levels": collected["missing_levels"]})


if __name__ == "__main__":
    main()
