"""
Do different explanation methods point at the same residues? (T08, descriptive only.)

Top-k IoU between two explanations of the same pair, averaged over the proteins of a cell,
reported beside what two independent random top-k sets would give (`chance_iou`) — an
IoU of 0.05 means little until it is read against a chance of 0.01. k is the pre-specified
10 (amendment §4). No test and no Holm here (amendment §3: agreement is descriptive).

    python -m src.evaluation.explanation_agreement --base moltrans --dataset davis \
        --levels random cold_drug --seeds 1 2 3 --checkpoint-dir ~/ColdSite-results/davis_binary \
        --out-dir results/methods_v2/agreement

Every method is scored on the same pairs: `collect_cell` reads the same test rows in the
same order for every model name, and the runner refuses to compare two methods whose
protein lists differ.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

K = 10


def top_k(weights, k: int = K) -> set:
    """Indices of the k largest weights. Ties break towards the lower index (stable sort),
    so the same explanation always gives the same set."""
    w = np.asarray(weights, dtype=float)
    if w.size < k:
        raise ValueError(f"an explanation of {w.size} residues has no top {k}")
    return set(np.argsort(-w, kind="stable")[:k].tolist())


def topk_iou(a, b, k: int = K) -> float:
    ta, tb = top_k(a, k), top_k(b, k)
    return len(ta & tb) / len(ta | tb)


def chance_iou(n: int, k: int = K) -> float:
    """Expected IoU of two independent uniformly random k-subsets of n residues: the
    overlap i is hypergeometric and IoU = i / (2k - i). Exact, not simulated."""
    if not 0 < k <= n:
        raise ValueError("need 0 < k <= n")
    total = math.comb(n, k)
    expected = 0.0
    for i in range(max(0, 2 * k - n), k + 1):
        p = math.comb(k, i) * math.comb(n - k, k - i) / total
        expected += p * i / (2 * k - i)
    return expected


def agreement(explanations: dict, k: int = K) -> dict:
    """explanations: method -> list over proteins of 1-D weights. Returns, per method pair,
    the mean top-k IoU over proteins and the mean chance IoU for those proteins' lengths."""
    names = sorted(explanations)
    lengths = {m: [np.asarray(w).size for w in explanations[m]] for m in names}
    n_proteins = {len(v) for v in lengths.values()}
    if len(n_proteins) != 1:
        raise ValueError(f"methods explain different numbers of proteins: "
                         f"{ {m: len(v) for m, v in lengths.items()} }")
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ious, chances = [], []
            for wa, wb in zip(explanations[a], explanations[b]):
                n = min(np.asarray(wa).size, np.asarray(wb).size)
                if n < k:
                    continue
                ious.append(topk_iou(np.asarray(wa)[:n], np.asarray(wb)[:n], k))
                chances.append(chance_iou(n, k))
            out[f"{a}|{b}"] = {"iou": float(np.mean(ious)) if ious else None,
                               "chance": float(np.mean(chances)) if chances else None,
                               "n_proteins": len(ious)}
    return out


METHOD_NAMES = {
    "coldsite_dti": ["coldsite_dti", "coldsite_dti_ig", "coldsite_dti_occlusion"],
    "hyperattentiondti": ["hyperattentiondti", "hyperattentiondti_ig",
                          "hyperattentiondti_occlusion", "hyperattentiondti_attngrad"],
    "moltrans": ["moltrans", "moltrans_ig", "moltrans_occlusion", "moltrans_attngrad",
                 "moltrans_rollout"],
    "drugban": ["drugban", "drugban_occlusion"],
}


def run_cell(base: str, dataset: str, level: str, seed: int, *, site_sets, checkpoint_dir,
             split_root="data/splits", max_proteins=None, device="cpu", k=K) -> dict:
    from src.evaluation.collect import collect_cell

    explanations, ids = {}, {}
    for name in METHOD_NAMES[base]:
        weights, _sites, used = collect_cell(
            name, dataset, level, seed, site_sets=site_sets, split_root=split_root,
            checkpoint_dir=checkpoint_dir, max_proteins=max_proteins, device=device,
            verbose=False)
        explanations[name], ids[name] = weights, used
    first = ids[METHOD_NAMES[base][0]]
    for name, used in ids.items():
        if used != first:
            raise RuntimeError(f"{name} scored different proteins from "
                               f"{METHOD_NAMES[base][0]}: agreement would compare unlike things")
    return {"base": base, "dataset": dataset, "level": level, "seed": seed, "k": k,
            "n_proteins": len(first), "pairs": agreement(explanations, k)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base", required=True, choices=sorted(METHOD_NAMES))
    ap.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-proteins", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ground-truth", default=None,
                    help="default data/<dataset>_ground_truth_sites.json")
    args = ap.parse_args(argv)

    from src.data.ground_truth import load_site_sets
    if os.path.abspath(args.out_dir).startswith(os.path.abspath("results") + os.sep) and \
            "methods_v2" not in args.out_dir:
        raise SystemExit("write extension outputs under results/methods_v2/, never over "
                         "the original results (plan: statistical_integrity_rules)")
    site_sets = load_site_sets(args.ground_truth or f"data/{args.dataset}_ground_truth_sites.json",
                               max_len=1000)
    os.makedirs(args.out_dir, exist_ok=True)
    for level in args.levels:
        for seed in args.seeds:
            path = os.path.join(args.out_dir,
                                f"agreement_{args.base}_{args.dataset}_{level}_seed{seed}.json")
            if os.path.exists(path):
                print(f"exists, skipping: {path}")
                continue
            cell = run_cell(args.base, args.dataset, level, seed, site_sets=site_sets,
                            checkpoint_dir=args.checkpoint_dir,
                            max_proteins=args.max_proteins, device=args.device)
            with open(path, "w") as handle:
                json.dump(cell, handle, indent=2)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
