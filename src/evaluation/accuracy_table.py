"""
Predictive accuracy of every trained cell (remediation task T04).

Three stages, so the expensive one runs once and everything after it reads files:

    predict    one test pass per cell, by the cell's own trainer's code (the same `_scores`
               that `clean_accuracy` uses), writing the per-row logits to
               results/accuracy_v2/predictions/ and a sidecar with the checkpoint's SHA-256.
               Resumable: a cell whose predictions and sidecar exist is skipped.
    tabulate   reads ONLY the predictions files; writes cells.csv, by_model.csv and
               predictions_manifest.csv (SHA-256 of every predictions file). Refuses to write
               unless every expected cell is present (84), so a partial table cannot be
               mistaken for the full one.
    localize   accuracy vs localization: Spearman rho between a cell's AUROC and its
               precision@10 from the committed ladders, with a percentile bootstrap CI over
               cells (`docs/PROTOCOL_AMENDMENT_v2.md` §5). A description, not a test.

    python -m src.evaluation.accuracy_table predict  --checkpoint-dir ~/ColdSite-results --device mps
    python -m src.evaluation.accuracy_table tabulate
    python -m src.evaluation.accuracy_table localize

Decision rule for MCC / F1 / accuracy: the trainers' own (`src/model/train.py`,
`compute_metrics`): probability >= 0.5, i.e. logit >= 0. Every cell's recorded test accuracy is
reproduced by it (`reproduces_recorded`). The binary label threshold is the dataset's
(`BINARY_THRESHOLD`), already applied when the test rows were read.

DeepDTA was trained on the binary task in all 36 cells here, so it is scored like the other
classifiers; regression metrics (MSE, CI, Pearson, Spearman) describe no checkpoint on disk.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from src.evaluation.clean_accuracy import REPRODUCE_TOLERANCE, _scores, seen_by_sequence
from src.evaluation.run_faithfulness import output_tag
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag
from src.model.dataset import BINARY_THRESHOLD

OUT_DIR = "results/accuracy_v2"
MODELS = {"davis": ("coldsite_dti", "hyperattentiondti", "moltrans", "drugban", "deepdta"),
          "kiba": ("coldsite_dti", "hyperattentiondti", "moltrans", "deepdta")}
LEVELS = {"davis": ("random", "cold_drug", "cold_target", "cold_pair"),
          "kiba": ("random", "cold_drug")}
SEEDS = (1, 2, 3)
ATTENTION_MODELS = ("coldsite_dti", "hyperattentiondti", "moltrans", "drugban")
LOGIT_THRESHOLD = 0.0                    # sigmoid(logit) >= 0.5, `train.compute_metrics`
K = "10"                                 # the pre-specified k (amendment §4)
N_RESAMPLES = 10000
CONFIDENCE = 0.95


def expected_cells() -> list:
    return [(d, m, l, s) for d in MODELS for m in MODELS[d] for l in LEVELS[d] for s in SEEDS]


def cell_id(dataset: str, model: str, level: str, seed: int) -> str:
    return f"{dataset}_{level}_{model}_seed{seed}"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- metrics ---------------------------------------------------------------------------

def binary_metrics(labels, logits) -> dict:
    """Threshold-free (AUROC, AUPR) and thresholded (accuracy, MCC, F1, precision, recall)."""
    y = (np.asarray(labels) >= 0.5).astype(int)
    z = np.asarray(logits, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError(f"{int((~np.isfinite(z)).sum())} non-finite logits")
    pred = (z >= LOGIT_THRESHOLD).astype(int)
    two_classes = len(np.unique(y)) == 2
    nan = float("nan")
    return {"rows": int(y.size), "positives": int(y.sum()), "positive_rate": float(y.mean()),
            "auroc": float(roc_auc_score(y, z)) if two_classes else nan,
            "auprc": float(average_precision_score(y, z)) if two_classes else nan,
            "accuracy": float((pred == y).mean()),
            "mcc": float(matthews_corrcoef(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "predicted_positive_rate": float(pred.mean())}


def write_predictions(path: str, frame: pd.DataFrame) -> str:
    """Byte-deterministic gzip CSV (mtime pinned to 0); returns its SHA-256."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(frame.to_csv(index=False, float_format="%.9g").encode())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)
    return sha256_file(path)


# -- stage 1: predict ------------------------------------------------------------------

def _dirs(checkpoint_dir: str, dataset: str) -> str:
    return os.path.join(checkpoint_dir, f"{dataset}_binary")


def predict_cell(dataset, model, level, seed, checkpoint_dir, out_dir, device) -> str:
    import torch
    ck_dir = _dirs(checkpoint_dir, dataset)
    ckpt = checkpoint_path(ck_dir, dataset, level, "binary", seed, model=model)
    res = results_path(ck_dir, run_tag(dataset, level, "binary", seed), model=model)
    recorded = json.load(open(res))
    split_dir = os.path.join("data/splits", dataset, level)
    test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    labels, logits = _scores(model, split_dir, dataset, ckpt, recorded, seed, device)
    if len(labels) != len(test):
        raise RuntimeError(f"{cell_id(dataset, model, level, seed)}: {len(labels)} scores for "
                           f"{len(test)} test rows")
    leaked = seen_by_sequence(dataset, level) if level in ("cold_target", "cold_pair") else frozenset()
    frame = pd.DataFrame({
        "row": np.arange(len(test)), "Drug_ID": test.Drug_ID.to_numpy(),
        "Target_ID": test.Target_ID.astype(str).to_numpy(),
        "seen_by_sequence": test.Target_ID.astype(str).isin(leaked).to_numpy(),
        "label": (np.asarray(labels) >= 0.5).astype(int), "logit": np.asarray(logits, float)})
    cid = cell_id(dataset, model, level, seed)
    pred_sha = write_predictions(os.path.join(out_dir, "predictions", cid + ".csv.gz"), frame)
    meta = {"cell": cid, "dataset": dataset, "model": model, "level": level, "seed": seed,
            "checkpoint_file": os.path.basename(ckpt), "checkpoint_sha256": sha256_file(ckpt),
            "predictions_sha256": pred_sha, "device": device, "torch": torch.__version__,
            "binary_threshold": BINARY_THRESHOLD[dataset], "test_rows": len(test),
            "recorded_test_metrics": recorded["test_metrics"]}
    with open(os.path.join(out_dir, "predictions", cid + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return cid


def cmd_predict(args) -> None:
    out = args.out_dir
    todo = [c for c in expected_cells()
            if (not args.datasets or c[0] in args.datasets.split(","))
            and (not args.models or c[1] in args.models.split(","))]
    for i, (d, m, l, s) in enumerate(todo, 1):
        cid = cell_id(d, m, l, s)
        done = all(os.path.exists(os.path.join(out, "predictions", cid + ext))
                   for ext in (".csv.gz", ".meta.json"))
        if done:
            print(f"[{i}/{len(todo)}] skip {cid}: already predicted", flush=True)
            continue
        try:
            predict_cell(d, m, l, s, args.checkpoint_dir, out, args.device)
            print(f"[{i}/{len(todo)}] ok   {cid}", flush=True)
        except Exception as exc:                          # a missing DGL must not hide the rest
            print(f"[{i}/{len(todo)}] FAIL {cid}: {type(exc).__name__}: {exc}", flush=True)


# -- stage 2: tabulate -----------------------------------------------------------------

def load_cell(out_dir: str, cid: str) -> tuple[pd.DataFrame, dict]:
    base = os.path.join(out_dir, "predictions", cid)
    meta = json.load(open(base + ".meta.json"))
    if sha256_file(base + ".csv.gz") != meta["predictions_sha256"]:
        raise RuntimeError(f"{cid}: predictions file does not match its recorded SHA-256")
    return pd.read_csv(base + ".csv.gz"), meta


def cell_rows(dataset, model, level, seed, frame: pd.DataFrame, meta: dict) -> list:
    """One row per view. `uncorrected` = every test row; `unseen_by_sequence` only where
    some test target is seen by sequence in training (DAVIS cold-target / cold-pair)."""
    rec = meta["recorded_test_metrics"]
    views = [("uncorrected", frame)]
    if frame.seen_by_sequence.any():
        views.append(("unseen_by_sequence", frame[~frame.seen_by_sequence]))
    out = []
    for view, part in views:
        m = binary_metrics(part.label, part.logit)
        row = {"dataset": dataset, "model": model, "level": level, "seed": seed, "view": view,
               **m, "rows_dropped": int(len(frame) - len(part)),
               "predictions_file": meta["cell"] + ".csv.gz"}
        if view == "uncorrected":
            row.update({"recorded_auroc": rec["auroc"], "recorded_auprc": rec["auprc"],
                        "recorded_accuracy": rec["accuracy"],
                        "reproduces_recorded": all(
                            abs(m[k] - rec[k]) <= REPRODUCE_TOLERANCE
                            for k in ("auroc", "auprc", "accuracy"))})
        out.append(row)
    return out


METRICS = ("auroc", "auprc", "accuracy", "mcc", "f1", "precision", "recall")


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD (ddof=1) over the seeds of each (dataset, model, level, view)."""
    rows = []
    for (d, m, l, v), g in cells.groupby(["dataset", "model", "level", "view"], sort=False):
        row = {"dataset": d, "model": m, "level": l, "view": v, "n_seeds": len(g)}
        for k in METRICS:
            row[f"{k}_mean"] = float(g[k].mean())
            row[f"{k}_sd"] = float(g[k].std(ddof=1)) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_tabulate(args) -> None:
    out, rows, missing, manifest = args.out_dir, [], [], []
    for d, m, l, s in expected_cells():
        cid = cell_id(d, m, l, s)
        try:
            frame, meta = load_cell(out, cid)
        except FileNotFoundError:
            missing.append(cid)
            continue
        rows += cell_rows(d, m, l, s, frame, meta)
        manifest.append({"cell": cid, "predictions_file": cid + ".csv.gz",
                         "predictions_sha256": meta["predictions_sha256"], "rows": len(frame),
                         "checkpoint_file": meta["checkpoint_file"],
                         "checkpoint_sha256": meta["checkpoint_sha256"],
                         "device": meta["device"], "torch": meta["torch"]})
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} of {len(expected_cells())} cells have no predictions "
                         f"(first: {missing[:3]}); refusing to write a partial table. "
                         f"--allow-partial writes it under a _partial suffix.")
    suffix = "_partial" if missing else ""
    cells = pd.DataFrame(rows)
    n_cells = cells[["dataset", "model", "level", "seed"]].drop_duplicates().shape[0]
    if not missing and n_cells != len(expected_cells()):
        raise SystemExit(f"row-count check failed: {n_cells} cells, expected {len(expected_cells())}")
    cells.to_csv(os.path.join(out, f"cells{suffix}.csv"), index=False, float_format="%.6f")
    summarise(cells).to_csv(os.path.join(out, f"by_model{suffix}.csv"), index=False,
                            float_format="%.6f")
    pd.DataFrame(manifest).to_csv(os.path.join(out, f"predictions_manifest{suffix}.csv"), index=False)
    bad = cells[cells.reproduces_recorded == False]          # noqa: E712  (NaN rows are views)
    print(f"{n_cells} cells ({len(missing)} missing); "
          f"{int((cells.reproduces_recorded == True).sum())} reproduce their recorded "  # noqa: E712
          f"AUROC/AUPR/accuracy, {len(bad)} do not"
          + (f": {list(bad.model + ' ' + bad.level + ' s' + bad.seed.astype(str))}" if len(bad) else ""))


# -- stage 3: accuracy vs localization -------------------------------------------------

def spearman_bootstrap(x, y, n_resamples: int = N_RESAMPLES, seed: int = 0,
                       confidence: float = CONFIDENCE) -> dict:
    """Spearman rho and a percentile CI from resampling cells (rows) with replacement.
    Resamples in which either variable is constant have no rho and are counted, not used."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rho = float(spearmanr(x, y)[0]) if n > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    rhos = []
    for idx in draws:
        xs, ys = x[idx], y[idx]
        if xs.std() > 0 and ys.std() > 0:
            rhos.append(spearmanr(xs, ys)[0])
    alpha = (1 - confidence) / 2
    low, high = (float(np.quantile(rhos, alpha)), float(np.quantile(rhos, 1 - alpha))) \
        if rhos else (float("nan"), float("nan"))
    return {"n_cells": n, "rho": rho, "low": low, "high": high, "n_resamples": n_resamples,
            "n_degenerate_resamples": n_resamples - len(rhos)}


def ladder_cell(ladder_dir: str, model: str, dataset: str, level: str, seed: int) -> dict | None:
    path = os.path.join(ladder_dir, f"ladder_{output_tag(model, dataset, seed)}.json")
    if not os.path.exists(path):
        return None
    entry = json.load(open(path)).get(level)
    if not entry or K not in entry.get("by_k", {}):
        return None
    k = entry["by_k"][K]
    return {"precision_at_10": k["precision_at_k"], "chance": k["chance"],
            "n_proteins": entry["n_proteins"]}


def cmd_localize(args) -> None:
    cells = pd.read_csv(os.path.join(args.out_dir, "cells.csv"))
    # accuracy on the same target policy the explanation metrics use (drop seen-by-sequence)
    key = ["dataset", "model", "level", "seed"]
    clean = cells[cells.view == "unseen_by_sequence"].set_index(key)
    every = cells[cells.view == "uncorrected"].set_index(key)
    rows = []
    for (d, m, l, s), row in every.iterrows():
        if m not in ATTENTION_MODELS:
            continue
        lad = ladder_cell(os.path.join("results", f"analysis_{d}_policyA"), m, d, l, s)
        if lad is None:
            continue
        auroc = clean.loc[(d, m, l, s)].auroc if (d, m, l, s) in clean.index else row.auroc
        rows.append({"dataset": d, "model": m, "level": l, "seed": s, "auroc": auroc,
                     "auroc_uncorrected": row.auroc, **lad,
                     "enrichment": lad["precision_at_10"] / lad["chance"] if lad["chance"] else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "localization_cells.csv"), index=False, float_format="%.6f")
    summary = []
    for d, g in table.groupby("dataset"):
        groups = [("all attention models", g)] + [(m, gm) for m, gm in g.groupby("model")]
        for name, gg in groups:
            for yname in ("precision_at_10", "enrichment"):
                ok = gg[[yname]].notna().all(axis=1)
                summary.append({"dataset": d, "group": name, "y": yname,
                                **spearman_bootstrap(gg.auroc[ok], gg[yname][ok])})
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "localization_spearman.csv"),
                                 index=False, float_format="%.6f")
    print(f"{len(table)} cells with a ladder; wrote localization_cells.csv, localization_spearman.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("predict")
    a.add_argument("--checkpoint-dir", required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--datasets", help="comma list; default all")
    a.add_argument("--models", help="comma list; default all")
    b = sub.add_parser("tabulate")
    b.add_argument("--allow-partial", action="store_true")
    sub.add_parser("localize")
    for s in (a, b, sub.choices["localize"]):
        s.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    {"predict": cmd_predict, "tabulate": cmd_tabulate, "localize": cmd_localize}[args.cmd](args)


if __name__ == "__main__":
    main()
