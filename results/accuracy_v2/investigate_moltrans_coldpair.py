"""T04 follow-up: why do MolTrans DAVIS cold-pair seeds 1 and 2 not reproduce their recorded test metrics?

Read-only with respect to every recorded result: it loads checkpoints and split files, scores them, and
writes ONE new file (results/accuracy_v2/moltrans_coldpair_investigation.json). It uses the pipeline's own
`clean_accuracy._scores`, so "current" is exactly what `accuracy_table predict` ran.

Arms (per cell):
  current      _scores() as is: torch.manual_seed(cell seed) then one pass on the device      x2  (repeatable?)
  no_dropout   torch.nn.functional.dropout forced to training=False (every dropout off, incl. the
               vendored `F.dropout(i_v, p)` that has no `training=` argument)               x2 on MPS, x1 on CPU
  seeded       as `current` but manual_seed(1000 + i)                                        i = 0..5
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.getcwd())
from src.evaluation.clean_accuracy import _scores                       # noqa: E402
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag  # noqa: E402

HOME = os.path.expanduser("~/ColdSite-results/davis_binary")
OUT = "results/accuracy_v2/moltrans_coldpair_investigation.json"
CELLS = [("cold_pair", 1), ("cold_pair", 2)]
_orig_dropout = F.dropout


def metrics(labels, logits):
    y = (np.asarray(labels) >= 0.5).astype(int)
    z = np.asarray(logits, float)
    return {"auroc": float(roc_auc_score(y, z)), "auprc": float(average_precision_score(y, z)),
            "accuracy": float(((z >= 0).astype(int) == y).mean())}


def score(level, seed, device, *, dropout_forced_off=False, rng_seed=None):
    ck = checkpoint_path(HOME, "davis", level, "binary", seed, model="moltrans")
    rec = json.load(open(results_path(HOME, run_tag("davis", level, "binary", seed), model="moltrans")))
    if dropout_forced_off:
        F.dropout = lambda x, p=0.5, training=True, inplace=False: _orig_dropout(x, p, False, inplace)
    try:
        # `_scores` seeds with `seed`; for the seeded arm we override by calling it with rng_seed
        labels, logits = _scores("moltrans", f"data/splits/davis/{level}", "davis", ck, rec,
                                 seed if rng_seed is None else rng_seed, device)
    finally:
        F.dropout = _orig_dropout
    return metrics(labels, logits), np.asarray(logits, float)


def main():
    result = {"device_default": "mps", "torch": torch.__version__, "cells": {}}
    for level, seed in CELLS:
        rec = json.load(open(results_path(HOME, run_tag("davis", level, "binary", seed), model="moltrans")))
        import pandas as pd
        saved = pd.read_csv(f"results/accuracy_v2/predictions/davis_{level}_moltrans_seed{seed}.csv.gz").logit.to_numpy()
        cell = {"recorded": rec["test_metrics"], "arms": {}}
        t = time.time()

        def arm(name, **kw):
            m, z = score(level, seed, kw.pop("device", "mps"), **kw)
            m["max_abs_logit_diff_vs_saved_predictions"] = float(np.abs(z - saved).max()) if len(z) == len(saved) else None
            cell["arms"].setdefault(name, []).append(m)
            print(f"cold_pair s{seed} {name:11s} auroc {m['auroc']:.6f} auprc {m['auprc']:.6f} "
                  f"acc {m['accuracy']:.6f}  |dlogit vs saved| {m['max_abs_logit_diff_vs_saved_predictions']}  "
                  f"({time.time() - t:.0f}s)", flush=True)

        arm("current"); arm("current")
        arm("no_dropout", dropout_forced_off=True); arm("no_dropout", dropout_forced_off=True)
        arm("no_dropout_cpu", dropout_forced_off=True, device="cpu")
        for i in range(6):
            arm("seeded", rng_seed=1000 + i)
        result["cells"][f"cold_pair_seed{seed}"] = cell
        json.dump(result, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
