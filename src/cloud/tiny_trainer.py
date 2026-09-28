"""A tiny trainer that uses the same resume machinery as the four real ones, for testing the
harness end to end on a CPU in seconds (tests/test_cloud_runner.py). It is not a model anyone
audits and no wave ever names it.

    python -m src.cloud.tiny_trainer --checkpoint-dir D --results-dir D --seed 1 --epochs 6
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.model import resume
from src.model.early_stopping import CheckpointSelector

RESUME_KEYS = ("seed", "epochs", "lr")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epoch-sleep", type=float, default=0.0)
    ap.add_argument("--stem", default="tiny_cell")
    ap.add_argument("--results-file", help="default <results-dir>/<stem>_results.json")
    ap.add_argument("--stop-after-epoch", type=int)
    args = ap.parse_args(argv)

    ckpt = os.path.join(args.checkpoint_dir, f"{args.stem}.pt")
    results = args.results_file or os.path.join(args.results_dir, f"{args.stem}_results.json")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 5, generator=generator)
    y = x @ torch.arange(1.0, 6.0) + 0.1 * torch.randn(64, generator=generator)
    model = nn.Linear(5, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    selector = CheckpointSelector(patience=100, min_epochs=1, n_epochs=args.epochs)
    run = resume.Resumable(ckpt, "cpu", vars(args), RESUME_KEYS, args.stop_after_epoch)
    start = run.begin(model, optimizer, selector)
    for epoch in (range(start, args.epochs + 1) if start else ()):
        for idx in torch.randperm(64).split(16):
            optimizer.zero_grad()
            ((model(x[idx]).squeeze(-1) - y[idx]) ** 2).mean().backward()
            optimizer.step()
        time.sleep(args.epoch_sleep)
        with torch.no_grad():
            val = float(((model(x).squeeze(-1) - y) ** 2).mean())
        best = ({"model_state": model.state_dict(), "epoch": epoch}
                if selector.consider(epoch, val) else None)
        run.end_epoch(epoch, finished=epoch == args.epochs, best_checkpoint=best)
        if run.interrupt_now(epoch):
            return
    with torch.no_grad():
        final = [p.detach().clone().flatten().tolist() for p in model.parameters()]
    with open(results, "w") as handle:
        json.dump({"test_metrics": {"auroc": 0.5, "auprc": 0.5}, "final_params": final}, handle)
    run.clear()


if __name__ == "__main__":
    main()
