"""
Train the vendored MolTrans on our splits.

Why a separate trainer
-----------------------
Same reasoning as `train_deepdta.py` and `train_hyperattentiondti.py`: MolTrans
tokenises with its own ESPF subword tables (`baselines/MolTrans/stream.py`),
not `src/model/dataset.py`'s vocabulary, so it needs its own loader. Everything
downstream -- the run tag, the checkpoint path, the results JSON, the metrics
-- comes from the same shared modules the other two use
(`checkpoint_naming.py`, `train.compute_metrics`), so a MolTrans cell lands in
the same shape as a DeepDTA or HyperAttentionDTI cell and nothing downstream
needs a MolTrans-specific case.

The batch-size reshape bug
---------------------------
`BIN_Interaction_Flat.forward` reshapes twice using `self.batch_size` -- a
config value, not the tensor's real first dimension:

    i_v = i.view(int(self.batch_size / self.gpus), -1, max_d, max_p)
    f   = f.view(int(self.batch_size / self.gpus), -1)

The full story is in `src.evaluation.baseline_adapters.MolTransAdapter.
_fit_batch_size` (it doesn't fail loudly -- the reshape "succeeds" with wrong
scores). `_fit_batch_size` below is the same fix, applied once per batch here
instead of once per inference call, so the last, partial batch of an epoch
is scored against its own size rather than whatever the previous batch left
`self.batch_size` set to.

`train_epoch=True` also keeps every loader's `drop_last` off except the
training one. The decoder's `nn.BatchNorm1d` layers need more than one row to
compute batch statistics in training mode, and only the last training batch
of an epoch can land on exactly one row -- dropping that one possible short
batch loses at most `batch_size - 1` rows out of tens of thousands. Doing the
same on validation or test would silently shrink an already-small split
(DAVIS cold-pair test is 1,144 rows) -- the vendored script drops the last
batch of every loader, which this trainer deliberately does not repeat.

Training recipe
----------------
Copied from `baselines/MolTrans/train.py`, not substituted with something
more familiar -- same reasoning as `train_hyperattentiondti.py`'s recipe note:
    Adam, lr 1e-4, config batch_size 16 (`BIN_config_DBPE`'s own default)
Loss is `BCEWithLogitsLoss` rather than the vendored `Sigmoid()` + `BCELoss()`
-- the same value, numerically stabler, and the convention every other
trainer in this repo already uses (see `src/model/train.py`'s note on the
same substitution). `compute_metrics` expects a raw logit and applies the
sigmoid itself, so this also keeps MolTrans's output shape consistent with
DeepDTA's and HyperAttentionDTI's.

Binary, necessarily
--------------------
Same reason as HyperAttentionDTI: the vendored decoder ends in `Linear(32, 1)`
scored with `BCELoss`, not a regression head. Labels come from the same
`BINARY_THRESHOLD` the other two trainers use (DAVIS pKd >= 7.0, KIBA >= 12.1).

Usage
-----
    python -m src.model.train_moltrans \\
        --split-dir data/splits/davis/cold_target \\
        --dataset davis --split cold_target --seed 1

Requires `subword_nmt`, which is not in requirements.txt:
    python -m pip install subword-nmt

STATUS: verified end to end against real DAVIS rows (2026-09-12): encodes,
trains, early-stops, writes a checkpoint and results JSON in the same shape
as the other two baseline trainers. Not yet run to a full grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model.early_stopping import DEFAULT_MIN_EPOCHS, CheckpointSelector
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag
from src.model.train import compute_metrics
# Shared with the other two models -- see the note in src.model.dataset.
from src.model.dataset import BINARY_THRESHOLD

VENDORED = os.path.join("baselines", "MolTrans")


def _import_vendored():
    path = os.path.abspath(VENDORED)
    if not os.path.isdir(path):
        raise SystemExit(
            f"{path} not found. Clone it first:\n"
            f"    cd baselines && git clone <MolTrans url> MolTrans"
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    from config import BIN_config_DBPE  # noqa: E402
    from models import BIN_Interaction_Flat  # noqa: E402

    previous = os.getcwd()
    os.chdir(path)          # stream.py opens './ESPF/...' at import time
    try:
        from stream import drug2emb_encoder, protein2emb_encoder  # noqa: E402
    finally:
        os.chdir(previous)
    return BIN_config_DBPE, BIN_Interaction_Flat, drug2emb_encoder, protein2emb_encoder


def _fit_batch_size(model, batch_size: int) -> None:
    """Same bug, same fix as MolTransAdapter._fit_batch_size -- see
    src/evaluation/baseline_adapters.py for the full explanation. Applied here
    once per training/eval batch rather than once per inference call.

    `self.gpus` is `torch.cuda.device_count()`, 0 on a CPU-only machine, which
    would divide by zero; floored at 1, same as the adapter.
    """
    gpus = max(int(getattr(model, "gpus", 0) or 0), 1)
    model.gpus = gpus
    model.batch_size = int(batch_size) * gpus


class MolTransDataset(Dataset):
    """Split CSV -> MolTrans's own ESPF subword encoding.

    Encoded once up front, not per __getitem__ -- same reasoning as
    HyperAttentionDataset and DeepDTADataset: KIBA is 118k rows, and
    re-tokenising a sequence on every access makes the loader, not the GPU,
    the bottleneck.
    """

    def __init__(self, csv_path: str, threshold: float, encoders):
        drug2emb_encoder, protein2emb_encoder = encoders
        frame = pd.read_csv(csv_path)
        for column in ("Drug", "Target", "Y"):
            if column not in frame.columns:
                raise KeyError(f"{csv_path} has no '{column}' column")

        drugs, drug_masks, proteins, protein_masks = [], [], [], []
        for smiles, sequence in zip(frame["Drug"], frame["Target"]):
            d, dm = drug2emb_encoder(str(smiles))
            p, pm = protein2emb_encoder(str(sequence))
            drugs.append(d)
            drug_masks.append(dm)
            proteins.append(p)
            protein_masks.append(pm)
        self.drugs = np.stack(drugs)
        self.drug_masks = np.stack(drug_masks)
        self.proteins = np.stack(proteins)
        self.protein_masks = np.stack(protein_masks)
        self.y = (frame["Y"].to_numpy(dtype=float) >= threshold).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.drugs[index]).long(),
                torch.from_numpy(self.proteins[index]).long(),
                torch.from_numpy(self.drug_masks[index]).long(),
                torch.from_numpy(self.protein_masks[index]).long(),
                torch.tensor(self.y[index], dtype=torch.float32))


def run_epoch(model, loader, loss_fn, device, optimizer=None,
              log_every: int = 20, label: str = ""):
    training = optimizer is not None
    model.train(training)
    total, n, logits, trues = 0.0, 0, [], []
    n_batches = len(loader)

    for batch_index, (d, p, dm, pm, y) in enumerate(loader, start=1):
        d, p, dm, pm, y = (d.to(device), p.to(device), dm.to(device),
                           pm.to(device), y.to(device))
        _fit_batch_size(model, d.shape[0])
        with torch.set_grad_enabled(training):
            out = model(d, p, dm, pm).squeeze(-1)
            loss = loss_fn(out, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            # Same clipping as train.py and train_deepdta.py.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total += float(loss.item()) * len(y)
        n += len(y)
        logits.append(out.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

        if log_every and (batch_index % log_every == 0 or batch_index == n_batches):
            print(f"\r    {label} batch {batch_index}/{n_batches} "
                  f"loss {total / max(n, 1):.4f}", end="", flush=True)
    if log_every:
        print()

    return total / max(n, 1), np.concatenate(trues), np.concatenate(logits)


def main():
    parser = argparse.ArgumentParser(description="Train MolTrans on our splits")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100,
                        help="vendored default is 50; we early-stop instead")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="vendored default (BIN_config_DBPE's own "
                             "config['batch_size'])")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="vendored default")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS,
        help="no checkpoint before this epoch; see src/model/early_stopping.py")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="exit immediately if this cell's results file "
                             "already exists, so a grid can be resumed after "
                             "an interruption without redoing work")
    args = parser.parse_args()

    tag = run_tag(args.dataset, args.split, "binary", args.seed)
    out_path = results_path(args.results_dir, tag, model="moltrans")
    if args.skip_if_done and os.path.exists(out_path):
        print(f"already done, skipping -> {out_path}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    BIN_config_DBPE, BIN_Interaction_Flat, drug_encoder, protein_encoder = _import_vendored()
    encoders = (drug_encoder, protein_encoder)
    threshold = BINARY_THRESHOLD[args.dataset]

    loaders, datasets = {}, {}
    for part in ("train", "valid", "test"):
        path = os.path.join(args.split_dir, f"{part}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run build_splits first.")
        datasets[part] = MolTransDataset(path, threshold, encoders)
        # drop_last on TRAIN ONLY -- see the module docstring. A dropped final
        # batch of at most batch_size-1 rows is invisible in a training curve;
        # a dropped final batch of validation or test rows is not.
        loaders[part] = DataLoader(datasets[part], batch_size=args.batch_size,
                                   shuffle=(part == "train"),
                                   drop_last=(part == "train"))
        positive = float(datasets[part].y.mean())
        print(f"  {part:5s} {len(datasets[part]):>7,} pairs  "
              f"{positive:.1%} positive  <- {path}")

    device = args.device
    print(f"\n  device        {device}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
             if device.startswith("cuda") and torch.cuda.is_available() else ""))
    print(f"  batch size    {args.batch_size} (vendored: 16)")
    if device == "cpu":
        print("  WARNING: running on CPU. Two 2-layer transformer encoders "
              "(384-dim, 545 protein tokens) per pair; a full split will "
              "take hours.")

    config = BIN_config_DBPE()
    config["batch_size"] = args.batch_size
    model = BIN_Interaction_Flat(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # BCEWithLogitsLoss, not the vendored Sigmoid() + BCELoss() -- see the
    # module docstring's "Training recipe" note.
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt = checkpoint_path(args.checkpoint_dir, args.dataset, args.split,
                           "binary", args.seed, model="moltrans")
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)

    selector = CheckpointSelector(patience=args.patience,
                                  min_epochs=args.min_epochs,
                                  n_epochs=args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, device,
                                     optimizer, label=f"epoch {epoch} train")
        val_loss, val_true, val_score = run_epoch(model, loaders["valid"], loss_fn,
                                                  device, label=f"epoch {epoch} valid")
        metrics = compute_metrics(val_true, val_score, "binary")
        print(f"  epoch {epoch:>3} train {train_loss:.4f} val {val_loss:.4f} "
              + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        if selector.consider(epoch, val_loss):
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "args": vars(args)}, ckpt)
        elif selector.should_stop(epoch):
            print(f"  early stop at epoch {epoch} (best {selector.best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model_state"])
    _loss, test_true, test_score = run_epoch(model, loaders["test"], loss_fn,
                                             device, label="test")
    test_metrics = compute_metrics(test_true, test_score, "binary")

    with open(out_path, "w") as handle:
        json.dump({"tag": tag, "model": "moltrans", "dataset": args.dataset,
                   "split": args.split, "task": "binary", "seed": args.seed,
                   "checkpoint": ckpt, "best_epoch": selector.best_epoch,
                   "selection": selector.summary(),
                   "batch_size": args.batch_size,
                   "test_positive_rate": float(datasets["test"].y.mean()),
                   "n_train_rows": len(datasets["train"]),
                   "test_metrics": test_metrics}, handle, indent=2)

    print("\nTest metrics:", json.dumps(test_metrics, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
