"""
Train the vendored HyperAttentionDTI on our splits.

Unlike DeepDTA, nothing is reimplemented here. `baselines/HpyerAttentionDTI/`
is PyTorch and runs as published, so this file imports `AttentionDTI` and its
tokeniser directly and only replaces the data plumbing: their loader expects
their own text format, and we need our four difficulty splits.

The training recipe is theirs, not mine
---------------------------------------
Copied from `HyperAttentionDTI_main.py` rather than substituted with something
more familiar, because an audit that quietly retrained a subject under a
different optimiser is measuring a model its authors never released:

    AdamW, lr 5e-5, weight decay 1e-4 on weights and 0 on biases
    CyclicLR, base_lr -> 10x base_lr, step_size_up = train_size // batch_size
    CrossEntropyLoss with an optional class weight
    batch size 32

Two of those deserve a note. The bias/weight split is why the optimiser is
built from two parameter groups rather than `model.parameters()`. And CyclicLR
steps **per batch**, not per epoch — stepping it per epoch instead would leave
the learning rate crawling through a fraction of one cycle for the whole run,
which trains, converges to something, and looks fine.

Binary, necessarily
-------------------
HyperAttentionDTI's head is `nn.Linear(512, 2)` with CrossEntropyLoss: it is a
classifier and cannot do regression. That is why the audit's cross-model
accuracy axis has to be binary — see `results.md`. Labels come from DeepDTA's
published thresholds, already verified against the real data: DAVIS pKd >= 7.0
(8.3% positive), KIBA >= 12.1 (21.0%).

Class weighting is off by default. The vendored code hardcodes weights per
dataset (0.3/0.7, 0.2/0.8) for its own benchmarks, and those do not correspond
to our splits' prevalence. `--class-weight balanced` computes it from the
actual training split instead; whichever is used has to be stated, because it
moves AUPRC.

Usage
-----
    python -m src.model.train_hyperattentiondti \\
        --split-dir data/splits/davis/cold_target \\
        --dataset davis --split cold_target --seed 1

STATUS: never executed -- written without a working torch install. Run one cell
and check the numbers before launching the grid.
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

from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag
from src.model.train import compute_metrics

BINARY_THRESHOLD = {"davis": 7.0, "kiba": 12.1}
VENDORED = os.path.join("baselines", "HpyerAttentionDTI")


def _import_vendored():
    path = os.path.abspath(VENDORED)
    if not os.path.isdir(path):
        raise SystemExit(
            f"{path} not found. Clone it first:\n"
            f"    cd baselines && git clone <HyperAttentionDTI url> HpyerAttentionDTI"
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    from dataset import CHARISOSMISET, CHARPROTSET, label_sequence, label_smiles
    from hyperparameter import hyperparameter
    from model import AttentionDTI
    return (AttentionDTI, hyperparameter, label_smiles, label_sequence,
            CHARISOSMISET, CHARPROTSET)


class HyperAttentionDataset(Dataset):
    """Split CSV -> HyperAttentionDTI's own integer encoding.

    Encoded once up front, not per __getitem__: KIBA is 118k rows and
    re-tokenising a 1000-character sequence on every access makes the loader
    the bottleneck rather than the GPU.
    """

    def __init__(self, csv_path: str, threshold: float):
        (_A, _h, label_smiles, label_sequence,
         CHARISOSMISET, CHARPROTSET) = _import_vendored()

        frame = pd.read_csv(csv_path)
        for column in ("Drug", "Target", "Y"):
            if column not in frame.columns:
                raise KeyError(f"{csv_path} has no '{column}' column")

        self.drugs = np.stack([label_smiles(s, CHARISOSMISET, 100)
                               for s in frame["Drug"]])
        self.proteins = np.stack([label_sequence(s, CHARPROTSET, 1000)
                                  for s in frame["Target"]])
        # CrossEntropyLoss wants int64 class indices, not floats
        self.y = (frame["Y"].to_numpy(dtype=float) >= threshold).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.drugs[index]),
                torch.from_numpy(self.proteins[index]),
                torch.tensor(self.y[index]))


def attention_memory_gb(batch_size: int) -> float:
    """Activation memory for ONE of the interaction tensors, in GB.

    `AttentionDTI.forward` materialises (batch, 85, 979, 160) — drug positions
    x protein positions x channels — and holds roughly four such tensors live
    (`d_layers`, `p_layers`, their sum after ReLU, and `atten_matrix`), before
    autograd saves what it needs for the backward pass.

    At the vendored batch size of 32 that is 1.7 GB per tensor, so ~7 GB of
    activations for a single step. This is the reason the model appears to hang
    rather than the reason it is slow: on a GPU it OOMs, and on CPU it simply
    grinds. Neither prints anything.
    """
    return batch_size * 85 * 979 * 160 * 4 / 1024 ** 3


def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None,
              log_every: int = 20, label: str = ""):
    training = optimizer is not None
    model.train(training)
    total, n, logits, trues = 0.0, 0, [], []
    n_batches = len(loader)

    for batch_index, (drug, protein, y) in enumerate(loader, start=1):
        drug, protein, y = drug.to(device), protein.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            out = model(drug, protein)
            loss = loss_fn(out, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # per BATCH, not per epoch -- see the module docstring
            if scheduler is not None:
                scheduler.step()

        total += float(loss.item()) * len(y)
        n += len(y)
        logits.append(out.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

        # Per-batch progress, not per-epoch. One epoch here is ~660 batches of a
        # 7GB-activation model; printing only at epoch boundaries makes a run
        # that is merely slow indistinguishable from one that is wedged.
        if log_every and (batch_index % log_every == 0 or batch_index == n_batches):
            print(f"\r    {label} batch {batch_index}/{n_batches} "
                  f"loss {total / max(n, 1):.4f}", end="", flush=True)
    if log_every:
        print()

    logits = np.concatenate(logits)
    # compute_metrics expects ONE logit and applies a sigmoid. This head emits
    # two. The difference (positive minus negative) is the log-odds, so
    # sigmoid(difference) equals the softmax probability of the positive class
    # exactly -- passing logits[:, 1] alone would not, and would quietly shift
    # every AUROC.
    score = logits[:, 1] - logits[:, 0]
    return total / max(n, 1), np.concatenate(trues), score


def main():
    parser = argparse.ArgumentParser(description="Train HyperAttentionDTI on our splits")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="vendored default is 32")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="vendored default is 5e-5")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--class-weight", choices=["none", "balanced"],
                        default="none",
                        help="'balanced' derives it from the training split. "
                             "The vendored hardcoded weights are for their "
                             "benchmarks, not our splits.")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

    # task is always binary: the head is Linear(512, 2)
    tag = run_tag(args.dataset, args.split, "binary", args.seed)
    out_path = results_path(args.results_dir, tag).replace(
        "_results.json", "_hyperattentiondti_results.json")
    if args.skip_if_done and os.path.exists(out_path):
        print(f"already done, skipping -> {out_path}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    AttentionDTI, hyperparameter, *_ = _import_vendored()
    threshold = BINARY_THRESHOLD[args.dataset]

    loaders, datasets = {}, {}
    for part in ("train", "valid", "test"):
        path = os.path.join(args.split_dir, f"{part}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run build_splits first.")
        datasets[part] = HyperAttentionDataset(path, threshold)
        loaders[part] = DataLoader(datasets[part], batch_size=args.batch_size,
                                   shuffle=(part == "train"))
        positive = float(datasets[part].y.mean())
        print(f"  {part:5s} {len(datasets[part]):>7,} pairs  "
              f"{positive:.1%} positive  <- {path}")

    device = args.device
    per_tensor = attention_memory_gb(args.batch_size)
    n_batches = max(len(datasets["train"]) // args.batch_size, 1)
    print(f"\n  device        {device}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
             if device.startswith("cuda") and torch.cuda.is_available() else ""))
    print(f"  batch size    {args.batch_size}  ->  {n_batches} batches/epoch")
    print(f"  interaction tensor  {per_tensor:.2f} GB each, ~{per_tensor * 4:.1f} GB "
          f"of activations per step")
    if device == "cpu":
        print("  WARNING: running on CPU. This model materialises a "
              "(batch, 85, 979, 160) tensor per step;\n"
              "           an epoch will take hours. Check torch.cuda.is_available().")
    elif per_tensor * 4 > 0.7 * torch.cuda.get_device_properties(0).total_memory / 1024**3:
        print(f"  WARNING: activations may not fit. Try --batch-size "
              f"{max(args.batch_size // 4, 4)}.")

    hp = hyperparameter()
    hp.Learning_rate, hp.Batch_size = args.lr, args.batch_size
    model = AttentionDTI(hp).to(device)

    # Their parameter grouping: weight decay on weights, none on biases.
    weight_p, bias_p = [], []
    for name, parameter in model.named_parameters():
        (bias_p if "bias" in name else weight_p).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": weight_p, "weight_decay": hp.weight_decay},
         {"params": bias_p, "weight_decay": 0}], lr=hp.Learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
        cycle_momentum=False,
        step_size_up=max(len(datasets["train"]) // args.batch_size, 1))

    weight = None
    if args.class_weight == "balanced":
        positive = float(datasets["train"].y.mean())
        weight = torch.tensor([positive, 1.0 - positive],
                              dtype=torch.float32, device=device)
        print(f"  class weight (balanced): {weight.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    ckpt = checkpoint_path(args.checkpoint_dir, args.dataset, args.split,
                           "binary", args.seed).replace(
        ".pt", "_hyperattentiondti.pt")
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)

    best_loss, best_epoch = float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, device,
                                     optimizer, scheduler,
                                     label=f"epoch {epoch} train")
        val_loss, val_true, val_score = run_epoch(model, loaders["valid"],
                                                  loss_fn, device,
                                                  label=f"epoch {epoch} valid")
        metrics = compute_metrics(val_true, val_score, "binary")
        print(f"  epoch {epoch:>3} train {train_loss:.4f} val {val_loss:.4f} "
              + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "args": vars(args)}, ckpt)
        elif epoch - best_epoch >= args.patience:
            print(f"  early stop at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model_state"])
    _loss, test_true, test_score = run_epoch(model, loaders["test"], loss_fn,
                                             device, label="test")
    test_metrics = compute_metrics(test_true, test_score, "binary")

    with open(out_path, "w") as handle:
        json.dump({"tag": tag, "model": "hyperattentiondti",
                   "dataset": args.dataset, "split": args.split,
                   "task": "binary", "seed": args.seed, "checkpoint": ckpt,
                   "best_epoch": best_epoch,
                   "class_weight": args.class_weight,
                   "test_positive_rate": float(datasets["test"].y.mean()),
                   "n_train_rows": len(datasets["train"]),
                   "test_metrics": test_metrics}, handle, indent=2)

    print("\nTest metrics:", json.dumps(test_metrics, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
