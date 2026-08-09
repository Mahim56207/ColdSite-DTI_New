"""
Track B (124AD0015) — training loop.

Standard PyTorch training loop: batch, forward, loss, backward, step.
The DataLoader / dataset class lives in src/model/dataset.py and reads
Track A's split files from data/splits/. See docs/02_GUIDE_124AD0015.md Step 4-5.

Usage
-----
    python -m src.model.train --dummy --epochs 3
    python -m src.model.train --split-dir data/splits/davis/cold_target \
        --dataset davis --split cold_target --task regression --epochs 100
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model.checkpoint_naming import (
    checkpoint_path as build_checkpoint_path,
    history_path,
    results_path,
    run_tag,
)
from src.model.coldsite_dti import ColdSiteDTI
from src.model.dataset import load_split, make_loader, random_dataset


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def concordance_index(y_true, y_pred) -> float:
    """Fraction of comparable pairs ranked correctly; ties count as half.

    O(n^2), so it is fine on validation and test sets of a few thousand rows but
    should not be called on the full training set every epoch.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    comparable = (y_true[:, None] - y_true[None, :]) > 0
    total = comparable.sum()
    if total == 0:
        return float("nan")
    diff = y_pred[:, None] - y_pred[None, :]
    return float((((diff > 0) & comparable).sum()
                  + 0.5 * ((diff == 0) & comparable).sum()) / total)


# Which key of compute_metrics() below is "accuracy" for the headline figure.
# Defined here because compute_metrics is what decides the keys exist at all;
# every consumer imports this rather than hardcoding a name. Two independent
# copies of it is how the grid's status table and Track C's hand-off end up
# reporting different quantities, both labelled "accuracy".
#
# CI and AUROC are both bounded [0, 1] and higher-is-better, so they share an
# axis sensibly with precision@k. MSE would invert the reading of the figure.
DEFAULT_ACCURACY_METRIC = {"regression": "ci", "binary": "auroc"}


def accuracy_metric_for(task: str) -> str:
    """The accuracy field for a task. Raises rather than guessing.

    A wrong guess here does not crash -- it makes every cell fail verification
    with "no 'ci' in test_metrics", which reads like a training failure rather
    than a metric-name mismatch.
    """
    if task not in DEFAULT_ACCURACY_METRIC:
        raise ValueError(
            f"no accuracy metric defined for task {task!r}; "
            f"known tasks: {sorted(DEFAULT_ACCURACY_METRIC)}")
    return DEFAULT_ACCURACY_METRIC[task]


def compute_metrics(y_true, y_pred, task: str) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if task == "binary":
        probs = 1.0 / (1.0 + np.exp(-y_pred))     # pred is a logit, not a probability
        if len(np.unique(y_true)) < 2:
            # one-class validation batch makes AUROC undefined; report NaN rather
            # than crashing an overnight run
            return {"auroc": float("nan"), "auprc": float("nan"),
                    "accuracy": float(((probs >= 0.5) == (y_true >= 0.5)).mean())}
        return {"auroc": float(roc_auc_score(y_true, probs)),
                "auprc": float(average_precision_score(y_true, probs)),
                "accuracy": float(((probs >= 0.5) == (y_true >= 0.5)).mean())}

    mse = float(np.mean((y_true - y_pred) ** 2))
    pearson = float("nan") if y_true.std() == 0 or y_pred.std() == 0 \
        else float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "pearson": pearson,
            "ci": concordance_index(y_true, y_pred)}


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for drug_batch, protein_batch, label_batch in dataloader:
        drug_batch = drug_batch.to(device)
        protein_batch = protein_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        pred, _attn = model(drug_batch, protein_batch)
        loss = loss_fn(pred.squeeze(-1), label_batch.float())
        loss.backward()
        # The BiLSTM plus attention will occasionally spike the gradient norm in
        # the first few hundred steps; clipping stops a long run from silently
        # turning into NaNs overnight.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * drug_batch.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, task="regression"):
    """Returns (mean loss, metrics dict)."""
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for drug_batch, protein_batch, label_batch in dataloader:
        drug_batch = drug_batch.to(device)
        protein_batch = protein_batch.to(device)
        label_batch = label_batch.to(device)

        pred, _attn = model(drug_batch, protein_batch)
        pred = pred.squeeze(-1)
        loss = loss_fn(pred, label_batch.float())

        total_loss += loss.item() * drug_batch.size(0)
        preds.append(pred.cpu().numpy())
        targets.append(label_batch.cpu().numpy())

    return (total_loss / len(dataloader.dataset),
            compute_metrics(np.concatenate(targets), np.concatenate(preds), task))


def run_training(drug_vocab_size, protein_vocab_size, train_loader, val_loader,
                 n_epochs=30, lr=1e-3, task="regression",
                 checkpoint_path="results/coldsite_dti_best.pt", patience=15):
    """Train one model on one split.

    checkpoint_path should always name the dataset, the split and the seed --
    build it with src.model.checkpoint_naming.checkpoint_path rather than by
    hand. The most expensive mistake available in this project is reporting a
    cold-target number that was actually produced by a model trained on
    cold-drug; the second most expensive is reporting a three-seed mean
    produced by three runs that overwrote each other.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    model = ColdSiteDTI(drug_vocab_size, protein_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # BCEWithLogitsLoss, not BCELoss: the model returns a raw logit and this
    # applies the sigmoid internally, which is numerically stabler.
    loss_fn = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    best_val_loss, best_epoch, history = float("inf"), -1, []

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device, task)
        scheduler.step(val_loss)

        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(f"Epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  {metric_str}")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "task": task, "val_metrics": val_metrics}, checkpoint_path)
            print(f"  -> saved new best checkpoint to {checkpoint_path}")

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, **val_metrics})

        if epoch - best_epoch >= patience:
            print(f"No improvement for {patience} epochs, stopping early")
            break

    with open(history_path(checkpoint_path), "w") as f:
        json.dump(history, f, indent=2)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ColdSite-DTI")
    parser.add_argument("--split-dir", help="e.g. data/splits/davis/cold_target")
    parser.add_argument("--dataset", default="dummy", help="davis | kiba | antiviral")
    parser.add_argument("--split", default="dummy",
                        help="warm | cold_drug | cold_target | cold_pair")
    parser.add_argument("--task", choices=["regression", "binary"],
                        default="regression")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-protein-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42,
                        help="training seed; appears in every output filename")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--train-subsample", type=int,
                        help="keep only this many training rows. For the "
                             "cold-pair volume-matched control ONLY; write it "
                             "to a separate --results-dir so it cannot be "
                             "confused with the run it controls for.")
    parser.add_argument("--dummy", action="store_true",
                        help="run on random data, no real splits needed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dummy:
        binary = args.task == "binary"
        train_ds = random_dataset(512, binary=binary, seed=1)
        val_ds = random_dataset(128, binary=binary, seed=2)
        test_ds = random_dataset(128, binary=binary, seed=3)
        train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
        val_loader = make_loader(val_ds, args.batch_size)
        test_loader = make_loader(test_ds, args.batch_size)
        drug_vocab, protein_vocab = train_ds.drug_vocab, train_ds.protein_vocab
    else:
        if not args.split_dir:
            parser.error("--split-dir is required unless --dummy is set")
        train_loader, val_loader, test_loader, drug_vocab, protein_vocab = load_split(
            args.split_dir, args.max_protein_len, args.batch_size,
            train_subsample=args.train_subsample, subsample_seed=args.seed)
        if args.train_subsample:
            print(f"VOLUME-MATCHED CONTROL: training on "
                  f"{len(train_loader.dataset)} rows "
                  f"(--train-subsample {args.train_subsample})")

    # The seed is part of the tag. Without it the three runs the master plan
    # requires per cell all write to one path, and the loss is silent.
    tag = run_tag(args.dataset, args.split, args.task, args.seed)
    checkpoint_path = build_checkpoint_path(
        args.results_dir, args.dataset, args.split, args.task, args.seed)

    model = run_training(
        drug_vocab_size=len(drug_vocab) + 2,        # +2 for PAD and UNK
        protein_vocab_size=len(protein_vocab) + 2,
        train_loader=train_loader, val_loader=val_loader,
        n_epochs=args.epochs, lr=args.lr, task=args.task,
        checkpoint_path=checkpoint_path,
    )

    # The final epoch is usually not the best one, so reload before testing.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.MSELoss() if args.task == "regression" else nn.BCEWithLogitsLoss()
    _test_loss, test_metrics = evaluate(model.to(device), test_loader, loss_fn,
                                        device, args.task)

    metrics_path = results_path(args.results_dir, tag)
    with open(metrics_path, "w") as f:
        # dataset/split/seed are recorded as fields as well as encoded in the
        # tag, so Track C never has to re-parse a filename to know what a
        # number came from.
        json.dump({"tag": tag, "dataset": args.dataset, "split": args.split,
                   "task": args.task, "seed": args.seed,
                   "checkpoint": checkpoint_path, "best_epoch": state["epoch"],
                   "train_subsample": args.train_subsample,
                   "n_train_rows": len(train_loader.dataset),
                   "test_metrics": test_metrics}, f, indent=2)
    print("\nTest metrics:", json.dumps(test_metrics, indent=2))
    print(f"Saved -> {metrics_path}")
