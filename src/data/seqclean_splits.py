"""
Sequence-clean cold splits, and a leak-keeping control matched to them (DAVIS).

Why
---
DeepDTA's DAVIS file gives every mutant the wild-type sequence, so the cold-target and
cold-pair splits -- which hold targets out by *name* -- put proteins in the test set that
training also contains under another name (`src/data/sequence_audit.py`; 12/88 and 11/88
test targets). The evaluation already scores the cold levels on unseen targets (option A,
`src/evaluation/clean_accuracy.py`). What re-scoring cannot remove is the leak into
*training* and *validation*. These splits remove it, so a model can be retrained without
it and compared on the very same test rows.

Two derived splits per cold level, both keeping the original test file unchanged:

    <level>_seqclean    validation rows whose sequence is in test are dropped; training rows
                        whose sequence is in validation or test are dropped. No sequence is
                        shared between any two parts.
    <level>_seqmatched  the control: the SAME number of training and validation rows, with
                        the SAME number of positives, removed at random from rows that do
                        NOT carry a held-out sequence. The leak stays; volume and class
                        balance match seqclean.

Removing the leaked sequences is not a neutral cut: on DAVIS it takes 16% of cold-target's
training rows, and they are binder-rich (8.8% -> 7.0% positive). seqclean minus seqmatched
is therefore the effect of the leak itself, at equal data volume and class balance; the
original split minus seqmatched is the effect of the smaller, less positive training set.

    python -m src.data.seqclean_splits      # writes data/splits/davis/<level>_seq{clean,matched}/

Deterministic: the control's removal sample uses a fixed seed, independent of the training
seed, so every training seed sees the same split.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.model.dataset import BINARY_THRESHOLD

LEVELS = ("cold_target", "cold_pair")
PARTS = ("train", "valid", "test")
CONTROL_SEED = 0


def _read(root: str, dataset: str, level: str) -> dict:
    return {p: pd.read_csv(os.path.join(root, dataset, level, f"{p}.csv")) for p in PARTS}


def seqclean(parts: dict) -> dict:
    test_seqs = set(parts["test"].Target)
    valid = parts["valid"][~parts["valid"].Target.isin(test_seqs)]
    held = test_seqs | set(valid.Target)
    train = parts["train"][~parts["train"].Target.isin(held)]
    return {"train": train, "valid": valid, "test": parts["test"]}


def _matched_removal(frame: pd.DataFrame, keep: pd.DataFrame, candidates: pd.Series,
                     positive: pd.Series, rng) -> pd.DataFrame:
    """Drop from `frame`, at random among `candidates`, as many positives and negatives as
    were dropped to get `keep`."""
    kept_pos = int(positive.loc[keep.index].sum())
    drop_pos = int(positive.sum()) - kept_pos
    drop_neg = (len(frame) - len(keep)) - drop_pos
    pool_pos = frame.index[candidates & positive]
    pool_neg = frame.index[candidates & ~positive]
    if drop_pos > len(pool_pos) or drop_neg > len(pool_neg):
        raise ValueError("not enough non-held-out rows to match the removal")
    drop = np.concatenate([rng.choice(pool_pos, drop_pos, replace=False),
                           rng.choice(pool_neg, drop_neg, replace=False)])
    return frame.drop(index=drop)


def seqmatched(parts: dict, clean: dict, threshold: float, seed: int = CONTROL_SEED) -> dict:
    rng = np.random.default_rng(seed)
    test_seqs = set(parts["test"].Target)
    # Rows the control may remove are the ones seqclean keeps: it must keep every
    # leaked row, so only rows sharing no sequence with a held-out part are eligible.
    eligible = {"valid": test_seqs, "train": test_seqs | set(parts["valid"].Target)}
    out = {"test": parts["test"]}
    for part in ("valid", "train"):
        frame = parts[part]
        positive = frame.Y >= threshold
        candidates = ~frame.Target.isin(eligible[part])
        out[part] = _matched_removal(frame, clean[part], candidates, positive, rng)
    return out


def check(parts: dict, clean: dict, matched: dict, threshold: float) -> dict:
    """The properties the design promises; raises if any fails."""
    seqs = {p: set(clean[p].Target) for p in PARTS}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = seqs[a] & seqs[b]
        if shared:
            raise AssertionError(f"seqclean: {len(shared)} sequences shared by {a} and {b}")
    for name, split in (("seqclean", clean), ("seqmatched", matched)):
        if not split["test"].equals(parts["test"]):
            raise AssertionError(f"{name}: test differs from the original")
        for p in ("train", "valid"):
            if not set(split[p].index) <= set(parts[p].index):
                raise AssertionError(f"{name}: {p} has rows the original does not")
    for p in ("train", "valid"):
        if len(matched[p]) != len(clean[p]):
            raise AssertionError(f"seqmatched {p}: {len(matched[p])} rows vs {len(clean[p])}")
        mp, cp = int((matched[p].Y >= threshold).sum()), int((clean[p].Y >= threshold).sum())
        if mp != cp:
            raise AssertionError(f"seqmatched {p}: {mp} positives vs {cp}")
    leak = len(set(matched["train"].Target) & set(parts["test"].Target))
    return {"train_rows": (len(parts["train"]), len(clean["train"])),
            "valid_rows": (len(parts["valid"]), len(clean["valid"])),
            "train_positives": int((clean["train"].Y >= threshold).sum()),
            "valid_positives": int((clean["valid"].Y >= threshold).sum()),
            "control_keeps_leaked_sequences": leak}


def build(root: str = "data/splits", dataset: str = "davis") -> dict:
    threshold = BINARY_THRESHOLD[dataset]
    report = {}
    for level in LEVELS:
        parts = _read(root, dataset, level)
        clean = seqclean(parts)
        matched = seqmatched(parts, clean, threshold)
        report[level] = check(parts, clean, matched, threshold)
        for suffix, split in (("seqclean", clean), ("seqmatched", matched)):
            out = os.path.join(root, dataset, f"{level}_{suffix}")
            os.makedirs(out, exist_ok=True)
            for p in PARTS:
                split[p].to_csv(os.path.join(out, f"{p}.csv"), index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default="data/splits")
    parser.add_argument("--dataset", default="davis", choices=("davis",))
    args = parser.parse_args()
    for level, r in build(args.root, args.dataset).items():
        print(f"{level}: train {r['train_rows'][0]} -> {r['train_rows'][1]}, valid "
              f"{r['valid_rows'][0]} -> {r['valid_rows'][1]} (positives: train "
              f"{r['train_positives']}, valid {r['valid_positives']}); test unchanged; "
              f"control keeps {r['control_keeps_leaked_sequences']} leaked sequences -- all checks pass")


if __name__ == "__main__":
    main()
