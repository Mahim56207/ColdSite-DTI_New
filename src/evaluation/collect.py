"""
The real collector — trained checkpoints in, `(weights, sites, target_ids)` out.

`run_audit.build_grid` takes a `collect_fn` and calls it once per
(model, dataset, level, seed) cell. Until now the only implementation was
`dummy_collect_fn`, so real mode raised `SystemExit` and there was no path from
a checkpoint to the audit table however much training had finished. This module
is that path.

Every model is tokenised with its own vendored tables
-----------------------------------------------------
The one thing this file must not do is standardise the models. Each adapter
already knows how its authors tokenise; the collector's job is to hand each one
a row from the test split in the form that adapter expects, and to keep the
protein's identity attached to the explanation that comes back:

    ColdSite-DTI       the split's own vocabulary, built from train.csv only
    HyperAttentionDTI  CHARISOSMISET / CHARPROTSET, its own label_* functions
    MolTrans           ESPF subword units, plus the token list explain() needs
    uniform_control    any encoding -- the output is flat by construction

DeepDTA is not collectable and that is correct: `provides_attention = False`.
Asking it for an explanation raises rather than returning a saliency map its
authors never published.

One pair per target, by default
-------------------------------
precision@k is a per-protein quantity. A test split holds many pairs per
protein, so collecting every row would put the same protein into the average
dozens of times, once per drug it was measured against. That inflates
`n_evaluated` from "proteins" to "pairs" and makes the permutation test look
far better powered than it is -- 400 correlated samples are not 400
observations.

`pairs_per_target=1` therefore takes one representative pair per protein.
Raise it to compare explanations across drugs for the same protein, which is a
different and also interesting question; just do not report the result as if n
were the number of proteins. **This is a measurement decision that belongs to
Track C** -- it is a parameter here rather than a constant so that the choice
is made explicitly and stated in Methods.

Usage
-----
    from src.evaluation.collect import make_collect_fn
    from src.evaluation.run_audit import build_grid

    collect = make_collect_fn(dataset="davis", task="binary",
                              ground_truth="data/davis_ground_truth_sites.json")
    results = build_grid(collect, ["coldsite_dti", "uniform_control"],
                         ["davis"], [1, 2, 3])
"""
from __future__ import annotations

import os

import numpy as np

from src.data.ground_truth import load_site_sets

# Imported for its registration side-effect: @register runs at import time, so
# without this the registry holds only coldsite_dti and uniform_control and
# every baseline reads as "unknown model" -- with an error telling you to write
# an adapter that already exists and already passes validate_adapter.
from src.evaluation import baseline_adapters  # noqa: F401
from src.evaluation.model_registry import _REGISTRY, get_model
from src.model.checkpoint_naming import MODEL_SUFFIX, checkpoint_path

# Models that need no checkpoint: the control is flat by construction, so
# "untrained" is not a defect and a missing file must not skip the cell.
CHECKPOINT_FREE = {"uniform_control"}

SMILES_COLUMN = "Drug"
SEQUENCE_COLUMN = "Target"
TARGET_ID_COLUMN = "Target_ID"


class MissingCell(Exception):
    """This cell cannot be collected yet. Carries the reason for the report."""


def _read_test_rows(split_dir: str, pairs_per_target: int,
                    rows_csv: str | None = None):
    """One row per (target, drug) pair from the test split, capped per target.

    Returns a list of (target_id, smiles, sequence). Rows keep the file's own
    order so a cap of 1 is deterministic rather than whichever pair pandas
    happened to group first.

    `rows_csv` evaluates a model on pairs from somewhere other than its own
    test split -- the non-kinase control panel above all. The vocabulary still
    comes from `split_dir`, because it is a property of the *trained model*:
    rebuilding it from the panel would give the model an embedding table it
    was never trained with.
    """
    import pandas as pd

    path = rows_csv or os.path.join(split_dir, "test.csv")
    if not os.path.exists(path):
        raise MissingCell(f"no test split at {path}")

    frame = pd.read_csv(path)
    for column in (TARGET_ID_COLUMN, SMILES_COLUMN, SEQUENCE_COLUMN):
        if column not in frame.columns:
            raise MissingCell(
                f"{path} has no {column!r} column (found {list(frame.columns)})")

    seen: dict = {}
    rows = []
    for target_id, smiles, sequence in zip(frame[TARGET_ID_COLUMN],
                                           frame[SMILES_COLUMN],
                                           frame[SEQUENCE_COLUMN]):
        count = seen.get(target_id, 0)
        if pairs_per_target and count >= pairs_per_target:
            continue
        seen[target_id] = count + 1
        rows.append((str(target_id), str(smiles), str(sequence).upper()))
    return rows


def _build_adapter(model_name: str, checkpoint: str | None, split_dir: str,
                   device: str, max_protein_len: int):
    """Instantiate one adapter, loading its checkpoint if it needs one.

    ColdSite-DTI is the only model whose *architecture* depends on the split:
    its drug vocabulary is built from that split's train.csv, so the embedding
    size differs per cell. Rebuilt here the same way the trainer built it --
    train rows only, never valid or test, or the cold splits leak the drugs
    they exist to hold out.
    """
    if model_name == "coldsite_dti":
        import pandas as pd

        from src.model.dataset import SMILES_COLUMNS, find_column
        from src.model.drug_encoder import build_smiles_vocab
        from src.model.protein_encoder import build_protein_vocab

        train_path = os.path.join(split_dir, "train.csv")
        if not os.path.exists(train_path):
            raise MissingCell(f"no train split at {train_path} (needed for the vocab)")
        train_df = pd.read_csv(train_path)
        drug_vocab = build_smiles_vocab(
            train_df[find_column(train_df, SMILES_COLUMNS, "SMILES")]
            .astype(str).tolist())
        protein_vocab = build_protein_vocab()
        # +2 for PAD and UNK, matching run_ladder and the trainer
        return get_model(model_name, checkpoint_path=checkpoint,
                         drug_vocab_size=len(drug_vocab) + 2,
                         protein_vocab_size=len(protein_vocab) + 2,
                         device=device), (drug_vocab, protein_vocab)

    if model_name == "uniform_control":
        from src.model.drug_encoder import build_smiles_vocab
        from src.model.protein_encoder import build_protein_vocab
        # the control needs an encoding only to measure the protein's length
        return get_model(model_name), (build_smiles_vocab(["C"]),
                                       build_protein_vocab())

    return get_model(model_name, checkpoint_path=checkpoint, device=device), None


def _explain_row(model_name: str, adapter, vocabs, smiles: str, sequence: str,
                 max_protein_len: int) -> np.ndarray:
    """One explanation, tokenised the way that model's own authors tokenise."""
    import torch

    if model_name in ("coldsite_dti", "uniform_control"):
        from src.model.drug_encoder import encode_smiles
        from src.model.protein_encoder import encode_protein

        drug_vocab, protein_vocab = vocabs
        drug = torch.tensor(encode_smiles(smiles, drug_vocab, 100), dtype=torch.long)
        protein = torch.tensor(
            encode_protein(sequence, protein_vocab, max_protein_len), dtype=torch.long)
        return np.asarray(adapter.explain(drug, protein), dtype=float)

    if model_name == "hyperattentiondti":
        drug, protein = type(adapter).encode(smiles, sequence)
        return np.asarray(adapter.explain(drug, protein), dtype=float)

    if model_name == "moltrans":
        drug, drug_mask, protein, protein_mask, tokens = type(adapter).encode(
            smiles, sequence)
        return np.asarray(
            adapter.explain(drug, protein, protein_tokens=tokens,
                            drug_mask=drug_mask, protein_mask=protein_mask),
            dtype=float)

    raise MissingCell(
        f"no tokenisation registered for {model_name!r}. Add one in "
        f"src/evaluation/collect.py::_explain_row — guessing an encoding "
        f"produces an array of the wrong length, which misaligns every "
        f"ground-truth index and yields a plausible wrong number.")


def collect_cell(model_name: str, dataset: str, level: str, seed: int, *,
                 site_sets: dict, task: str = "binary",
                 split_root: str = "data/splits",
                 checkpoint_dir: str = "results",
                 max_protein_len: int = 1000,
                 pairs_per_target: int = 1,
                 max_proteins: int | None = None,
                 rows_csv: str | None = None,
                 device: str = "cpu",
                 verbose: bool = True):
    """One grid cell. Returns (weights, sites, target_ids), or raises MissingCell.

    A protein with no usable ground truth is skipped rather than scored against
    an empty site set, which would count as a zero and drag every mean down.
    """
    adapter_cls = _REGISTRY.get(model_name)
    if adapter_cls is None:
        raise MissingCell(
            f"unknown model {model_name!r}. Registered: {sorted(_REGISTRY)}")

    if not getattr(adapter_cls, "provides_attention", True):
        raise MissingCell(
            f"{model_name} has provides_attention = False — it anchors the "
            f"accuracy axis and has no explanation to collect")

    split_dir = os.path.join(split_root, dataset, level)

    checkpoint = None
    if model_name not in CHECKPOINT_FREE:
        if model_name not in MODEL_SUFFIX:
            raise MissingCell(
                f"{model_name} has no checkpoint suffix registered; add it to "
                f"MODEL_SUFFIX in src/model/checkpoint_naming.py")
        checkpoint = checkpoint_path(checkpoint_dir, dataset, level, task, seed,
                                     model=model_name)
        if not os.path.exists(checkpoint):
            raise MissingCell(f"no checkpoint at {checkpoint}")

    rows = _read_test_rows(split_dir, pairs_per_target, rows_csv)
    adapter, vocabs = _build_adapter(model_name, checkpoint, split_dir, device,
                                     max_protein_len)

    weights, sites, used_ids = [], [], []
    skipped_no_sites = 0
    for target_id, smiles, sequence in rows:
        site_set = site_sets.get(target_id)
        if site_set is None or not site_set.usable:
            skipped_no_sites += 1
            continue
        weights.append(_explain_row(model_name, adapter, vocabs, smiles,
                                    sequence, max_protein_len))
        sites.append(site_set.positions)
        used_ids.append(target_id)
        if max_proteins and len(weights) >= max_proteins:
            break

    if not weights:
        raise MissingCell(
            f"{len(rows)} test rows, none with usable ground truth "
            f"(skipped {skipped_no_sites}) — check that the ground-truth file "
            f"matches this dataset's Target_ID spelling")

    if verbose:
        print(f"  {model_name}/{dataset}/{level}/seed{seed}: "
              f"{len(weights)} proteins ({skipped_no_sites} without usable sites)")
    return weights, sites, used_ids


def make_collect_fn(dataset: str, ground_truth: str, task: str = "binary",
                    split_root: str = "data/splits",
                    checkpoint_dir: str = "results",
                    max_protein_len: int = 1000,
                    pairs_per_target: int = 1,
                    max_proteins: int | None = None,
                    rows_csv: str | None = None,
                    device: str = "cpu",
                    skipped: list | None = None):
    """A `collect_fn` for `run_audit.build_grid`, over real checkpoints.

    Returns None for a cell that cannot be collected yet, which is what
    `build_grid` records as missing. The reason is appended to `skipped` so the
    report can say *why* a cell is absent — "no checkpoint" and "no usable
    ground truth" call for very different responses, and a bare count of
    missing cells cannot tell them apart.
    """
    site_sets = load_site_sets(ground_truth, max_len=max_protein_len)

    def collect(model_name, dataset_name, level, seed):
        try:
            return collect_cell(
                model_name, dataset_name, level, seed,
                site_sets=site_sets, task=task, split_root=split_root,
                checkpoint_dir=checkpoint_dir, max_protein_len=max_protein_len,
                pairs_per_target=pairs_per_target, max_proteins=max_proteins,
                rows_csv=rows_csv, device=device)
        except MissingCell as reason:
            if skipped is not None:
                skipped.append(
                    f"{model_name}/{dataset_name}/{level}/seed{seed}: {reason}")
            return None

    return collect
