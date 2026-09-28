"""The training command for one cell, exactly as the notebooks built it.

The recipes that produced the 84 existing cells live in `notebooks/kaggle_binary_grid.ipynb`
(DeepDTA, ColdSite-DTI, HyperAttentionDTI, MolTrans) and `notebooks/runner_patch.py` (DrugBAN).
They are restated here so a harness cell and a notebook cell are the same command; a test
compares the flags with the notebooks' own text, so a change to either side fails loudly.
Batch sizes are those the notebooks pick for a 16 GB card (`big = memory >= 14 GB`), which is
what a T4 is; the pre-flight refuses any other card, so no other branch is needed.

AMP is a property of the wave, not of a model: DAVIS was trained in full precision, KIBA with
`--amp` (MolTrans excepted, see `src/model/train_moltrans.py`), and a cell must be trained
under the protocol of its dataset.
"""
from __future__ import annotations

import sys

from src.cloud.config import Cell

COLDSITE_BATCH = 64
HAT_BATCH, HAT_ACCUM = 32, 1
DEEPDTA_BATCH = 256
MOLTRANS_BATCH = 16
DRUGBAN_BATCH, DRUGBAN_LR, DRUGBAN_LOG_EVERY = 64, "5e-05", 100

TRAINER_MODULE = {"deepdta": "src.model.train_deepdta", "coldsite_dti": "src.model.run_grid",
                  "hyperattentiondti": "src.model.train_hyperattentiondti",
                  "moltrans": "src.model.train_moltrans", "drugban": "src.model.train_drugban"}


def cell_dirs(results_root: str, cell: Cell) -> str:
    """Where a cell's files live: the layout of ~/ColdSite-results (`<dataset>_binary/`)."""
    return f"{results_root.rstrip('/')}/{cell.dataset}_binary"


def uses_amp(cell: Cell) -> bool:
    """KIBA is trained with mixed precision, DAVIS in full precision (CLAUDE.md §4).
    MolTrans is full precision on both: its hand-written LayerNorm underflows in float16
    (src/model/train_moltrans.py)."""
    return cell.dataset == "kiba" and cell.model != "moltrans"


def train_command(cell: Cell, results_root: str, python: str | None = None) -> list:
    out = cell_dirs(results_root, cell)
    split_dir = f"data/splits/{cell.dataset}/{cell.level}"
    amp = ["--amp"] if uses_amp(cell) else []
    py = python or sys.executable
    common = ["--checkpoint-dir", out, "--results-dir", out]
    if cell.model == "deepdta":
        return [py, "-u", "-m", TRAINER_MODULE["deepdta"], "--split-dir", split_dir,
                "--dataset", cell.dataset, "--split", cell.level, "--task", "binary",
                "--seed", str(cell.seed), "--batch-size", str(DEEPDTA_BATCH),
                "--min-epochs", "10", "--epochs", "100", "--patience", "10",
                *common, "--skip-if-done", *amp]
    if cell.model == "coldsite_dti":
        # patience is not a train.py flag; ColdSite-DTI's is fixed at 15 in run_training
        return [py, "-u", "-m", TRAINER_MODULE["coldsite_dti"], "--datasets", cell.dataset,
                "--splits", cell.level, "--seeds", str(cell.seed), "--task", "binary",
                "--epochs", "100", "--min-epochs", "10", "--batch-size", str(COLDSITE_BATCH),
                "--results-dir", out, *amp]
    if cell.model == "hyperattentiondti":
        return [py, "-u", "-m", TRAINER_MODULE["hyperattentiondti"], "--split-dir", split_dir,
                "--dataset", cell.dataset, "--split", cell.level, "--seed", str(cell.seed),
                "--batch-size", str(HAT_BATCH), "--accum-steps", str(HAT_ACCUM),
                "--patience", "15", "--min-epochs", "10", "--epochs", "100",
                *common, "--skip-if-done", *amp]
    if cell.model == "moltrans":
        return [py, "-u", "-m", TRAINER_MODULE["moltrans"], "--split-dir", split_dir,
                "--dataset", cell.dataset, "--split", cell.level, "--seed", str(cell.seed),
                "--batch-size", str(MOLTRANS_BATCH), "--min-epochs", "10", "--epochs", "100",
                "--patience", "15", *common, "--skip-if-done", *amp]
    if cell.model == "drugban":
        return [py, "-u", "-m", TRAINER_MODULE["drugban"], "--split-dir", split_dir,
                "--dataset", cell.dataset, "--split", cell.level, "--seed", str(cell.seed),
                "--batch-size", str(DRUGBAN_BATCH), "--lr", DRUGBAN_LR, "--min-epochs", "10",
                "--epochs", "100", "--patience", "15", "--log-every", str(DRUGBAN_LOG_EVERY),
                *common, "--skip-if-done", *amp]
    raise ValueError(f"no recipe for {cell.model!r}")
