"""
Checkpoint selection with a minimum-epoch floor — shared by all three trainers.

Why this exists
---------------
Every trainer saved the checkpoint at minimum validation loss, from epoch 1.
On DAVIS that produced this, measured over three seeds of DeepDTA:

    level         checkpoint taken at   val AUROC there   test AUROC
    random               epoch 15.7           0.944          0.931
    cold_target          epoch  9.3           0.885          0.894
    cold_drug            epoch  3.3           0.600          0.727
    cold_pair            epoch  2.3           0.613          0.696

DAVIS cold-pair validation is 264 rows, roughly 13 positives. Validation loss
bottoms out almost immediately and then wanders, so the saved checkpoint came
from epoch 2 while the warm cell trained to epoch 16.

That is survivable for the accuracy table. It is not survivable for the
explanation axis, because `run_ladder` and `run_audit` read attention out of
exactly these checkpoints. Cold-pair precision@k would be measured on a model
that got 2 epochs against a warm model that got 16, and the resulting
degradation would partly be "this checkpoint is undertrained" rather than
"explanations do not survive cold-start" -- a confound sitting directly on the
paper's central claim.

The floor
---------
No checkpoint is taken, and early stopping cannot fire, before `min_epochs`.
The checkpoint is then the best validation loss among epochs >= min_epochs.

Why one module rather than three copies
---------------------------------------
The audit compares models to each other, so the training recipe has to differ
between them only where their authors' recipes differ. A floor implemented
three times is a floor that can drift in two of them -- the same reasoning
that put run tags in `checkpoint_naming.py`. The three loops are not even
indexed alike (`train.py` counts from 0, the other two from 1), which is
exactly the kind of detail that makes three hand-written copies disagree.

Epochs here are always 1-indexed: `epoch=1` is the first pass over the data.
`train.py` converts at the call site.
"""
from __future__ import annotations

# 10 is chosen against the measurement above: cold_target already settles by
# epoch ~9, and random by ~16, so a floor of 10 leaves the well-behaved levels
# untouched while preventing the epoch-2 checkpoints on the sparse ones. It is
# a parameter, not a constant, and whichever value is used has to be stated in
# Methods -- it changes which checkpoint the explanation axis reads.
DEFAULT_MIN_EPOCHS = 10


class CheckpointSelector:
    """Best-val-loss selection, floored at `min_epochs`.

        selector = CheckpointSelector(patience=10, min_epochs=10, n_epochs=100)
        for epoch in range(1, n_epochs + 1):
            ...
            if selector.consider(epoch, val_loss):
                torch.save(...)
            if selector.should_stop(epoch):
                break

    `min_epochs=1` reproduces the unfloored behaviour exactly, which is what
    makes the change auditable: the old numbers are reproducible from the new
    code by passing 1.
    """

    def __init__(self, patience: int, min_epochs: int = DEFAULT_MIN_EPOCHS,
                 n_epochs: int | None = None):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_epochs < 1:
            raise ValueError(f"min_epochs must be >= 1, got {min_epochs}")

        # A floor above the budget would mean no checkpoint is ever written and
        # the cell would look untrained rather than fail. Clamping keeps a short
        # run (a smoke test, or a grid deliberately capped low) working.
        self.floor = min(min_epochs, n_epochs) if n_epochs else min_epochs
        self.requested_min_epochs = min_epochs
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = -1

    def consider(self, epoch: int, val_loss: float) -> bool:
        """Record this epoch. True when the caller should save a checkpoint."""
        if epoch < self.floor:
            return False
        if val_loss < self.best_loss:
            self.best_loss, self.best_epoch = val_loss, epoch
            return True
        return False

    def should_stop(self, epoch: int) -> bool:
        """True once `patience` epochs have passed with no improvement.

        Never fires before the floor. Without that guard the counter would run
        against `best_epoch = -1` and stop the run at epoch `patience - 1`,
        before a single checkpoint had been written.
        """
        if epoch < self.floor or self.best_epoch < 0:
            return False
        return epoch - self.best_epoch >= self.patience

    def summary(self) -> dict:
        """What Methods has to report: which epoch the audited weights are from."""
        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": (None if self.best_loss == float("inf")
                              else self.best_loss),
            "min_epochs": self.requested_min_epochs,
            "min_epochs_applied": self.floor,
        }
