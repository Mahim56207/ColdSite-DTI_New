"""
Epoch-level resume, shared by all four trainers.

Why it exists
-------------
A Kaggle commit is stopped at 11 hours, and on KIBA a HyperAttentionDTI or
MolTrans cell can need more than that (`results/speed_test_kiba_t4.md`). Before
this module an interrupted cell restarted from epoch 1 on the next commit, so a
cell longer than one commit could never finish at all.

What it does
------------
At the end of every epoch the trainer writes `<checkpoint>_resume.pt`: model,
optimiser, scheduler and GradScaler state, the checkpoint selector's state,
the epoch reached, the history so far, and every random-number generator. On
the next start, if that file exists and the cell has no results yet, training
continues at the next epoch. At most the epoch in progress is lost. The file is
written atomically (temporary file, then rename), so a process killed mid-save
leaves the previous epoch's file intact rather than a corrupt one. It is
deleted once the cell's results are written.

The best-so-far checkpoint is written through the same path (`end_epoch`'s
`best_checkpoint`), atomically and after the resume file, so the notebook's
11-hour kill can land at any instant and leave a checkpoint that agrees with
the selector -- see `end_epoch`.

Continuing is exact on a CPU: restoring the RNG state reproduces the
uninterrupted run bit for bit, which the tests assert. On a GPU the result is
as reproducible as an uninterrupted run already is -- cuDNN is not
deterministic either way.

A resume file records the settings it was written under. Continuing with
different ones (batch size, learning rate, --amp, ...) would train one cell
under two protocols, so the trainer refuses and says which setting differs.

Naming: `..._resume.pt` is matched by the notebooks' restore step (which copies
every `.pt`), so it is carried from one commit to the next, and it is ignored by
`checkpoint_naming.discover_checkpoints` (its stem does not parse as a run tag),
so nothing downstream mistakes it for a trained cell.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

RESUME_SUFFIX = "_resume.pt"


def resume_path(checkpoint_path: str) -> str:
    if not checkpoint_path.endswith(".pt"):
        raise ValueError(f"expected a .pt checkpoint path, got {checkpoint_path!r}")
    return checkpoint_path[: -len(".pt")] + RESUME_SUFFIX


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)


def save(path: str, state: dict) -> None:
    """Atomic: a kill during the write leaves the previous file, not half of this one."""
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path: str, device) -> dict | None:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def clear(path: str) -> None:
    for candidate in (path, f"{path}.tmp"):
        if os.path.exists(candidate):
            os.remove(candidate)


class Resumable:
    """The bookkeeping around one trainer's epoch loop.

        run = Resumable(ckpt, device, vars(args), RESUME_KEYS, args.stop_after_epoch)
        start = run.begin(model, optimizer, selector, scheduler=..., scaler=...)
        for epoch in (range(start, n_epochs + 1) if start else ()):
            ...train, validate...
            best = {"model_state": ..., "epoch": ...} if selector.consider(...) else None
            stopping = best is None and selector.should_stop(epoch)
            run.end_epoch(epoch, finished=stopping or epoch == n_epochs,
                          best_checkpoint=best)      # writes the checkpoint too
            if stopping: break
            if run.interrupt_now(epoch): return          # testing only
        ...test, write results...
        run.clear()

    `begin` returns the epoch to start from (1 for a fresh cell), or None when
    the saved state says training already finished and only the test pass is
    left -- a process killed between early stopping and writing its results.
    `extra` carries anything else a trainer needs back (ColdSite-DTI's history).
    """

    def __init__(self, checkpoint: str, device, args: dict, keys,
                 stop_after_epoch: int | None = None):
        self.checkpoint = checkpoint
        self.path = resume_path(checkpoint)
        self.device = device
        self.args = dict(args)
        self.keys = tuple(keys)
        self.stop_after_epoch = stop_after_epoch
        self.resumed_from = None
        self.extra: dict = {}
        self._parts: dict = {}

    def begin(self, model, optimizer, selector, scheduler=None, scaler=None):
        self._parts = {"model": model, "optimizer": optimizer, "selector": selector,
                       "scheduler": scheduler, "scaler": scaler}
        state = load(self.path, self.device)
        if state is None:
            return 1
        check_compatible(state["args"], self.args, self.keys)
        self._repair_checkpoint(state)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        selector.load_state_dict(state["selector_state"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        restore_rng(state["rng"])
        self.extra = state.get("extra", {})
        self.resumed_from = state["epoch"]
        print(f"  RESUMING after epoch {state['epoch']} "
              f"(best so far: epoch {selector.best_epoch}) <- {self.path}", flush=True)
        return None if state["finished"] else state["epoch"] + 1

    def end_epoch(self, epoch: int, finished: bool, best_checkpoint: dict | None = None,
                  **extra) -> None:
        """Save this epoch's state; then, if the epoch was the best so far, the
        checkpoint `best_checkpoint` (the dict the trainer would torch.save).

        The order is what makes a kill at any moment recoverable. The checkpoint
        is written only after a resume file that already holds its weights, so
        a kill between the two leaves a checkpoint one best-epoch stale that
        `begin` rewrites from the resume file. Written the other way round, a
        kill between them would leave the checkpoint from an epoch the selector
        never recorded, with the weights it did record already overwritten.
        """
        self.extra.update(extra)
        parts = self._parts
        meta = ({k: v for k, v in best_checkpoint.items() if k != "model_state"}
                if best_checkpoint is not None else None)
        save(self.path, {
            "epoch": epoch, "finished": bool(finished), "args": self.args,
            "model_state": parts["model"].state_dict(),
            "optimizer_state": parts["optimizer"].state_dict(),
            "selector_state": parts["selector"].state_dict(),
            "scheduler_state": (parts["scheduler"].state_dict()
                                if parts["scheduler"] is not None else None),
            "scaler_state": (parts["scaler"].state_dict()
                             if parts["scaler"] is not None else None),
            "best_checkpoint_meta": meta,
            "rng": capture_rng(), "extra": self.extra})
        if best_checkpoint is not None:
            save(self.checkpoint, best_checkpoint)

    def _repair_checkpoint(self, state: dict) -> None:
        """A kill between the two saves of `end_epoch`: rewrite the checkpoint."""
        meta = state.get("best_checkpoint_meta")
        if meta is None:
            return
        on_disk = load(self.checkpoint, "cpu")
        if on_disk is not None and on_disk.get("epoch") == meta.get("epoch"):
            return
        print(f"  checkpoint is older than the resume file (killed between the two "
              f"saves); rewriting it from epoch {meta.get('epoch')} -> {self.checkpoint}",
              flush=True)
        save(self.checkpoint, {**meta, "model_state": state["model_state"]})

    def interrupt_now(self, epoch: int) -> bool:
        """--stop-after-epoch: behave as if the process were killed here."""
        if self.stop_after_epoch and epoch >= self.stop_after_epoch:
            print(f"  stopping after epoch {epoch} (--stop-after-epoch); "
                  f"state saved -> {self.path}", flush=True)
            return True
        return False

    def clear(self) -> None:
        clear(self.path)


def check_compatible(saved: dict, current: dict, keys) -> None:
    """Refuse to continue a cell under different settings from the ones it began with."""
    differing = [(k, saved.get(k), current.get(k)) for k in keys
                 if saved.get(k) != current.get(k)]
    if differing:
        detail = "; ".join(f"{k}: was {a!r}, now {b!r}" for k, a, b in differing)
        raise SystemExit(
            f"Refusing to resume: this cell was started with different settings ({detail}). "
            f"Continuing would train one cell under two protocols. Restore the original "
            f"settings, or delete the resume file to start the cell again.")
