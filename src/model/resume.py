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

import hashlib
import json
import os
import random
import signal
import time

import numpy as np
import torch

RESUME_SUFFIX = "_resume.pt"
START_SUFFIX = "_start.json"

# Cloud-harness hooks (src/cloud/runner.py). All of them are inert unless the launcher sets
# these variables, so a trainer run by hand, or by the older notebooks, behaves exactly as it
# did before they existed. None of them touches a weight, an optimiser state or a random
# number; they read state, write side files, and decide when to stop *between* epochs.
ENV_HARNESS = "COLDSITE_HARNESS"          # "1": record the start, honour SIGTERM/SIGINT, write status
ENV_STOP_AT = "COLDSITE_STOP_AT"          # unix seconds: do not start an epoch that would end after this
ENV_STATUS = "COLDSITE_STATUS_JSON"       # path of the per-cell status file, rewritten every epoch
ENV_CONTEXT = "COLDSITE_CELL_CONTEXT"     # JSON: {"cell": ..., "account": ..., "gpu": ...}


def _context() -> dict:
    raw = os.environ.get(ENV_CONTEXT)
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {"context_unreadable": raw}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resume_path(checkpoint_path: str) -> str:
    if not checkpoint_path.endswith(".pt"):
        raise ValueError(f"expected a .pt checkpoint path, got {checkpoint_path!r}")
    return checkpoint_path[: -len(".pt")] + RESUME_SUFFIX


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    # PyTorch takes generator states only as CPU ByteTensors, CUDA's included. A state
    # loaded with map_location="cuda" fails here ("RNG state must be a
    # torch.ByteTensor") -- first seen resuming on a Colab T4, 2026-09-13.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])


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
        self.clock = time.time                 # injectable, for tests
        self.harness = os.environ.get(ENV_HARNESS) == "1"
        stop_at = os.environ.get(ENV_STOP_AT)
        self.stop_at = float(stop_at) if stop_at else None
        self.signalled: str | None = None      # set by the first SIGTERM / SIGINT
        self.epoch_seconds: list = []          # durations measured in THIS process
        self._epoch_started: float | None = None
        self._session_started = self.clock()

    def begin(self, model, optimizer, selector, scheduler=None, scaler=None):
        self._parts = {"model": model, "optimizer": optimizer, "selector": selector,
                       "scheduler": scheduler, "scaler": scaler}
        # Loaded onto the CPU whatever the training device: load_state_dict copies the
        # weights onto the model's device and moves the optimiser state to its
        # parameters', while the RNG states must stay CPU tensors (restore_rng).
        self._epoch_started = self.clock()
        if self.harness:
            self._install_signal_handlers()
        state = load(self.path, "cpu")
        if state is None:
            if self.harness:
                self._record_start(model)
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
        now = self.clock()
        if self._epoch_started is not None:
            self.epoch_seconds.append(now - self._epoch_started)
        self._epoch_started = now
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
        if self.harness and os.environ.get(ENV_STATUS):
            self._write_status(epoch, finished)

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
        """Stop here? True after a SIGTERM/SIGINT, past --stop-after-epoch, or when the
        next epoch is projected to end after the deadline.

        Called after `end_epoch`, so the state on disk is this epoch's and nothing is lost.
        The projection is `now + the longest epoch measured in this process`: pre-declared
        and deliberately pessimistic, so a cell is never left mid-epoch when the session
        ends. (Before the first epoch of a process there is nothing to project from; the
        launcher's own margin covers that one.)
        """
        if self.signalled:
            print(f"  stopping after epoch {epoch}: {self.signalled} received; "
                  f"state saved -> {self.path}", flush=True)
            return True
        if self.stop_after_epoch and epoch >= self.stop_after_epoch:
            print(f"  stopping after epoch {epoch} (--stop-after-epoch); "
                  f"state saved -> {self.path}", flush=True)
            return True
        if self.stop_at is not None and self.epoch_seconds:
            projected = self.clock() + max(self.epoch_seconds)
            if projected > self.stop_at:
                print(f"  stopping after epoch {epoch}: the next epoch (~"
                      f"{max(self.epoch_seconds) / 60:.1f} min) would end "
                      f"{(projected - self.stop_at) / 60:.1f} min past the session deadline; "
                      f"state saved -> {self.path}", flush=True)
                return True
        return False

    # -- harness hooks ---------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """First SIGTERM/SIGINT: finish the epoch in progress, save, exit cleanly. A second
        one is not caught, so an operator can still force the process down. Mid-epoch state
        is not resumable (the data order lives in the loader), so the epoch-end save is the
        checkpoint; a harder kill still leaves the previous epoch's file, written atomically."""
        def handler(signum, _frame):
            self.signalled = signal.Signals(signum).name
            print(f"\n  {self.signalled}: finishing this epoch, then saving and stopping",
                  flush=True)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_DFL)
        try:
            signal.signal(signal.SIGTERM, handler)
            signal.signal(signal.SIGINT, handler)
        except ValueError:            # not the main thread: nothing to install
            pass

    def _record_start(self, model) -> None:
        """What this run started from: seed, the RNG fingerprints and a hash of the initial
        weights, written once, when a cell begins. Two seeds that share an initial-weight
        hash are one seed under two names (src/model/integrity.py)."""
        from src.model.integrity import initial_weight_hash, rng_state
        context = _context()
        record = {"seed": self.args.get("seed", context.get("seed")), "checkpoint": os.path.basename(self.checkpoint),
                  "initial_weight_hash": initial_weight_hash(model),
                  "rng_state_sha256": rng_state(), "torch": torch.__version__,
                  "context": context}
        path = self.checkpoint[: -len(".pt")] + START_SUFFIX
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(record, handle, indent=1)
        os.replace(tmp, path)

    def _write_status(self, epoch: int, finished: bool) -> None:
        """One JSON per cell, rewritten each epoch: where the cell is and what is on disk."""
        now = self.clock()
        mean = sum(self.epoch_seconds) / len(self.epoch_seconds) if self.epoch_seconds else None
        total = self.args.get("epochs") or self.args.get("n_epochs")
        status = {
            **_context(), "epoch": epoch, "finished_training": bool(finished),
            "elapsed_s": round(now - self._session_started, 1),
            "last_epoch_s": round(self.epoch_seconds[-1], 1) if self.epoch_seconds else None,
            "mean_epoch_s": round(mean, 1) if mean else None,
            "projected_finish_unix_upper_bound": (
                round(now + mean * max(int(total) - epoch, 0), 1) if mean and total else None),
            "note": "upper bound: assumes no early stopping",
            "checkpoint": {"path": self.checkpoint,
                           "sha256": _sha256(self.checkpoint)
                           if os.path.exists(self.checkpoint) else None},
            "resume_file": {"path": self.path, "sha256": _sha256(self.path)},
        }
        path = os.environ[ENV_STATUS]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as handle:
            json.dump(status, handle, indent=1)
        os.replace(tmp, path)

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
