"""
Does `--seed 2` actually train a different model from `--seed 1`?

On 2026-09-13 it did not. MolTrans's vendored `models.py` runs `torch.manual_seed(1)`
and `np.random.seed(1)` at import time, and the trainer seeded *before* that import, so
the DAVIS grid's three MolTrans seeds all trained as seed 1 -- identical AUROCs at every
level. The fix was one line moved; nothing in the code could have told you it was wrong.

This module gives that a fingerprint:

    rng_fingerprint()          a hash of the python / numpy / torch RNG states
    initial_weight_hash(model) a hash of a freshly constructed model's parameters
    rng_delta(before, after)   which of the three generators a piece of code moved

`initial_weight_hash` is what a seed is *for*: two seeds that produce the same initial
weights are one seed wearing two names, whatever the filename says.

Nothing here is used by the training loop. It is the instrument the guard tests measure
with, and it is available to any future run that wants to record what it started from --
the 84 cells trained before it existed carry no such record, and none can be recovered
(a finished checkpoint holds only `model_state, epoch, task, val_metrics`).
"""
from __future__ import annotations

import contextlib
import hashlib
import random

import numpy as np
import torch


def _digest(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


def rng_state() -> dict:
    """The three generators' states, each as a hash. Comparable across processes."""
    torch_state = torch.get_rng_state().numpy().tobytes()
    numpy_state = np.random.get_state()
    numpy_bytes = b"".join([
        str(numpy_state[0]).encode(), np.asarray(numpy_state[1]).tobytes(),
        str(numpy_state[2:]).encode()])
    return {"python": _digest(repr(random.getstate()).encode()),
            "numpy": _digest(numpy_bytes),
            "torch": _digest(torch_state)}


def rng_fingerprint() -> str:
    """One hash over all three generators."""
    state = rng_state()
    return _digest(*(state[name].encode() for name in ("python", "numpy", "torch")))


def rng_delta(before: dict, after: dict) -> list:
    """Which generators moved between two `rng_state()` calls."""
    return sorted(name for name in before if before[name] != after[name])


def initial_weight_hash(model) -> str:
    """A hash over a model's parameters, in a fixed order.

    Buffers are included: a module whose randomness lives in a registered buffer rather
    than a Parameter would otherwise look seed-independent.
    """
    parts = []
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().to("cpu")
        if array.dtype in (torch.bfloat16, torch.float16):
            array = array.float()
        parts.append(name.encode())
        parts.append(np.ascontiguousarray(array.numpy()).tobytes())
    return _digest(*parts)


@contextlib.contextmanager
def preserve_rng():
    """Leave python / numpy / torch RNG state as it was found.

    Wrap the import of a vendored module that seeds at import time (MolTrans's
    `models.py` runs `torch.manual_seed(1)` and `np.random.seed(1)`), so that a `--seed`
    set before the import is still the seed in force after it.
    """
    py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
