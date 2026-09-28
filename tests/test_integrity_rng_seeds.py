"""T02 (a), (b): a seed must mean something, and importing vendored code must not change it.

2026-09-13: MolTrans's vendored models.py ran torch.manual_seed(1) at import, after the
trainer had seeded, so the DAVIS grid's three seeds trained as one. These pin both halves:
importing leaves every generator where it was, and distinct seeds give distinct starting
weights.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.model.integrity import (initial_weight_hash, preserve_rng, rng_delta,
                                 rng_fingerprint, rng_state)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each snippet imports one vendored module the way the trainers do. Adapters are not
# here: constructing one builds a randomly initialised model, which legitimately consumes
# torch RNG. An adapter that reseeds during construction is caught instead by
# test_distinct_seeds_give_distinct_initial_weights_for_each_model.
IMPORTS = {
    "moltrans_trainer": "from src.model.train_moltrans import _import_vendored as f; f()",
    "hyperattentiondti_trainer":
        "from src.model.train_hyperattentiondti import _import_vendored as f; f()",
}


def _vendored_present(name):
    return os.path.isdir(os.path.join(ROOT, "baselines", name))


def _run(order):
    """Import the named modules in this order, in a fresh process; report what moved."""
    code = ("from src.model.integrity import rng_state, rng_delta\n"
            "moved = []\n")
    for name in order:
        code += (f"b = rng_state()\n{IMPORTS[name]}\n"
                 f"moved.append(({name!r}, rng_delta(b, rng_state())))\n")
    code += "print(moved)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
    assert out.returncode == 0, out.stderr[-1500:]
    return eval(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("order", [
    ["moltrans_trainer", "hyperattentiondti_trainer"],
    ["hyperattentiondti_trainer", "moltrans_trainer"],
])
def test_importing_vendored_code_leaves_every_generator_alone(order):
    if not (_vendored_present("MolTrans") and _vendored_present("HpyerAttentionDTI")):
        pytest.skip("vendored baselines not present")
    for name, moved in _run(order):
        assert moved == [], f"importing {name} moved the {moved} RNG"


def test_a_seed_set_before_the_moltrans_import_survives_it():
    """The bug itself: seed, import, and the seed must still be in force."""
    if not _vendored_present("MolTrans"):
        pytest.skip("vendored MolTrans not present")
    code = ("import torch, numpy as np\n"
            "torch.manual_seed(7); np.random.seed(7)\n"
            "a = (torch.rand(3).tolist(), np.random.rand(3).tolist())\n"
            "torch.manual_seed(7); np.random.seed(7)\n"
            "from src.model.train_moltrans import _import_vendored; _import_vendored()\n"
            "b = (torch.rand(3).tolist(), np.random.rand(3).tolist())\n"
            "print(a == b)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip().endswith("True")


def test_preserve_rng_restores_all_three_generators():
    import random
    before = rng_state()
    with preserve_rng():
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        assert rng_delta(before, rng_state()) == ["numpy", "python", "torch"]
    assert rng_delta(before, rng_state()) == []


def test_rng_fingerprint_moves_when_any_generator_moves():
    base = rng_fingerprint()
    torch.rand(1)
    assert rng_fingerprint() != base


# --------------------------------------------------------------- initial weights

def _fresh_weights(build, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return initial_weight_hash(build())


def test_hash_helper_separates_seeds_and_repeats_a_seed():
    build = lambda: torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 2))
    assert _fresh_weights(build, 1) == _fresh_weights(build, 1)
    assert len({_fresh_weights(build, s) for s in (1, 2, 3)}) == 3


def test_the_hash_covers_buffers():
    """A module whose randomness sits in a buffer would otherwise look seed-independent."""
    class Buf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("noise", torch.randn(4))
    assert _fresh_weights(Buf, 1) != _fresh_weights(Buf, 2)


HASH_SNIPPET = """
import sys, torch, numpy as np
from src.evaluation.model_registry import model_class
from src.model.integrity import initial_weight_hash
name, seed = sys.argv[1], int(sys.argv[2])
cls = model_class(name)
torch.manual_seed(seed); np.random.seed(seed)      # the trainers' order: seed, then build
print(initial_weight_hash(cls(checkpoint_path=None, device="cpu").model))
"""


def _hash_in_fresh_process(name, seed):
    """A fresh interpreter, so a vendored import that seeds at import time actually runs
    between the seed and the build -- in a warm process it has already happened."""
    out = subprocess.run([sys.executable, "-c", HASH_SNIPPET, name, str(seed)],
                         capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": ROOT})
    if out.returncode != 0:
        pytest.skip(f"{name} unavailable here: {out.stderr.strip().splitlines()[-1:]}")
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("name", ["coldsite_dti", "hyperattentiondti", "moltrans", "deepdta"])
def test_distinct_seeds_give_distinct_initial_weights_for_each_model(name):
    hashes = [_hash_in_fresh_process(name, s) for s in (1, 2, 3)]
    assert len(set(hashes)) == 3, f"{name}: three seeds produced a repeated hash"
    assert _hash_in_fresh_process(name, 1) == hashes[0], \
        f"{name}: the same seed did not reproduce its initial weights"
