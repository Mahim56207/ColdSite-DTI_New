"""Tests for the MolTrans trainer (src/model/train_moltrans.py)."""
import os
import subprocess
import sys

import pandas as pd
import pytest
import torch

SOURCE = "data/splits/davis/cold_target"
VENDORED = "baselines/MolTrans"


def _tiny_split(root):
    root.mkdir()
    for part, n in (("train", 24), ("valid", 12), ("test", 12)):
        pd.read_csv(os.path.join(SOURCE, f"{part}.csv")).head(n).to_csv(
            root / f"{part}.csv", index=False)
    return root


def _train(split, out, seed):
    """Run the trainer in its own process: the vendored import only reseeds once."""
    out.mkdir()
    subprocess.run(
        [sys.executable, "-m", "src.model.train_moltrans", "--split-dir", str(split),
         "--dataset", "davis", "--split", "cold_target", "--seed", str(seed),
         "--epochs", "1", "--min-epochs", "1", "--batch-size", "8", "--device", "cpu",
         "--checkpoint-dir", str(out), "--results-dir", str(out)],
        check=True, capture_output=True)
    ckpt = out / f"coldsite_dti_davis_cold_target_binary_seed{seed}_moltrans.pt"
    return torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]


@pytest.mark.slow
@pytest.mark.skipif(not (os.path.isdir(SOURCE) and os.path.isdir(VENDORED)),
                    reason="needs the DAVIS splits and the vendored MolTrans")
def test_the_seed_reaches_moltrans(tmp_path):
    """Regression: the vendored models.py seeds torch with 1 when imported, and the
    trainer used to seed before importing it, so every --seed trained as seed 1. The
    DAVIS grid's three MolTrans seeds came out identical to four decimals."""
    split = _tiny_split(tmp_path / "split")
    one = _train(split, tmp_path / "s1", 1)
    two = _train(split, tmp_path / "s2", 2)
    assert any(not torch.equal(one[k], two[k]) for k in one), \
        "seeds 1 and 2 produced identical weights: --seed is not reaching MolTrans"
