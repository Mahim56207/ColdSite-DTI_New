"""Tests for the sequence-clean cold splits and their matched control."""
import os

import numpy as np
import pandas as pd
import pytest

from src.data.seqclean_splits import check, seqclean, seqmatched

T = 7.0


def _toy(seed=0):
    """Test holds proteins A and B; training holds A's twin (same sequence, other name)."""
    rng = np.random.default_rng(seed)
    seq = {"A": "MKA", "A_mut": "MKA", "B": "MKB", "C": "MKC", "D": "MKD", "E": "MKE",
           "V": "MKV", "W": "MKW", "V_mut": "MKB"}   # V_mut: a validation protein with a test sequence
    def rows(names, n):
        out = []
        for name in names:
            for i in range(n):
                out.append({"Drug_ID": f"d{i}", "Drug": "C", "Target_ID": name, "Target": seq[name],
                            "Y": float(rng.choice([5.0, 8.0], p=[0.8, 0.2]))})
        return pd.DataFrame(out)
    # eligible pools several times larger than what has to be removed, as on DAVIS
    return {"train": rows(["A_mut", "C", "D", "E"], 40), "valid": rows(["V", "W", "V_mut"], 20),
            "test": rows(["A", "B"], 10)}


def test_the_clean_split_shares_no_sequence_and_keeps_the_test_set():
    parts = _toy()
    clean = seqclean(parts)
    assert "MKA" not in set(clean["train"].Target)            # A's twin is gone
    assert "MKB" not in set(clean["valid"].Target)            # V_mut is gone
    assert clean["test"].equals(parts["test"])
    matched = seqmatched(parts, clean, T)
    report = check(parts, clean, matched, T)
    assert report["control_keeps_leaked_sequences"] == 1      # the control still leaks A


def test_the_control_matches_volume_and_positives_and_keeps_every_leaked_row():
    parts = _toy(1)
    clean = seqclean(parts)
    matched = seqmatched(parts, clean, T)
    for p in ("train", "valid"):
        assert len(matched[p]) == len(clean[p])
        assert (matched[p].Y >= T).sum() == (clean[p].Y >= T).sum()
    assert (matched["train"].Target_ID == "A_mut").sum() == 40
    assert (matched["valid"].Target_ID == "V_mut").sum() == 20


def test_the_checks_catch_a_leak():
    parts = _toy()
    clean = seqclean(parts)
    broken = dict(clean, train=pd.concat([clean["train"], parts["train"][parts["train"].Target == "MKA"]]))
    with pytest.raises(AssertionError, match="shared"):
        check(parts, broken, seqmatched(parts, clean, T), T)


def test_the_control_is_the_same_split_every_time():
    parts = _toy(2)
    clean = seqclean(parts)
    a, b = seqmatched(parts, clean, T), seqmatched(parts, clean, T)
    assert a["train"].equals(b["train"]) and a["valid"].equals(b["valid"])


@pytest.mark.skipif(not os.path.isdir("data/splits/davis/cold_target"), reason="needs DAVIS splits")
def test_davis_splits_pass_every_check():
    from src.data.seqclean_splits import _read
    for level in ("cold_target", "cold_pair"):
        parts = _read("data/splits", "davis", level)
        clean = seqclean(parts)
        report = check(parts, clean, seqmatched(parts, clean, T), T)
        assert report["control_keeps_leaked_sequences"] > 0
        assert report["train_rows"][1] < report["train_rows"][0]
