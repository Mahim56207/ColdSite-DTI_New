"""The leakage retraining comparison: which rows count, and which difference is the result.

The arithmetic here is the whole finding, so it is pinned on small hand-made cases: the
model loading is `clean_accuracy`'s and tested there.
"""
import numpy as np
import pandas as pd

from src.evaluation.leakage_retrain import (SUBSETS, report, split_name, subset_masks,
                                            subset_metrics)


def _test_csv(tmp_path, targets):
    path = tmp_path / "test.csv"
    pd.DataFrame({"Drug_ID": [f"d{i}" for i in range(len(targets))],
                  "Drug": ["C"] * len(targets), "Target_ID": targets,
                  "Target": ["MK"] * len(targets), "Y": [5.0] * len(targets)}).to_csv(path, index=False)
    return str(path)


def test_the_leaked_rows_are_the_ones_whose_target_was_seen_by_sequence(tmp_path):
    csv = _test_csv(tmp_path, ["A", "B", "A", "C"])
    masks = subset_masks(csv, frozenset({"A"}))
    assert list(masks["leaked"]) == [True, False, True, False]
    assert list(masks["unleaked"]) == [False, True, False, True]
    assert masks["all"].all() and len(masks["all"]) == 4


def test_a_numeric_target_id_still_matches_a_leaked_name(tmp_path):
    """DAVIS target ids read back as ints where the audit holds them as strings."""
    csv = _test_csv(tmp_path, [101, 102])
    assert list(subset_masks(csv, frozenset({101}))["leaked"]) == [True, False]


def test_a_subset_with_one_class_is_dropped_not_scored():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    masks = {"all": np.ones(4, bool), "leaked": np.array([True, True, False, False]),
             "unleaked": np.array([False, False, True, True])}
    out = subset_metrics(labels, scores, masks)
    assert set(out) == {"all"}                      # both halves are single-class
    assert out["all"]["auroc"] == 1.0 and out["all"]["n"] == 4 and out["all"]["positives"] == 2


def _cell(level, arm, seed, all_, leaked, unleaked, rows=100, recorded=None):
    return {"model": "deepdta", "dataset": "davis", "level": level, "arm": arm, "seed": seed,
            "n_train_rows": rows, "recorded_auroc": recorded if recorded is not None else all_,
            "rescored_auroc": all_, "reproduces_recorded": True,
            "subsets": {"all": {"auroc": all_, "n": 20, "positives": 5, "auprc": 0.5},
                        "leaked": {"auroc": leaked, "n": 5, "positives": 2, "auprc": 0.5},
                        "unleaked": {"auroc": unleaked, "n": 15, "positives": 3, "auprc": 0.5}}}


def test_the_report_states_the_leak_as_matched_minus_clean_per_seed():
    # the leak helps the leaked targets by 0.10 and does nothing elsewhere
    cells = [_cell("cold_target", "original", 1, 0.90, 0.95, 0.88, rows=210),
             _cell("cold_target", "seqmatched", 1, 0.88, 0.93, 0.86, rows=180),
             _cell("cold_target", "seqclean", 1, 0.86, 0.83, 0.86, rows=180)]
    page = report(cells, "deepdta")
    assert "seqmatched − seqclean = the leak" in page
    assert "| 1 | +0.020 | +0.100 | +0.000 |" in page        # per-seed row
    assert "| **mean** | **+0.020** | **+0.100** | **+0.000** |" in page
    assert "original − seqmatched = fewer training rows" in page
    assert "PASS: seqclean and seqmatched trained on the same number of rows" in page


def test_a_control_that_is_not_volume_matched_fails_its_check():
    cells = [_cell("cold_pair", "seqmatched", 1, 0.7, 0.7, 0.7, rows=180),
             _cell("cold_pair", "seqclean", 1, 0.7, 0.7, 0.7, rows=120)]
    assert "FAIL: seqclean and seqmatched trained on the same number of rows" in report(cells, "deepdta")


def test_a_cell_that_does_not_reproduce_its_recorded_auroc_is_named():
    cells = [_cell("cold_pair", "seqclean", 2, 0.70, 0.70, 0.70)]
    cells[0]["reproduces_recorded"] = False
    cells[0]["recorded_auroc"] = 0.80
    page = report(cells, "deepdta")
    assert "FAIL" in page and "cold_pair seqclean s2 0.8000 vs 0.7000" in page


def test_split_names_are_the_ones_on_disk():
    assert split_name("cold_target", "original") == "cold_target"
    assert split_name("cold_target", "seqclean") == "cold_target_seqclean"
    assert SUBSETS == ("all", "leaked", "unleaked")
