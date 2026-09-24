"""T04 generator: metrics, deterministic predictions files, refusal of partial tables, the
localization bootstrap. Synthetic inputs, used only here."""
import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.evaluation import accuracy_table as at


def test_expected_cells_are_the_84_trained_cells():
    cells = at.expected_cells()
    assert len(cells) == len(set(cells)) == 84
    assert sum(c[0] == "davis" for c in cells) == 60
    assert sum(c[0] == "kiba" for c in cells) == 24
    assert not any(c[0] == "kiba" and c[1] == "drugban" for c in cells)   # P08: none trained


def test_binary_metrics_hand_computed():
    labels = [1, 1, 1, 0, 0, 0, 0, 0]
    logits = [2.0, 1.0, -1.0, 0.5, -0.5, -2.0, -3.0, -1.0]   # predicted positive: rows 0, 1, 3
    m = at.binary_metrics(labels, logits)
    assert (m["rows"], m["positives"]) == (8, 3)
    assert m["accuracy"] == pytest.approx(6 / 8)              # TP 2, FP 1, FN 1, TN 4
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["mcc"] == pytest.approx((2 * 4 - 1 * 1) / np.sqrt(3 * 3 * 5 * 5))
    assert m["auroc"] == pytest.approx(12.5 / 15)             # 15 pos-neg pairs: 12 ordered, 1 tie (-1 vs -1)


def test_the_threshold_is_the_trainers_logit_zero():
    m = at.binary_metrics([1, 0], [0.0, -1e-9])              # logit 0 counts as positive
    assert m["predicted_positive_rate"] == 0.5 and m["accuracy"] == 1.0


def test_non_finite_logits_are_refused_and_one_class_gives_nan_not_a_crash():
    with pytest.raises(ValueError):
        at.binary_metrics([0, 1], [np.nan, 1.0])
    m = at.binary_metrics([0, 0, 0], [-1.0, 1.0, 2.0])
    assert np.isnan(m["auroc"]) and np.isnan(m["auprc"]) and m["accuracy"] == pytest.approx(1 / 3)


def test_predictions_file_is_byte_deterministic(tmp_path):
    frame = pd.DataFrame({"row": [0, 1], "label": [1, 0], "logit": [0.123456789012, -2.5]})
    a = at.write_predictions(str(tmp_path / "a" / "x.csv.gz"), frame)
    b = at.write_predictions(str(tmp_path / "b" / "x.csv.gz"), frame)
    assert a == b == at.sha256_file(str(tmp_path / "a" / "x.csv.gz"))
    back = pd.read_csv(tmp_path / "a" / "x.csv.gz")
    assert back.logit.tolist() == [0.123456789, -2.5]      # float_format="%.9g"
    with gzip.open(tmp_path / "a" / "x.csv.gz", "rt") as f:
        assert f.readline().strip() == "row,label,logit"


def _fake_cell(out, dataset, model, level, seed, leaked=False, corrupt=False):
    rng = np.random.default_rng(seed)
    n = 60
    label = np.tile([0, 0, 1], n // 3)
    logit = np.where(label == 1, 1.0, -1.0) + rng.normal(size=n)
    seen = np.zeros(n, bool)
    if leaked:
        seen[:12] = True
    frame = pd.DataFrame({"row": range(n), "Drug_ID": 1, "Target_ID": "T", "seen_by_sequence": seen,
                          "label": label, "logit": logit})
    cid = at.cell_id(dataset, model, level, seed)
    sha = at.write_predictions(os.path.join(out, "predictions", cid + ".csv.gz"), frame)
    m = at.binary_metrics(label, logit)
    meta = {"cell": cid, "predictions_sha256": "0" * 64 if corrupt else sha, "device": "cpu",
            "torch": "x", "checkpoint_file": "c.pt", "checkpoint_sha256": "1" * 64,
            "recorded_test_metrics": {k: m[k] for k in ("auroc", "auprc", "accuracy")}}
    with open(os.path.join(out, "predictions", cid + ".meta.json"), "w") as f:
        json.dump(meta, f)


def test_cell_rows_add_the_unseen_view_only_where_something_leaked(tmp_path):
    _fake_cell(str(tmp_path), "davis", "deepdta", "cold_target", 1, leaked=True)
    _fake_cell(str(tmp_path), "davis", "deepdta", "random", 1)
    frame, meta = at.load_cell(str(tmp_path), "davis_cold_target_deepdta_seed1")
    rows = at.cell_rows("davis", "deepdta", "cold_target", 1, frame, meta)
    assert [r["view"] for r in rows] == ["uncorrected", "unseen_by_sequence"]
    assert rows[0]["rows"] == 60 and rows[1]["rows"] == 48 and rows[1]["rows_dropped"] == 12
    assert rows[0]["reproduces_recorded"] is True
    frame, meta = at.load_cell(str(tmp_path), "davis_random_deepdta_seed1")
    assert [r["view"] for r in at.cell_rows("davis", "deepdta", "random", 1, frame, meta)] == ["uncorrected"]


def test_a_predictions_file_that_no_longer_matches_its_hash_is_refused(tmp_path):
    _fake_cell(str(tmp_path), "davis", "deepdta", "random", 1, corrupt=True)
    with pytest.raises(RuntimeError, match="SHA-256"):
        at.load_cell(str(tmp_path), "davis_random_deepdta_seed1")


def test_tabulate_refuses_a_partial_grid(tmp_path):
    _fake_cell(str(tmp_path), "davis", "deepdta", "random", 1)
    args = type("A", (), {"out_dir": str(tmp_path), "allow_partial": False})()
    with pytest.raises(SystemExit, match="refusing to write a partial table"):
        at.cmd_tabulate(args)
    assert not (tmp_path / "cells.csv").exists()
    args.allow_partial = True
    at.cmd_tabulate(args)                                   # only under a _partial suffix
    assert (tmp_path / "cells_partial.csv").exists() and not (tmp_path / "cells.csv").exists()


def test_tabulate_writes_all_84_and_the_row_count_matches(tmp_path, capsys):
    for d, m, l, s in at.expected_cells():
        _fake_cell(str(tmp_path), d, m, l, s, leaked=(d == "davis" and l in ("cold_target", "cold_pair")))
    at.cmd_tabulate(type("A", (), {"out_dir": str(tmp_path), "allow_partial": False})())
    cells = pd.read_csv(tmp_path / "cells.csv")
    assert cells[["dataset", "model", "level", "seed"]].drop_duplicates().shape[0] == 84
    assert (cells.view == "unseen_by_sequence").sum() == 5 * 2 * 3
    by_model = pd.read_csv(tmp_path / "by_model.csv")
    assert (by_model.n_seeds == 3).all()
    man = pd.read_csv(tmp_path / "predictions_manifest.csv")
    assert len(man) == 84 and man.predictions_sha256.str.len().eq(64).all()
    assert "84 cells (0 missing); 84 reproduce" in capsys.readouterr().out


def test_summary_sd_is_the_sample_sd():
    cells = pd.DataFrame({"dataset": "davis", "model": "m", "level": "random", "view": "uncorrected",
                          **{k: [0.6, 0.7, 0.8] for k in at.METRICS}})
    s = at.summarise(cells).iloc[0]
    assert s.auroc_mean == pytest.approx(0.7) and s.auroc_sd == pytest.approx(0.1)


def test_spearman_bootstrap_planted_signal_and_reproducibility():
    x = np.arange(20.0)
    up = at.spearman_bootstrap(x, x * 2 + 1, n_resamples=500)
    assert up["rho"] == pytest.approx(1.0) and up["low"] == pytest.approx(1.0)
    down = at.spearman_bootstrap(x, -x, n_resamples=500)
    assert down["rho"] == pytest.approx(-1.0) and down["high"] == pytest.approx(-1.0)
    noise = np.random.default_rng(5).normal(size=40)
    a = at.spearman_bootstrap(np.arange(40.0), noise, n_resamples=500)
    assert a == at.spearman_bootstrap(np.arange(40.0), noise, n_resamples=500)   # fixed seed
    assert a["low"] < 0 < a["high"]                                              # no signal planted


def test_spearman_constant_input_is_nan_not_a_crash():
    r = at.spearman_bootstrap(np.ones(10), np.arange(10.0), n_resamples=50)
    assert np.isnan(r["rho"]) and r["n_degenerate_resamples"] == 50
