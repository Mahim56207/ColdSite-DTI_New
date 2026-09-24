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


def _fake_recorded(tmp_path, monkeypatch, level="cold_target", seed=1):
    """A checkpoint dir with one DrugBAN `_results.json`, a clean-accuracy file, and a 5-row split."""
    from src.model.checkpoint_naming import results_path, run_tag
    ck = tmp_path / "ck"
    (ck / "davis_binary").mkdir(parents=True)
    res = results_path(str(ck / "davis_binary"), run_tag("davis", level, "binary", seed), model="drugban")
    json.dump({"test_metrics": {"auroc": 0.85, "auprc": 0.47, "accuracy": 0.93}}, open(res, "w"))
    split = tmp_path / "data" / "splits" / "davis" / level
    split.mkdir(parents=True)
    pd.DataFrame({"Drug_ID": range(5)}).to_csv(split / "test.csv", index=False)
    clean = [{"model": "drugban", "dataset": "davis", "level": level, "seed": seed,
              "unseen_by_sequence": {"auroc": 0.81, "auprc": 0.27, "rows": 4, "positive_rate": 0.06},
              "rows_dropped": 1}]
    json.dump(clean, open(tmp_path / "clean.json", "w"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(at, "CLEAN_ACCURACY", str(tmp_path / "clean.json"))
    return str(ck)


def test_recorded_only_cell_has_recorded_metrics_and_empty_mcc_f1(tmp_path, monkeypatch):
    ck = _fake_recorded(tmp_path, monkeypatch)
    rows, man = at.recorded_cell_rows("davis", "drugban", "cold_target", 1, ck)
    assert [r["view"] for r in rows] == ["uncorrected", "unseen_by_sequence"]
    a, b = rows
    assert (a["auroc"], a["auprc"], a["accuracy"], a["rows"]) == (0.85, 0.47, 0.93, 5)
    assert (b["auroc"], b["auprc"], b["rows"], b["rows_dropped"]) == (0.81, 0.27, 4, 1)
    for r in rows:
        assert r["source"] == "recorded"
        assert all(np.isnan(r[k]) for k in ("mcc", "f1", "precision", "recall"))
    assert np.isnan(b["accuracy"])                      # not held by any file for the unseen view
    assert man["device"] == "recorded" and len(man["checkpoint_sha256"]) == 64


def test_recorded_only_cell_with_no_clean_entry_gets_no_unseen_view(tmp_path, monkeypatch):
    ck = _fake_recorded(tmp_path, monkeypatch, level="random")
    assert [r["view"] for r in at.recorded_cell_rows("davis", "drugban", "random", 1, ck)[0]] == ["uncorrected"]


def test_scope_names_a_dataset_table_so_it_cannot_pass_for_the_full_one():
    cells, tag = at._scope(type("A", (), {"datasets": "davis"})())
    assert len(cells) == 60 and tag == "_davis"
    cells, tag = at._scope(type("A", (), {})())
    assert len(cells) == 84 and tag == ""


def test_dataset_scoped_tabulate_mixes_predicted_and_recorded_cells(tmp_path, monkeypatch, capsys):
    """A full DAVIS table: 48 predicted cells + DrugBAN from recorded files, MCC/F1 empty for DrugBAN only."""
    out = tmp_path / "out"
    davis = [c for c in at.expected_cells() if c[0] == "davis"]
    for d, m, l, s in davis:
        if m != "drugban":
            _fake_cell(str(out), d, m, l, s, leaked=(l in ("cold_target", "cold_pair")))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(at, "CLEAN_ACCURACY", str(tmp_path / "clean.json"))
    (tmp_path / "ck" / "davis_binary").mkdir(parents=True)
    from src.model.checkpoint_naming import results_path, run_tag
    clean = []
    for l in at.LEVELS["davis"]:
        split = tmp_path / "data" / "splits" / "davis" / l
        split.mkdir(parents=True)
        pd.DataFrame({"Drug_ID": range(5)}).to_csv(split / "test.csv", index=False)
        for s in at.SEEDS:
            res = results_path(str(tmp_path / "ck" / "davis_binary"), run_tag("davis", l, "binary", s),
                               model="drugban")
            json.dump({"test_metrics": {"auroc": .8, "auprc": .3, "accuracy": .9}}, open(res, "w"))
            if l in ("cold_target", "cold_pair"):
                clean.append({"model": "drugban", "dataset": "davis", "level": l, "seed": s,
                              "unseen_by_sequence": {"auroc": .7, "auprc": .2, "rows": 4, "positive_rate": .1},
                              "rows_dropped": 1})
    json.dump(clean, open(tmp_path / "clean.json", "w"))
    args = type("A", (), {"out_dir": str(out), "allow_partial": False, "datasets": "davis",
                          "checkpoint_dir": str(tmp_path / "ck")})()
    at.cmd_tabulate(args)
    cells = pd.read_csv(out / "cells_davis.csv")
    assert cells[["model", "level", "seed"]].drop_duplicates().shape[0] == 60
    assert not (out / "cells.csv").exists()
    db, other = cells[cells.model == "drugban"], cells[cells.model != "drugban"]
    assert db.mcc.isna().all() and db.f1.isna().all() and (db.source == "recorded").all()
    assert other.mcc.notna().all() and (other.source == "predictions").all()
    by = pd.read_csv(out / "by_model_davis.csv")
    assert by[by.model == "drugban"].auroc_mean.notna().all()
    assert (by.n_seeds == 3).all()
    assert "; 18 rows are recorded-only" in capsys.readouterr().out       # 12 uncorrected + 6 unseen


def test_spearman_bootstrap_ci_is_finite_when_a_resample_repeats_one_inexact_float():
    """Regression, found on MolTrans/KIBA (6 cells, seed 0, 10,000 resamples): one resample drew the same cell
    six times. Its std() is 1.1e-16 > 0 although the values are identical, so the old `std() > 0` guard let it
    through, spearmanr returned NaN, and the percentile CI came out NaN. The guard is now the exact range."""
    x = np.array([0.916780, 0.915991, 0.923356, 0.802837, 0.812213, 0.820903])
    y = np.array([0.020379, 0.052607, 0.021801, 0.027830, 0.062736, 0.016509])
    assert np.std(np.full(6, x[3])) > 0 and np.ptp(np.full(6, x[3])) == 0      # the trap itself
    r = at.spearman_bootstrap(x, y, n_resamples=10000, seed=0)
    assert np.isfinite(r["low"]) and np.isfinite(r["high"])
    assert r["n_degenerate_resamples"] >= 1                                     # counted, not used
