"""Confidence intervals by resampling proteins.

An interval is easy to compute and easy to compute wrongly, and a wrong one is worse
than none: it would let a cell resting on six pairs look as settled as one resting on
349. These tests use cases whose answer is known in advance -- a constant, a coin, a
sample of one -- and pin the property that matters most, that resampling is over
proteins rather than over (protein, seed) pairs.
"""
import numpy as np
import pytest

from src.evaluation.bootstrap_ci import (bootstrap_mean, collect_per_protein,
                                         percentile_ci, report)


def test_a_constant_has_an_interval_of_zero_width():
    out = bootstrap_mean({f"p{i}": [0.3] for i in range(20)})
    assert out["mean"] == pytest.approx(0.3)
    assert out["low"] == pytest.approx(0.3) and out["high"] == pytest.approx(0.3)


def test_the_interval_brackets_the_mean_and_narrows_with_more_proteins():
    rng = np.random.default_rng(0)
    wide = bootstrap_mean({f"p{i}": [float(v)] for i, v in enumerate(rng.random(10))})
    narrow = bootstrap_mean({f"p{i}": [float(v)] for i, v in enumerate(rng.random(400))})
    for out in (wide, narrow):
        assert out["low"] <= out["mean"] <= out["high"]
    assert (narrow["high"] - narrow["low"]) < (wide["high"] - wide["low"]) / 3


def test_it_covers_the_truth_about_as_often_as_it_claims():
    """The point of an interval: 95% of them should contain the population mean."""
    rng = np.random.default_rng(7)
    covered = 0
    trials = 200
    for t in range(trials):
        sample = rng.binomial(1, 0.3, size=120).astype(float)   # true mean 0.3
        out = bootstrap_mean({f"p{i}": [float(v)] for i, v in enumerate(sample)},
                             n_resamples=400, seed=t)
        covered += out["low"] <= 0.3 <= out["high"]
    assert 0.88 <= covered / trials <= 1.0, f"coverage {covered / trials:.2f}"


def test_one_protein_measured_three_times_is_still_one_protein():
    """Treating (protein, seed) as the unit would shrink the interval by sqrt(3) and
    claim a precision the data does not have."""
    rng = np.random.default_rng(1)
    values = {f"p{i}": [float(v)] * 3 for i, v in enumerate(rng.random(40))}
    paired = bootstrap_mean(values, seed=3)
    flat = bootstrap_mean({f"p{i}_{s}": [v[0]] for i, v in values.items() for s in range(3)},
                          seed=3)
    assert (flat["high"] - flat["low"]) < (paired["high"] - paired["low"]) * 0.75
    assert paired["seeds"] == 3


def test_the_seeds_of_a_protein_are_averaged_not_stacked():
    out = bootstrap_mean({"p": [0.0, 1.0], "q": [0.5, 0.5]})
    assert out["mean"] == pytest.approx(0.5)         # each protein contributes its mean
    assert out["n"] == 2


def test_no_proteins_is_not_an_interval_of_zero():
    out = bootstrap_mean({})
    assert out["n"] == 0
    assert np.isnan(out["mean"]) and np.isnan(out["low"])


def test_it_is_the_same_interval_every_time_and_barely_moves_with_the_seed():
    """Reproducible for a given seed, and stable across seeds: 10,000 resamples leave
    the interval settled to about three decimals, so a reported CI is a property of the
    data rather than of the random draw."""
    values = {f"p{i}": [float(i % 3) / 2] for i in range(30)}
    assert bootstrap_mean(values, seed=5) == bootstrap_mean(values, seed=5)
    a, b = bootstrap_mean(values, seed=5), bootstrap_mean(values, seed=6)
    assert abs(a["low"] - b["low"]) < 0.01 and abs(a["high"] - b["high"]) < 0.01


def test_percentile_ci_is_the_requested_width():
    samples = np.linspace(0.0, 1.0, 1001)
    low, high = percentile_ci(samples, confidence=0.90)
    assert low == pytest.approx(0.05, abs=1e-3) and high == pytest.approx(0.95, abs=1e-3)


def _ladder(tmp_path, name, level="random", scores=(0.1, 0.2), ids=("a", "b"), k="10"):
    import json
    payload = {level: {"n_proteins": len(ids), "ids": list(ids),
                       "by_k": {k: {"precision_at_k": float(np.mean(scores)),
                                    "per_protein": list(scores)}}}}
    (tmp_path / name).write_text(json.dumps(payload))


def test_it_reads_the_seeds_of_a_cell_and_pairs_them_by_protein(tmp_path):
    _ladder(tmp_path, "ladder_davis_seed1.json", scores=(0.0, 0.4))
    _ladder(tmp_path, "ladder_davis_seed2.json", scores=(0.2, 0.2))
    per_protein, missing = collect_per_protein(str(tmp_path), "coldsite_dti", "davis",
                                               [1, 2], "random")
    assert per_protein == {"a": [0.0, 0.2], "b": [0.4, 0.2]}
    assert missing == []


def test_a_ladder_without_per_protein_scores_is_named_not_skipped(tmp_path):
    import json
    (tmp_path / "ladder_davis_seed1.json").write_text(
        json.dumps({"random": {"by_k": {"10": {"precision_at_k": 0.02}}}}))
    per_protein, missing = collect_per_protein(str(tmp_path), "coldsite_dti", "davis",
                                               [1], "random")
    assert per_protein == {}
    assert "predates per-protein scores" in missing[0]


def test_the_report_names_the_widest_interval(tmp_path):
    rows = [{"ground_truth": "uniprot", "model": "m", "level": "random", "missing": [],
             "n": 349, "mean": 0.02, "low": 0.015, "high": 0.025, "n_resamples": 10},
            {"ground_truth": "uniprot", "model": "m", "level": "cold_pair", "missing": [],
             "n": 6, "mean": 0.13, "low": 0.0, "high": 0.33, "n_resamples": 10}]
    text = report(rows)
    assert "cold-pair" in text and "6 proteins" in text
    assert "0.015–0.025" in text
