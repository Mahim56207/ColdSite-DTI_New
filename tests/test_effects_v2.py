"""T05: effect sizes and confidence intervals (`src/evaluation/effects_v2.py`).

Each statistic is pinned on a case whose answer is known -- a planted enrichment, a planted
zero, a resampling that must equal the one the repo already committed -- because an interval
that is merely plausible is the failure this task exists to avoid. Synthetic data appears only
here, labelled, and never in a result.
"""
import json

import numpy as np
import pytest
import torch

from src.evaluation import effects_v2 as fx
from src.evaluation import seed_agreement
from src.evaluation.bootstrap_ci import bootstrap_mean
from src.evaluation.faithfulness import batch_faithfulness
from src.evaluation.run_faithfulness import collect_pairs
from src.evaluation.significance_test import _chance_precision_sample
from src.model.coldsite_dti import ColdSiteDTI
from src.model.dataset import make_loader, random_dataset


# --------------------------------------------------------------------------
# chance, per protein
# --------------------------------------------------------------------------

def test_per_protein_chance_is_sites_in_window_over_window():
    assert fx.per_protein_chance(12, 300) == pytest.approx(0.04)
    assert np.isnan(fx.per_protein_chance(3, 0))


def test_the_exact_chance_is_what_the_permutation_null_draws():
    """`_chance_precision_sample` is the null the p-values use; its mean is sites / N."""
    rng = np.random.default_rng(0)
    sites = set(range(40, 52))
    draws = [_chance_precision_sample(rng, sites, 300, 10) for _ in range(20000)]
    assert np.mean(draws) == pytest.approx(fx.per_protein_chance(12, 300), abs=0.002)
    # ... and the hypergeometric variance behind the Monte-Carlo tolerance is right too
    assert np.var(draws) == pytest.approx(fx.chance_variance(12, 300, 10), rel=0.05)


def test_the_mc_tolerance_shrinks_with_more_trials_and_more_proteins():
    v = [fx.chance_variance(10, 300, 10)] * 50
    assert fx.mc_standard_error(v, 10000) < fx.mc_standard_error(v, 500)
    assert fx.mc_standard_error(v * 4, 500) < fx.mc_standard_error(v, 500)


def test_window_lengths_are_capped_at_the_scored_window(tmp_path):
    root = tmp_path / "davis" / "random"
    root.mkdir(parents=True)
    (root / "test.csv").write_text(
        "Drug_ID,Drug,Target_ID,Target,Y\n"
        f"1,C,SHORT,{'A' * 120},1\n2,C,SHORT,{'A' * 120},0\n3,C,LONG,{'A' * 1500},1\n")
    assert fx.load_lengths("davis", "random", str(tmp_path)) == {"SHORT": 120, "LONG": 1000}


def test_sites_outside_the_window_do_not_count_toward_chance():
    class Sites:
        positions = {5, 6, 900, 1200}
    out = fx.protein_chances(["P"], {"P": Sites()}, {"P": 1000})
    assert out["P"][2] == 3 and out["P"][0] == pytest.approx(0.003)


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------

def _cell(n=60, seeds=3, enrich=2.0, noise=0.0, seed=1):
    """A planted cell: every protein's precision is `enrich` x its chance (+ noise)."""
    rng = np.random.default_rng(seed)
    chance = {f"P{i}": float(rng.uniform(0.01, 0.06)) for i in range(n)}
    precision = {i: [max(0.0, enrich * c + noise * rng.normal()) for _ in range(seeds)]
                 for i, c in chance.items()}
    return precision, chance


def test_a_planted_2x_enrichment_is_recovered_exactly_without_noise():
    precision, chance = _cell(enrich=2.0, noise=0.0)
    row = fx.bootstrap_effect(precision, chance, n_resamples=2000)
    assert row["enrichment"] == pytest.approx(2.0)
    assert row["enrichment_low"] == pytest.approx(2.0)
    assert row["enrichment_high"] == pytest.approx(2.0)


def test_a_planted_null_has_an_interval_that_covers_one():
    """Precision drawn around chance: the interval must include 1 and be non-degenerate."""
    precision, chance = _cell(n=120, enrich=1.0, noise=0.03)
    row = fx.bootstrap_effect(precision, chance, n_resamples=4000)
    assert row["enrichment_low"] < 1.0 < row["enrichment_high"]
    assert row["enrichment_high"] > row["enrichment_low"]
    assert row["excess_low"] < 0.0 < row["excess_high"]


def test_a_planted_signal_has_an_interval_that_excludes_one():
    precision, chance = _cell(n=120, enrich=3.0, noise=0.03)
    row = fx.bootstrap_effect(precision, chance, n_resamples=4000)
    assert row["enrichment_low"] > 1.0
    assert row["excess_low"] > 0.0


def test_the_precision_interval_is_the_one_bootstrap_ci_already_computes():
    """Same draws, same percentiles: the committed results/ci_davis.json intervals still hold."""
    precision, chance = _cell(n=80, enrich=1.5, noise=0.02, seed=7)
    row = fx.bootstrap_effect(precision, chance, n_resamples=10000)
    old = bootstrap_mean(precision, n_resamples=10000)
    assert (row["precision"], row["precision_low"], row["precision_high"]) == \
        (old["mean"], old["low"], old["high"])


def test_the_bootstrap_is_reproducible_and_seed_dependent():
    precision, chance = _cell(n=50, noise=0.02)
    a = fx.bootstrap_effect(precision, chance, n_resamples=500)
    b = fx.bootstrap_effect(precision, chance, n_resamples=500)
    c = fx.bootstrap_effect(precision, chance, n_resamples=500, seed=1)
    assert a == b
    assert a["enrichment_low"] != c["enrichment_low"]


def test_a_protein_carries_all_its_seeds_into_a_resample():
    """Three seeds of one protein are one unit: n counts proteins, not protein-seeds."""
    precision, chance = _cell(n=10, seeds=3)
    assert fx.bootstrap_effect(precision, chance, n_resamples=100)["n"] == 10


def test_a_cell_with_no_annotated_chance_gets_no_ratio():
    row = fx.bootstrap_effect({"A": [0.0], "B": [0.0]}, {"A": 0.0, "B": 0.0}, n_resamples=100)
    assert np.isnan(row["enrichment"]) and row["enrichment_low"] != row["enrichment_low"]


def test_the_amendments_resample_count_is_enforced_by_the_cli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["effects_v2", "--n-resamples", "500"])
    with pytest.raises(SystemExit):
        fx.main()


# --------------------------------------------------------------------------
# seed spread against distance from chance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("precisions,chance", [
    ([0.010, 0.030, 0.020], 0.020),      # spread 0.02 > |0.02 - 0.02| = 0
    ([0.041, 0.040, 0.042], 0.020),      # spread 0.002 < 0.021
    ([0.030, 0.010, 0.020], 0.030),      # spread 0.02 > 0.01
    ([0.010, 0.012, 0.011], 0.030),      # BELOW chance: spread 0.002 < |0.011 - 0.030|
])
def test_spread_rule_matches_seed_agreement(precisions, chance):
    mine = fx.seed_spread(precisions, chance)["spread_exceeds_signal"]
    table = {("m", "davis", "random"): [(p, 0.5, chance) for p in precisions]}
    assert mine == bool(seed_agreement.summarise(table)["spread_exceeds_signal"])


# --------------------------------------------------------------------------
# faithfulness delta, resampled by target
# --------------------------------------------------------------------------

def test_a_positive_delta_is_load_bearing_by_interval_and_a_zero_one_is_not():
    rng = np.random.default_rng(3)
    positive = {f"T{i}": [0.5 + 0.1 * rng.normal()] for i in range(60)}
    null = {f"T{i}": [0.1 * rng.normal()] for i in range(60)}
    up = fx.bootstrap_delta(positive, n_resamples=2000)
    flat = fx.bootstrap_delta(null, n_resamples=2000)
    assert up["load_bearing_sign"] and up["load_bearing_ci"]
    assert flat["delta_low"] < 0.0 < flat["delta_high"] and not flat["load_bearing_ci"]


def _faith_file(path, level, pairs, delta):
    payload = {"model": "coldsite_dti", "levels": {level: {
        "comprehensiveness_delta": delta, "n_pairs": len(pairs), "per_pair": pairs}}}
    path.write_text(json.dumps(payload))


def test_pairs_are_grouped_by_target_before_they_are_resampled(tmp_path):
    """Two targets, 50 pairs of one and 2 of the other: the CI is over 2 targets, not 52 pairs."""
    pairs = ([{"id": "BIG", "comprehensiveness_delta": 1.0, "sufficiency_delta": 0.0}] * 50
             + [{"id": "SMALL", "comprehensiveness_delta": -1.0, "sufficiency_delta": 0.0}] * 2)
    pair_mean = float(np.mean([p["comprehensiveness_delta"] for p in pairs]))
    for seed in (1, 2):
        _faith_file(tmp_path / f"faithfulness_davis_seed{seed}.json", "random", pairs, pair_mean)
    rows, hashes, problems = fx.faithfulness_cells(str(tmp_path), n_resamples=500)
    row = rows[0]
    assert row["n_targets"] == 2 and row["seeds"] == 2 and not problems
    assert row["delta"] == pytest.approx(0.0)                    # (1 + -1) / 2, by target
    assert row["pair_mean_delta"] == pytest.approx(pair_mean)    # the committed way, by pair
    assert row["max_abs_diff_vs_summary"] < 1e-12
    assert len(hashes) == 2


def test_a_file_without_per_pair_values_is_reported_not_skipped(tmp_path):
    (tmp_path / "faithfulness_davis_seed1.json").write_text(json.dumps(
        {"levels": {"random": {"comprehensiveness_delta": 0.2, "n_pairs": 3}}}))
    rows, _hashes, problems = fx.faithfulness_cells(str(tmp_path), n_resamples=100)
    assert rows == [] and "no per_pair" in problems[0]


# --------------------------------------------------------------------------
# recording the pairs changes no number
# --------------------------------------------------------------------------

@pytest.fixture
def model():
    torch.manual_seed(0)
    return ColdSiteDTI(70, 28).eval()


@pytest.fixture
def loader():
    return make_loader(random_dataset(12, binary=False, seed=3), batch_size=4)


def test_recording_pairs_leaves_every_mean_and_the_verdict_untouched(model, loader):
    drugs, proteins, attentions = collect_pairs(model, loader, max_pairs=5)
    plain = batch_faithfulness(model, drugs, proteins, attentions, k=5, n_random_trials=2)
    recorded = batch_faithfulness(model, drugs, proteins, attentions, k=5, n_random_trials=2,
                                  ids=[f"T{i}" for i in range(5)])
    assert "per_pair" not in plain
    assert {k: v for k, v in recorded.items() if k != "per_pair"} == plain
    assert [p["id"] for p in recorded["per_pair"]] == [f"T{i}" for i in range(5)]
    deltas = [p["comprehensiveness_delta"] for p in recorded["per_pair"]]
    assert np.mean(deltas) == pytest.approx(plain["comprehensiveness_delta"])


def test_ids_follow_their_rows_through_the_keep_mask(model, loader):
    """Row j's id must stay row j's id when other rows are skipped by the sequence policy."""
    names = [f"row{j}" for j in range(12)]
    keep = [j % 3 != 0 for j in range(12)]           # drops rows 0, 3, 6, 9
    ids_out = []
    _d, _p, attentions = collect_pairs(model, loader, max_pairs=99, keep=keep,
                                       target_ids=names, ids_out=ids_out)
    assert ids_out == [n for n, k in zip(names, keep) if k]
    assert len(ids_out) == len(attentions) == 8


def test_token_space_level_records_targets_and_leaves_means_alone(monkeypatch):
    import pandas as pd

    from src.evaluation import token_faithfulness as tf

    def fake_attention(_adapter, _drug, _sequence):
        weights = np.zeros(40)
        weights[:5] = 1.0
        return {"drug": torch.zeros(1, 4), "drug_mask": torch.ones(1, 4),
                "protein": torch.ones(1, 40), "protein_mask": torch.ones(1, 40, dtype=torch.long),
                "tokens": ["A"] * 40, "n_real": 40}, weights

    class Adapter:
        importance = np.r_[np.full(5, 10.0), np.zeros(35)]

        def predict(self, drug, protein, drug_mask=None, protein_mask=None):
            return float((self.importance * np.asarray(protein_mask).reshape(-1)).sum())

    monkeypatch.setattr(tf, "token_attention", fake_attention)
    rows = pd.DataFrame({"Target_ID": ["A", "B", "C"], "Drug": ["C"] * 3, "Target": ["M" * 50] * 3})
    plain = tf.level(Adapter(), rows, k=5, n_random_trials=3)
    recorded = tf.level(Adapter(), rows, k=5, n_random_trials=3, record_pairs=True)
    assert "per_pair" not in plain
    assert {k: v for k, v in recorded.items() if k != "per_pair"} == plain
    assert [p["id"] for p in recorded["per_pair"]] == ["A", "B", "C"]


# --------------------------------------------------------------------------
# the original verdicts come back
# --------------------------------------------------------------------------

def _audit(path, raw, corrected):
    path.write_text(json.dumps({"p_values_raw": raw, "p_values_corrected": corrected}))


def test_holm_reproduction_accepts_the_true_verdicts_and_rejects_a_flipped_one(tmp_path):
    from src.evaluation.aggregate import holm_bonferroni

    raw = {"a": 0.0001, "b": 0.02, "c": 0.4, "d": 0.9}
    truth = holm_bonferroni(raw)
    good = tmp_path / "good.json"
    _audit(good, raw, truth)
    assert fx.reproduce_audit(str(good))["ok"]
    forged = json.loads(json.dumps(truth))
    forged["b"]["significant"] = not forged["b"]["significant"]
    bad = tmp_path / "bad.json"
    _audit(bad, raw, forged)
    assert not fx.reproduce_audit(str(bad))["ok"]


def test_faithfulness_reproduction_catches_a_flag_that_contradicts_its_delta(tmp_path):
    good = {"levels": {"random": {"comprehensiveness_delta": 0.3, "explanation_is_load_bearing": True},
                       "cold_pair": {"comprehensiveness_delta": -0.1, "explanation_is_load_bearing": False}}}
    (tmp_path / "faithfulness_davis_seed1.json").write_text(json.dumps(good))
    assert fx.reproduce_faithfulness([str(tmp_path)])["ok"]
    good["levels"]["cold_pair"]["explanation_is_load_bearing"] = True
    (tmp_path / "faithfulness_davis_seed1.json").write_text(json.dumps(good))
    assert not fx.reproduce_faithfulness([str(tmp_path)])["ok"]


def test_a_rerun_is_compared_with_the_committed_file_of_the_same_name(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir(), old.mkdir()
    pairs = [{"id": f"T{i}", "comprehensiveness_delta": 0.5, "sufficiency_delta": 0.0}
             for i in range(6)]
    _faith_file(new / "faithfulness_davis_seed1.json", "random", pairs, 0.5)
    _faith_file(old / "faithfulness_davis_seed1.json", "random", pairs, 0.5 - 1e-3)
    rows, _h, _p = fx.faithfulness_cells(str(new), n_resamples=100, committed_dirs=(str(old),))
    assert rows[0]["max_abs_diff_vs_committed"] == pytest.approx(1e-3)
    rows, _h, _p = fx.faithfulness_cells(str(new), n_resamples=100)
    assert np.isnan(rows[0]["max_abs_diff_vs_committed"])


# --------------------------------------------------------------------------
# two-way bootstrap (targets AND seeds resampled)
# --------------------------------------------------------------------------

def _by_seed(precision: dict) -> dict:
    return {p: {s + 1: v for s, v in enumerate(values)} for p, values in precision.items()}


def test_two_way_equals_target_only_when_every_seed_agrees():
    """Identical seeds carry no seed variance: the target draws come first and match exactly."""
    rng = np.random.default_rng(11)
    base = {f"P{i}": float(rng.uniform(0, 0.1)) for i in range(40)}
    precision = {p: [v, v, v] for p, v in base.items()}
    chance = {p: 0.02 for p in base}
    one = fx.bootstrap_effect(precision, chance, n_resamples=3000)
    two = fx.bootstrap_effect_2d(_by_seed(precision), chance, n_resamples=3000)
    for key in ("precision", "precision_low", "precision_high", "enrichment",
                "enrichment_low", "enrichment_high"):
        assert two[key] == pytest.approx(one[key], abs=1e-12)


def test_two_way_sees_seed_variance_that_the_target_bootstrap_hides():
    """Every protein identical within a seed, seeds far apart: target-only is degenerate."""
    precision = {f"P{i}": [0.02, 0.06, 0.10] for i in range(30)}
    chance = {p: 0.02 for p in precision}
    one = fx.bootstrap_effect(precision, chance, n_resamples=3000)
    two = fx.bootstrap_effect_2d(_by_seed(precision), chance, n_resamples=3000)
    assert one["enrichment_low"] == pytest.approx(one["enrichment_high"])
    assert two["enrichment"] == pytest.approx(one["enrichment"])
    assert two["enrichment_low"] < one["enrichment_low"] < two["enrichment_high"]
    assert two["enrichment_low"] <= 1.0 + 1e-9                      # all three drawn as seed 1


def test_two_way_point_values_match_the_target_only_ones_on_real_shaped_data():
    precision, chance = _cell(n=60, enrich=1.5, noise=0.03, seeds=3, seed=5)
    one = fx.bootstrap_effect(precision, chance, n_resamples=2000)
    two = fx.bootstrap_effect_2d(_by_seed(precision), chance, n_resamples=2000)
    assert two["precision"] == pytest.approx(one["precision"])
    assert two["enrichment"] == pytest.approx(one["enrichment"])
    assert two["enrichment_high"] - two["enrichment_low"] >= \
        one["enrichment_high"] - one["enrichment_low"] - 1e-9


def test_two_way_skips_a_seed_a_protein_lacks():
    by_seed = {"A": {1: 0.1, 2: 0.1, 3: 0.1}, "B": {1: 0.3, 3: 0.3}}
    row = fx.bootstrap_effect_2d(by_seed, {"A": 0.1, "B": 0.1}, n_resamples=500)
    assert row["precision"] == pytest.approx(0.2)
    assert np.isfinite(row["precision_low"]) and np.isfinite(row["enrichment_high"])


def test_two_way_delta_sees_seed_variance():
    by_seed = {f"T{i}": {1: 0.0, 2: 0.5, 3: 1.0} for i in range(30)}
    flat = {t: list(v.values()) for t, v in by_seed.items()}
    one = fx.bootstrap_delta(flat, n_resamples=3000)
    two = fx.bootstrap_delta_2d(by_seed, n_resamples=3000)
    assert one["delta_low"] == pytest.approx(one["delta_high"]) == pytest.approx(0.5)
    assert two["delta"] == pytest.approx(0.5)
    assert two["delta_low"] < 0.5 < two["delta_high"] and not two["load_bearing_ci"]


def test_the_two_way_mode_refuses_to_overwrite_the_committed_tables(monkeypatch):
    monkeypatch.setattr("sys.argv", ["effects_v2", "--resample", "seeds_and_targets",
                                     "--out-dir", "results/effects_v2"])
    with pytest.raises(SystemExit):
        fx.main()
