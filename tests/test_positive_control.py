"""The positive control must itself be able to fail.

A control that passes whatever it is given is not a control. These pin that
the oracle really ranks sites first, that noise really is noise, that a site
outside the sequence the model sees turns the oracle check red, and -- on the
real splits, when present -- that the audit's own pipeline scores the oracle
at its ceiling.
"""
import json
import os

import numpy as np
import pytest
import torch

from src.evaluation.positive_control import (
    PlantedModel,
    dosed_explanation,
    equivalent_dose,
    faithfulness_curve,
    minimum_detectable_dose,
    plausibility_curve,
    report,
    run,
    sites_outside_sequence,
    verdicts,
    _read_ladder,
)
from src.evaluation.residue_space import MASK_CODE

AA = "ACDEFGHIKLMNPQRSTVWY"


def _rows(n=60, length=300, n_sites=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sequence = "".join(rng.choice(list(AA), size=length))
        sites = set(int(s) for s in rng.choice(length, size=n_sites, replace=False))
        rows.append((f"T{i}", sequence, sites))
    return rows


# --------------------------------------------------------------------------
# the dose
# --------------------------------------------------------------------------

def test_the_oracle_ranks_every_site_above_every_non_site():
    rng = np.random.default_rng(0)
    sites = {3, 50, 51, 199}
    weights = dosed_explanation(200, sites, 1.0, rng)
    lowest_site = min(weights[s] for s in sites)
    highest_other = max(w for i, w in enumerate(weights) if i not in sites)
    assert lowest_site > highest_other


def test_dose_zero_leaves_sites_indistinguishable_from_noise():
    weights = dosed_explanation(200, {3, 50, 199}, 0.0, np.random.default_rng(0))
    assert weights.max() < 1.0


def test_dose_outside_zero_one_is_refused():
    with pytest.raises(ValueError):
        dosed_explanation(10, {1}, 1.5, np.random.default_rng(0))


# --------------------------------------------------------------------------
# plausibility
# --------------------------------------------------------------------------

def test_oracle_scores_its_ceiling_and_noise_scores_chance():
    curve = plausibility_curve(_rows(), doses=(0.0, 1.0), k=10, n_trials=200)
    assert curve[1.0]["normalised"] == pytest.approx(1.0)
    assert curve[1.0]["p_value"] < 0.05
    assert abs(curve[0.0]["precision_at_k"] - curve[0.0]["chance"]) < 0.02


def test_precision_rises_with_dose():
    curve = plausibility_curve(_rows(), doses=(0.0, 0.2, 0.5, 1.0), k=10, n_trials=100)
    values = [curve[d]["precision_at_k"] for d in (0.0, 0.2, 0.5, 1.0)]
    assert values == sorted(values)


def test_a_site_outside_the_seen_sequence_fails_even_when_the_oracle_cannot_see_it():
    """The check the DAVIS re-numbering would have failed before it was fixed.

    This protein has 12 real sites, more than k = 10, so the oracle still fills
    its top 10 with real ones and scores its ceiling. That is exactly why the
    misplaced site is counted directly rather than inferred from the score.
    """
    rows = _rows(n=30)
    target, sequence, sites = rows[0]
    rows[0] = (target, sequence, sites | {len(sequence) + 5})   # past the end
    assert sites_outside_sequence(rows) == 1

    curve = plausibility_curve(rows, doses=(0.0, 1.0), k=10, n_trials=50)
    assert curve[1.0]["normalised"] == pytest.approx(1.0)       # blind to it

    results = {"levels": {"random": {"plausibility": curve, "faithfulness": {},
                                     "sites_outside_sequence": 1}}}
    hard_failures = [m for ok, hard, m in verdicts(results) if hard and not ok]
    assert any("outside the sequence" in m for m in hard_failures)


def test_window_cut_counts_as_outside_only_past_the_window():
    rows = [("T", "A" * 1500, {10, 1200})]
    assert sites_outside_sequence(rows, max_len=1000) == 1
    assert sites_outside_sequence(rows, max_len=2000) == 0


def test_minimum_detectable_and_equivalent_dose():
    curve = {0.0: {"precision_at_k": 0.02, "p_value": 0.4},
             0.1: {"precision_at_k": 0.10, "p_value": 0.2},
             0.2: {"precision_at_k": 0.20, "p_value": 0.01},
             1.0: {"precision_at_k": 0.90, "p_value": 0.001}}
    assert minimum_detectable_dose(curve) == 0.2
    assert equivalent_dose(curve, 0.06) == pytest.approx(0.05)
    assert equivalent_dose(curve, 0.95) is None            # above the oracle


def test_a_lucky_small_dose_is_not_quoted_as_the_resolution():
    """0.005 came out significant by the luck of one draw, but 0.01 did not;
    the resolution is where detection becomes reliable, 0.02."""
    curve = {0.0: {"p_value": 0.5}, 0.005: {"p_value": 0.01},
             0.01: {"p_value": 0.3}, 0.02: {"p_value": 0.001}, 1.0: {"p_value": 0.001}}
    assert minimum_detectable_dose(curve) == 0.02


def test_equivalent_dose_refuses_a_non_monotone_curve():
    curve = {0.0: {"precision_at_k": 0.05}, 0.5: {"precision_at_k": 0.03},
             1.0: {"precision_at_k": 0.9}}
    assert equivalent_dose(curve, 0.04) is None


def test_ladder_json_is_read_with_string_k_keys(tmp_path):
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"random": {"by_k": {"10": {"precision_at_k": 0.04}}}}))
    assert _read_ladder(str(path), 10) == {"random": 0.04}


# --------------------------------------------------------------------------
# faithfulness
# --------------------------------------------------------------------------

def test_planted_model_depends_on_sites_and_nothing_else():
    model = PlantedModel(seed=0)
    drug, protein = model.add_pair("ACDEFGHIKL", {2, 7}, max_len=1000)
    baseline = model.predict(drug, protein)
    masked_site = protein.clone()
    masked_site[0, 2] = MASK_CODE
    masked_other = protein.clone()
    masked_other[0, 4] = MASK_CODE
    assert model.predict(drug, masked_site) < baseline
    assert model.predict(drug, masked_other) == baseline


def test_planted_oracle_is_load_bearing_and_noise_is_not():
    curve = faithfulness_curve(_rows(n=40), doses=(0.0, 1.0), k=10)
    assert curve[1.0]["explanation_is_load_bearing"]
    assert abs(curve[0.0]["comprehensiveness_delta"]) < 0.2 * curve[1.0]["comprehensiveness_delta"]


# --------------------------------------------------------------------------
# the real data, when it is here
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", ["davis", "kiba"])
def test_real_pipeline_scores_the_oracle_at_its_ceiling(dataset):
    split = f"data/splits/{dataset}/cold_pair"
    truth = f"data/{dataset}_ground_truth_sites.json"
    if not (os.path.isdir(split) and os.path.exists(truth)):
        pytest.skip(f"{dataset} splits or ground truth not present")
    results = run(dataset, truth, levels=("cold_pair",), doses=(0.0, 1.0),
                  n_trials=50, verbose=False)
    failures = [m for ok, hard, m in results["verdicts"] if hard and not ok]
    assert not failures, failures
    assert "Positive control" in report(results)


def test_a_missing_equivalent_dose_says_why():
    from src.evaluation.positive_control import equivalent_dose_with_reason

    rising = {0.0: {"precision_at_k": 0.02}, 1.0: {"precision_at_k": 0.9}}
    bumpy = {0.0: {"precision_at_k": 0.05}, 0.5: {"precision_at_k": 0.03},
             1.0: {"precision_at_k": 0.9}}
    assert equivalent_dose_with_reason(rising, 0.01) == (None, "at or below chance")
    assert equivalent_dose_with_reason(rising, 0.95) == (None, "above the oracle")
    assert equivalent_dose_with_reason(bumpy, 0.04)[1].startswith("curve not monotone")
