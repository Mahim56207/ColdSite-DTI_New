"""Tests for the faithfulness runner and the accuracy hand-off to Track C.

The algorithm itself is covered by tests/test_faithfulness.py. What is tested
here is the plumbing around it, where the failure modes are silent rather than
loud: attention that does not line up with the tensor it masks, and an accuracy
axis that disagrees with the training runs it claims to report.
"""
import json

import numpy as np
import pytest
import torch

from src.evaluation.run_faithfulness import (
    DEFAULT_ACCURACY_METRIC,
    LEVELS,
    collect_accuracy,
    collect_pairs,
    faithfulness_for_level,
    faithfulness_table,
    run_dummy,
    write_accuracy_json,
)
from src.model.checkpoint_naming import results_path, run_tag
from src.model.coldsite_dti import ColdSiteDTI
from src.model.dataset import make_loader, random_dataset

DRUG_VOCAB, PROTEIN_VOCAB = 70, 28


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ColdSiteDTI(DRUG_VOCAB, PROTEIN_VOCAB).eval()


@pytest.fixture
def loader():
    return make_loader(random_dataset(12, binary=False, seed=3), batch_size=4)


# --------------------------------------------------------------------------
# collecting the inputs -- the alignment trap
# --------------------------------------------------------------------------

def test_collected_attention_is_one_weight_per_real_residue(model, loader):
    """The trap the whole contract is built around.

    batch_faithfulness masks protein positions using indices taken from the
    attention array. If attention came back at padded length, every masked
    position past the real sequence would be off, and nothing would raise.
    """
    drugs, proteins, attentions = collect_pairs(model, loader, max_pairs=5)
    assert len(drugs) == len(proteins) == len(attentions)
    for protein, attention in zip(proteins, attentions):
        real_length = int((protein[0] != 0).sum().item())
        assert attention.size == real_length, (
            f"{attention.size} weights for {real_length} real residues")


def test_collect_pairs_respects_the_budget(model, loader):
    _drugs, _proteins, attentions = collect_pairs(model, loader, max_pairs=3)
    assert len(attentions) == 3


def test_collect_pairs_returns_single_row_tensors(model, loader):
    """batch_faithfulness masks one pair at a time; a whole batch would mask
    every protein in it with one protein's top-k."""
    drugs, proteins, _attentions = collect_pairs(model, loader, max_pairs=2)
    assert all(d.dim() == 2 and d.shape[0] == 1 for d in drugs)
    assert all(p.dim() == 2 and p.shape[0] == 1 for p in proteins)


def test_level_summary_carries_the_delta_and_the_control(model, loader):
    summary = faithfulness_for_level(model, loader, k=5, n_random_trials=2,
                                     max_pairs=3)
    for key in ("comprehensiveness", "comprehensiveness_random",
                "comprehensiveness_delta", "explanation_is_load_bearing"):
        assert key in summary
    assert summary["comprehensiveness_delta"] == pytest.approx(
        summary["comprehensiveness"] - summary["comprehensiveness_random"])


def test_an_empty_level_does_not_crash_the_run(model):
    """A cell with no usable pairs must report zero, not take the run down."""
    summary = faithfulness_for_level(model, [], k=5, n_random_trials=1)
    assert summary["n_pairs"] == 0
    assert not summary["explanation_is_load_bearing"]
    assert np.isnan(summary["comprehensiveness_delta"])


# --------------------------------------------------------------------------
# the accuracy hand-off to Track C
# --------------------------------------------------------------------------

def _write_run(results_dir, dataset, level, task, seed, metrics):
    tag = run_tag(dataset, level, task, seed)
    path = results_path(str(results_dir), tag)
    with open(path, "w") as f:
        json.dump({"tag": tag, "dataset": dataset, "split": level,
                   "task": task, "seed": seed, "test_metrics": metrics}, f)
    return path


def test_accuracy_is_read_from_the_trainers_own_files(tmp_path):
    """Derived, not recomputed. A recomputed accuracy axis could disagree with
    the numbers the training runs reported and nobody would see it."""
    expected = {"random": 0.88, "cold_drug": 0.80,
                "cold_target": 0.75, "cold_pair": 0.69}
    for level, ci in expected.items():
        _write_run(tmp_path, "davis", level, "regression", 1,
                   {"mse": 0.5, "ci": ci, "pearson": 0.7})

    collected = collect_accuracy(str(tmp_path), "davis", "regression", 1)
    assert collected["accuracy"] == expected
    assert collected["metric"] == "ci"
    assert collected["missing_levels"] == []


def test_accuracy_reads_the_right_seed(tmp_path):
    """The whole reason the seed is in the filename."""
    _write_run(tmp_path, "davis", "random", "regression", 1, {"ci": 0.90})
    _write_run(tmp_path, "davis", "random", "regression", 2, {"ci": 0.60})
    assert collect_accuracy(str(tmp_path), "davis", "regression",
                            1)["accuracy"]["random"] == 0.90
    assert collect_accuracy(str(tmp_path), "davis", "regression",
                            2)["accuracy"]["random"] == 0.60


def test_a_missing_level_is_nan_and_named_not_zero(tmp_path):
    """A zero would draw as a real point on the headline figure."""
    _write_run(tmp_path, "davis", "random", "regression", 1, {"ci": 0.9})
    collected = collect_accuracy(str(tmp_path), "davis", "regression", 1)
    assert np.isnan(collected["accuracy"]["cold_pair"])
    assert "cold_pair" in collected["missing_levels"]


def test_a_missing_metric_is_reported_with_what_was_available(tmp_path):
    _write_run(tmp_path, "davis", "random", "regression", 1, {"mse": 0.4})
    collected = collect_accuracy(str(tmp_path), "davis", "regression", 1)
    assert np.isnan(collected["accuracy"]["random"])
    assert any("no 'ci'" in m for m in collected["missing_levels"])


def test_the_default_accuracy_metric_matches_the_task():
    """CI and AUROC are bounded [0,1] and read 'higher is better', so they can
    share an axis with precision@k. MSE would invert the figure."""
    assert DEFAULT_ACCURACY_METRIC["regression"] == "ci"
    assert DEFAULT_ACCURACY_METRIC["binary"] == "auroc"


def test_binary_runs_hand_over_auroc(tmp_path):
    _write_run(tmp_path, "davis", "random", "binary", 1,
               {"auroc": 0.83, "auprc": 0.7, "accuracy": 0.8})
    collected = collect_accuracy(str(tmp_path), "davis", "binary", 1)
    assert collected["metric"] == "auroc"
    assert collected["accuracy"]["random"] == 0.83


def test_the_metric_can_be_overridden(tmp_path):
    _write_run(tmp_path, "davis", "random", "regression", 1,
               {"ci": 0.9, "pearson": 0.55})
    collected = collect_accuracy(str(tmp_path), "davis", "regression", 1,
                                 metric="pearson")
    assert collected["accuracy"]["random"] == 0.55


def test_the_handoff_file_is_the_shape_run_ladder_reads(tmp_path):
    """run_ladder --accuracy-json expects a flat {level: value} mapping, and
    plot_degradation_curve indexes it by level. Anything nested silently
    produces a figure with no accuracy line."""
    accuracy = {"random": 0.9, "cold_drug": 0.8,
                "cold_target": 0.7, "cold_pair": 0.6}
    path = write_accuracy_json(accuracy, str(tmp_path), "davis_seed1")
    loaded = json.load(open(path))
    assert loaded == accuracy
    assert all(isinstance(v, float) for v in loaded.values())
    assert set(loaded) == set(LEVELS)


def test_the_handoff_actually_drives_the_headline_figure(tmp_path):
    """End to end across the seam: our accuracy file, Track C's figure."""
    from src.evaluation.run_ladder import evaluate_level, write_results

    accuracy = json.load(open(write_accuracy_json(
        {"random": 0.9, "cold_drug": 0.8, "cold_target": 0.7, "cold_pair": 0.6},
        str(tmp_path), "davis_seed1")))

    sites = {40, 41, 42}
    weights = [np.eye(1, 300, 40).ravel() for _ in range(3)]
    results = {level: evaluate_level(weights, [sites] * 3, k_values=(10,),
                                     n_trials=50) for level in LEVELS}
    _json_path, _table, figure = write_results(
        results, str(tmp_path), "davis_seed1", k=10, accuracy=accuracy)
    assert figure is not None, "accuracy hand-off did not satisfy run_ladder"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def test_the_table_leads_with_the_delta_not_the_raw_number():
    summaries = {"random": {"comprehensiveness": 0.40,
                            "comprehensiveness_random": 0.05,
                            "comprehensiveness_delta": 0.35,
                            "sufficiency": 0.1, "sufficiency_random": 0.2,
                            "aopc": 0.3, "n_pairs": 100,
                            "explanation_is_load_bearing": True}}
    table = faithfulness_table(summaries)
    assert "**0.3500**" in table
    assert "random control" in table


def test_a_non_load_bearing_level_is_stated_plainly():
    """A negative delta is the paper's most interesting possible finding. It
    must not be rendered as a blank or an error."""
    summaries = {"cold_pair": {"comprehensiveness": 0.04,
                               "comprehensiveness_random": 0.06,
                               "comprehensiveness_delta": -0.02,
                               "sufficiency": 0.1, "sufficiency_random": 0.1,
                               "aopc": 0.05, "n_pairs": 100,
                               "explanation_is_load_bearing": False}}
    table = faithfulness_table(summaries)
    assert "**-0.0200**" in table and "**no**" in table


# --------------------------------------------------------------------------
# dummy mode
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_dummy_run_covers_every_level_and_tags_its_output(tmp_path):
    out = run_dummy(out_dir=str(tmp_path), n_pairs=2, protein_length=60,
                    k=5, n_random_trials=1)
    assert set(out["summaries"]) == set(LEVELS)
    written = [p.name for p in tmp_path.iterdir()]
    assert written and all("DUMMY_PLACEHOLDER" in name for name in written), written


@pytest.mark.slow
def test_an_untrained_model_is_not_load_bearing_in_the_dummy_run(tmp_path):
    """The sanity floor, at runner level rather than metric level.

    An untrained model's attention is arbitrary, so masking its top-k should be
    no more disruptive than masking random residues. A large positive delta
    here means the runner is measuring a masking artefact -- padding, index
    drift, tie ordering -- and every real number produced afterwards is
    suspect.
    """
    out = run_dummy(out_dir=str(tmp_path), n_pairs=3, protein_length=80,
                    k=5, n_random_trials=2)
    for level, summary in out["summaries"].items():
        assert abs(summary["comprehensiveness_delta"]) < 0.5, (
            f"{level}: untrained model looks load-bearing "
            f"(delta={summary['comprehensiveness_delta']:.3f})")


# --------------------------------------------------------------------------
# the audit seam: faithfulness must work on adapters, not just our own model
# --------------------------------------------------------------------------

def test_faithfulness_accepts_a_registry_adapter(model):
    """Under the audit framing the baselines arrive as ExplainableDTIModel
    adapters, whose contract is predict() -> float rather than ColdSite-DTI's
    forward() -> (pred, attention). Without this, faithfulness could only ever
    measure our own model."""
    from src.evaluation.faithfulness import batch_faithfulness
    from src.evaluation.model_registry import get_model

    adapter = get_model("coldsite_dti")
    torch.manual_seed(4)
    drug = torch.randint(2, DRUG_VOCAB, (1, 30))
    protein = torch.randint(2, PROTEIN_VOCAB, (1, 100))
    attention = np.asarray(adapter.explain(drug, protein))

    summary = batch_faithfulness(adapter, [drug], [protein], [attention],
                                 k=5, n_random_trials=2)
    assert np.isfinite(summary["comprehensiveness_delta"])
