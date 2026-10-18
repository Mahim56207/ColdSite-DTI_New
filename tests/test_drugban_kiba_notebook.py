"""E2 Kaggle notebook (notebooks/kaggle_drugban_kiba_analysis.ipynb): it is what its builder makes, every cell compiles, it pins the commit that holds both verdicts, it checks the
checkpoints' markers and the KIBA splits before analysing, and its embedded statistics reproduce the exact per-seed p-values already recorded for HyperAttentionDTI on KIBA
(the same numbers scripts/certification reproduced), including the Holm step over a two-cell family."""
import importlib.util
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "kaggle_drugban_kiba_analysis.ipynb")
PRIMARY = os.path.join(ROOT, "results", "analysis_kiba_policyA")


def _builder():
    spec = importlib.util.spec_from_file_location("bdk", os.path.join(ROOT, "notebooks", "build_drugban_kiba_analysis_nb.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_the_notebook_is_what_the_builder_makes():
    b = _builder()
    assert b.build(b.DEFAULT_COMMIT) == json.load(open(NB))


def test_every_code_cell_compiles():
    for i, src in enumerate(_code(json.load(open(NB)))):
        compile(src, f"cell{i}", "exec")


def test_it_pins_a_commit_checks_before_analysing_and_cannot_touch_the_primary_results():
    text = "\n".join(_code(json.load(open(NB))))
    commit = _builder().DEFAULT_COMMIT
    assert re.fullmatch(r"[0-9a-f]{40}", commit) and commit in text
    assert "'--extension'" in text and "'--out-dir', OUT" in text and "analysis_kiba_E2" in text
    assert "results/analysis_kiba_policyA" not in text
    assert "markers.state" in text and "MANIFEST.json" in text and "DGL does not import" in text
    assert "torch==2.6.0" in text                                    # the environment ACC3 trained in


def _stats():
    ns = {}
    exec(_builder().STATS_CODE, ns)
    return ns


@pytest.mark.skipif(not os.path.exists(os.path.join(PRIMARY, "ladder_hyperattentiondti_kiba_seed1.json")), reason="primary KIBA ladders absent")
def test_the_embedded_statistics_reproduce_the_recorded_exact_p_values(tmp_path):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        ns = _stats()
        text = ns["e2_report"](PRIMARY, "ladder_hyperattentiondti_kiba_seed{s}.json", label="HAT", out_dir=str(tmp_path))
    finally:
        os.chdir(cwd)
    rnd = [l for l in text.splitlines() if " random " in l and "exact p" in l][0]
    cold = [l for l in text.splitlines() if "cold_drug" in l and "exact p" in l][0]
    assert "0.9730 0.6432 0.0000" in rnd and "marks ..*" in rnd and "seeds DISAGREE" in rnd
    assert "0.9065 0.8035 0.2655" in cold and "marks ..." in cold and "DISAGREE" not in cold
    summary = json.load(open(tmp_path / "E2_summary.json"))
    assert len(summary["median_p"]) == 2 and summary["holm_adjusted"][0] >= summary["median_p"][0]


def test_holm_matches_a_hand_computed_case():
    ns = _stats()
    adj, rej = ns["holm"]([0.001, 0.04])
    assert adj == pytest.approx([0.002, 0.04]) and rej == [True, True]
    adj, rej = ns["holm"]([0.03, 0.04])
    assert adj == pytest.approx([0.06, 0.06]) and rej == [False, False]
