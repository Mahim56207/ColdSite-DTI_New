"""The drug-specific Kaggle notebook: its arms, and what it must not shadow.

This notebook reuses the analysis notebook's GPU check, clone, split verification, input
check and parallel runner by copying those cells at build time. Copies drift, and the
runner defines a `spread()` that an ill-named helper here would shadow -- so both are
pinned, along with the arms themselves, since a missing swapped arm would leave the
drug-specific numbers with nothing to be compared against.
"""
import json
import os

import pytest

NB = "notebooks/kaggle_drug_sites_davis.ipynb"
ANALYSIS = "notebooks/kaggle_analysis_davis.ipynb"


@pytest.fixture(scope="module")
def code_cells():
    return ["".join(c["source"]) for c in json.load(open(NB))["cells"]
            if c["cell_type"] == "code"]


def _strip_magics(text):
    out = []
    for line in text.splitlines():
        bare = line.lstrip()
        out.append(" " * (len(line) - len(bare)) + "pass" if bare.startswith("!") else line)
    return "\n".join(out)


def test_every_code_cell_is_valid_python(code_cells):
    for i, cell in enumerate(code_cells):
        compile(_strip_magics(cell), f"cell{i}", "exec")


def test_the_three_arms_are_all_there(code_cells):
    joined = "\n".join(code_cells)
    assert "RUN_DRUG" in joined and "RUN_SWAPPED" in joined and "RUN_ONE_PER_PROTEIN" in joined
    assert "_drug_sites.json" in joined and "_drug_sites_swapped.json" in joined
    # the swapped arm is what makes the drug arm interpretable
    assert "drug - swapped" in joined


def test_the_drug_arm_keeps_every_pair_and_the_third_arm_keeps_one(code_cells):
    stage = [c for c in code_cells if "ARMS = []" in c][0]
    assert "'drug', GT_DRUG, 0" in stage
    assert "'swapped', GT_SWAP, 0" in stage
    assert "'one pair per protein', GT_DRUG, 1" in stage


def test_the_comparison_helper_does_not_shadow_the_runners_spread(code_cells):
    """The runner deals jobs to GPUs with spread(); a mean-and-sd helper of the same name
    would break a re-run of the scoring cell."""
    joined = "\n".join(code_cells)
    assert "def spread(jobs):" in joined
    assert "def mean_sd(" in joined
    assert "def spread(values)" not in joined


def test_it_reads_the_ladder_json_the_way_run_ladder_writes_it(code_cells):
    """{level: {'by_k': {'10': {'precision_at_k', 'chance', 'n_evaluated'}}}}."""
    stage = [c for c in code_cells if "def at_k(" in c][0]
    assert "['by_k']" in stage or ".get('by_k', {})" in stage
    assert "precision_at_k" in stage and "chance" in stage and "n_evaluated" in stage


def test_the_shared_cells_are_copies_of_the_analysis_notebooks(code_cells):
    """They are copied at build time, so a change there must be rebuilt here."""
    theirs = ["".join(c["source"]) for c in json.load(open(ANALYSIS))["cells"]]
    shared = [c for c in theirs if "EXPECTED_AUROC = {" in c or "def run_parallel(" in c]
    assert len(shared) == 2, "the analysis notebook's input check and runner moved"
    for cell in shared:
        assert cell in code_cells, "rebuild notebooks/kaggle_drug_sites_davis.ipynb"


def test_it_trains_nothing(code_cells):
    joined = "\n".join(code_cells)
    for trainer in ("train_deepdta", "train_moltrans", "train_hyperattentiondti", "run_grid"):
        assert trainer not in joined


def test_deepdta_is_not_scored_here(code_cells):
    """It has no attention: it is the accuracy anchor, not an audited model."""
    settings = [c for c in code_cells if "RUN_DRUG" in c][0]
    assert "MODELS = ['coldsite_dti', 'hyperattentiondti', 'moltrans']" in settings


@pytest.mark.skipif(not os.path.exists("data/davis_drug_sites_swapped.json"),
                    reason="needs the built drug sites and their swapped control")
def test_the_swapped_file_holds_the_same_pairs_it_can_compare(code_cells):
    sites = json.load(open("data/davis_drug_sites.json"))
    swapped = json.load(open("data/davis_drug_sites_swapped.json"))
    assert set(swapped) <= set(sites)
    for key in swapped:
        assert swapped[key] != sites[key], f"{key} was swapped with itself"
