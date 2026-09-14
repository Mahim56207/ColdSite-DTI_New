"""The readout-sensitivity notebook.

Its whole purpose is comparing readouts of one checkpoint against both ground truths, so
the failure that matters is a plan that quietly drops half the work: a `break` placed
inside the ground-truth loop cancelled every KLIFS ladder while still reporting "every
readout scored".
"""
import json
import os

import pytest

from src.model.checkpoint_naming import base_model_name

NB = "notebooks/kaggle_readouts_davis.ipynb"
ANALYSIS = "notebooks/kaggle_analysis_davis.ipynb"
VARIANTS = ["coldsite_dti_selfattn", "hyperattentiondti_maxchannel",
            "hyperattentiondti_receptive", "moltrans_maxhead", "moltrans_firstlayer"]
PUBLISHED = ["coldsite_dti", "hyperattentiondti", "moltrans"]


@pytest.fixture(scope="module")
def code_cells():
    return ["".join(c["source"]) for c in json.load(open(NB))["cells"]
            if c["cell_type"] == "code"]


def _strip(text):
    out = []
    for line in text.splitlines():
        bare = line.lstrip()
        out.append(" " * (len(line) - len(bare)) + "pass" if bare.startswith("!") else line)
    return "\n".join(out)


def test_every_code_cell_is_valid_python(code_cells):
    for i, cell in enumerate(code_cells):
        compile(_strip(cell), f"cell{i}", "exec")


def _jobs(code_cells, **over):
    stage = [c for c in code_cells if "OUT_R = " in c][0]
    scope = {"os": os, "json": json, "DATASET": "davis", "SEEDS": [1, 2, 3],
             "WORK": "/tmp/w", "VARIANTS": VARIANTS, "PUBLISHED": PUBLISHED,
             "RUN_UNIPROT": True, "RUN_KLIFS": True, "RUN_FAITHFULNESS": False,
             "GT": "gt.json", "GT_KLIFS": "klifs.json",
             "COMMON": ["--checkpoint-dir", "/tmp/R"], "base_model_name": base_model_name,
             "py": lambda m, *a: ["python", "-u", "-m", m, *map(str, a)],
             "run_parallel": lambda *a, **k: False, "spread": lambda j: {0: j}}
    scope.update(over)
    exec(stage, scope)
    return scope["jobs"]


def test_both_ground_truths_are_scored_for_every_readout(code_cells):
    jobs = _jobs(code_cells)
    ladders = [c for _n, c, _e in jobs if c[3].endswith("run_ladder")]
    assert len(ladders) == 48                     # 8 readouts x 3 seeds x 2 ground truths
    assert {c[c.index("--ground-truth") + 1] for c in ladders} == {"gt.json", "klifs.json"}
    for readout in VARIANTS + PUBLISHED:
        mine = [c for c in ladders if c[5] == readout]
        assert len(mine) == 6, readout


def test_asking_for_faithfulness_does_not_cancel_the_klifs_ladders(code_cells):
    """The exact bug this test exists for: 24 ladders instead of 48, silently."""
    jobs = _jobs(code_cells, RUN_FAITHFULNESS=True)
    ladders = [c for _n, c, _e in jobs if c[3].endswith("run_ladder")]
    faith = [c for _n, c, _e in jobs if c[3].endswith("run_faithfulness")]
    assert len(ladders) == 48
    assert {c[c.index("--ground-truth") + 1] for c in ladders} == {"gt.json", "klifs.json"}
    assert len(faith) == 18                       # 6 readouts (no ColdSite) x 3 seeds


def test_faithfulness_skips_the_readouts_whose_masking_is_not_wired(code_cells):
    faith = [c for _n, c, _e in _jobs(code_cells, RUN_FAITHFULNESS=True)
             if c[3].endswith("run_faithfulness")]
    assert not [c for c in faith if base_model_name(c[5]) == "coldsite_dti"]


def test_the_published_readouts_are_scored_in_the_same_run(code_cells):
    """Comparing against numbers from another run would confound the readout with the
    device and the code state."""
    jobs = _jobs(code_cells)
    assert {c[5] for _n, c, _e in jobs} == set(VARIANTS + PUBLISHED)


def test_every_variant_resolves_to_a_model_being_scored():
    for variant in VARIANTS:
        assert base_model_name(variant) in PUBLISHED


def test_the_shared_cells_are_copies_of_the_analysis_notebooks(code_cells):
    theirs = ["".join(c["source"]) for c in json.load(open(ANALYSIS))["cells"]]
    shared = [c for c in theirs if "EXPECTED_AUROC = {" in c or "def run_parallel(" in c]
    assert len(shared) == 2
    for cell in shared:
        assert cell in code_cells, "rebuild notebooks/kaggle_readouts_davis.ipynb"


def test_it_trains_nothing_and_avoids_the_statistics_module(code_cells):
    joined = "\n".join(code_cells)
    for trainer in ("train_deepdta", "train_moltrans", "train_hyperattentiondti", "run_grid"):
        assert trainer not in joined
    assert "st.stdev" not in joined and "import statistics" not in joined
