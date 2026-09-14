"""The integrated-gradients Kaggle notebook.

Part 6's question -- attention's fault or the models' -- is only answered if the IG
variants read the SAME checkpoints as the attention runs and are scored against BOTH
ground truths. The notebook also has to leave out the one job the pipeline refuses
(ColdSite-DTI's IG faithfulness), or the run reports a failure every time.
"""
import json
import os

import pytest

NB = "notebooks/kaggle_integrated_gradients_davis.ipynb"
ANALYSIS = "notebooks/kaggle_analysis_davis.ipynb"


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
    stage = [c for c in code_cells if "OUT_IG = " in c][0]
    scope = {"os": os, "DATASET": "davis", "SEEDS": [1, 2, 3], "MAX_PAIRS": 200,
             "MODELS": ["coldsite_dti", "hyperattentiondti", "moltrans"], "WORK": "/tmp/w",
             "RUN_UNIPROT": True, "RUN_KLIFS": True, "RUN_FAITHFULNESS": True, "N_GPU": 2,
             "COMMON": ["--checkpoint-dir", "/tmp/R"], "GT": "gt.json",
             "GT_KLIFS": "klifs.json",
             "py": lambda m, *a: ["python", "-u", "-m", m, *map(str, a)],
             "run_parallel": lambda *a, **k: False, "spread": lambda j: {0: j}}
    scope.update(over)
    exec(stage, scope)
    return scope["jobs"]


def test_it_scores_both_ground_truths_for_every_model(code_cells):
    jobs = _jobs(code_cells)
    ladders = [c for _n, c, _e in jobs if c[3].endswith("run_ladder")]
    assert len(ladders) == 18                      # 3 models x 3 seeds x 2 ground truths
    for model in ("coldsite_dti_ig", "hyperattentiondti_ig", "moltrans_ig"):
        mine = [c for c in ladders if c[5] == model]
        assert len(mine) == 6, model
        assert {c[c.index("--ground-truth") + 1] for c in mine} == {"gt.json", "klifs.json"}


def test_coldsites_ig_faithfulness_is_left_out(code_cells):
    """run_faithfulness refuses it (its masking does not go through residue_space), so
    asking for it would report a failed job on every run."""
    jobs = _jobs(code_cells)
    faith = [c for _n, c, _e in jobs if c[3].endswith("run_faithfulness")]
    assert {c[5] for c in faith} == {"hyperattentiondti_ig", "moltrans_ig"}
    assert len(faith) == 6


def test_switching_an_arm_off_removes_exactly_that_arm(code_cells):
    assert not [c for _n, c, _e in _jobs(code_cells, RUN_KLIFS=False)
                if "klifs.json" in c]
    assert not [c for _n, c, _e in _jobs(code_cells, RUN_FAITHFULNESS=False)
                if c[3].endswith("run_faithfulness")]


def test_the_variants_are_checked_before_any_gpu_time_is_spent(code_cells):
    """A path pointing at the drug side would attribute the wrong input and still return
    an array of the right shape, so section 7 resolves all three first."""
    check = [c for c in code_cells if "embedding_module" in c]
    assert len(check) == 1
    assert "model_suffix(variant) == model_suffix(m)" in check[0]
    assert "available_models()" in check[0]


def test_the_comparison_reads_the_attention_runs_from_the_repository(code_cells):
    compare = [c for c in code_cells if "IG - attention" in c][0]
    assert "results/analysis_" in compare
    assert "'uniprot'" in compare and "'klifs'" in compare


def test_the_shared_cells_are_copies_of_the_analysis_notebooks(code_cells):
    theirs = ["".join(c["source"]) for c in json.load(open(ANALYSIS))["cells"]]
    shared = [c for c in theirs if "EXPECTED_AUROC = {" in c or "def run_parallel(" in c]
    assert len(shared) == 2
    for cell in shared:
        assert cell in code_cells, "rebuild notebooks/kaggle_integrated_gradients_davis.ipynb"


def test_it_trains_nothing(code_cells):
    joined = "\n".join(code_cells)
    for trainer in ("train_deepdta", "train_moltrans", "train_hyperattentiondti", "run_grid"):
        assert trainer not in joined
