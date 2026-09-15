"""The KIBA notebook's plan and commands, checked before 100 GPU-hours are spent on them.

Every KIBA cell costs 5 to 9 hours on a T4 and the whole grid is ~101 GPU-hours over four
accounts, so a mistake here is not a re-run of a script -- it is a day of somebody's
Kaggle quota. What these tests protect:

* the plan trains each of the 24 cells exactly once, and no two accounts share one;
* both GPUs of every account get the same number of hours (the session lasts as long as
  the slower queue, so an unbalanced pair wastes quota);
* every flag in every generated command is one the target trainer actually accepts --
  the failure mode is `error: unrecognized arguments`, discovered eleven hours in;
* the recorded split sizes and class balance match the real KIBA files;
* the settings cell runs before the repo is cloned (it must not import from `src`).
"""
import json
import os
import pathlib
import subprocess
import sys

import pytest

NOTEBOOK = pathlib.Path("notebooks/kaggle_kiba_4accounts.ipynb")
REPO = pathlib.Path(__file__).resolve().parents[1]


def cells():
    return json.loads((REPO / NOTEBOOK).read_text())["cells"]


def source(index):
    return "".join(cells()[index]["source"])


def code_cells():
    return [(i, "".join(c["source"])) for i, c in enumerate(cells())
            if c["cell_type"] == "code"]


def settings_cell():
    for index, text in code_cells():
        if "ACCOUNT = " in text and "ACCOUNTS = {" in text:
            return index, text
    raise AssertionError("no settings cell found")


def run_settings(account="A"):
    """Execute the settings cell as Kaggle would, and return its namespace."""
    _index, text = settings_cell()
    text = text.replace("ACCOUNT = 'A'", f"ACCOUNT = {account!r}", 1)
    namespace = {}
    exec(compile(text, "settings", "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

def test_the_plan_is_exactly_the_18_cells_once_each():
    ns = run_settings()
    flat = [c for queues in ns["ACCOUNTS"].values() for q in queues.values() for c in q]
    assert len(flat) == 18, f"{len(flat)} cells in the plan"
    assert len(set(flat)) == 18, "a cell appears twice -- it would be trained twice"
    assert sorted(flat) == sorted(ns["EVERY_CELL"])


def test_coldsite_dti_is_deliberately_not_replicated_on_kiba():
    """Cut to fit the compute (-27 GPU-h). It is our own model and its DAVIS verdict is
    unambiguous, so the replication is aimed at the two published models. If it ever
    reappears in the plan, the paper's claim about which models are audited on two
    datasets changes and Limitations has to change with it."""
    ns = run_settings()
    assert "coldsite_dti" not in ns["KIBA_MODELS"]
    trained = {m for queues in ns["ACCOUNTS"].values() for q in queues.values()
               for m, _lv, _s in q}
    assert trained == {"deepdta", "hyperattentiondti", "moltrans"}, trained
    assert "coldsite_dti" in ns["HOURS"], (
        "keep its cost so the model stays complete if it is added back")


def test_every_account_pair_of_gpus_is_balanced():
    ns = run_settings()
    hours = ns["HOURS"]
    for name, queues in ns["ACCOUNTS"].items():
        loads = [sum(hours[m][0] for m, _lv, _s in q) for q in queues.values()]
        assert len(loads) == 2, f"account {name} has {len(loads)} queues, expected 2"
        assert max(loads) - min(loads) <= 0.2, (
            f"account {name}: GPUs differ by {max(loads) - min(loads):.1f} h")


def test_no_two_accounts_train_the_same_cell():
    ns = run_settings()
    owned = {name: {c for q in queues.values() for c in q}
             for name, queues in ns["ACCOUNTS"].items()}
    names = sorted(owned)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not owned[a] & owned[b], f"{a} and {b} both train {owned[a] & owned[b]}"


def test_no_account_needs_more_than_a_week_of_quota():
    """Kaggle gives ~30 h a week. The wall time is what a session consumes; report it."""
    ns = run_settings()
    hours = ns["HOURS"]
    for name, queues in ns["ACCOUNTS"].items():
        wall = max(sum(hours[m][0] for m, _lv, _s in q) for q in queues.values())
        assert wall <= 20, f"account {name} needs {wall:.1f} h of wall time on one T4"


def test_each_account_letter_selects_its_own_cells():
    for account in "ABCD":
        ns = run_settings(account)
        assert ns["ACCOUNT"] == account
        mine = {c for q in ns["MY_CELLS"].values() for c in q}
        assert mine == {c for q in ns["ACCOUNTS"][account].values() for c in q}
        assert ns["TOTAL_CELLS"] == len(mine)


def test_an_unknown_account_letter_is_refused():
    _index, text = settings_cell()
    text = text.replace("ACCOUNT = 'A'", "ACCOUNT = 'E'", 1)
    with pytest.raises(AssertionError, match="ACCOUNT must be one of"):
        exec(compile(text, "settings", "exec"), {})


def test_the_settings_are_the_kiba_protocol():
    ns = run_settings()
    assert ns["DATASET"] == "kiba"
    assert ns["TASK"] == "binary"
    assert ns["AMP_BY_MODEL"] == {"deepdta": True, "hyperattentiondti": True,
                                  "moltrans": False}, ns["AMP_BY_MODEL"]
    assert ns["BRANCH"] == "main", "kiba-resume is merged and now behind main"
    assert ns["LEVELS"] == ["random", "cold_drug"]
    assert ns["SEEDS"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# cell ordering: nothing may import the repo before it is cloned
# ---------------------------------------------------------------------------

def imports_src(body):
    for line in body.splitlines():
        head = line.split("#", 1)[0].strip()
        if head.startswith(("from src.", "from src ", "import src")):
            return True
    return False


def test_nothing_imports_the_repo_before_the_clone():
    clone = next(i for i, text in code_cells() if "git clone" in text)
    for index, text in code_cells():
        if imports_src(text):
            assert index > clone, (
                f"cell {index} imports src but the clone is cell {clone}: on Kaggle that "
                "is ModuleNotFoundError before anything runs")


def test_the_settings_cell_runs_with_no_repo_on_the_path():
    _index, text = settings_cell()
    proc = subprocess.run([sys.executable, "-c", text], capture_output=True, text=True,
                          cwd="/tmp")
    assert proc.returncode == 0, proc.stderr
    assert "account A" in proc.stdout


# ---------------------------------------------------------------------------
# the commands
# ---------------------------------------------------------------------------

def runner_namespace(account="A", n_gpu=2):
    """Execute the runner cell with the globals the earlier cells would have set."""
    ns = run_settings(account)
    index = next(i for i, text in code_cells() if "def moltrans_cmd" in text)
    ns.update({
        "os": os, "N_GPU": n_gpu, "WORK": "/tmp", "RESULTS": "/tmp/results",
        "START": 0.0, "COLDSITE_BATCH": 64, "HAT_BATCH": 32, "HAT_ACCUM": 1,
        "DEEPDTA_BATCH": 256, "MOLTRANS_BATCH": 16,
    })
    import time as _time
    ns["time"] = _time
    ns["START"] = _time.time()
    exec(compile(source(index), "runner", "exec"), ns)
    return ns


def test_one_queue_per_gpu_covering_exactly_this_accounts_cells():
    for account in "ABCD":
        ns = runner_namespace(account)
        queues = ns["QUEUES"]
        assert sorted(queues) == ["0", "1"], queues.keys()
        built = sum(len(q) for q in queues.values())
        assert built == ns["TOTAL_CELLS"], (
            f"account {account}: {built} commands for {ns['TOTAL_CELLS']} cells")


def test_one_command_is_one_cell():
    """ColdSite-DTI goes through run_grid, which can train a whole sub-grid. If it were
    handed more than one split or seed, one command would silently train cells this
    account does not own -- and another account would train them too."""
    for account in "ABCD":
        ns = runner_namespace(account)
        for queue in ns["QUEUES"].values():
            for cmd in queue:
                if "src.model.run_grid" in cmd:
                    splits = cmd[cmd.index("--splits") + 1]
                    seeds = cmd[cmd.index("--seeds") + 1]
                    assert "," not in splits, f"run_grid given several splits: {splits}"
                    assert "," not in seeds, f"run_grid given several seeds: {seeds}"


def test_moltrans_never_gets_mixed_precision_and_the_others_always_do():
    """The 2026-09-14 failure: MolTrans under float16 on KIBA went NaN at batch ~4,040 of
    epoch 1, because its hand-rolled LayerNorm divides by sqrt(var + 1e-12) and 1e-12
    underflows in float16. A NaN in the forward pass is in the weights from then on, so
    six cells would be unusable. This is the test that stops --amp coming back."""
    seen = set()
    for account in "ABCD":
        ns = runner_namespace(account)
        for gpu, cells_ in ns["MY_CELLS"].items():
            for model, split, seed in cells_:
                match = [c for c in ns["QUEUES"][str(gpu)]
                         if split in c and str(seed) in c
                         and any(model.replace("coldsite_dti", "run_grid") in part
                                 or model in part for part in c)]
                assert match, f"no command for {model} {split} s{seed}"
                for cmd in match:
                    if model == "moltrans":
                        assert "--amp" not in cmd, (
                            "MolTrans must train in full precision on KIBA: "
                            f"{' '.join(cmd)}")
                    else:
                        assert "--amp" in cmd, f"{model} lost its --amp: {cmd}"
                seen.add(model)
    assert seen == {"deepdta", "hyperattentiondti", "moltrans"}, seen


def test_every_command_is_a_kiba_binary_command():
    """HyperAttentionDTI and MolTrans are classifiers in their published form and have no
    --task flag at all; DeepDTA and ColdSite-DTI do regression too, so theirs must say
    binary explicitly or they would train the wrong objective."""
    for account in "ABCD":
        ns = runner_namespace(account)
        for queue in ns["QUEUES"].values():
            for cmd in queue:
                assert "kiba" in cmd, f"not a KIBA command: {cmd}"
                if "--task" in cmd:
                    assert cmd[cmd.index("--task") + 1] == "binary", cmd
                else:
                    assert any(m in cmd for m in ("src.model.train_moltrans",
                                                  "src.model.train_hyperattentiondti")), (
                        f"no --task, and this trainer is not binary-only: {cmd}")


def test_deepdta_is_told_binary():
    """DeepDTA does regression too, so an omitted --task would train the wrong
    objective and still produce a plausible-looking results file."""
    saw = False
    for account in "ABCD":
        ns = runner_namespace(account)
        for queue in ns["QUEUES"].values():
            for cmd in queue:
                if "src.model.train_deepdta" in cmd:
                    assert cmd[cmd.index("--task") + 1] == "binary", cmd
                    saw = True
    assert saw, "no DeepDTA command in any account"


def test_the_commands_cover_the_cells_the_plan_lists():
    """A command's split and seed must match the cell it was built for."""
    for account in "ABCD":
        ns = runner_namespace(account)
        for gpu, cells_ in ns["MY_CELLS"].items():
            commands = ns["QUEUES"][str(gpu)]
            for model, split, seed in cells_:
                def names(cmd):
                    if "src.model.run_grid" in cmd:
                        return (cmd[cmd.index("--splits") + 1],
                                cmd[cmd.index("--seeds") + 1])
                    return (cmd[cmd.index("--split") + 1], cmd[cmd.index("--seed") + 1])
                assert any(names(c) == (split, str(seed)) for c in commands), (
                    f"account {account} GPU{gpu}: no command for {model} {split} s{seed}")


MODULE_OF = {
    "deepdta": "src.model.train_deepdta",
    "hyperattentiondti": "src.model.train_hyperattentiondti",
    "moltrans": "src.model.train_moltrans",
}


@pytest.mark.parametrize("module", sorted(set(MODULE_OF.values())))
def test_every_flag_is_one_the_trainer_accepts(module):
    """The bug this prevents: `error: unrecognized arguments`, eleven hours in. Each
    trainer has its own CLI -- train.py has no --patience, train_deepdta has no
    --train-subsample -- so the flags are checked against the real parsers."""
    help_text = subprocess.run([sys.executable, "-m", module, "--help"],
                               capture_output=True, text=True, cwd=REPO).stdout
    assert help_text, f"{module} --help printed nothing"
    used = set()
    for account in "ABCD":
        ns = runner_namespace(account)
        for queue in ns["QUEUES"].values():
            for cmd in queue:
                if module in cmd:
                    used |= {a for a in cmd if a.startswith("--")}
    assert used, f"no command in any account uses {module}"
    for flag in sorted(used):
        assert flag in help_text, f"{module} does not accept {flag}"


def test_a_single_gpu_session_is_refused():
    """On one GPU the account would need twice the wall time and overrun its quota."""
    with pytest.raises(AssertionError, match="GPU T4 x2"):
        runner_namespace("A", n_gpu=1)


# ---------------------------------------------------------------------------
# the data the notebook asserts on
# ---------------------------------------------------------------------------

def split_check_cell():
    for index, text in code_cells():
        if "EXPECTED = {" in text and "POSITIVE_RATE" in text:
            return text
    raise AssertionError("no split-verification cell")


def test_the_recorded_split_sizes_are_the_real_ones():
    pd = pytest.importorskip("pandas")
    namespace = {}
    text = split_check_cell()
    body = text[text.index("EXPECTED = {"):text.index("POSITIVE_RATE")]
    exec(compile(body, "sizes", "exec"), namespace)
    for level, expected in namespace["EXPECTED"].items():
        directory = REPO / "data" / "splits" / "kiba" / level
        if not directory.is_dir():
            pytest.skip(f"no local splits for kiba/{level}")
        sizes = tuple(len(pd.read_csv(directory / f"{part}.csv"))
                      for part in ("train", "valid", "test"))
        assert sizes == expected, f"{level}: real {sizes}, notebook says {expected}"


def test_the_recorded_positive_rates_are_the_real_ones():
    pd = pytest.importorskip("pandas")
    from src.model.dataset import BINARY_THRESHOLD
    text = split_check_cell()
    line = next(l for l in text.splitlines() if l.startswith("POSITIVE_RATE"))
    rates = eval(line.split("=", 1)[1].split("#")[0].strip())
    for level, claimed in rates.items():
        path = REPO / "data" / "splits" / "kiba" / level / "test.csv"
        if not path.exists():
            pytest.skip(f"no local splits for kiba/{level}")
        real = (pd.read_csv(path)["Y"] >= BINARY_THRESHOLD["kiba"]).mean()
        assert abs(real - claimed) < 0.01, f"{level}: real {real:.3f}, notebook {claimed}"


def speed_row(model, amp):
    """The row of results/speed_test_kiba_t4.md that a model's cost must come from.

    The file has two tables whose rows both start "| <model> | amp+benchmark |" -- the
    s/batch one and the projected-hours one -- so the search starts at the hours table.
    Which row applies depends on the precision that model actually trains in: MolTrans
    is full precision on KIBA, so its costs are the fp32 row, not the amp row.
    """
    measured = REPO / "results" / "speed_test_kiba_t4.md"
    if not measured.exists():
        pytest.skip("speed test not in this checkout")
    text = measured.read_text()
    table = text[text.index("## KIBA hours per cell"):]
    want = "amp" if amp else "fp32"
    for line in table.splitlines():
        if not line.startswith(f"| {model} |"):
            continue
        setting = line.split("|")[2].strip()
        if (setting.startswith("amp") if amp else setting == "fp32"):
            return line
    raise AssertionError(f"no {want} row for {model} in the hours table")


def test_the_hour_estimates_match_the_measured_speed_test():
    """HOURS drives the whole partition, so it must be the measured table, not a guess --
    and the row it comes from must match the precision that model trains in."""
    ns = run_settings()
    for model, (at36, at25) in ns["HOURS"].items():
        if model not in ns["AMP_BY_MODEL"]:
            continue                      # ColdSite-DTI: costed but not trained on KIBA
        row = speed_row(model, ns["AMP_BY_MODEL"][model])
        # "| model | amp+benchmark | min/epoch | h at 25 / 36 ep | ..."
        cell = row.split("|")[4]
        low, high = (float(x) for x in cell.split("/"))
        assert (low, high) == (at25, at36), (
            f"{model}: notebook says {at25}/{at36} h, speed test says {low}/{high}")


# ---------------------------------------------------------------------------
# the progress report: measured epoch rate, and when this finishes
# ---------------------------------------------------------------------------

def fresh_state(ns, gpu="0", key="moltrans"):
    return {"model": "MolTrans", "split": "random", "seed": "1", "epoch": "-", "cell": 1,
            "total": 2, "gpu": gpu, "key": key, "sec_per_epoch": None,
            "timed_epochs": 0, "last_epoch": None, "last_epoch_at": None}


def test_the_first_epoch_is_not_counted_as_a_rate():
    """It carries the encoding of 83,000 KIBA rows, so timing it would slow the whole
    projection down and predict a finish that never comes."""
    ns = runner_namespace()
    w = fresh_state(ns)
    ns["note_epoch"](w, 1, 1000.0)
    assert w["sec_per_epoch"] is None and w["timed_epochs"] == 0
    assert ns["eta"](w, 1000.0, 1e12) == [], "no ETA may be claimed from one epoch"
    ns["note_epoch"](w, 2, 1600.0)
    assert w["sec_per_epoch"] == 600.0 and w["timed_epochs"] == 1


def test_a_repeated_epoch_number_is_not_a_new_boundary():
    """HyperAttentionDTI and MolTrans print 'epoch 1 train batch 7/10' for every batch."""
    ns = runner_namespace()
    w = fresh_state(ns)
    ns["note_epoch"](w, 1, 1000.0)
    for t in (1010.0, 1020.0, 1030.0):        # batch lines inside epoch 1
        ns["note_epoch"](w, 1, t)
    assert w["timed_epochs"] == 0, "batch lines were mistaken for epochs"
    ns["note_epoch"](w, 2, 1600.0)
    assert w["sec_per_epoch"] == 600.0, "the rate must span epoch 1 -> 2, not a batch"


def test_the_rate_is_a_mean_over_the_epochs_it_timed():
    ns = runner_namespace()
    w = fresh_state(ns)
    for number, when in ((1, 0.0), (2, 600.0), (3, 1200.0), (4, 2400.0)):
        ns["note_epoch"](w, number, when)
    assert w["timed_epochs"] == 3
    assert w["sec_per_epoch"] == pytest.approx((600 + 600 + 1200) / 3)


def test_the_report_names_the_measured_rate_and_the_projection():
    ns = runner_namespace()
    w = fresh_state(ns, key="moltrans")
    rate = ns["MIN_PER_EPOCH"]["moltrans"] * 60      # fp32 on KIBA: 20.7 min
    ns["note_epoch"](w, 1, 0.0)
    ns["note_epoch"](w, 2, rate)                     # exactly the projected rate
    lines = "\n".join(ns["eta"](w, rate, 1e12))
    assert f"{rate / 60:.1f} min/epoch measured" in lines
    assert f"{rate / 60:.1f} projected" in lines
    assert "x1.00" in lines, lines


def test_a_slower_gpu_shows_up_as_drift():
    ns = runner_namespace()
    w = fresh_state(ns, key="hyperattentiondti")
    ns["note_epoch"](w, 1, 0.0)
    ns["note_epoch"](w, 2, 12.1 * 60 * 1.5)    # 50% slower than the benchmark
    lines = "\n".join(ns["eta"](w, 0.0, 1e12))
    assert "x1.50" in lines, lines


def test_the_report_says_when_this_cell_ends_at_both_epoch_counts():
    ns = runner_namespace()
    w = fresh_state(ns)
    ns["note_epoch"](w, 1, 0.0)
    ns["note_epoch"](w, 10, 9 * 600.0)         # 10 epochs done, 600 s each
    lines = "\n".join(ns["eta"](w, 9 * 600.0, 1e12))
    assert "this cell ends" in lines
    assert "min " in lines and "median " in lines
    # 15 epochs to the floor, 26 to the median, at 600 s
    assert "+2.5 h" in lines, lines
    assert "+4.3 h" in lines, lines


def test_the_report_counts_the_cells_still_queued_behind_this_one():
    ns = runner_namespace("A")                 # A: 2 cells per GPU
    w = fresh_state(ns)
    w["cell"] = 1                              # on the first of two
    ns["note_epoch"](w, 1, 0.0)
    ns["note_epoch"](w, 2, 600.0)
    lines = "\n".join(ns["eta"](w, 600.0, 1e12))
    assert "1 cell(s) after this one" in lines, lines
    w["cell"] = 2                              # on the last
    lines = "\n".join(ns["eta"](w, 600.0, 1e12))
    assert "last cell of this queue" in lines, lines


def test_it_says_when_another_commit_is_needed():
    ns = runner_namespace("A")
    w = fresh_state(ns)
    w["cell"] = 1
    ns["note_epoch"](w, 1, 0.0)
    ns["note_epoch"](w, 2, 600.0)
    now = 600.0
    soon = "\n".join(ns["eta"](w, now, now + 600))          # 10 minutes left
    assert "more commit(s)" in soon, soon
    plenty = "\n".join(ns["eta"](w, now, now + 100 * 3600))
    assert "inside this commit" in plenty, plenty


def test_every_model_has_a_projected_epoch_rate():
    ns = run_settings()
    assert set(ns["MIN_PER_EPOCH"]) == set(ns["HOURS"])
    for model, (at36, _at25) in ns["HOURS"].items():
        implied = ns["MIN_PER_EPOCH"][model] * ns["EPOCHS_TYPICAL"] / 60
        assert abs(implied - at36) < 0.15, (
            f"{model}: {ns['MIN_PER_EPOCH'][model]} min/epoch over "
            f"{ns['EPOCHS_TYPICAL']} epochs is {implied:.2f} h, but HOURS says {at36}")


def test_the_projected_epoch_rate_is_the_measured_one():
    ns = run_settings()
    for model, claimed in ns["MIN_PER_EPOCH"].items():
        if model not in ns["AMP_BY_MODEL"]:
            continue
        row = speed_row(model, ns["AMP_BY_MODEL"][model])
        assert float(row.split("|")[3]) == claimed, (
            f"{model}: notebook says {claimed} min/epoch, speed test says "
            f"{row.split('|')[3].strip()}")
