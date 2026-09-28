"""T10: the wave partition and its budget. The plan is generated from files, so the tests recompute
the arithmetic independently and check the structural promises the plan makes: every cell once,
DrugBAN alone on its account, balanced GPUs, and no cell that already exists."""
import json
import math
import os

import pytest

from src.cloud import budget, preflight
from src.cloud.config import Cell, load_waves

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVES = os.path.join(ROOT, "config", "waves.json")
BUDGET = os.path.join(ROOT, "config", "wave_budget.json")
needs_committed = pytest.mark.skipif(not os.path.isdir(os.path.join(budget.COMMITTED, "davis_binary")),
                                     reason="~/ColdSite-results not on this machine")


def _cells():
    waves = load_waves(WAVES)
    return {a: [c for q in qs.values() for c in q] for a, qs in waves["accounts"].items()}


def test_three_accounts_and_every_cell_exactly_once():
    cells = _cells()
    assert sorted(cells) == ["ACC1", "ACC2", "ACC3"]
    ids = [c.id for cs in cells.values() for c in cs]
    assert len(ids) == len(set(ids)) == 30                      # load_waves would also refuse a repeat
    davis = {(m, l, s) for m in ("coldsite_dti", "hyperattentiondti", "moltrans")
             for l in budget.LEVELS for s in budget.SEEDS_DAVIS}
    planned = {(c.model, c.level, c.seed) for cs in cells.values() for c in cs if c.dataset == "davis"}
    assert planned == davis                                      # 3 models x 4 levels x seeds 4-5 = 24
    kiba = {(c.level, c.seed) for cs in cells.values() for c in cs if c.dataset == "kiba"}
    assert kiba == set(budget.DRUGBAN_KIBA)                      # DrugBAN x {random, cold_drug} x 1-3 = 6


def test_seeds_are_split_across_accounts_and_drugban_sits_alone():
    cells = _cells()
    assert {c.seed for c in cells["ACC1"]} == {4} and {c.seed for c in cells["ACC2"]} == {5}
    assert {c.model for c in cells["ACC3"]} == {"drugban"}
    assert all(c.model != "drugban" for a in ("ACC1", "ACC2") for c in cells[a])


def test_the_plan_never_names_a_cell_that_already_exists():
    """Seeds 4-5 were never trained, and DrugBAN has no KIBA cell (P08): a new cell must not collide
    with any of the 84 (or any DrugBAN cell)."""
    for cs in _cells().values():
        for c in cs:
            assert not (c.dataset == "davis" and c.seed <= 3), c.id
            assert not (c.dataset == "kiba" and c.model != "drugban"), c.id
    if os.path.isdir(os.path.join(budget.COMMITTED, "kiba_binary")):
        kiba_done = os.listdir(os.path.join(budget.COMMITTED, "kiba_binary"))
        assert not [f for f in kiba_done if "drugban" in f]


def test_each_account_passes_the_harness_wave_check():
    for account in ("ACC1", "ACC2", "ACC3"):
        check, _ = preflight.check_wave(WAVES, account)
        assert check.ok, check.detail


def test_the_speed_table_is_read_from_its_first_table_not_the_projection_below_it():
    speeds = budget.parse_speed(os.path.join(ROOT, budget.SPEED_MD))
    assert speeds == {"coldsite_dti": 0.387, "hyperattentiondti": 0.552, "moltrans": 0.228}


@needs_committed
def test_the_generated_files_are_what_the_generator_produces_now():
    waves, budget_now = budget.build()
    assert json.load(open(WAVES)) == json.loads(json.dumps(waves))
    assert json.load(open(BUDGET)) == json.loads(json.dumps(budget_now))


@needs_committed
def test_a_cells_hours_are_the_documented_product_recomputed_by_hand():
    b = json.load(open(BUDGET))
    speeds = b["inputs"]["seconds_per_batch_fp32"]
    rows = json.load(open(os.path.join(budget.COMMITTED, "davis_binary",
                                       "davis_cold_pair_binary_seed1_hyperattentiondti_results.json")))["n_train_rows"]
    epochs = [json.load(open(os.path.join(budget.COMMITTED, "davis_binary",
              f"davis_cold_pair_binary_seed{s}_hyperattentiondti_results.json")))["best_epoch"] + 15
              for s in (1, 2, 3)]
    epoch_s = math.ceil(rows / 32) * speeds["hyperattentiondti"] * b["inputs"]["overhead"]["hyperattentiondti"]
    cell = b["cells"]["davis_cold_pair_hyperattentiondti_seed4"]
    assert cell["hours_mean"] == pytest.approx(epoch_s * sum(epochs) / 3 / 3600, abs=0.01)
    assert cell["hours_low"] <= cell["hours_mean"] <= cell["hours_high"]


def test_the_overhead_is_never_below_one_and_matches_the_kaggle_logs_it_came_from():
    b = json.load(open(BUDGET))
    for model, factor in b["inputs"]["overhead"].items():
        assert factor >= 1.0
        computed = budget.batches(model, 21039) * b["inputs"]["seconds_per_batch_fp32"][model]
        assert computed * factor == pytest.approx(max(computed, budget.KAGGLE_EPOCH_SECONDS[model]), rel=1e-2)


def test_gpus_of_a_measured_account_are_balanced_within_five_percent():
    b = json.load(open(BUDGET))
    for account in ("ACC1", "ACC2"):
        g = b["per_account"][account]["gpu_hours_mean_by_gpu"]
        assert abs(g["gpu0"] - g["gpu1"]) <= 0.05 * max(g.values())


def test_the_required_quota_keeps_the_fifteen_percent_reserve():
    b = json.load(open(BUDGET))
    for account in ("ACC1", "ACC2"):
        p = b["per_account"][account]
        assert p["required_weekly_quota_gpu_h_for_high"] * (1 - budget.RESERVE) == pytest.approx(
            p["gpu_hours_high"], abs=0.1)


def test_drugban_cells_carry_no_invented_hours():
    b = json.load(open(BUDGET))
    for cid, e in b["cells"].items():
        if "drugban" in cid:
            assert e["hours_mean"] is None and "unmeasured" in e["note"]
    assert b["per_account"]["ACC3"]["measured_cells"] == 0
    assert "required_weekly_quota_gpu_h_for_mean" not in b["per_account"]["ACC3"]


@needs_committed
def test_the_method_lands_near_the_reports_kiba_aggregate_it_is_only_checked_against():
    check = json.load(open(BUDGET))["crosscheck_kiba"]
    assert abs(check["difference_percent"]) <= 15.0
    assert check["report_gpu_hours"] == 101.0


def test_a_commit_holds_two_gpus_for_the_session_limit_less_the_margin():
    b = json.load(open(BUDGET))
    assert b["gpu_hours_per_commit_per_account"] == pytest.approx(2 * (11 - 45 / 60))
