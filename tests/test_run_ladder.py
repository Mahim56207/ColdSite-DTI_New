"""Tests for the experiment-ladder runner."""
import numpy as np
import pytest

from src.evaluation.run_ladder import (
    LEVELS,
    evaluate_level,
    ladder_table,
    write_results,
)

SITES = {40, 41, 42, 142, 143, 144}


def perfect(length=300, sites=SITES):
    attention = np.zeros(length)
    attention[sorted(sites)] = 1.0
    return attention


def test_levels_are_in_increasing_difficulty_order():
    assert LEVELS == ("random", "cold_drug", "cold_target", "cold_pair")


def test_evaluate_level_reports_every_requested_k():
    result = evaluate_level([perfect()] * 5, [SITES] * 5,
                            k_values=(5, 10, 20), n_trials=100)
    assert set(result["by_k"]) == {5, 10, 20}
    assert result["n_proteins"] == 5


def test_evaluate_level_flags_a_perfect_explanation_as_significant():
    result = evaluate_level([perfect()] * 8, [SITES] * 8,
                            k_values=(10,), n_trials=200)
    entry = result["by_k"][10]
    assert entry["significant"]
    assert entry["precision_at_k"] > entry["chance"]


def test_evaluate_level_does_not_flag_noise():
    rng = np.random.default_rng(0)
    weights = [rng.random(300) for _ in range(8)]
    result = evaluate_level(weights, [SITES] * 8, k_values=(10,), n_trials=200)
    assert not result["by_k"][10]["significant"]


def test_evaluate_level_carries_the_skip_counts_through():
    weights = [perfect(), np.random.rand(300), np.random.rand(4)]
    sites = [SITES, set(), {0, 1}]
    entry = evaluate_level(weights, sites, k_values=(10,), n_trials=50)["by_k"][10]
    assert entry["n_skipped_no_sites"] == 1
    assert entry["n_skipped_too_short"] == 1
    assert entry["n_evaluated"] == 1


def test_ladder_table_has_one_row_per_level():
    results = {level: evaluate_level([perfect()] * 3, [SITES] * 3,
                                     k_values=(10,), n_trials=50)
               for level in LEVELS}
    table = ladder_table(results, k=10)
    for label in ("Warm", "Cold-Drug", "Cold-Target", "Cold-Pair"):
        assert label in table


def test_ladder_table_reports_ceiling_alongside_raw_precision():
    """Raw precision is not comparable across proteins without its ceiling."""
    results = {"random": evaluate_level([perfect()] * 3, [SITES] * 3,
                                        k_values=(10,), n_trials=50)}
    table = ladder_table(results, k=10)
    assert "ceiling" in table and "normalised" in table


def test_ladder_table_tolerates_a_missing_level():
    """A partially finished run must still produce a readable table."""
    results = {"random": evaluate_level([perfect()] * 3, [SITES] * 3,
                                        k_values=(10,), n_trials=50)}
    table = ladder_table(results, k=10)
    assert "Warm" in table and "Cold-Pair" not in table


def test_write_results_emits_json_and_markdown(tmp_path):
    results = {level: evaluate_level([perfect()] * 3, [SITES] * 3,
                                     k_values=(10,), n_trials=50)
               for level in LEVELS}
    json_path, table_path, figure_path = write_results(
        results, str(tmp_path), "unit_test", k=10)
    assert json_path.endswith(".json") and table_path.endswith(".md")
    assert figure_path is None, "no accuracy supplied, so no figure should be drawn"
    assert "Cold-Pair" in open(table_path).read()


def test_headline_figure_is_only_drawn_when_accuracy_is_supplied(tmp_path):
    """The fidelity curve alone is half the paper's claim.

    The whole contribution is fidelity plotted *against* accuracy across the
    ladder. Drawing a one-line figure would quietly ship an incomplete result.
    """
    results = {level: evaluate_level([perfect()] * 3, [SITES] * 3,
                                     k_values=(10,), n_trials=50)
               for level in LEVELS}
    accuracy = dict(zip(LEVELS, [0.9, 0.8, 0.78, 0.7]))
    _json, _table, figure = write_results(results, str(tmp_path), "with_acc",
                                          k=10, accuracy=accuracy)
    assert figure is not None and figure.endswith(".png")


def test_dummy_outputs_are_tagged_so_they_cannot_pass_as_results(tmp_path):
    from src.evaluation.run_ladder import run_dummy
    run_dummy(out_dir=str(tmp_path), n_proteins=4, protein_length=120)
    written = [p.name for p in tmp_path.iterdir()]
    assert all("DUMMY_PLACEHOLDER" in name for name in written), written


# --------------------------------------------------------------------------
# one pair per protein, and the row alignment it depends on
# --------------------------------------------------------------------------

class _RowModel:
    """Explains row i with a one-hot at position i, so the output names its row."""

    def eval(self):
        return self

    def to(self, _device):
        return self

    def explain(self, drug_batch, protein_batch):
        return [np.eye(50)[int(row)] for row in drug_batch[:, 0]]


def _loader(n_rows, batch=4):
    import torch
    rows = torch.arange(n_rows).reshape(-1, 1)
    return [(rows[i:i + batch], rows[i:i + batch], None) for i in range(0, n_rows, batch)]


def _sites(*targets):
    from src.data.ground_truth import SiteSet
    return {t: SiteSet(target_id=t, positions={1, 2}) for t in targets}


def test_ladder_keeps_the_first_pair_of_each_protein_in_file_order():
    from src.evaluation.run_ladder import collect_explanations

    ids = ["A", "B", "A", "C", "B", "A", "C"]        # rows 0..6
    weights, _sites_out, used = collect_explanations(
        _RowModel(), _loader(len(ids)), ids, _sites("A", "B", "C"))
    assert used == ["A", "B", "C"]
    assert [int(np.argmax(w)) for w in weights] == [0, 1, 3], \
        "each protein must be scored on its FIRST test row"


def test_zero_keeps_every_pair():
    from src.evaluation.run_ladder import collect_explanations

    ids = ["A", "B", "A", "C", "B", "A", "C"]
    _w, _s, used = collect_explanations(_RowModel(), _loader(len(ids)), ids,
                                        _sites("A", "B", "C"), pairs_per_target=0)
    assert used == ids


def test_protein_without_sites_is_skipped_not_scored():
    from src.evaluation.run_ladder import collect_explanations

    ids = ["A", "X", "A", "X"]
    _w, _s, used = collect_explanations(_RowModel(), _loader(len(ids)), ids, _sites("A"))
    assert used == ["A"]


def test_ladder_and_audit_choose_the_same_pairs(tmp_path):
    """The point of the change: the headline figure and the audit table score
    identical rows, so their numbers can be compared directly."""
    import pandas as pd

    from src.evaluation.collect import _read_test_rows
    from src.evaluation.run_ladder import collect_explanations

    from src.evaluation.exclusions import protein_key

    ids = ["A", "B", "A", "C", "B", "A", "C"]
    # B is a second name for A's sequence (DAVIS's mutants): one protein under the policy
    seqs = {"A": "MKV", "B": "MKV", "C": "MKW"}
    pd.DataFrame({"Target_ID": ids, "Drug": [f"C{i}" for i in range(len(ids))],
                  "Target": [seqs[t] for t in ids]}).to_csv(tmp_path / "test.csv", index=False)

    for policy in (True, False):
        audit_rows = _read_test_rows(str(tmp_path), pairs_per_target=1, policy=policy)
        audit_drugs = [smiles for _t, smiles, _seq in audit_rows]
        weights, _s, _u = collect_explanations(
            _RowModel(), _loader(len(ids)), ids, _sites("A", "B", "C"),
            keys=[protein_key(t, seqs[t], policy) for t in ids])
        ladder_drugs = [f"C{int(np.argmax(w))}" for w in weights]
        assert ladder_drugs == audit_drugs, f"policy={policy}"
    assert len(audit_drugs) == 3 and len(ladder_drugs) == 3          # policy off: by name


def test_ladder_table_states_what_n_counts():
    per_protein = {"random": evaluate_level([perfect()], [SITES], k_values=(10,),
                                            n_trials=20, pairs_per_target=1)}
    per_pair = {"random": evaluate_level([perfect()], [SITES], k_values=(10,),
                                         n_trials=20, pairs_per_target=0)}
    assert "`n` = proteins" in ladder_table(per_protein, k=10)
    assert "`n` = test pairs" in ladder_table(per_pair, k=10)


# ----------------------------------------------------------------------------
# explaining only the rows that are scored (2026-09-14)
# ----------------------------------------------------------------------------

def test_rows_to_score_keeps_the_first_pair_of_each_protein():
    from src.evaluation.run_ladder import rows_to_score

    class Sites:
        usable = True
        positions = {1, 2}

    site_sets = {"A": Sites(), "B": Sites()}
    targets = ["A", "A", "B", "A", "B"]
    assert rows_to_score(targets, site_sets, pairs_per_target=1) == [0, 2]
    assert rows_to_score(targets, site_sets, pairs_per_target=0) == [0, 1, 2, 3, 4]
    # A's first two rows are 0 and 1; B's are 2 and 4 -- the cap is per protein,
    # counted in file order, not a cap on the first rows of the file
    assert rows_to_score(targets, site_sets, pairs_per_target=2) == [0, 1, 2, 4]


def test_rows_to_score_skips_excluded_and_unusable_and_missing():
    from src.evaluation.run_ladder import rows_to_score

    class Sites:
        def __init__(self, usable=True):
            self.usable = usable
            self.positions = {1}

    site_sets = {"A": Sites(), "B": Sites(usable=False)}
    targets = ["A", "B", "C"]                      # C has no ground truth at all
    assert rows_to_score(targets, site_sets, pairs_per_target=0) == [0]
    assert rows_to_score(targets, site_sets, pairs_per_target=0,
                         exclude=frozenset({"A"})) == []


def test_rows_to_score_understands_a_drug_specific_ground_truth():
    from src.evaluation.run_ladder import rows_to_score

    class Sites:
        usable = True
        positions = {5}

    site_sets = {"d1|A": Sites(), "d2|B": Sites()}
    targets = ["A", "A", "B"]
    drugs = ["d1", "d2", "d1"]                     # only row 0 is a crystallised pair
    assert rows_to_score(targets, site_sets, pairs_per_target=0, drug_ids=drugs) == [0]


def test_the_collector_scores_exactly_the_selected_rows(monkeypatch):
    """The collector and the runner must agree on which rows matter, or the runner's
    filtered loader would drop rows the collector still expects."""
    import numpy as np
    import torch

    from src.evaluation.run_ladder import collect_explanations, rows_to_score

    class Sites:
        usable = True
        positions = {0}

    class Model:
        calls = 0

        def eval(self):
            return self

        def to(self, _device):
            return self

        def explain(self, drug, protein):
            Model.calls += 1
            return np.ones((len(drug), 4))

    site_sets = {"A": Sites(), "B": Sites()}
    targets = ["A", "A", "B", "B"]
    batches = [(torch.zeros(2, 2), torch.zeros(2, 4), torch.zeros(2))] * 2
    weights, _sites, ids = collect_explanations(Model(), batches, targets, site_sets,
                                                pairs_per_target=1)
    assert ids == ["A", "B"]
    assert len(weights) == len(rows_to_score(targets, site_sets, pairs_per_target=1))


def test_a_batch_with_nothing_to_score_costs_no_forward_pass():
    """The saving itself: 6,011 rows were explained to score 349 before this."""
    import numpy as np
    import torch

    from src.evaluation.run_ladder import collect_explanations

    class Sites:
        usable = True
        positions = {0}

    class Counter:
        def __init__(self):
            self.calls = 0

        def eval(self):
            return self

        def to(self, _device):
            return self

        def explain(self, drug, protein):
            self.calls += 1
            return np.ones((len(drug), 4))

    # rows 0-1 are protein A, rows 2-3 protein A again: with a cap of 1, the second
    # batch holds nothing to score
    model = Counter()
    batches = [(torch.zeros(2, 2), torch.zeros(2, 4), torch.zeros(2))] * 2
    collect_explanations(model, batches, ["A", "A", "A", "A"], {"A": Sites()},
                         pairs_per_target=1)
    assert model.calls == 1, "the empty batch should have been skipped"


def test_the_per_protein_cap_counts_scorable_rows_not_the_first_rows():
    """The cap must not be spent on a row that has no ground truth.

    With a drug-specific ground truth a protein's first test row is usually a drug it was
    never crystallised with; consuming the slot there scored 1 protein of 29 at DAVIS
    random and none at all at the cold levels (Kaggle, 2026-09-14).
    """
    from src.evaluation.run_ladder import rows_to_score

    class Sites:
        usable = True
        positions = {3}

    # protein P is tested against three drugs; only the last has a co-crystal
    site_sets = {"d3|P": Sites()}
    targets = ["P", "P", "P"]
    drugs = ["d1", "d2", "d3"]
    assert rows_to_score(targets, site_sets, pairs_per_target=1, drug_ids=drugs) == [2]


def test_the_cap_still_keeps_one_row_per_protein_when_several_are_scorable():
    from src.evaluation.run_ladder import rows_to_score

    class Sites:
        usable = True
        positions = {3}

    site_sets = {"d1|P": Sites(), "d2|P": Sites(), "d1|Q": Sites()}
    targets = ["P", "P", "Q"]
    drugs = ["d1", "d2", "d1"]
    assert rows_to_score(targets, site_sets, pairs_per_target=1, drug_ids=drugs) == [0, 2]
    assert rows_to_score(targets, site_sets, pairs_per_target=0, drug_ids=drugs) == [0, 1, 2]


def test_a_protein_keyed_ground_truth_is_unaffected_by_the_order():
    """Every row of a protein shares its site set, so consulting it before the cap
    changes nothing -- verified on the real DAVIS ladders, all 60 numbers identical."""
    from src.evaluation.run_ladder import rows_to_score

    class Sites:
        def __init__(self, usable=True):
            self.usable = usable
            self.positions = {1}

    site_sets = {"A": Sites(), "B": Sites(usable=False)}
    targets = ["A", "A", "B", "B"]
    assert rows_to_score(targets, site_sets, pairs_per_target=1) == [0]
    assert rows_to_score(targets, site_sets, pairs_per_target=0) == [0, 1]
