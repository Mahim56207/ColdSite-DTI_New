"""T02 (f): a cell that loses rows its own encoder refuses must fail loudly past a limit.

DrugBAN's loader caps a drug at 290 atoms, so the collector skips such rows -- but only up to
UNENCODABLE_LIMIT, above which the cell raises rather than score a different population under
the same name.

What the rule actually is, measured on the real cells (2026-09-24): the limit is
max(UNENCODABLE_LIMIT_MIN, 1% of rows). Explanation cells hold one row per protein, so ~60-370
rows, and the floor of 3 dominates: 0.8% on DAVIS random (372 rows), 4.0% on DAVIS cold-target
(75), 1.3% on KIBA random (228), 5.0% on the 60-protein non-kinase panel. The 1% ceiling holds
for the fraction constant, not for the effective rate on small cells. These tests pin the rule as
it is, so that loosening it fails; whether to tighten it is the author's call (see the ledger).
"""
import pytest

from src.evaluation import collect as collect_module
from src.evaluation.collect import (UNENCODABLE_LIMIT_FRACTION, UNENCODABLE_LIMIT_MIN,
                                    MissingCell, collect_cell, load_site_sets)

GROUND_TRUTH = "data/davis_ground_truth_sites.json"
SPLIT = "data/splits/davis/random"


def _have_data():
    import os
    return os.path.exists(GROUND_TRUTH) and os.path.isdir(SPLIT)


def test_the_limit_constants_cannot_be_loosened_silently():
    assert UNENCODABLE_LIMIT_FRACTION <= 0.01
    assert UNENCODABLE_LIMIT_MIN == 3


def test_drugbans_atom_cap_is_the_published_290():
    from src.evaluation.drugban_adapter import DRUGBAN_MAX_DRUG_NODES
    assert DRUGBAN_MAX_DRUG_NODES == 290


def _refuse_first(monkeypatch, n):
    """Make the encoder refuse the first `n` rows it is asked about."""
    real = collect_module._explain_row
    state = {"calls": 0}

    def flaky(model_name, adapter, vocabs, smiles, sequence, max_protein_len):
        state["calls"] += 1
        if state["calls"] <= n:
            raise ValueError("322 atoms exceeds DrugBAN's DRUG.MAX_NODES = 290; "
                             "their dataloader cannot encode this drug")
        return real(model_name, adapter, vocabs, smiles, sequence, max_protein_len)

    monkeypatch.setattr(collect_module, "_explain_row", flaky)


def _cell():
    return collect_cell("uniform_control", "davis", "random", 1,
                        site_sets=load_site_sets(GROUND_TRUTH, max_len=1000),
                        device="cpu", verbose=False)


def _limit():
    rows = collect_module._read_test_rows(SPLIT, 1, None, policy=True, with_drug=False)
    return max(UNENCODABLE_LIMIT_MIN, int(UNENCODABLE_LIMIT_FRACTION * len(rows)))


def test_losing_exactly_the_limit_is_tolerated(monkeypatch, capsys):
    if not _have_data():
        pytest.skip("DAVIS splits or ground truth not present")
    limit = _limit()
    _refuse_first(monkeypatch, limit)
    weights, _sites, _ids = _cell()
    assert weights, "the cell was lost although it was within the limit"
    assert f"skipped {limit} row(s)" in capsys.readouterr().out


def test_losing_one_more_than_the_limit_fails_the_cell(monkeypatch):
    if not _have_data():
        pytest.skip("DAVIS splits or ground truth not present")
    limit = _limit()
    _refuse_first(monkeypatch, limit + 1)
    with pytest.raises(MissingCell) as excinfo:
        _cell()
    message = str(excinfo.value)
    assert f"could not encode {limit + 1}" in message and f"limit {limit}" in message
    assert "MAX_NODES" in message, "the reason the rows were refused must reach the error"
