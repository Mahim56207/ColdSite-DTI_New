"""Tests for the sequence policy (src/evaluation/exclusions.py), option A of 2026-09-13."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.exclusions import describe, excluded_target_ids, protein_key

DAVIS = "data/splits/davis"
needs_davis = pytest.mark.skipif(
    not (os.path.isdir(DAVIS) and os.path.exists("data/davis_klifs_pocket_report.json")),
    reason="needs the DAVIS splits and the KLIFS report")


@needs_davis
def test_the_leaked_and_pocketless_targets_are_excluded_where_they_should_be():
    cold = excluded_target_ids("davis", "cold_target")
    warm = excluded_target_ids("davis", "random")
    assert {"MET", "EGFR(G719S)", "TYK2(JH1domain-catalytic)"} <= cold      # seen by sequence
    assert {"RET", "EPHA6", "ROCK2"} <= warm                                 # no kinase pocket
    assert "MET" not in warm, "nothing is 'seen by sequence' where targets are shared by design"
    assert excluded_target_ids("davis", "cold_target", policy=False) == frozenset()


@needs_davis
def test_every_excluded_cold_target_really_is_in_training_by_sequence():
    for level in ("cold_target", "cold_pair"):
        train = set(pd.read_csv(f"{DAVIS}/{level}/train.csv").Target)
        test = pd.read_csv(f"{DAVIS}/{level}/test.csv").drop_duplicates("Target_ID")
        seq = dict(zip(test.Target_ID.astype(str), test.Target))
        leaked = excluded_target_ids("davis", level) - excluded_target_ids("davis", "random")
        assert leaked and all(seq[t] in train for t in leaked if t in seq)
        kept = set(seq) - excluded_target_ids("davis", level)
        assert not any(seq[t] in train for t in kept), "a kept cold target is seen by sequence"


@needs_davis
def test_one_protein_is_one_sequence():
    from src.evaluation.collect import _read_test_rows
    new = _read_test_rows(f"{DAVIS}/random", pairs_per_target=1)
    old = _read_test_rows(f"{DAVIS}/random", pairs_per_target=1, policy=False)
    assert len({seq for _t, _s, seq in new}) == len(new), "a sequence was counted twice"
    assert len(old) == 442 and len({seq for _t, _s, seq in old}) < len(old)
    info = describe("davis", "random")
    assert len(new) == info["distinct_sequences_kept"]


def test_the_policy_can_be_switched_off():
    assert protein_key("ABL1(T315I)", "mkq", policy=True) == "MKQ"
    assert protein_key("ABL1(T315I)", "mkq", policy=False) == "ABL1(T315I)"


class _Echo(torch.nn.Module):
    """explain() returns the protein row as its own attention, so rows are traceable."""
    def explain(self, drug, protein):
        return [p.float().numpy() for p in protein]


def test_the_ladder_path_skips_and_dedupes_by_key():
    from src.evaluation.run_ladder import collect_explanations
    from src.data.ground_truth import build_site_set
    proteins = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    loader = [(torch.zeros(4, 2), proteins, torch.zeros(4))]
    ids = ["A", "A_mut", "B", "C"]
    keys = ["seqA", "seqA", "seqB", "seqC"]                 # A_mut is A's sequence
    sites = {t: build_site_set(t, [{"start": 1, "end": 1, "type": "Binding site"}])
             for t in ids}
    w, _s, used = collect_explanations(_Echo(), loader, ids, sites, keys=keys,
                                       exclude=frozenset({"C"}))
    assert used == ["A", "B"]
    assert [int(x[0]) for x in w] == [1, 3]


def test_the_faithfulness_path_skips_masked_rows():
    from src.evaluation.run_faithfulness import collect_pairs
    proteins = torch.tensor([[1, 1], [2, 2], [3, 3]])
    loader = [(torch.zeros(3, 2), proteins, torch.zeros(3))]
    _d, p, _a = collect_pairs(_Echo(), loader, keep=[True, False, True])
    assert [int(x[0, 0]) for x in p] == [1, 3]


@needs_davis
def test_accuracy_uses_the_unseen_auroc_only_where_the_level_leaks(tmp_path):
    import json
    from src.evaluation.run_faithfulness import collect_accuracy
    from src.model.checkpoint_naming import results_path, run_tag
    for level, auroc in (("random", 0.93), ("cold_target", 0.91), ("cold_pair", 0.73)):
        with open(results_path(str(tmp_path), run_tag("davis", level, "binary", 1)), "w") as f:
            json.dump({"test_metrics": {"auroc": auroc}}, f)
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps([{"model": "coldsite_dti", "dataset": "davis", "level": "cold_target",
                                  "seed": 1, "unseen_by_sequence": {"auroc": 0.88}}]))
    out = collect_accuracy(str(tmp_path), "davis", "binary", 1, clean_accuracy_path=str(clean))
    assert out["accuracy"]["random"] == 0.93 and out["sources"]["random"] == "recorded test set"
    assert out["accuracy"]["cold_target"] == 0.88
    assert out["sources"]["cold_target"] == "targets unseen by sequence"
    assert out["accuracy"]["cold_pair"] == 0.73 and out["uncorrected_levels"] == ["cold_pair"]


def test_several_dropout_passes_are_averaged_and_judged_by_their_range():
    from src.evaluation.clean_accuracy import combine_draws, merge_cells, reproduces
    draws = [({"auroc": a, "auprc": 0.1, "rows": 10}, {"auroc": a - 0.05, "auprc": 0.1, "rows": 8})
             for a in (0.562, 0.575, 0.570)]
    every, unseen = combine_draws(draws)
    assert abs(every["auroc"] - 0.569) < 1e-9 and len(every["auroc_draws"]) == 3
    assert reproduces(0.567, [0.562, 0.575, 0.570])            # the MolTrans case
    assert not reproduces(0.600, [0.562, 0.575, 0.570])
    assert reproduces(0.9039, [0.9039]) and not reproduces(0.91, [0.90])
    old = [{"model": "deepdta", "dataset": "davis", "level": "cold_pair", "seed": 1, "v": 1},
           {"model": "moltrans", "dataset": "davis", "level": "cold_pair", "seed": 1, "v": 1}]
    new = [{"model": "moltrans", "dataset": "davis", "level": "cold_pair", "seed": 1, "v": 2}]
    merged = merge_cells(old, new)
    assert len(merged) == 2 and {c["model"]: c["v"] for c in merged} == {"deepdta": 1, "moltrans": 2}
