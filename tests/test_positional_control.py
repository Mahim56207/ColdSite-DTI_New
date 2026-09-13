"""Tests for the positional control (src/evaluation/positional_control.py).

Two planted cases with known answers: attention that only has a positional habit
must NOT beat borrowed maps, and attention that knows each protein's own sites MUST.
"""
import numpy as np

from src.evaluation.positional_control import borrow, borrowed_null, fixed_window
from src.evaluation.significance_test import permutation_test_batch


def _proteins(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [int(rng.integers(300, 600)) for _ in range(n)], rng


def test_a_pure_positional_habit_is_caught():
    """Attention always on the first residues, sites clustered there: the uniform
    null calls it significant, the borrowed null must not."""
    lengths, rng = _proteins()
    weights, sites = [], []
    for n in lengths:
        w = rng.random(n) * 0.01
        w[:15] += 1.0                                  # the habit, same for every protein
        weights.append(w)
        sites.append(set(rng.choice(40, size=8, replace=False).tolist()))   # N-terminal sites
    uniform = permutation_test_batch(weights, sites, k=10, n_trials=300, seed=0)
    assert uniform["p_value"] < 0.05, "setup: the uniform null should be fooled"
    borrowed = borrowed_null(weights, sites, k=10, mode="absolute", n_trials=300)
    assert borrowed["p_value"] > 0.05
    assert abs(borrowed["observed"] - borrowed["null_mean"]) < 0.05


def test_protein_specific_knowledge_survives():
    """Attention peaks on each protein's own sites, which sit at protein-specific places."""
    lengths, rng = _proteins(seed=1)
    weights, sites = [], []
    for n in lengths:
        s = set(rng.choice(n, size=8, replace=False).tolist())
        w = rng.random(n) * 0.01
        w[list(s)] += 1.0
        weights.append(w)
        sites.append(s)
    for mode in ("absolute", "relative"):
        borrowed = borrowed_null(weights, sites, k=10, mode=mode, n_trials=300)
        assert borrowed["observed"] > 0.7
        assert borrowed["null_mean"] < 0.1
        assert borrowed["p_value"] < 0.01


def test_borrowing_keeps_the_length():
    w = np.arange(5.0)
    assert borrow(w, 8, "absolute").size == 8 and borrow(w, 3, "absolute").size == 3
    assert borrow(w, 8, "absolute")[5:].max() < w.min()      # padding never outranks real attention
    assert borrow(w, 9, "relative").size == 9


def test_fixed_windows():
    weights = [np.zeros(50)]
    sites = [set(range(5))]
    assert fixed_window(weights, sites, 10, "first") == 0.5
    assert fixed_window(weights, sites, 10, "last") == 0.0


def _his_case(seed, attend_sites):
    """Sites are histidines. Attention either likes every histidine (a residue-type
    habit) or only the histidines that are sites (knowledge of where they are)."""
    from src.evaluation.positional_control import residue_type_null  # noqa: F401
    rng = np.random.default_rng(seed)
    weights, sites, seqs = [], [], []
    for _ in range(40):
        n = int(rng.integers(300, 500))
        letters = rng.choice(list("ACDEFGIKLMNPQRSTVWY"), size=n)
        his = rng.choice(n, size=30, replace=False)
        letters[his] = "H"
        site = set(his[:8].tolist())                       # 8 of the 30 histidines bind
        w = rng.random(n) * 0.01
        w[list(site) if attend_sites else his] += 1.0
        weights.append(w); sites.append(site); seqs.append("".join(letters))
    return weights, sites, seqs


def test_a_residue_type_habit_is_caught():
    from src.evaluation.positional_control import residue_type_null
    w, s, q = _his_case(0, attend_sites=False)
    result = residue_type_null(w, s, q, k=10, n_trials=300)
    assert result["p_value"] > 0.05 and abs(result["observed"] - result["null_mean"]) < 0.05


def test_knowing_which_histidines_survives():
    from src.evaluation.positional_control import residue_type_null
    w, s, q = _his_case(1, attend_sites=True)
    result = residue_type_null(w, s, q, k=10, n_trials=300)
    assert result["observed"] > 0.7 and result["null_mean"] < 0.4 and result["p_value"] < 0.01


def _domain_case(seed, knows_pocket):
    """A 'domain' stretch holds a sparse 'pocket'. Attention either covers the whole
    domain evenly (knows the domain) or sits on the pocket (knows the pocket)."""
    rng = np.random.default_rng(seed)
    weights, sites = [], []
    for _ in range(40):
        n = int(rng.integers(500, 800))
        lo = int(rng.integers(50, n - 300))
        domain = np.arange(lo, lo + 250)
        pocket = set(rng.choice(domain, size=60, replace=False).tolist()) | {lo, lo + 249}
        w = rng.random(n) * 0.01
        w[list(pocket) if knows_pocket else domain] += 1.0
        weights.append(w); sites.append(pocket)
    return weights, sites


def test_knowing_only_the_domain_is_caught():
    from src.evaluation.positional_control import span_null
    w, s = _domain_case(0, knows_pocket=False)
    result = span_null(w, s, k=10, n_trials=300)
    assert result["p_value"] > 0.05 and result["top_k_in_span"] > 0.95


def test_knowing_the_pocket_survives():
    from src.evaluation.positional_control import span_null
    w, s = _domain_case(1, knows_pocket=True)
    result = span_null(w, s, k=10, n_trials=300)
    assert result["observed"] > 0.9 and result["null_mean"] < 0.4 and result["p_value"] < 0.01
