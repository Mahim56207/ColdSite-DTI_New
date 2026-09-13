"""Tests for the KLIFS pocket ground truth (src/data/klifs_pocket.py)."""
import random

from src.data.ground_truth import build_site_set
from src.data.klifs_pocket import KLIFS_REGIONS, POCKET_LENGTH, _features, choose_domain, place_pocket

AMINO = "ACDEFGHIKLMNPQRSTVWY"


def _case(seed=0, length=500):
    """A random sequence and a KLIFS-shaped pocket: each region a consecutive stretch,
    insertions between regions, and two positions the kinase lacks ('_')."""
    rng = random.Random(seed)
    sequence = "".join(rng.choice(AMINO) for _ in range(length))
    pocket, positions, start = [], [], 60
    missing = {25, 70}                                    # 0-based pocket indices given as '_'
    for _name, first, last in KLIFS_REGIONS:
        for i in range(first - 1, last):
            if i in missing:
                pocket.append("_")
                continue
            if i == 32:                                   # CDK2's case: one residue inserted inside b.l
                start += 1
            pocket.append(sequence[start])
            positions.append(start)
            start += 1
        start += rng.choice((0, 0, 2, 5, 12))             # an insertion, or none, before the next region
    assert len(pocket) == POCKET_LENGTH
    return sequence, "".join(pocket), positions


def test_a_pocket_is_placed_where_it_came_from():
    for seed in range(5):
        sequence, pocket, positions = _case(seed)
        placed, identity = place_pocket(pocket, sequence)
        assert placed == positions and identity == 1.0


def test_a_point_mutation_inside_the_pocket_is_tolerated():
    sequence, pocket, positions = _case(7)
    p = positions[10]          # away from the insertion: next to one, a mismatch is inherently ambiguous
    mutant = sequence[:p] + ("A" if sequence[p] != "A" else "G") + sequence[p + 1:]
    placed, identity = place_pocket(pocket, mutant)
    assert placed == positions and 0.95 <= identity < 1.0


def test_a_pocket_from_another_protein_is_rejected():
    sequence, pocket, _ = _case(1)
    other, _, _ = _case(2)
    _, identity = place_pocket(pocket, other)
    assert identity < 0.95


def test_features_survive_the_ground_truth_loader():
    """The loader drops any feature type it does not know; the pocket must not be dropped."""
    features = _features([10, 11, 12, 40])
    assert features == [
        {"start": 11, "end": 13, "type": "Binding site", "description": "KLIFS ATP pocket", "source": "KLIFS"},
        {"start": 41, "end": 41, "type": "Binding site", "description": "KLIFS ATP pocket", "source": "KLIFS"}]
    site_set = build_site_set("X", features, max_len=1000)
    assert site_set.usable and set(site_set.positions) == {10, 11, 12, 40}


def test_the_domain_is_taken_from_the_name():
    first, second = ({"name": "JAK1-b"}, [100, 101]), ({"name": "JAK1"}, [800, 801])
    assert choose_domain("JAK1(JH2domain-pseudokinase)", [second, first])[0] == [first]
    assert choose_domain("JAK1(JH1domain-catalytic)", [first, second])[0] == [second]
    assert choose_domain("RSK1(KinDom.1-N-terminal)", [second, first])[0] == [first]
    assert choose_domain("RPS6KA4(KinDom.2-C-terminal)", [first, second])[0] == [second]
    assert len(choose_domain("GCN2", [first, second])[0]) == 2
