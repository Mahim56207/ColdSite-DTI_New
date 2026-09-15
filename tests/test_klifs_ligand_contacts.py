"""Drug-specific ground truth from KLIFS interaction fingerprints.

The fingerprint is 85 positions x 7 interaction types in one string, and the pocket a
kinase actually has is shorter than 85 wherever KLIFS wrote a gap. Both are easy to get
off by one, and a silent off-by-one would move every drug's "binding site" -- so the
decoding, the agreement rule and the KLIFS-to-DAVIS mapping are pinned here on cases
small enough to check by hand, plus the real ABL1-imatinib structure when the network is
reachable.
"""
import json
import os

import pytest

from src.data.klifs_ligand_contacts import (N_TYPES, agreed, contacts, pair_key,
                                            placement)
from src.data.klifs_pocket import POCKET_LENGTH


def _ifp(positions, kind=0):
    """A fingerprint with `kind` set at each of `positions` (0-based)."""
    bits = ["0"] * (POCKET_LENGTH * N_TYPES)
    for p in positions:
        bits[p * N_TYPES + kind] = "1"
    return "".join(bits)


def test_a_contact_is_any_interaction_type_at_that_position():
    for kind in range(N_TYPES):
        assert contacts(_ifp([0, 7, 84], kind)) == [0, 7, 84]


def test_two_types_at_one_position_is_still_one_contact():
    bits = list(_ifp([3]))
    bits[3 * N_TYPES + 4] = "1"
    assert contacts("".join(bits)) == [3]


def test_no_contacts_is_empty_not_everything():
    assert contacts("0" * (POCKET_LENGTH * N_TYPES)) == []


def test_a_fingerprint_of_the_wrong_length_is_refused():
    """Better to lose a structure than to read positions out of a shifted string."""
    with pytest.raises(ValueError, match="expected"):
        contacts("0" * 100)


def test_one_structure_carries_its_own_contacts():
    assert agreed([[1, 2, 3]]) == [1, 2, 3]


def test_a_contact_must_hold_in_half_the_structures():
    # 4 structures: position 1 in all, 2 in two, 3 in one
    per = [[1, 2], [1, 2], [1, 3], [1]]
    assert agreed(per) == [1, 2]                       # 3 is in 1 of 4
    assert agreed(per, min_fraction=0.25) == [1, 2, 3]
    assert agreed(per, min_fraction=1.0) == [1]


def test_three_structures_need_two_agreeing_not_one_and_a_half():
    per = [[5], [5], [6]]
    assert agreed(per) == [5]


def test_no_structures_is_no_sites():
    assert agreed([]) == []


def test_the_pair_key_names_the_drug_and_the_target():
    assert pair_key("5291", "ABL1(T315I)") == "5291|ABL1(T315I)"


def test_a_gap_in_the_kinases_pocket_shifts_nothing():
    """KLIFS writes '_' where a kinase lacks a pocket residue. The fingerprint still has
    85 positions, so column 2 here is the third KLIFS position, not the third residue."""
    entry = {"name": "TEST", "kinase_ID": 1, "pocket": "AB_CD" + "_" * 80}
    sequence = "XXABCDXX"
    mapping, fields = placement([entry], sequence, sequence, "TEST")
    assert fields["status"] == "placed"
    # columns 0,1,3,4 carry residues A,B,C,D at sequence positions 2,3,4,5
    assert mapping == {0: 2, 1: 3, 3: 4, 4: 5}
    assert 2 not in mapping                            # the gap has no residue


def test_a_pocket_that_does_not_fit_the_sequence_is_rejected():
    entry = {"name": "TEST", "kinase_ID": 1, "pocket": "WWWWW" + "_" * 80}
    mapping, fields = placement([entry], "AAAAAAAAAA", "AAAAAAAAAA", "TEST")
    assert mapping == {} and fields["status"] == "pocket_not_placed"
    assert fields["rejected"][0]["klifs"] == "TEST"


@pytest.mark.skipif(not os.path.exists("data/davis_drug_sites.json")
                    or not os.path.exists("src/data/baselines/deepdta/data/davis/proteins.txt"),
                    reason="needs the fetched drug sites and DeepDTA's protein file")
def test_the_built_pairs_are_inside_their_protein_and_look_like_a_pocket():
    sites = json.load(open("data/davis_drug_sites.json"))
    report = json.load(open("data/davis_drug_sites_report.json"))
    from src.data.align_ground_truth import paths
    proteins = json.load(open(paths("davis")["proteins"]))
    assert sites, "no pairs were built"
    for key, features in sites.items():
        drug, target = key.split("|")
        assert target in proteins, key
        length = len(proteins[target])
        residues = {r for f in features for r in range(f["start"], f["end"] + 1)}
        assert max(residues) <= length, f"{key}: site past the end of the sequence"
        assert min(residues) >= 1, f"{key}: site before the sequence starts"
        # a drug touches part of the pocket, never most of a kinase
        assert len(residues) <= 60, f"{key}: {len(residues)} contacted residues"
        assert report[key]["structures"] >= 1
        assert report[key]["pdb_entries"], f"{key}: no PDB entry recorded"
        assert all(f["source"] == "KLIFS-IFP" for f in features), f"{key}: wrong provenance"


@pytest.mark.skipif(not os.path.exists("data/davis_drug_sites.json"),
                    reason="needs the fetched drug sites")
def test_a_drugs_contacts_sit_inside_the_pocket_ground_truth():
    """Contacts are pocket positions, so they must fall in the KLIFS pocket of the same
    protein -- the two ground truths are built in one coordinate frame."""
    sites = json.load(open("data/davis_drug_sites.json"))
    pocket = json.load(open("data/davis_klifs_pocket_sites.json"))
    checked = 0
    for key, features in sites.items():
        target = key.split("|")[1]
        spans = pocket.get(target)
        if not spans:
            continue
        allowed = {r for s in spans for r in range(s["start"], s["end"] + 1)}
        mine = {r for f in features for r in range(f["start"], f["end"] + 1)}
        assert mine <= allowed, f"{key}: {len(mine - allowed)} contacts outside the pocket"
        checked += 1
    assert checked, "no pair could be compared with the pocket ground truth"


def test_the_swap_control_keeps_the_protein_and_changes_the_drug():
    from src.data.klifs_ligand_contacts import swap_drugs
    sites = {"A|P1": ["a"], "B|P1": ["b"], "C|P1": ["c"]}
    swapped = swap_drugs(sites)
    assert set(swapped) == set(sites)                     # same pairs, same protein
    assert all(swapped[k] != sites[k] for k in sites)     # no pair keeps its own sites
    assert sorted(map(str, swapped.values())) == sorted(map(str, sites.values()))


def test_a_protein_with_one_crystallised_drug_cannot_be_swapped():
    """Nothing to swap with, so the pair leaves both arms rather than being compared
    against itself."""
    from src.data.klifs_ligand_contacts import swap_drugs
    assert swap_drugs({"A|P1": ["a"]}) == {}
    swapped = swap_drugs({"A|P1": ["a"], "B|P1": ["b"], "C|P2": ["c"]})
    assert set(swapped) == {"A|P1", "B|P1"}


def test_the_swap_is_the_same_every_time():
    from src.data.klifs_ligand_contacts import swap_drugs
    sites = {f"{d}|P": [d] for d in "abcdef"}
    assert swap_drugs(sites) == swap_drugs(sites)


def test_coverage_counts_only_pairs_that_are_in_that_split(tmp_path):
    """The scorable count is the honest limit on the measurement, so it must count pairs
    present in the split's own test file -- not every crystallised pair."""
    import pandas as pd
    from src.data.klifs_ligand_contacts import coverage
    level = tmp_path / "davis" / "random"
    level.mkdir(parents=True)
    pd.DataFrame({"Drug_ID": ["A", "B", "A"], "Drug": ["C"] * 3,
                  "Target_ID": ["P1", "P1", "P2"], "Target": ["MK"] * 3,
                  "Y": [5.0] * 3}).to_csv(level / "test.csv", index=False)
    sites = {"A|P1": [], "A|P2": [], "B|P9": []}          # B|P9 is not in this split
    rows = coverage(sites, str(tmp_path), "davis", swapped={"A|P1": []})
    assert len(rows) == 1                                  # only 'random' exists here
    assert rows[0] == {"level": "random", "test_rows": 3, "scorable": 2,
                       "proteins": 2, "drugs": 1, "swappable": 1}
