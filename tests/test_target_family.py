"""Tests for the kinase-confound control."""
import pytest

from src.evaluation.target_family import (
    KINASE,
    NON_KINASE,
    UNKNOWN,
    classify_target,
    confound_report,
    stratified_indices,
    stratify,
)


@pytest.mark.parametrize("target", [
    "ABL1", "ABL1(T315I)p", "EGFR", "CDK4", "MAPK1", "JAK2", "SRC",
    "PIK3CA", "MTOR", "AURKA", "FLT3", "MET",
])
def test_kinase_gene_symbols_are_recognised(target):
    assert classify_target(target) == KINASE


@pytest.mark.parametrize("target", [
    "HIV-1 protease", "HIV-1 reverse transcriptase",
    "SARS-CoV-2 main protease", "Influenza neuraminidase",
    "SARS-CoV-2 RNA-dependent RNA polymerase", "SARS-CoV-2 nsp5",
])
def test_antiviral_control_targets_are_non_kinase(target):
    assert classify_target(target) == NON_KINASE


def test_non_kinase_patterns_win_over_gene_symbol_collisions():
    """A false KINASE label would contaminate the control arm irrecoverably."""
    assert classify_target("MET protease") == NON_KINASE
    assert classify_target("RET-like proteinase") == NON_KINASE


def test_the_word_kinase_is_enough():
    assert classify_target("Casein kinase II") == KINASE
    assert classify_target("XYZ123", "serine/threonine kinase") == KINASE


def test_uniprot_accessions_are_unknown_without_a_mapping():
    """KIBA uses accessions, so it cannot be stratified as-is.

    This is not a bug -- it is the reason the guides ask Track A for a
    UniProt -> gene-name mapping before the control can run on KIBA.
    """
    for accession in ("O00141", "P00533", "Q9Y6K9"):
        assert classify_target(accession) == UNKNOWN


def test_empty_and_missing_ids_are_unknown():
    assert classify_target("") == UNKNOWN
    assert classify_target("   ") == UNKNOWN


def test_stratify_partitions_every_input():
    targets = ["ABL1", "HIV-1 protease", "O00141", "EGFR"]
    groups = stratify(targets)
    assert groups[KINASE] == ["ABL1", "EGFR"]
    assert groups[NON_KINASE] == ["HIV-1 protease"]
    assert groups[UNKNOWN] == ["O00141"]
    assert sum(len(v) for v in groups.values()) == len(targets)


def test_stratified_indices_align_with_input_order():
    targets = ["ABL1", "HIV-1 protease", "EGFR"]
    idx = stratified_indices(targets)
    assert idx[KINASE] == [0, 2]
    assert idx[NON_KINASE] == [1]


def test_control_is_unusable_below_twenty_non_kinase_targets():
    report = confound_report(["ABL1"] * 100 + ["HIV-1 protease"] * 5)
    assert not report["control_is_usable"]
    assert "limitation" in report["note"]


def test_control_becomes_usable_with_enough_distinct_non_kinase_targets():
    targets = ["ABL1"] * 50 + [f"HIV-{i} protease" for i in range(25)]
    report = confound_report(targets)
    assert report["control_is_usable"]
    assert report["distinct_non_kinase"] == 25


def test_report_counts_sum_to_the_total():
    report = confound_report(["ABL1", "HIV-1 protease", "O00141"])
    assert (report["n_kinase"] + report["n_non_kinase"]
            + report["n_unknown"]) == report["n_total"]


def test_committed_datasets_currently_have_no_control_arm():
    """Guard on the real state. Delete once the antiviral rebuild lands."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data/davis_ground_truth_sites.json"
    if not path.exists():
        pytest.skip("ground truth not present")
    report = confound_report(list(json.loads(path.read_text())))
    assert report["distinct_non_kinase"] == 0, (
        "DAVIS now has non-kinase targets -- the control arm changed, "
        "update this test and re-read the confound section of the guides"
    )
