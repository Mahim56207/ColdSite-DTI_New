"""
Tests for the accession -> gene-symbol path that makes the kinase-confound
control run on KIBA.

The failure this guards against is not a crash. If the mapping silently does
nothing, `target_family` reports zero non-kinase targets and
`control_is_usable: False`, which reads exactly like "KIBA has no suitable
targets" rather than "the map was never wired in". The audit would then be
published with a control arm covering one dataset out of two, and nothing in the
output would say so.
"""
import json

import pytest

from src.data.build_gene_map import build_mapping, coverage
from src.data.resolve_unmapped import unresolved_targets, write_template
from src.evaluation.target_family import (
    KINASE,
    NON_KINASE,
    UNKNOWN,
    classify_target,
    clear_gene_map,
    confound_report,
    get_gene_map,
    load_gene_map,
    set_gene_map,
    stratified_indices,
)


@pytest.fixture(autouse=True)
def _no_leaking_global_map():
    """The registry is module-level, so a test that registers a map would
    otherwise change the answers of every test that runs after it."""
    clear_gene_map()
    yield
    clear_gene_map()


# ---------------------------------------------------------------------------
# build_gene_map
# ---------------------------------------------------------------------------

def test_build_mapping_keeps_targets_that_resolved_to_a_name():
    provenance = {
        "O00141": {"gene_name": "SGK1", "protein_name": "Serine/threonine kinase"},
        "P00533": {"gene_name": "EGFR", "protein_name": "Epidermal growth factor receptor"},
    }
    mapping = build_mapping(provenance)
    assert set(mapping) == {"O00141", "P00533"}
    assert mapping["O00141"]["gene_name"] == "SGK1"


def test_build_mapping_omits_targets_with_no_name_at_all():
    """Absent, not blank. A blank entry classifies the same as a missing one but
    hides the fact that coverage is incomplete."""
    provenance = {
        "P00533": {"gene_name": "EGFR", "protein_name": ""},
        "Q99999": {"gene_name": "", "protein_name": ""},
        "BROKEN": None,
    }
    mapping = build_mapping(provenance)
    assert set(mapping) == {"P00533"}


def test_coverage_counts_the_gap_rather_than_hiding_it():
    provenance = {"A": {"gene_name": "EGFR", "protein_name": "x"},
                  "B": {"gene_name": "", "protein_name": ""}}
    stats = coverage(provenance, build_mapping(provenance))
    assert stats["targets_in_provenance"] == 2
    assert stats["targets_with_gene_symbol"] == 1
    assert stats["targets_unresolved"] == 1
    assert stats["pct_with_gene_symbol"] == 50.0


# ---------------------------------------------------------------------------
# target_family with a map
# ---------------------------------------------------------------------------

def test_kiba_accession_is_still_unknown_without_a_map():
    """The original behaviour, pinned. This is the state the project was in."""
    assert classify_target("O00141") == UNKNOWN


def test_kiba_accession_classifies_once_a_map_is_registered():
    set_gene_map({"O00141": "SGK3 Serine/threonine-protein kinase Sgk3"})
    assert classify_target("O00141") == KINASE


def test_symbol_alone_is_enough_when_the_description_never_says_kinase():
    """Some UniProt entries name the family only in the gene symbol. Matching
    the description alone would miss every one of them."""
    set_gene_map({"P00533": "EGFR Proto-oncogene c-ErbB-1"})
    assert classify_target("P00533") == KINASE


def test_non_kinase_still_wins_over_a_kinase_looking_symbol():
    """Contamination of the control arm is the one unrecoverable error, so the
    non-kinase check must keep priority even when a map is present."""
    set_gene_map({"X1": "MET 3C-like proteinase"})
    assert classify_target("X1") == NON_KINASE


def test_an_unmapped_accession_is_unaffected_by_a_partial_map():
    set_gene_map({"O00141": "SGK3 kinase"})
    assert classify_target("Q9UNKNOWN") == UNKNOWN


def test_explicit_map_argument_overrides_the_registry():
    set_gene_map({"X1": "EGFR kinase"})
    assert classify_target("X1", gene_map={}) == UNKNOWN
    assert classify_target("X1", gene_map={"X1": "HIV-1 protease"}) == NON_KINASE


def test_stratified_indices_uses_the_registry_too():
    """run_audit calls this several frames deep without passing a map, so the
    registry has to reach it or the stratified figure silently loses KIBA."""
    set_gene_map({"O00141": "SGK3 kinase", "P12345": "HIV-1 protease"})
    idx = stratified_indices(["O00141", "P12345", "ZZZ"])
    assert idx[KINASE] == [0]
    assert idx[NON_KINASE] == [1]
    assert idx[UNKNOWN] == [2]


def test_confound_report_flips_to_usable_once_the_map_lands():
    accessions = [f"P{i:05d}" for i in range(25)]
    before = confound_report(accessions)
    assert before["n_non_kinase"] == 0
    assert before["control_is_usable"] is False

    mapping = {a: "HIV-1 protease" for a in accessions}
    after = confound_report(accessions, gene_map=mapping)
    assert after["distinct_non_kinase"] == 25
    assert after["control_is_usable"] is True


# ---------------------------------------------------------------------------
# load_gene_map
# ---------------------------------------------------------------------------

def test_load_gene_map_accepts_both_file_shapes(tmp_path):
    rich = tmp_path / "rich.json"
    rich.write_text(json.dumps(
        {"O00141": {"gene_name": "SGK3", "protein_name": "kinase Sgk3"}}))
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"O00141": "SGK3 kinase"}))

    assert "SGK3" in load_gene_map(str(rich), register=False)["O00141"]
    assert "SGK3" in load_gene_map(str(flat), register=False)["O00141"]


def test_load_gene_map_registers_by_default(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"O00141": "SGK3 kinase"}))
    load_gene_map(str(path))
    assert get_gene_map()["O00141"] == "SGK3 kinase"
    assert classify_target("O00141") == KINASE


def test_load_gene_map_says_how_to_build_it_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_gene_map"):
        load_gene_map(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------
# resolve_unmapped
# ---------------------------------------------------------------------------

def test_unresolved_separates_no_entry_from_entry_without_sites():
    provenance = {
        "GOOD": {"resolution": "accession"},
        "MISSING": {"resolution": "not_found"},
        "BROKE": {"resolution": "failed"},
        "EMPTY": {"resolution": "accession"},
    }
    sites = {"GOOD": [{"start": 1, "end": 1}], "MISSING": [], "BROKE": [], "EMPTY": []}
    out = unresolved_targets(provenance, sites)
    assert out == {"MISSING": "not_found", "BROKE": "failed", "EMPTY": "no_sites"}
    assert "GOOD" not in out


def test_template_preserves_decisions_already_made(tmp_path):
    """A second --template run after a partial hand-edit must not wipe the
    accessions already filled in."""
    path = tmp_path / "overrides.json"
    provenance = {"T1": {"resolution": "not_found"}, "T2": {"resolution": "not_found"}}
    sites = {"T1": [], "T2": []}

    write_template(provenance, sites, str(path))
    edited = json.loads(path.read_text())
    edited["T1"]["uniprot_accession"] = "P00533"
    edited["T2"]["unresolvable"] = "no UniProt entry for this construct"
    path.write_text(json.dumps(edited))

    write_template(provenance, sites, str(path))
    again = json.loads(path.read_text())
    assert again["T1"]["uniprot_accession"] == "P00533"
    assert again["T2"]["unresolvable"] == "no UniProt entry for this construct"


def test_template_keeps_a_decision_for_a_target_that_now_resolves(tmp_path):
    path = tmp_path / "overrides.json"
    write_template({"T1": {"resolution": "not_found"}}, {"T1": []}, str(path))
    edited = json.loads(path.read_text())
    edited["T1"]["uniprot_accession"] = "P00533"
    path.write_text(json.dumps(edited))

    # T1 now resolves on its own, so it is no longer in the unresolved list
    write_template({"T1": {"resolution": "accession"}},
                   {"T1": [{"start": 1, "end": 1}]}, str(path))
    assert json.loads(path.read_text())["T1"]["uniprot_accession"] == "P00533"
