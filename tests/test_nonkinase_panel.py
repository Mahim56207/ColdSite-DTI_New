"""
Tests for the non-kinase control panel.

What the panel is for
---------------------
DAVIS and KIBA are kinase panels. The v2 master plan names the resulting
confound "the objection most likely to sink the paper": a cold-target kinase
still shares the ATP pocket of the several hundred kinases already in training,
so a good cold-target precision@k could be pocket recognition rather than an
explanation generalising. The control arm is a set of proteins that do *not*
share that pocket.

The plan's arm was five antiviral proteins. `confound_report` gates the control
at >=20 distinct non-kinase targets, so five could never clear it, and a real
BindingDB release made it worse: all 18,149 SARS-CoV-2 rows are filed under the
7,096-residue "Replicase polyprotein 1ab", leaving three usable proteins.

The subtle failure this file mostly guards
------------------------------------------
"Non-kinase" is not the property that matters. *Not binding a nucleotide* is.
HSP90, DNA gyrase B, helicases, myosins and NADPH-dependent oxidoreductases are
all non-kinases with nucleotide pockets. Admitting them would put the very thing
being controlled for inside the control arm, and the stratified panel would then
show "no difference between families" for a reason with nothing to do with the
paper's claim — an answer that looks like evidence and is not.

That check runs against UniProt's annotated ligand, not the protein's name,
because the name does not reliably say: "Heat shock protein HSP 90-alpha"
mentions no nucleotide anywhere.
"""
import json

import pytest

from src.data.build_nonkinase_panel import (
    MAX_SEQUENCE_LENGTH,
    binding_ligands,
    panel_verdict,
)
from src.evaluation.target_family import (
    KINASE,
    NON_KINASE,
    UNKNOWN,
    classify_target,
    clear_family_map,
    confound_report,
    load_family_map,
    set_family_map,
)


def sites(*ligands):
    return {"features": [{"type": "Binding site", "ligand": {"name": name}}
                         for name in ligands]}


@pytest.fixture(autouse=True)
def _no_leaking_registry():
    clear_family_map()
    yield
    clear_family_map()


# ---------------------------------------------------------------------------
# what belongs in the arm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gene,protein,ligand", [
    ("PTGS2", "Prostaglandin G/H synthase 2", "heme"),
    ("CA2", "Carbonic anhydrase 2", "Zn(2+)"),
    ("BRD4", "Bromodomain-containing protein 4", "acetyl-lysine"),
    ("HDAC1", "Histone deacetylase 1", "Zn(2+)"),
    ("ESR1", "Estrogen receptor", "estradiol"),
    ("F2", "Prothrombin", "Na(+)"),
    ("ACHE", "Acetylcholinesterase", "choline"),
])
def test_genuine_non_kinases_are_kept(gene, protein, ligand):
    usable, reason = panel_verdict(gene, protein, sites(ligand))
    assert usable, reason


# ---------------------------------------------------------------------------
# kinases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gene,protein", [
    ("EGFR", "Epidermal growth factor receptor"),
    ("PDPK1", "3-phosphoinositide-dependent protein kinase 1"),
    ("PIK3CA", "PI3-kinase subunit alpha"),
    ("CSNK1D", "Casein kinase I isoform delta"),
])
def test_kinases_are_rejected(gene, protein):
    usable, reason = panel_verdict(gene, protein, sites("heme"))
    assert not usable
    assert reason == "kinase"


# ---------------------------------------------------------------------------
# the real trap: non-kinases that bind a nucleotide anyway
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gene,protein,ligand", [
    ("HSP90AA1", "Heat shock protein HSP 90-alpha", "ATP"),
    ("GYRB", "DNA gyrase subunit B", "ATP"),
    ("DDX3X", "ATP-dependent RNA helicase DDX3X", "ATP"),
    ("MYH7", "Myosin-7", "ATP"),
    ("POLB", "DNA polymerase beta", "dCTP"),
    ("NUDT1", "Nudix hydrolase 1", "8-oxo-dGTP"),
    ("IMPDH2", "Inosine-5'-monophosphate dehydrogenase 2", "NAD(+)"),
    ("DHFR", "Dihydrofolate reductase", "NADPH"),
    ("MAT2A", "Methionine adenosyltransferase 2A", "S-adenosyl-L-methionine"),
])
def test_nucleotide_binders_are_rejected_even_though_they_are_not_kinases(
        gene, protein, ligand):
    usable, reason = panel_verdict(gene, protein, sites(ligand))
    assert not usable, f"{gene} would contaminate the control arm"
    assert reason.startswith("binds_nucleotide")


@pytest.mark.parametrize("ligand", [
    "dGTP", "8-oxo-dGTP", "2'-deoxy-ATP", "AMP-PNP", "dCTP", "dTTP",
])
def test_deoxy_and_modified_forms_are_caught(ligand):
    """`\\bGTP\\b` cannot match inside "dGTP" -- `d` is a word character. A
    left-hand word boundary here would let every deoxynucleotide binder through
    while appearing to test for them."""
    usable, reason = panel_verdict("X", "Some protein", sites(ligand))
    assert not usable, f"{ligand} slipped through"


@pytest.mark.parametrize("ligand", ["NADPH", "NADH", "NADP(+)", "FADH2"])
def test_reduced_cofactor_forms_are_caught(ligand):
    """The trailing-letter guard rejects "NADPH" if only "NADP" is listed."""
    assert not panel_verdict("X", "Some protein", sites(ligand))[0]


def test_one_nucleotide_site_disqualifies_the_whole_target():
    """A protein with a drug pocket AND a nucleotide pocket is still a target
    whose attention could be riding on nucleotide-pocket recognition."""
    entry = sites("heme", "Zn(2+)", "ATP")
    usable, reason = panel_verdict("X", "Multi-site protein", entry)
    assert not usable
    assert reason.startswith("binds_nucleotide")


def test_an_entry_with_no_recommended_name_is_rejected():
    """No name means no evidence either way, and a control arm is the wrong
    place to guess."""
    usable, reason = panel_verdict("Q9XXXX", "", sites("heme"))
    assert not usable
    assert reason == "unnamed_entry"


def test_binding_ligands_ignores_non_binding_feature_types():
    entry = {"features": [
        {"type": "Binding site", "ligand": {"name": "heme"}},
        {"type": "Chain", "ligand": {"name": "ATP"}},
        {"type": "Site", "ligand": {"name": "ATP"}},
    ]}
    assert binding_ligands(entry) == ["heme"]


def test_binding_ligands_survives_missing_blocks():
    assert binding_ligands({}) == []
    assert binding_ligands({"features": [{"type": "Binding site"}]}) == []


# ---------------------------------------------------------------------------
# the family map -- how the panel's verdict reaches confound_report
# ---------------------------------------------------------------------------

def test_an_accession_is_unknown_without_an_assignment():
    """The regexes classify kinases by naming convention and a short antiviral
    list. A prostaglandin synthase is neither, so it reads UNKNOWN and counts
    toward no arm at all. This is why the assignment has to be explicit."""
    assert classify_target("P35354") == UNKNOWN


def test_an_explicit_assignment_is_honoured():
    set_family_map({"P35354": NON_KINASE})
    assert classify_target("P35354") == NON_KINASE


def test_an_explicit_assignment_beats_the_regexes():
    """The builder had the full UniProt entry and its annotated ligands. That
    is strictly more evidence than a name-shaped guess."""
    set_family_map({"ABL1": NON_KINASE})
    assert classify_target("ABL1") == NON_KINASE
    clear_family_map()
    assert classify_target("ABL1") == KINASE


def test_the_panel_flips_the_gate_that_five_antivirals_could_not():
    accessions = [f"P{i:05d}" for i in range(45)]
    assert confound_report(accessions)["control_is_usable"] is False

    set_family_map({a: NON_KINASE for a in accessions})
    report = confound_report(accessions)
    assert report["distinct_non_kinase"] == 45
    assert report["control_is_usable"] is True


def test_five_antiviral_targets_still_do_not_clear_the_gate():
    """Pins the finding that motivated the panel: the master plan's control arm
    cannot pass its own threshold, and no amount of extraction work fixes it."""
    five = ["HIV-1 protease", "HIV-1 reverse transcriptase",
            "Influenza neuraminidase", "SARS-CoV-2 Mpro", "SARS-CoV-2 RdRp"]
    report = confound_report(five)
    assert report["n_non_kinase"] == 5
    assert report["control_is_usable"] is False


def test_load_family_map_round_trips(tmp_path):
    path = tmp_path / "families.json"
    path.write_text(json.dumps({"P35354": "non_kinase"}))
    load_family_map(str(path))
    assert classify_target("P35354") == NON_KINASE


def test_load_family_map_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_nonkinase_panel"):
        load_family_map(str(tmp_path / "missing.json"))


# ---------------------------------------------------------------------------
# ligands travel with the ground truth
# ---------------------------------------------------------------------------

def test_features_keep_their_ligand_and_their_coordinates():
    from src.data.build_nonkinase_panel import features_with_ligands

    entry = {"features": [
        {"type": "Binding site",
         "location": {"start": {"value": 10}, "end": {"value": 10}},
         "ligand": {"name": "Na(+)"}},
        {"type": "Binding site",
         "location": {"start": {"value": 20}, "end": {"value": 22}},
         "ligand": {"name": "serotonin"}},
        {"type": "Chain",
         "location": {"start": {"value": 1}, "end": {"value": 600}},
         "ligand": {"name": "ATP"}},
    ]}
    features = features_with_ligands(entry)
    assert len(features) == 2, "non-binding feature types must not be kept"
    assert features[0]["ligand"] == "Na(+)"
    assert features[1]["start"] == 20 and features[1]["end"] == 22


def test_ligands_are_reported_but_never_filtered():
    """The panel must not silently drop ion sites. Zinc in carbonic anhydrase
    and in the HDACs IS the drug site -- their inhibitors chelate it directly --
    so a blanket metal-site rule would be as wrong as counting every ion. The
    call is per-protein, so it is surfaced for a human rather than decided by a
    regex."""
    from src.data.build_nonkinase_panel import features_with_ligands

    entry = {"features": [
        {"type": "Binding site",
         "location": {"start": {"value": 94}, "end": {"value": 94}},
         "ligand": {"name": "Zn(2+)"}},
    ]}
    assert len(features_with_ligands(entry)) == 1


def test_features_without_a_location_are_skipped_not_defaulted():
    from src.data.build_nonkinase_panel import features_with_ligands

    entry = {"features": [
        {"type": "Binding site", "location": {}, "ligand": {"name": "Zn(2+)"}},
    ]}
    assert features_with_ligands(entry) == []


def test_the_window_limit_is_the_models_not_an_arbitrary_number():
    """A site past the model's input window can never be retrieved, so the
    protein scores a guaranteed zero for a reason unrelated to explanation
    quality. The panel is kept inside the window instead."""
    assert MAX_SEQUENCE_LENGTH == 1000
