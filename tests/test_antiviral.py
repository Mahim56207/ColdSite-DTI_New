"""Tests for the antiviral subset builder.

The previous pipeline wrote a file containing 1 of the 5 required targets and
reported success. These tests pin the matcher against the naming variants
BindingDB actually uses, so a silent single-target file cannot recur.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.extract_antiviral import (
    REQUIRED_TARGETS,
    _parse_affinity,
    classify_target,
    to_p_scale,
    verify_coverage,
)


# --------------------------------------------------------------------------
# target matching
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "SARS-CoV-2 3C-like proteinase",
    "SARS-CoV-2 main protease",
    "3C-like proteinase (SARS-CoV-2)",
    "SARS-CoV-2 nsp5",
    "sars cov 2 Mpro",
    "2019-nCoV main protease",
])
def test_mpro_naming_variants_all_match(name):
    assert classify_target(name) == "SARS-CoV-2 Mpro"


@pytest.mark.parametrize("name", [
    "SARS-CoV-2 RNA-dependent RNA polymerase",
    "SARS-CoV-2 RdRp",
    "SARS-CoV-2 nsp12",
    "RNA-directed RNA polymerase (SARS-CoV-2)",
])
def test_rdrp_naming_variants_all_match(name):
    assert classify_target(name) == "SARS-CoV-2 RdRp"


@pytest.mark.parametrize("name", [
    "HIV-1 protease",
    "HIV protease",
    "Protease (HIV-1)",
    "Human immunodeficiency virus type 1 protease",
])
def test_hiv_protease_variants_all_match(name):
    assert classify_target(name) == "HIV-1 protease"


@pytest.mark.parametrize("name", [
    "HIV-1 reverse transcriptase",
    "Reverse transcriptase (HIV-1)",
    "Human immunodeficiency virus type 1 reverse transcriptase",
])
def test_hiv_rt_variants_all_match(name):
    assert classify_target(name) == "HIV-1 reverse transcriptase"


@pytest.mark.parametrize("name", [
    "Influenza A neuraminidase",
    "Neuraminidase (H1N1)",
    "H5N1 neuraminidase",
    "Neuraminidase",
])
def test_neuraminidase_variants_all_match(name):
    assert classify_target(name) == "Influenza neuraminidase"


def test_hiv_protease_and_rt_are_not_confused():
    """Both names contain 'HIV-1'; the discriminator is the enzyme word."""
    assert classify_target("HIV-1 protease") == "HIV-1 protease"
    assert classify_target("HIV-1 reverse transcriptase") == "HIV-1 reverse transcriptase"


@pytest.mark.parametrize("name", [
    "Epidermal growth factor receptor",
    "Carbonic anhydrase II",
    "Beta-secretase 1",
    "Hepatitis C virus NS5B polymerase",
    "SARS-CoV-2 spike glycoprotein",
])
def test_unrelated_targets_are_not_matched(name):
    assert classify_target(name) is None


def test_non_string_target_names_do_not_crash():
    assert classify_target(None) is None
    assert classify_target(float("nan")) is None
    assert classify_target(12345) is None


def test_all_five_targets_still_have_a_spec():
    """The SARS-CoV-2 specs stay in place even though they are not required, so
    a future BindingDB release that names the domains properly is picked up
    without anyone having to remember to re-add them."""
    from src.data.extract_antiviral import ALL_TARGETS
    assert len(ALL_TARGETS) == 5


def test_sars_cov_2_is_optional_and_the_reason_is_recorded():
    """Not a suppressed failure. BindingDB files SARS-CoV-2 against the 7,096-
    residue replicase polyprotein, so Mpro and RdRp cannot be told apart by
    target name -- 3 rows in 18,149 carry a domain range. The reason travels
    with the code because it has to end up in the Methods section."""
    from src.data.extract_antiviral import (
        OPTIONAL_TARGETS,
        OPTIONAL_TARGET_REASON,
    )
    assert set(OPTIONAL_TARGETS) == {"SARS-CoV-2 Mpro", "SARS-CoV-2 RdRp"}
    assert set(REQUIRED_TARGETS) == {
        "HIV-1 protease", "HIV-1 reverse transcriptase", "Influenza neuraminidase"}
    assert "polyprotein" in OPTIONAL_TARGET_REASON
    assert "assay" in OPTIONAL_TARGET_REASON.lower()


# --------------------------------------------------------------------------
# organism-aware matching
#
# Every case below is a real string from a real BindingDB release (2026-07-31),
# taken from the near-miss diagnostic of a scan that found only 2 of 5 targets.
# The three missing targets are named bare -- "3C-like protease",
# "RNA-directed RNA polymerase", "Reverse transcriptase" -- with the organism
# living in a separate column.
#
# The second block is the reason this is not solved by loosening the name
# patterns. Those are the strings a loose polymerase-or-transcriptase pattern
# would swallow, and every one of them is a non-kinase protein that would land
# in the antiviral subset. That subset is the kinase-confound control arm, so
# contaminating it does not merely add noise -- it produces a control that
# appears to answer the objection the v2 master plan calls "most likely to sink
# the paper" while actually answering nothing.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,organism,expected", [
    ("3C-like protease", "Severe acute respiratory syndrome coronavirus 2", "SARS-CoV-2 Mpro"),
    ("3C-like proteinase", "2019-nCoV", "SARS-CoV-2 Mpro"),
    ("nsp5", "SARS-CoV-2", "SARS-CoV-2 Mpro"),
    ("RNA-directed RNA polymerase", "Severe acute respiratory syndrome coronavirus 2", "SARS-CoV-2 RdRp"),
    ("nsp12", "SARS-CoV-2", "SARS-CoV-2 RdRp"),
    ("Protease", "Human immunodeficiency virus 1", "HIV-1 protease"),
    ("Reverse transcriptase", "Human immunodeficiency virus 1", "HIV-1 reverse transcriptase"),
    ("Reverse transcriptase/RNaseH", "Human immunodeficiency virus 1", "HIV-1 reverse transcriptase"),
    ("Neuraminidase", "Influenza A virus", "Influenza neuraminidase"),
])
def test_bare_names_classify_once_the_organism_column_is_supplied(name, organism, expected):
    assert classify_target(name, organism) == expected


@pytest.mark.parametrize("name,organism", [
    # human enzymes a loose 'polymerase' pattern eats -- 8,065 rows for PARP1
    # alone in the release this was built against
    ("Poly [ADP-ribose] polymerase 1", "Homo sapiens"),
    ("Poly [ADP-ribose] polymerase tankyrase-2", "Homo sapiens"),
    ("DNA polymerase theta", "Homo sapiens"),
    ("DNA polymerase theta [1-894]", "Homo sapiens"),
    ("DNA-directed RNA polymerase, mitochondrial", "Homo sapiens"),
    ("Telomerase reverse transcriptase", "Homo sapiens"),
    ("Sialidase-1", "Homo sapiens"),
    # right family, wrong pathogen
    ("RNA-directed RNA polymerase", "Hepatitis C virus"),
    ("RNA-directed RNA polymerase L", "Respiratory syncytial virus"),
    ("Reverse transcriptase protein", "Hepatitis B virus"),
    ("Reverse transcriptase", "Moloney murine leukemia virus"),
    ("Sialidase", "Vibrio cholerae"),
    ("Neuraminidase", "Newcastle disease virus"),
    ("DNA polymerase catalytic subunit", "Human herpesvirus 1"),
    # right pathogen family, wrong species -- SARS-CoV-1 and MERS both have a
    # 3C-like protease, and it is a different protein with a different pocket
    ("3C-like protease", "SARS coronavirus"),
    ("3C-like protease", "Middle East respiratory syndrome coronavirus"),
    ("3C protease", "Human rhinovirus"),
    ("Protease", "Human immunodeficiency virus 2"),
    # influenza, but the wrong influenza protein
    ("Polymerase basic protein 2", "Influenza A virus"),
    ("Polymerase acidic protein", "Influenza A virus"),
])
def test_near_neighbours_are_rejected_even_with_the_organism(name, organism):
    assert classify_target(name, organism) is None


@pytest.mark.parametrize("name", [
    "HIV WT-C pol protein (wild-type clade C)",
    "HIV WT-A pol protein (wild-type clade A)",
])
def test_pol_polyprotein_is_claimed_by_neither_bucket(name):
    """gag-pol carries protease, RT and integrase in one chain. Assigning it to
    either target would be a guess, and a guess in the control arm is the one
    error this project cannot undo later."""
    assert classify_target(name, "Human immunodeficiency virus 1") is None


def test_a_contradicting_organism_beats_a_permissive_name():
    """'Neuraminidase' alone is allowed through as influenza, since that is
    what a bare neuraminidase in BindingDB overwhelmingly is. But the exemption
    must not survive an organism that says otherwise."""
    assert classify_target("Neuraminidase") == "Influenza neuraminidase"
    assert classify_target("Neuraminidase", "Homo sapiens") is None
    assert classify_target("Neuraminidase", "Clostridium perfringens") is None


def test_organism_defaults_to_empty_so_self_describing_names_still_work():
    assert classify_target("SARS-CoV-2 main protease") == "SARS-CoV-2 Mpro"
    assert classify_target("HIV-1 reverse transcriptase") == "HIV-1 reverse transcriptase"


def test_non_string_organism_does_not_crash():
    assert classify_target("Protease", None) is None
    assert classify_target("Protease", float("nan")) is None


# --------------------------------------------------------------------------
# affinity handling
# --------------------------------------------------------------------------

def test_p_scale_conversion_matches_the_davis_convention():
    assert to_p_scale(10) == pytest.approx(8.0)      # 10 nM  -> pKd 8
    assert to_p_scale(1) == pytest.approx(9.0)       # 1 nM   -> pKd 9
    assert to_p_scale(10_000) == pytest.approx(5.0)  # 10 uM  -> pKd 5


def test_p_scale_rejects_non_positive_values():
    assert np.isnan(to_p_scale(0))
    assert np.isnan(to_p_scale(-5))
    assert np.isnan(to_p_scale(float("nan")))


def test_p_scale_compresses_the_range_that_would_dominate_mse():
    """Raw nM spans 10 orders of magnitude; p-scale spans about 10 units."""
    raw = [7.9e-4, 4.6, 1e7]
    converted = [to_p_scale(v) for v in raw]
    assert max(raw) / min(raw) > 1e9
    assert max(converted) - min(converted) < 15


@pytest.mark.parametrize("raw,expected", [
    (">10000", 10000.0),
    ("<0.5", 0.5),
    ("~25", 25.0),
    (" 42 ", 42.0),
    ("", np.nan),
    ("n/a", np.nan),
    (None, np.nan),
])
def test_affinity_qualifiers_are_stripped(raw, expected):
    result = _parse_affinity(raw)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


# --------------------------------------------------------------------------
# coverage verification
# --------------------------------------------------------------------------

def _frame(targets):
    return pd.DataFrame({"antiviral_target": targets})


def test_verify_coverage_passes_when_all_five_present():
    counts = verify_coverage(_frame(list(REQUIRED_TARGETS) * 3))
    assert all(counts[t] == 3 for t in REQUIRED_TARGETS)


def test_verify_coverage_raises_on_the_single_target_file_that_shipped():
    """This is exactly the state the committed data/processed file was in:
    614 rows, all HIV-1 protease. Two of the three required targets absent."""
    with pytest.raises(RuntimeError, match="2 of 3 required targets"):
        verify_coverage(_frame(["HIV-1 protease"] * 614))


def test_missing_sars_cov_2_alone_does_not_block_the_write():
    """The three obtainable targets are enough to write the case-study file.
    Blocking on a target that provably cannot be extracted from this release
    would only push someone toward --allow-partial, which is worse."""
    counts = verify_coverage(_frame(
        ["HIV-1 protease"] * 5
        + ["HIV-1 reverse transcriptase"] * 5
        + ["Influenza neuraminidase"] * 5))
    assert counts["HIV-1 protease"] == 5
    assert "SARS-CoV-2 Mpro" not in counts


def test_verify_coverage_names_the_missing_targets():
    with pytest.raises(RuntimeError, match="Influenza neuraminidase"):
        verify_coverage(_frame([t for t in REQUIRED_TARGETS
                                if t != "Influenza neuraminidase"]))


def test_allow_partial_lets_a_deliberate_partial_run_through():
    counts = verify_coverage(_frame(["HIV-1 protease"] * 10), strict=False)
    assert counts["HIV-1 protease"] == 10


def test_committed_antiviral_file_is_still_incomplete():
    """Guard on the real artefact. Delete this test once the file is rebuilt.

    This test is DESIGNED to fail the moment Track A's antiviral rebuild
    succeeds. That is not a regression -- it is the guard doing its job. See
    the assertion message for what to do.
    """
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data/processed/antiviral_clean.csv"
    if not path.exists():
        pytest.skip("antiviral file not present")
    df = pd.read_csv(path)
    if "antiviral_target" not in df.columns:
        assert df["Target_ID"].nunique() < 5, (
            "GOOD NEWS, NOT A BUG: data/processed/antiviral_clean.csv now has "
            "5+ distinct targets, so Track A's antiviral rebuild has "
            "succeeded.\n"
            "This test is a deliberate guard on the old broken artefact (614 "
            "rows, all HIV-1 protease) and is supposed to fail at exactly this "
            "point.\n"
            "ACTION: delete this test. Then confirm the control arm with\n"
            "  python -c \"import pandas as pd; "
            "from src.evaluation.target_family import confound_report; "
            "print(confound_report(pd.read_csv("
            "'data/processed/antiviral_clean.csv')['Target_ID'].tolist()))\"\n"
            "and check control_is_usable is True "
            "(docs/PART2_GUIDE_124AD0008.md Priority 1)."
        )
