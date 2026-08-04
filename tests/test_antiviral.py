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


def test_every_required_target_has_a_pattern():
    assert len(REQUIRED_TARGETS) == 5


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
    """This is exactly the state the committed data/processed file is in."""
    with pytest.raises(RuntimeError, match="4 of 5 required targets"):
        verify_coverage(_frame(["HIV-1 protease"] * 614))


def test_verify_coverage_names_the_missing_targets():
    with pytest.raises(RuntimeError, match="Influenza neuraminidase"):
        verify_coverage(_frame([t for t in REQUIRED_TARGETS
                                if t != "Influenza neuraminidase"]))


def test_allow_partial_lets_a_deliberate_partial_run_through():
    counts = verify_coverage(_frame(["HIV-1 protease"] * 10), strict=False)
    assert counts["HIV-1 protease"] == 10


def test_committed_antiviral_file_is_still_incomplete():
    """Guard on the real artefact. Delete this test once the file is rebuilt."""
    import os
    path = "data/processed/antiviral_clean.csv"
    if not os.path.exists(path):
        pytest.skip("antiviral file not present")
    df = pd.read_csv(path)
    if "antiviral_target" not in df.columns:
        assert df["Target_ID"].nunique() < 5, (
            "file now has 5+ targets -- rebuild verified, update this test"
        )
