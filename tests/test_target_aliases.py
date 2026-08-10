"""
Tests for the DAVIS-shorthand -> UniProt-gene-symbol alias table.

The alias table is the one place in the ground-truth pipeline where a human
asserts that two names refer to the same protein. If an assertion is wrong, the
consequence is not a crash -- it is another protein's binding sites attached to
this target, scored as correct answers, forever. So the tests here are less
about behaviour than about keeping the table honest: no accessions, no fuzzy
matching, no silent fallback, and every failure carrying the symbol that was
tried so it can be corrected rather than merely retried.
"""
import re

import pytest

from src.data.target_aliases import (
    DAVIS_TARGET_ALIASES,
    HUMAN,
    M_TUBERCULOSIS,
    P_FALCIPARUM,
    resolve_alias,
)


def test_the_three_failure_causes_are_each_covered():
    """One representative of each cause the module documents."""
    assert resolve_alias("ABL1p") == ("ABL1", HUMAN)              # phospho, no bracket
    assert resolve_alias("AMPK-alpha1") == ("PRKAA1", HUMAN)      # lab shorthand
    assert resolve_alias("PKNB(Mtuberculosis)") == ("pknB", M_TUBERCULOSIS)  # organism


def test_non_human_targets_do_not_get_the_human_taxon():
    """The bug this catches: stripping '(Pfalciparum)' as if it were a mutation
    bracket, then searching human UniProt for a malaria kinase."""
    for target in ("PFCDPK1(Pfalciparum)", "PFPK5(Pfalciparum)"):
        _, taxon = resolve_alias(target)
        assert taxon == P_FALCIPARUM
    assert resolve_alias("PKNB(Mtuberculosis)")[1] == M_TUBERCULOSIS


def test_unknown_targets_return_none_rather_than_a_guess():
    assert resolve_alias("EGFR") is None
    assert resolve_alias("O00141") is None
    assert resolve_alias("") is None


def test_matching_is_exact_not_fuzzy():
    """A near-miss must fall through to the normal path, not silently resolve.
    Fuzzy matching here is how one protein's sites end up on another target."""
    assert resolve_alias("AMPK-alpha") is None
    assert resolve_alias("p38") is None
    assert resolve_alias("ABL1") is None      # the real symbol needs no alias
    assert resolve_alias("abl1p") is None     # case-sensitive on purpose


def test_whitespace_is_tolerated():
    assert resolve_alias("  ABL1p  ") == ("ABL1", HUMAN)


def test_no_entry_is_a_uniprot_accession():
    """Every value must be a gene symbol handed to UniProt's search. An
    accession would resolve to *something* even when wrong, and wrong-but-
    resolving is the failure mode this table exists to avoid."""
    accession = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
    for target, (symbol, _) in DAVIS_TARGET_ALIASES.items():
        assert not accession.match(symbol), (
            f"{target} maps to {symbol}, which looks like a UniProt accession. "
            f"Use a gene symbol so a wrong entry fails loudly."
        )


def test_every_alias_has_a_real_symbol_and_a_real_taxon():
    for target, value in DAVIS_TARGET_ALIASES.items():
        assert isinstance(value, tuple) and len(value) == 2, target
        symbol, taxon = value
        assert isinstance(symbol, str) and symbol.strip(), target
        assert isinstance(taxon, int) and taxon > 0, target


def test_the_p38_family_maps_to_four_distinct_genes():
    """p38 alpha/beta/gamma/delta are MAPK14/11/12/13. Collapsing any two of
    them onto one gene would give two DAVIS targets identical site lists and
    quietly halve the diversity of the evaluation set."""
    genes = {resolve_alias(f"p38-{greek}")[0]
             for greek in ("alpha", "beta", "gamma", "delta")}
    assert genes == {"MAPK14", "MAPK11", "MAPK12", "MAPK13"}


def test_the_cdk4_complexes_deliberately_share_one_gene():
    """Both cyclin D complexes resolve to CDK4 on purpose -- the kinase subunit
    carries the pocket. This is the same wild-type approximation applied to the
    ABL1 point mutants, and it is pinned here so it stays a documented decision
    rather than becoming an unnoticed duplicate."""
    assert resolve_alias("CDK4-cyclinD1") == resolve_alias("CDK4-cyclinD3")
    assert resolve_alias("CDK4-cyclinD1")[0] == "CDK4"


@pytest.mark.parametrize("target", sorted(DAVIS_TARGET_ALIASES))
def test_every_alias_is_reachable_through_the_public_function(target):
    assert resolve_alias(target) == DAVIS_TARGET_ALIASES[target]


def test_alias_failures_are_visible_to_the_resolver():
    """An alias that UniProt does not recognise must land in the unresolved
    list, not disappear. It is a wrong guess, which is worse than a missing one,
    and the symbol that was tried has to survive into the report."""
    from src.data.resolve_unmapped import unresolved_targets

    provenance = {
        "GOOD": {"resolution": "alias(PRKAA1)"},
        "BAD": {"resolution": "alias_not_found(NOTAGENE)"},
        "ERRORED": {"resolution": "alias_failed(PRKAA1)"},
    }
    sites = {"GOOD": [{"start": 1, "end": 1}], "BAD": [], "ERRORED": []}
    out = unresolved_targets(provenance, sites)

    assert "GOOD" not in out
    assert out["BAD"] == "alias_not_found(NOTAGENE)"
    assert out["ERRORED"] == "alias_failed(PRKAA1)"
    assert "NOTAGENE" in out["BAD"], "the attempted symbol must survive"
