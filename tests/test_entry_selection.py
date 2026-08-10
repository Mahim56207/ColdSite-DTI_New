"""
Tests for how the fetcher picks among UniProt search results.

These are regression tests for two failures observed in a real DAVIS fetch, not
hypotheticals. With `size=1` and "take the first result":

    IKK-epsilon -> IKBKE -> Q96MC9  "Putative uncharacterized protein IKBKE-AS1"
    PRKCH       -> PRKCH -> C0HM02  "PRKCH upstream open reading frame 2"

Both are reviewed UniProt entries. Both are filed under a gene name that matches
the query. Neither is the kinase. Neither raised anything — each produced an
accession, a gene name, and an empty binding-site list, so the target scored
nothing and silently left the evaluation.

That is the worst failure shape in this project: not a crash, not a visible gap,
but a wrong answer wearing the costume of a right one.
"""
import pytest

from src.data.fetch_binding_sites import choose_entry


def entry(accession, genes, name, length, synonyms=()):
    record = {
        "primaryAccession": accession,
        "sequence": {"length": length},
        "proteinDescription": {"recommendedName": {"fullName": {"value": name}}},
        "genes": [{"geneName": {"value": genes[0]}}],
    }
    if synonyms:
        record["genes"][0]["synonyms"] = [{"value": s} for s in synonyms]
    for extra in genes[1:]:
        record["genes"].append({"geneName": {"value": extra}})
    return record


def test_antisense_transcript_does_not_win_over_the_kinase():
    """The IKBKE case. 'IKBKE-AS1' is a different symbol, so exact matching
    alone is enough here."""
    results = [
        entry("Q96MC9", ["IKBKE-AS1"], "Putative uncharacterized protein IKBKE-AS1", 106),
        entry("Q14164", ["IKBKE"], "Inhibitor of nuclear factor kappa-B kinase subunit epsilon", 716),
    ]
    chosen, how = choose_entry(results, "IKBKE")
    assert chosen["primaryAccession"] == "Q14164"
    assert how == "exact"


def test_upstream_orf_does_not_win_over_the_kinase():
    """The PRKCH case, which exact matching cannot solve: the uORF really is
    filed under gene PRKCH. Length is the only separator, and it is decisive —
    52 residues against 683."""
    results = [
        entry("C0HM02", ["PRKCH"], "PRKCH upstream open reading frame 2", 52),
        entry("P24723", ["PRKCH"], "Protein kinase C eta type", 683),
    ]
    chosen, how = choose_entry(results, "PRKCH")
    assert chosen["primaryAccession"] == "P24723"
    assert how.startswith("exact_longest_of_")


def test_result_order_does_not_decide_the_answer():
    """The original bug was order-dependence. Both orderings must agree."""
    good = entry("P24723", ["PRKCH"], "Protein kinase C eta type", 683)
    bad = entry("C0HM02", ["PRKCH"], "PRKCH upstream open reading frame 2", 52)
    assert choose_entry([bad, good], "PRKCH")[0] is good
    assert choose_entry([good, bad], "PRKCH")[0] is good


def test_a_matching_synonym_counts_as_exact():
    results = [entry("P00519", ["ABL1"], "Tyrosine-protein kinase ABL1", 1130,
                     synonyms=("JTK7",))]
    chosen, how = choose_entry(results, "JTK7")
    assert chosen["primaryAccession"] == "P00519"
    assert how == "exact"


def test_symbol_matching_is_case_insensitive():
    """M. tuberculosis PknB is filed lowercase; the alias table writes 'pknB'."""
    results = [entry("P9WI73", ["pknB"], "Serine/threonine-protein kinase PknB", 626)]
    assert choose_entry(results, "PKNB")[0]["primaryAccession"] == "P9WI73"


def test_no_symbol_match_is_labelled_inexact_not_silently_accepted():
    """Still returns a best guess, because a target with no entry at all is
    worse than one flagged for review — but the label has to make it auditable
    via `resolve_unmapped --audit`."""
    results = [entry("X11111", ["SOMETHINGELSE"], "Unrelated protein", 300)]
    chosen, how = choose_entry(results, "IKBKE")
    assert chosen["primaryAccession"] == "X11111"
    assert how == "inexact_of_1"


def test_empty_results_return_none():
    assert choose_entry([], "IKBKE") == (None, "none")


def test_entries_without_a_genes_block_do_not_crash():
    """Normal for some unreviewed and non-human records."""
    bare = {"primaryAccession": "Q00000", "sequence": {"length": 400}}
    chosen, how = choose_entry([bare], "IKBKE")
    assert chosen is bare
    assert how == "inexact_of_1"


def test_entries_without_a_sequence_block_do_not_crash():
    a = {"primaryAccession": "A", "genes": [{"geneName": {"value": "IKBKE"}}]}
    b = entry("B", ["IKBKE"], "real one", 716)
    assert choose_entry([a, b], "IKBKE")[0]["primaryAccession"] == "B"


@pytest.mark.parametrize("query", ["IKBKE", "ikbke", "  IKBKE  "])
def test_query_whitespace_and_case_are_normalised(query):
    results = [entry("Q14164", ["IKBKE"], "the kinase", 716)]
    assert choose_entry(results, query)[1] == "exact"
