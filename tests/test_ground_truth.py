"""Tests for the Track A -> Track C ground-truth bridge.

The off-by-one regression test at the bottom is the important one: it pins the
single failure mode that would silently cost the paper about a third of its
headline number while looking like a real result.
"""
import json

import numpy as np
import pytest

from src.data.ground_truth import (
    SiteSet,
    TruncationPolicy,
    build_site_set,
    coverage_report,
    expand_feature,
    is_variant_id,
    load_site_sets,
    normalise_target_id,
)
from src.evaluation.precision_at_k import precision_at_k


# --------------------------------------------------------------------------
# coordinate conversion
# --------------------------------------------------------------------------

def test_single_residue_shifts_by_one():
    # UniProt residue 41 is array index 40
    assert expand_feature({"start": 41, "end": 41}) == {40}


def test_inclusive_range_expands_to_every_position():
    # UniProt 41..43 inclusive -> indices 40, 41, 42
    assert expand_feature({"start": 41, "end": 43}) == {40, 41, 42}


def test_missing_end_is_treated_as_a_single_residue():
    assert expand_feature({"start": 100}) == {99}


def test_nested_uniprot_location_format_is_supported():
    feature = {"location": {"start": {"value": 5}, "end": {"value": 7}}}
    assert expand_feature(feature) == {4, 5, 6}


def test_reversed_range_is_normalised_not_dropped():
    assert expand_feature({"start": 43, "end": 41}) == {40, 41, 42}


def test_feature_with_no_position_is_dropped_silently():
    assert expand_feature({"description": "somewhere"}) == set()


def test_first_residue_maps_to_index_zero_not_minus_one():
    assert expand_feature({"start": 1, "end": 1}) == {0}


# --------------------------------------------------------------------------
# feature-type filtering
# --------------------------------------------------------------------------

def test_binding_and_active_sites_are_kept():
    features = [
        {"start": 10, "end": 10, "type": "Binding site"},
        {"start": 20, "end": 20, "type": "Active site"},
        {"start": 30, "end": 32, "type": "Nucleotide binding"},
    ]
    result = build_site_set("X", features)
    assert result.positions == {9, 19, 29, 30, 31}
    assert result.n_dropped_feature_type == 0


def test_uniprot_site_catch_all_type_is_rejected():
    features = [
        {"start": 10, "end": 10, "type": "Binding site"},
        {"start": 99, "end": 99, "type": "Site", "description": "Cleavage; by caspase-3"},
    ]
    result = build_site_set("X", features)
    assert result.positions == {9}
    assert result.n_dropped_feature_type == 1


def test_unrelated_feature_types_are_rejected():
    features = [{"start": 5, "end": 5, "type": "Modified residue"}]
    assert build_site_set("X", features).positions == set()


def test_description_fallback_used_only_when_type_is_absent():
    # the currently committed JSON has no 'type' key at all
    features = [
        {"start": 10, "end": 10, "description": "Proton acceptor"},
        {"start": 99, "end": 99, "description": "Cleavage; by caspase-3"},
        {"start": 50, "end": 50, "description": "Breakpoint for translocation"},
    ]
    result = build_site_set("X", features)
    assert result.positions == {9}
    assert result.n_dropped_description == 2


def test_description_filter_can_be_switched_off():
    features = [{"start": 99, "end": 99, "description": "Cleavage; by caspase-3"}]
    result = build_site_set("X", features, filter_descriptions=False)
    assert result.positions == {98}


# --------------------------------------------------------------------------
# truncation policy
# --------------------------------------------------------------------------

def _features_across_window():
    return [
        {"start": 100, "end": 100, "type": "Binding site"},    # index 99, inside
        {"start": 1500, "end": 1500, "type": "Binding site"},  # index 1499, outside
    ]


def test_exclude_policy_drops_out_of_window_sites():
    result = build_site_set("X", _features_across_window(), max_len=1000,
                            truncation=TruncationPolicy.EXCLUDE)
    assert result.positions == {99}
    assert result.n_dropped_truncation == 1


def test_keep_policy_retains_out_of_window_sites():
    result = build_site_set("X", _features_across_window(), max_len=1000,
                            truncation=TruncationPolicy.KEEP)
    assert result.positions == {99, 1499}
    assert result.n_dropped_truncation == 0


def test_error_policy_raises_loudly():
    with pytest.raises(ValueError, match="past max_len"):
        build_site_set("X", _features_across_window(), max_len=1000,
                       truncation=TruncationPolicy.ERROR)


def test_unknown_truncation_policy_is_rejected():
    with pytest.raises(ValueError, match="truncation must be one of"):
        build_site_set("X", [], truncation="whatever")


def test_range_straddling_the_window_keeps_only_the_visible_part():
    features = [{"start": 998, "end": 1003, "type": "Binding site"}]
    result = build_site_set("X", features, max_len=1000,
                            truncation=TruncationPolicy.EXCLUDE)
    assert result.positions == {997, 998, 999}
    assert result.n_dropped_truncation == 3


def test_protein_with_every_site_out_of_window_is_unusable():
    features = [{"start": 1500, "end": 1500, "type": "Binding site"}]
    result = build_site_set("X", features, max_len=1000)
    assert not result.usable


# --------------------------------------------------------------------------
# variant IDs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("ABL1(T315I)p", "ABL1"),
    ("ABL1(E255K)", "ABL1"),
    ("ABL1", "ABL1"),
    ("O00141", "O00141"),
    ("CDK4-cyclinD1", "CDK4-cyclinD1"),
])
def test_variant_id_normalisation(raw, expected):
    assert normalise_target_id(raw) == expected


def test_accession_ending_in_p_is_not_mangled():
    # only strip a trailing 'p' when a variant bracket was actually removed
    assert normalise_target_id("MAP2K1p") == "MAP2K1p"


def test_variant_flag_is_recorded():
    assert is_variant_id("ABL1(T315I)")
    assert not is_variant_id("ABL1")
    assert build_site_set("ABL1(T315I)", []).is_variant
    assert build_site_set("ABL1(T315I)", []).resolved_from == "ABL1"


# --------------------------------------------------------------------------
# file loading
# --------------------------------------------------------------------------

def test_load_site_sets_drops_unusable_by_default(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "GOOD": [{"start": 10, "end": 10, "type": "Binding site"}],
        "EMPTY": [],
        "ALL_FILTERED": [{"start": 5, "end": 5, "type": "Site"}],
    }))
    kept = load_site_sets(str(path))
    assert set(kept) == {"GOOD"}

    everything = load_site_sets(str(path), drop_unusable=False)
    assert set(everything) == {"GOOD", "EMPTY", "ALL_FILTERED"}


def test_coverage_report_counts_drops_from_removed_targets(tmp_path):
    """Regression: an earlier version summed counters over the FILTERED mapping,
    so a protein whose sites were all discarded took its own drop counts out of
    the total -- hiding the exact losses the report exists to surface."""
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "GOOD": [{"start": 10, "end": 10, "type": "Binding site"}],
        "ALL_TRUNCATED": [{"start": 5000, "end": 5002, "type": "Binding site"}],
    }))
    report = coverage_report(load_site_sets(str(path), max_len=1000,
                                            drop_unusable=False))
    assert report["targets_in_file"] == 2
    assert report["targets_usable"] == 1
    assert report["targets_dropped_entirely"] == 1
    assert report["dropped_truncation"] == 3     # not 0


def test_site_set_behaves_like_a_set_for_the_metric():
    site_set = SiteSet("X", positions={1, 2, 3})
    assert len(site_set) == 3
    assert 2 in site_set
    assert sorted(site_set) == [1, 2, 3]


# --------------------------------------------------------------------------
# THE regression test
# --------------------------------------------------------------------------

def test_off_by_one_regression(tmp_path):
    """A model with perfect attention must score exactly 1.0.

    UniProt says residues 41-43 and 143-145. If those numbers are fed to
    precision_at_k without the 1-indexed -> 0-indexed conversion, a perfect
    model scores 0.667 instead of 1.0. That is not a crash and not an outlier;
    it is a plausible-looking number that would go straight into the paper.
    """
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "TARGET": [
            {"start": 41, "end": 43, "type": "Binding site"},
            {"start": 143, "end": 145, "type": "Binding site"},
        ]
    }))

    sites = load_site_sets(str(path))["TARGET"]
    true_indices = [40, 41, 42, 142, 143, 144]
    assert sites.positions == set(true_indices)

    attention = np.zeros(300)
    attention[true_indices] = 1.0

    via_adapter = precision_at_k(attention, sites.positions, k=6)
    assert via_adapter == 1.0

    raw_uniprot_numbers = {41, 42, 43, 143, 144, 145}
    naive = precision_at_k(attention, raw_uniprot_numbers, k=6)
    assert naive == pytest.approx(4 / 6)
    assert naive < via_adapter
