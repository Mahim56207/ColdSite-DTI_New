"""
Shared bridge between Track A's UniProt ground-truth JSON and Track C's
precision@k evaluation.

This module exists because the two sides of the project speak different
coordinate systems, and nothing was translating between them:

    Track A  (UniProt)  -> 1-indexed, inclusive [start, end] residue RANGES
    Track C  (metric)   -> 0-indexed set of individual array POSITIONS

Feeding UniProt positions straight into `precision_at_k` costs roughly a third
of the score on a model with perfect attention (see tests/test_ground_truth.py
::test_off_by_one_regression). That error is silent and looks exactly like a
real scientific finding, so every consumer of the ground truth must go through
this module rather than reading the JSON directly.

Usage
-----
    from src.data.ground_truth import load_site_sets

    sites = load_site_sets("data/davis_ground_truth_sites.json", max_len=1000)
    positions = sites["AAK1"].positions        # 0-indexed set, ready for precision@k
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# feature-type filtering
# ---------------------------------------------------------------------------

# UniProt feature types that describe a place where something actually binds.
# 'Nucleotide binding' is included deliberately: DAVIS and KIBA are kinase
# panels and the ATP pocket is annotated under that type, not under
# 'Binding site'.
BINDING_FEATURE_TYPES = frozenset({
    "Binding site",
    "Active site",
    "Nucleotide binding",
})

# UniProt's 'Site' type is a catch-all. It carries genuine binding annotations
# alongside things like protease cleavage points and chromosomal breakpoints,
# which are not drug pockets and inflate precision@k if counted as hits.
# It is NOT in BINDING_FEATURE_TYPES.
EXCLUDED_FEATURE_TYPES = frozenset({"Site"})

# Fallback only. Ground-truth files generated before the fetcher recorded
# feature types have no 'type' key, so the only lever left is the description
# string. This is a stopgap with known false negatives -- regenerate the JSON
# with src/data/fetch_binding_sites.py to filter properly on type instead.
NON_BINDING_DESCRIPTION = re.compile(
    r"cleavage|breakpoint|translocation|proteolytic|"
    r"autoinhibit|dimeri[sz]|interaction with|required for interaction",
    re.IGNORECASE,
)


class TruncationPolicy:
    """What to do with annotated sites that fall past the model's input window.

    The model truncates proteins at `max_protein_len` residues, so attention
    simply does not exist past that point. A site beyond the cut can never be
    retrieved.

    Note the asymmetry, which is easy to get wrong: dropping or keeping these
    sites does NOT change raw precision@k, because precision counts hits among
    the top-k *attended* positions and there is no attention out there. What it
    changes is (a) whether the protein is evaluated at all, and (b) the
    achievable ceiling. A protein whose every annotated site is past the cut
    scores a guaranteed 0.0, which deflates the split average for a reason that
    has nothing to do with explanation quality.

    EXCLUDE  -- drop out-of-window sites; drop the protein entirely if none
                remain. This is the honest default: it measures explanation
                quality on the residues the model can actually see.
    KEEP     -- retain out-of-window sites. The protein stays in the average
                and its ceiling reflects the unreachable sites.
    ERROR    -- raise. Use when you believe truncation should never bite.

    Whichever you pick must be stated in the Methods section. Silently
    switching between them changes the headline number.
    """

    EXCLUDE = "exclude"
    KEEP = "keep"
    ERROR = "error"

    ALL = (EXCLUDE, KEEP, ERROR)


# ---------------------------------------------------------------------------
# ID normalisation
# ---------------------------------------------------------------------------

_VARIANT_SUFFIX = re.compile(r"\s*\(.*?\)\s*")
_PHOSPHO_SUFFIX = re.compile(r"p$")


def normalise_target_id(target_id: str) -> str:
    """DAVIS-style variant IDs -> the base gene symbol UniProt was queried with.

    'ABL1(T315I)p' -> 'ABL1',  'ABL1' -> 'ABL1',  'O00141' -> 'O00141'

    DAVIS names point mutants and phosphorylated forms of the same kinase
    separately. UniProt has no entry for those, so the fetcher resolves them to
    the wild-type accession -- which means every ABL1 variant inherits an
    identical site list.
    """
    base = _VARIANT_SUFFIX.sub("", str(target_id)).strip()
    if base != str(target_id).strip():
        # only strip a trailing 'p' when a variant bracket was actually removed,
        # so real accessions ending in 'p' are left alone
        base = _PHOSPHO_SUFFIX.sub("", base)
    return base


def is_variant_id(target_id: str) -> bool:
    """True if this ID names a mutant/phospho form rather than the wild type."""
    return normalise_target_id(target_id) != str(target_id).strip()


# ---------------------------------------------------------------------------
# the site set
# ---------------------------------------------------------------------------

@dataclass
class SiteSet:
    """0-indexed annotated positions for one protein, plus what was discarded.

    `positions` is what `precision_at_k` consumes. The counters exist so the
    paper can report exactly how much ground truth was dropped and why, rather
    than quietly losing it.
    """

    target_id: str
    positions: set = field(default_factory=set)
    n_raw_features: int = 0
    n_dropped_feature_type: int = 0
    n_dropped_description: int = 0
    # Sites excluded because of the ligand they coordinate -- cotransport
    # sodium and chloride above all. Zero unless exclude_ligands was passed.
    n_dropped_ligand: int = 0
    n_dropped_truncation: int = 0
    # Positions past max_len, counted regardless of policy. n_dropped_truncation
    # only increments under EXCLUDE, because only then are they removed; this
    # records that truncation BIT at all, which is what Methods has to report.
    n_out_of_window: int = 0
    # True when the window is the sole reason this target has nothing usable.
    # Decided here rather than in coverage_report, which does not know max_len.
    dropped_by_truncation: bool = False
    is_variant: bool = False
    resolved_from: str = ""

    def __len__(self) -> int:
        return len(self.positions)

    def __contains__(self, pos) -> bool:
        return pos in self.positions

    def __iter__(self):
        return iter(self.positions)

    @property
    def usable(self) -> bool:
        """False if nothing survived -- such proteins must be skipped, not scored 0."""
        return len(self.positions) > 0


# --------------------------------------------------------------------------
# ligand-aware filtering — the cotransport-ion question
# --------------------------------------------------------------------------
#
# Three panel targets carry most of their annotated sites on cotransport ions:
# SLC6A3 14 of 20 positions (11 Na+, 3 chloride), SLC6A4 10 of 16 (Na+), DRD4
# 2 of 4 (Na+). No drug binds a sodium-coordination residue, so those positions
# inflate precision@k exactly the way UniProt's `Site` catch-all did.
#
# The obvious fix -- drop every metal or ion -- is wrong, and this is the whole
# subtlety. **Zinc cuts the other way.** Carbonic anhydrase and HDAC inhibitors
# chelate the catalytic zinc directly, so for those proteins zinc *is* the drug
# site, and excluding it would discard the correct answer. The panel holds 76
# zinc features against 31 sodium.
#
# So the default excludes nothing. This is a measurement decision that belongs
# to Track C (STATUS.md, A13), and the honest way to settle it is to report
# both numbers rather than to bake one in. COTRANSPORT_IONS is the shortlist
# the argument above supports; it is a suggestion, not a policy.
COTRANSPORT_IONS = ("na(+)", "chloride", "k(+)", "cl(-)")


def ligand_name(feature: dict):
    """The ligand a feature is annotated against, or None.

    UniProt records it as a nested object; the fetcher preserves that shape.
    Older files may carry a bare string, so both are read.
    """
    ligand = feature.get("ligand")
    if isinstance(ligand, dict):
        return ligand.get("name")
    return ligand


def ligand_matches(name, patterns) -> bool:
    """Case-insensitive containment against any pattern. `None` never matches."""
    if name is None:
        return False
    lowered = str(name).lower()
    return any(str(pattern).lower() in lowered for pattern in patterns)


def ligand_breakdown(path: str) -> dict:
    """{ligand name: position count} over a ground-truth file.

    Track C's input for the cotransport-ion decision: it says how much of the
    ground truth each ligand accounts for, so the choice is made against the
    size of the effect rather than in the abstract.
    """
    counts: dict = {}
    for _target_id, features in load_ground_truth(path).items():
        for feature in features:
            name = ligand_name(feature) or "(none recorded)"
            counts[name] = counts.get(name, 0) + len(expand_feature(feature))
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def expand_feature(feature: dict) -> set:
    """One UniProt feature record -> the 0-indexed positions it covers.

    UniProt ranges are 1-indexed and inclusive at both ends, so residues 41..43
    become array indices {40, 41, 42}.
    """
    start = feature.get("start")
    end = feature.get("end", start)
    if start is None:
        loc = feature.get("location", {})
        start = loc.get("start", {}).get("value") if isinstance(loc, dict) else None
        end = loc.get("end", {}).get("value") if isinstance(loc, dict) else start
    if start is None:
        return set()
    if end is None:
        end = start
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    return set(range(start - 1, end))  # -1 converts 1-indexed -> 0-indexed


def build_site_set(
    target_id: str,
    features: Iterable[dict],
    max_len: int | None = None,
    truncation: str = TruncationPolicy.EXCLUDE,
    filter_types: bool = True,
    filter_descriptions: bool = True,
    exclude_ligands: tuple = (),
) -> SiteSet:
    """Turn one target's raw feature list into a 0-indexed SiteSet."""
    if truncation not in TruncationPolicy.ALL:
        raise ValueError(
            f"truncation must be one of {TruncationPolicy.ALL}, got {truncation!r}"
        )

    features = list(features)
    result = SiteSet(
        target_id=target_id,
        n_raw_features=len(features),
        is_variant=is_variant_id(target_id),
        resolved_from=normalise_target_id(target_id),
    )

    for feature in features:
        ftype = feature.get("type")
        if filter_types and ftype is not None:
            if ftype in EXCLUDED_FEATURE_TYPES or ftype not in BINDING_FEATURE_TYPES:
                result.n_dropped_feature_type += 1
                continue
        # no recorded type -> fall back to the description heuristic
        if filter_descriptions and ftype is None:
            if NON_BINDING_DESCRIPTION.search(feature.get("description", "") or ""):
                result.n_dropped_description += 1
                continue

        if exclude_ligands and ligand_name(feature) is not None:
            if ligand_matches(ligand_name(feature), exclude_ligands):
                result.n_dropped_ligand += 1
                continue

        positions = expand_feature(feature)
        if max_len is not None:
            out_of_window = {p for p in positions if p >= max_len}
            if out_of_window:
                result.n_out_of_window += len(out_of_window)
                if truncation == TruncationPolicy.ERROR:
                    raise ValueError(
                        f"{target_id}: site at 1-indexed residue "
                        f"{min(out_of_window) + 1} falls past max_len={max_len}"
                    )
                if truncation == TruncationPolicy.EXCLUDE:
                    result.n_dropped_truncation += len(out_of_window)
                    positions -= out_of_window
        result.positions |= positions

    # Would this target have had a usable site inside the window? Computed the
    # same way under either policy, so the reported figure does not silently
    # change when the policy does.
    if max_len is not None and result.n_out_of_window > 0:
        result.dropped_by_truncation = not any(
            p < max_len for p in result.positions)

    return result


def load_ground_truth(path: str) -> dict:
    """Read a raw ground-truth JSON exactly as written by the fetcher."""
    with open(path) as f:
        return json.load(f)


def load_site_sets(
    path: str,
    max_len: int | None = None,
    truncation: str = TruncationPolicy.EXCLUDE,
    filter_types: bool = True,
    filter_descriptions: bool = True,
    drop_unusable: bool = True,
    exclude_ligands: tuple = (),
) -> dict:
    """Ground-truth JSON path -> {target_id: SiteSet}, ready for precision@k.

    `drop_unusable=True` removes proteins with no surviving annotated site.
    Keeping them would score a guaranteed 0.0 and drag the split average down
    for a reason unrelated to how good the explanations are.

    `exclude_ligands` drops sites by the ligand they coordinate — see
    COTRANSPORT_IONS and the reasoning above it. Empty by default: which
    ligands count as drug sites is a measurement decision, and baking one in
    would hide it. Whatever is passed has to be stated in Methods.
    """
    raw = load_ground_truth(path)
    out = {}
    for target_id, features in raw.items():
        site_set = build_site_set(
            target_id, features, max_len=max_len, truncation=truncation,
            filter_types=filter_types, filter_descriptions=filter_descriptions,
            exclude_ligands=exclude_ligands,
        )
        if drop_unusable and not site_set.usable:
            continue
        out[target_id] = site_set
    return out


def coverage_report(site_sets: dict) -> dict:
    """Summary numbers for the Methods section.

    Pass the FULL mapping (drop_unusable=False). Passing an already-filtered
    mapping silently understates every drop counter, because a protein whose
    sites were all discarded takes its counters out of the total with it --
    which hides exactly the losses this report exists to surface.
    """
    usable = {k: v for k, v in site_sets.items() if v.usable}
    return {
        "targets_in_file": len(site_sets),
        "targets_usable": len(usable),
        "targets_dropped_entirely": len(site_sets) - len(usable),
        "variant_ids": sum(1 for s in usable.values() if s.is_variant),
        "distinct_wild_type_accessions": len({s.resolved_from for s in usable.values()}),
        "total_positions": sum(len(s) for s in usable.values()),
        "raw_features": sum(s.n_raw_features for s in site_sets.values()),
        "dropped_feature_type": sum(s.n_dropped_feature_type for s in site_sets.values()),
        "dropped_description": sum(s.n_dropped_description for s in site_sets.values()),
        "dropped_ligand": sum(s.n_dropped_ligand for s in site_sets.values()),
        "dropped_truncation": sum(s.n_dropped_truncation for s in site_sets.values()),
        # Per-TARGET truncation impact, which the report previously could not
        # express. `targets_dropped_entirely` counts every unusable target
        # whatever the cause, so quoting it as the truncation figure overstates
        # it: 2 of KIBA's 10 lose their sites to filtering, not to the window.
        "positions_out_of_window": sum(
            s.n_out_of_window for s in site_sets.values()),
        "targets_affected_by_truncation": sum(
            1 for s in site_sets.values() if s.n_out_of_window > 0),
        "targets_dropped_by_truncation": sum(
            1 for s in site_sets.values() if s.dropped_by_truncation),
    }


if __name__ == "__main__":
    for path in ("data/davis_ground_truth_sites.json",
                 "data/kiba_ground_truth_sites.json"):
        sets = load_site_sets(path, max_len=1000, drop_unusable=False)
        print(f"\n{path}")
        for key, value in coverage_report(sets).items():
            print(f"  {key:34s} {value}")
