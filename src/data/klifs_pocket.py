"""
A second ground truth: the KLIFS ATP pocket, in DAVIS-sequence coordinates.

Why a second ground truth
-------------------------
The primary ground truth is UniProt's binding-site annotation of each protein. It
is protein-level, sparse (about a dozen residues a kinase) and uneven: some kinases
carry a curated nucleotide-binding stretch, others a single active-site residue. A
reviewer can fairly ask whether a model's attention misses "the binding site" or
only misses UniProt's annotation of it.

KLIFS (Kanev et al., Nucleic Acids Res. 2021; klifs.net) defines the same 85-residue
ATP-binding pocket for every human kinase from its structural alignment: the
residues lining the cleft where ATP-competitive inhibitors bind -- and DAVIS's 68
compounds are kinase inhibitors, most of them ATP-competitive. It is structure-derived, uniform across kinases and 85
residues wide, so it answers the objection from the other side: if attention does
not land in the pocket either, the null is not an artefact of UniProt's annotation.

How it is built
---------------
1. KLIFS gives each kinase its UniProt accession and its 85-character pocket
   sequence (`_` or `-` where the kinase has no residue at that pocket position).
2. The pocket's residues are placed on the UniProt canonical sequence in order
   (`place_pocket`, a small dynamic programme: every pocket residue is placed,
   consecutive pocket residues prefer consecutive sequence positions). A placement
   below MIN_IDENTITY is rejected rather than trusted.
3. UniProt numbering is carried onto the DAVIS sequence by the same
   `align_ground_truth.residue_map` the UniProt sites go through, so both ground
   truths live in the same coordinates by the same rule.
4. Eleven kinases have two kinase domains, which KLIFS lists separately ("JAK1",
   "JAK1-b"), while DAVIS holds the full-length sequence and names the assayed domain.
   The domain is taken from the name: JH2 / KinDom.1 / N-terminal = the first domain
   along the chain, JH1 / KinDom.2 / C-terminal = the second.

    python -m src.data.klifs_pocket              # writes data/davis_klifs_pocket_sites.json

The output has the ground-truth file's shape (target -> [{start, end, type,
description}], 1-based, DAVIS numbering), so `load_site_sets`, `run_ladder
--ground-truth` and `run_audit` read it unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

import numpy as np

from src.data.align_ground_truth import paths, residue_map

KLIFS_API = "https://klifs.net/api"
POCKET_LENGTH = 85
# KLIFS marks a pocket position the kinase lacks with "_" in some entries and "-" in
# others (CDK2). Reading "-" as a residue shifted every pocket residue after it.
GAPS = frozenset("_-")
# KLIFS's pocket is taken from UniProt's canonical sequence, so a correct placement
# matches every residue; a few mismatches allow for sequence-version drift, and
# anything lower means the pocket was placed somewhere it does not belong.
MIN_IDENTITY = 0.95
# "Binding site", not a new type: load_site_sets keeps only BINDING_FEATURE_TYPES and
# silently drops every other feature, so a type of its own would empty every target.
FEATURE_TYPE = "Binding site"

# KLIFS's 85 pocket positions in their 19 structural regions (van Linden et al., J. Med.
# Chem. 2014), as (name, first, last), 1-based and inclusive. Inside a region the
# residues are consecutive along the chain; insertions fall between regions. Placing
# with that constraint removes the ambiguity a free placement has wherever the same
# residue sits on either side of a gap.
KLIFS_REGIONS = (("I", 1, 3), ("g.l", 4, 9), ("II", 10, 13), ("III", 14, 19),
                 ("aC", 20, 30), ("b.l", 31, 37), ("IV", 38, 41), ("V", 42, 44),
                 ("GK", 45, 45), ("hinge", 46, 48), ("linker", 49, 52), ("aD", 53, 59),
                 ("aE", 60, 64), ("VI", 65, 67), ("c.l", 68, 75), ("VII", 76, 78),
                 ("VIII", 79, 79), ("xDFG", 80, 83), ("a.l", 84, 85))
REGION_OF = [name for name, first, last in KLIFS_REGIONS for _ in range(first, last + 1)]
assert len(REGION_OF) == 85

FIRST_DOMAIN = re.compile(r"JH2|KinDom\.?1|N-terminal", re.I)
SECOND_DOMAIN = re.compile(r"JH1|KinDom\.?2|C-terminal", re.I)


def out_paths(dataset: str = "davis") -> dict:
    return {"sites": f"data/{dataset}_klifs_pocket_sites.json",
            "report": f"data/{dataset}_klifs_pocket_report.json",
            "klifs": "data/klifs_human_kinases.json"}


# --------------------------------------------------------------------------
# KLIFS
# --------------------------------------------------------------------------

def fetch_klifs(cache: str) -> list:
    """Every human kinase KLIFS knows, with its UniProt accession and pocket. Cached."""
    if os.path.exists(cache):
        return json.load(open(cache))
    names = json.load(urllib.request.urlopen(f"{KLIFS_API}/kinase_names?species=Human",
                                             timeout=120))
    ids = [n["kinase_ID"] for n in names]
    info = []
    for i in range(0, len(ids), 100):
        url = f"{KLIFS_API}/kinase_information?kinase_ID=" + ",".join(map(str, ids[i:i + 100]))
        info += json.load(urllib.request.urlopen(url, timeout=120))
    keep = ("kinase_ID", "name", "HGNC", "family", "group", "uniprot", "pocket")
    info = [{k: entry.get(k) for k in keep} for entry in info]
    with open(cache, "w") as f:
        json.dump(info, f, indent=1)
    return info


def usable_pocket(pocket) -> bool:
    return isinstance(pocket, str) and len(pocket) == POCKET_LENGTH


# --------------------------------------------------------------------------
# placing a pocket on a sequence
# --------------------------------------------------------------------------

def place_pocket(pocket: str, sequence: str, match: float = 3.0, mismatch: float = -2.0,
                 jump_between: float = -1.0, jump_within: float = -4.0) -> tuple[list[int], float]:
    """0-based sequence positions of the pocket's residues, and the identity.

    Every non-gap pocket residue is placed, in order, on a strictly increasing
    position. Consecutive residues of one KLIFS region prefer to sit side by side: a
    jump inside a region costs `jump_within`, more than a jump between regions
    (`jump_between`) and a little less than one mismatch (match 3, mismatch -2). KLIFS
    takes the pocket from the UniProt canonical sequence, which is what it is placed
    on, so a correct placement matches every residue; a mismatch means an insertion
    inside a region was not taken. Checked against KLIFS's own residue numbers
    (2026-09-13): with a jump dearer than a mismatch, PIM2, GCN2 and PIK3CA each had one
    or two pocket residues one position off. Returns ([], 0.0)
    if it does not fit.
    """
    index = [i for i, c in enumerate(pocket) if c not in GAPS]
    residues = "".join(pocket[i] for i in index)
    m, n = len(residues), len(sequence)
    if m == 0 or m > n:
        return [], 0.0
    seq = np.frombuffer(sequence.encode(), dtype=np.uint8)
    neg = -1e9
    score = np.full((m, n), neg)
    back = np.full((m, n), -1, dtype=np.int64)
    score[0] = np.where(seq == ord(residues[0]), match, mismatch)
    for i in range(1, m):
        prev = score[i - 1]
        # best earlier position strictly before j-1, for a jump
        prefix_best = np.maximum.accumulate(prev)
        # an argmax of prev[:j+1] for every j: the latest index that attains the running max
        prefix_arg = np.maximum.accumulate(np.where(prev >= prefix_best, np.arange(n), 0))
        step = np.where(seq == ord(residues[i]), match, mismatch)
        adjacent = np.concatenate([[neg], prev[:-1]])                 # from j-1
        same_region = REGION_OF[index[i - 1]] == REGION_OF[index[i]]
        # a '_' between two residues of one region is a residue this kinase lacks
        contiguous = same_region and index[i] == index[i - 1] + 1
        jump = jump_within if contiguous else jump_between
        jumped = np.concatenate([[neg, neg], prefix_best[:-2]]) + jump  # from < j-1
        from_jump = np.concatenate([[-1, -1], prefix_arg[:-2]])
        take_adjacent = adjacent >= jumped
        score[i] = np.maximum(adjacent, jumped) + step
        back[i] = np.where(take_adjacent, np.arange(n) - 1, from_jump)
    j = int(np.argmax(score[m - 1]))
    if score[m - 1, j] <= neg / 2:
        return [], 0.0
    positions = [j]
    for i in range(m - 1, 0, -1):
        j = int(back[i, j])
        positions.append(j)
    positions.reverse()
    identity = float(np.mean([sequence[p] == r for p, r in zip(positions, residues)]))
    return positions, identity


def _features(davis_positions: list[int]) -> list[dict]:
    """0-based DAVIS positions -> 1-based contiguous features in the ground-truth shape."""
    out = []
    for p in sorted(set(davis_positions)):
        if out and p + 1 == out[-1]["end"] + 1:
            out[-1]["end"] = p + 1
        else:
            out.append({"start": p + 1, "end": p + 1, "type": FEATURE_TYPE,
                        "description": "KLIFS ATP pocket", "source": "KLIFS"})
    return out


def choose_domain(target: str, placed: list[tuple[dict, list[int]]]):
    """For a two-domain kinase, the placement matching the domain the name gives."""
    if len(placed) < 2:
        return placed, None
    placed = sorted(placed, key=lambda item: min(item[1]))      # along the chain
    if FIRST_DOMAIN.search(target):
        return placed[:1], "first domain (from the name)"
    if SECOND_DOMAIN.search(target):
        return placed[-1:], "second domain (from the name)"
    return placed, "both domains (the name does not say which)"


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(dataset: str = "davis") -> tuple[dict, dict]:
    p, o = paths(dataset), out_paths(dataset)
    klifs = fetch_klifs(o["klifs"])
    by_accession = {}
    for entry in klifs:
        if usable_pocket(entry.get("pocket")):
            by_accession.setdefault(entry["uniprot"], []).append(entry)

    provenance = json.load(open(p["provenance"]))
    uniprot_sequences = json.load(open(p["sequences"]))
    held = json.load(open(p["proteins"]))

    sites, report = {}, {}
    for target, davis in held.items():
        accession = provenance.get(target, {}).get("uniprot_accession")
        entries = by_accession.get(accession, [])
        uniprot = uniprot_sequences.get(accession)
        if not entries:
            sites[target], report[target] = [], {"status": "not_in_klifs",
                                                 "accession": accession}
            continue
        if uniprot is None:
            sites[target], report[target] = [], {"status": "no_uniprot_sequence",
                                                 "accession": accession}
            continue

        placed, rejected = [], []
        for entry in entries:
            positions, identity = place_pocket(entry["pocket"], uniprot)
            if identity >= MIN_IDENTITY:
                placed.append((entry, positions))
            else:
                rejected.append({"klifs": entry["name"], "identity": round(identity, 3)})
        placed, domain_rule = choose_domain(target, placed)
        if not placed:
            sites[target], report[target] = [], {"status": "pocket_not_placed",
                                                 "accession": accession,
                                                 "rejected": rejected}
            continue

        mapping = ({i: i for i in range(len(uniprot))} if uniprot == davis
                   else residue_map(uniprot, davis)[0])
        uniprot_positions = [q for _entry, positions in placed for q in positions]
        davis_positions = [mapping[q] for q in uniprot_positions if q in mapping]
        sites[target] = _features(davis_positions)
        report[target] = {
            "status": "identical_sequence" if uniprot == davis else "remapped",
            "accession": accession, "klifs": [e["name"] for e, _ in placed],
            "domain_rule": domain_rule, "pocket_residues": len(uniprot_positions),
            "residues_in_davis": len(set(davis_positions)),
            "dropped": len(uniprot_positions) - len(davis_positions),
            "first_davis_residue": min(davis_positions) + 1 if davis_positions else None,
            "last_davis_residue": max(davis_positions) + 1 if davis_positions else None,
            "rejected": rejected}
    return sites, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="davis", choices=("davis", "kiba"))
    args = parser.parse_args()
    sites, report = build(args.dataset)
    o = out_paths(args.dataset)
    with open(o["sites"], "w") as f:
        json.dump(sites, f, indent=1)
    with open(o["report"], "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)

    from collections import Counter
    counts = Counter(r["status"] for r in report.values())
    n_res = [r["residues_in_davis"] for r in report.values() if "residues_in_davis" in r]
    print(f"{o['sites']}: {len(sites)} targets")
    for status, n in counts.most_common():
        print(f"  {status:22s} {n}")
    if n_res:
        print(f"  pocket residues per target: median {int(np.median(n_res))}, "
              f"min {min(n_res)}, max {max(n_res)}")
    print(f"report -> {o['report']}")


if __name__ == "__main__":
    main()
