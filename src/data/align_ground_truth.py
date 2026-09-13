"""Put the DAVIS binding-site annotations into the coordinates of the DAVIS sequences.

UniProt numbers residues along its canonical sequence; the model reads DAVIS's own
sequence. For most targets the two are identical and a UniProt residue number is also
a position in what the model saw. For 54 targets with annotated sites they are not:
DAVIS holds a fragment (RET: 458 residues against UniProt's 1,114), a longer isoform
(PIM1: 404 against 313) or a construct with extra residues (ABL1 and its sixteen
variants: 1,167 against 1,130). Some fragments are NOT the kinase domain: DAVIS's RET
(and its three mutants) is residues 1-430, the extracellular part, while the kinase
pocket is at 728-896; see `src/data/klifs_pocket.py` and the KLIFS report for the ten
targets whose DAVIS sequence lacks most or all of the ATP pocket (checked 2026-09-13). There a UniProt residue number points at
the wrong residue of the DAVIS sequence, or past its end, and precision@k scores the
model against sites it was never shown -- pulling every level toward chance.

Each DAVIS sequence is aligned to its UniProt sequence and every annotated residue is
carried across one at a time:

  * inside an aligned stretch of at least MIN_BLOCK identical residues, it maps across;
  * a single-residue substitution between two such stretches maps too -- the point
    mutants (ABL1(T315I) and the rest) differ from wild type at exactly the residues
    that matter most, the gatekeeper among them, and dropping those would be wrong;
  * anything else -- outside a fragment, inside a divergent region -- is dropped.

A target whose UniProt sequence has changed length since the sites were fetched keeps
no sites at all: its residue numbers were assigned against a sequence that no longer
exists, and there is nothing to align them to.

The original UniProt-numbered file is kept beside the aligned one, and a per-target
report records what happened to every site.

    python -m src.data.align_ground_truth                  # DAVIS
    python -m src.data.align_ground_truth --dataset kiba   # KIBA

Fetching needs network access to rest.uniprot.org; the sequences are then saved, so
applying again works offline.

KIBA
----
KIBA's targets are keyed by UniProt accession, so it cannot carry another protein's
sites the way four DAVIS gene names did. Its sequences can still differ from UniProt's
canonical one: on 2026-09-12, 219 of its 221 annotated targets matched UniProt's
length and two did not (PIM1, the same longer isoform DAVIS holds, and SGK2). The
same alignment applies, with the same one-writer rule: the fetch writes
`kiba_ground_truth_sites_uniprot.json`, and only this module writes
`kiba_ground_truth_sites.json`.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import urllib.request

ALIGNED_DATASETS = ("davis", "kiba")

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/accessions?accessions={}&format=fasta"


def paths(dataset: str) -> dict:
    """Every file one dataset's alignment reads or writes."""
    if dataset not in ALIGNED_DATASETS:
        raise ValueError(f"no alignment step for {dataset!r}; known: {ALIGNED_DATASETS}")
    return {
        "ground_truth": f"data/{dataset}_ground_truth_sites.json",           # derived here
        "original": f"data/{dataset}_ground_truth_sites_uniprot.json",       # the source
        "provenance": f"data/{dataset}_ground_truth_sites_provenance.json",
        "sequences": f"data/{dataset}_uniprot_sequences.json",
        "report": f"data/{dataset}_ground_truth_alignment.json",
        "proteins": f"src/data/baselines/deepdta/data/{dataset}/proteins.txt",
    }


# DAVIS's paths under their original names, for anything importing them.
_DAVIS = paths("davis")
GROUND_TRUTH, ORIGINAL, PROVENANCE = _DAVIS["ground_truth"], _DAVIS["original"], _DAVIS["provenance"]
SEQUENCES, REPORT, DAVIS_PROTEINS = _DAVIS["sequences"], _DAVIS["report"], _DAVIS["proteins"]


def uniprot_numbered_path(dataset: str) -> str:
    """Where the fetch and the manual overrides write UniProt-numbered sites.

    For an aligned dataset that is its `_uniprot` file, and the file evaluation reads
    is derived from it by this module -- so each file has one writer, and nothing can
    be aligned twice. Any other dataset keeps its single file.
    """
    if dataset in ALIGNED_DATASETS:
        return paths(dataset)["original"]
    return f"data/{dataset}_ground_truth_sites.json"

# An identical stretch shorter than this is not trusted as alignment: short exact
# matches turn up by chance inside regions that do not correspond at all.
MIN_BLOCK = 10
# The longest mismatch mapped as a substitution, and only between trusted stretches.
MAX_SUBSTITUTION = 1


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def residue_map(uniprot: str, davis: str) -> tuple[dict, int]:
    """{uniprot 0-based index: davis 0-based index}, and how many are substitutions.

    autojunk=False is essential. difflib's default treats any character above 1% of
    the sequence as junk and skips it when matching -- which, with an alphabet of
    twenty amino acids, is every residue, and the alignment silently comes back empty.
    """
    matcher = difflib.SequenceMatcher(None, uniprot, davis, autojunk=False)
    ops = matcher.get_opcodes()
    trusted = [tag == "equal" and (i2 - i1) >= MIN_BLOCK for tag, i1, i2, _j1, _j2 in ops]

    mapping, substitutions = {}, 0
    for k, (tag, i1, i2, j1, j2) in enumerate(ops):
        if trusted[k]:
            mapping.update(zip(range(i1, i2), range(j1, j2)))
        elif (tag == "replace" and i2 - i1 == j2 - j1 <= MAX_SUBSTITUTION
              and 0 < k < len(ops) - 1 and trusted[k - 1] and trusted[k + 1]):
            mapping.update(zip(range(i1, i2), range(j1, j2)))
            substitutions += i2 - i1
    return mapping, substitutions


def _runs(positions: list[int]) -> list[tuple[int, int]]:
    """Sorted positions -> contiguous (first, last) runs."""
    runs = []
    for p in positions:
        if runs and p == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], p)
        else:
            runs.append((p, p))
    return runs


def map_features(features: list, mapping: dict) -> tuple[list, int, int]:
    """UniProt-numbered features -> DAVIS-numbered features.

    Positions are 1-based inclusive on both sides, as in the source file. A feature
    whose residues map to more than one contiguous stretch is split, so no aligned
    feature ever spans residues the source did not annotate. Returns the features and
    how many annotated residues were kept and dropped.
    """
    out, kept, dropped = [], 0, 0
    for feature in features:
        mapped = []
        for residue in range(feature["start"], feature["end"] + 1):
            j = mapping.get(residue - 1)
            if j is None:
                dropped += 1
            else:
                mapped.append((j + 1, residue))
                kept += 1
        for first, last in _runs(sorted(p for p, _ in mapped)):
            src = [r for p, r in mapped if first <= p <= last]
            out.append({**feature, "start": first, "end": last,
                        "uniprot_start": min(src), "uniprot_end": max(src)})
    return out, kept, dropped


def align_target(features: list, uniprot: str | None, davis: str,
                 recorded_length: int | None, dataset: str = "davis") -> tuple[list, dict]:
    """One target: its aligned features and a report entry.

    `davis` is the sequence the dataset holds for this target, whichever dataset it is.
    """
    n_residues = sum(f["end"] - f["start"] + 1 for f in features)
    entry = {f"{dataset}_length": len(davis),
             "uniprot_length": len(uniprot) if uniprot else None,
             "residues_before": n_residues}

    if not features:
        return [], {**entry, "status": "no_sites", "residues_after": 0}
    if uniprot is None:
        return [], {**entry, "status": "no_uniprot_sequence", "residues_after": 0}
    if recorded_length is not None and len(uniprot) != recorded_length:
        # The sites were numbered against a sequence of recorded_length residues.
        return [], {**entry, "status": "uniprot_sequence_changed",
                    "recorded_length": recorded_length, "residues_after": 0}
    if uniprot == davis:
        return [dict(f) for f in features], {**entry, "status": "identical",
                                             "residues_after": n_residues}

    mapping, substitutions = residue_map(uniprot, davis)
    aligned, kept, dropped = map_features(features, mapping)
    return aligned, {**entry, "status": "remapped" if aligned else "no_sites_survive",
                     "residues_after": kept, "residues_dropped": dropped,
                     "aligned_residues": len(mapping), "substitutions_mapped": substitutions}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def parse_fasta(text: str) -> dict:
    sequences, accession, chunks = {}, None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if accession:
                sequences[accession] = "".join(chunks)
            accession, chunks = line[1:].split("|")[1], []
        elif line.strip():
            chunks.append(line.strip())
    if accession:
        sequences[accession] = "".join(chunks)
    return sequences


def fetch_sequences(accessions: list, batch: int = 100) -> dict:
    sequences = {}
    for start in range(0, len(accessions), batch):
        chunk = accessions[start:start + batch]
        with urllib.request.urlopen(UNIPROT_FASTA.format(",".join(chunk)), timeout=120) as r:
            sequences.update(parse_fasta(r.read().decode()))
        print(f"  fetched {len(sequences)}/{len(accessions)}", flush=True)
    missing = sorted(set(accessions) - set(sequences))
    if missing:
        print(f"  UniProt returned nothing for {len(missing)}: {missing[:10]}", flush=True)
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="davis", choices=ALIGNED_DATASETS)
    parser.add_argument("--refetch", action="store_true",
                        help="fetch sequences again even if the cached file exists")
    args = parser.parse_args()
    p = paths(args.dataset)

    # The UniProt-numbered file is the source; the aligned file is derived from it.
    # On the first run the current file IS the original and is set aside. After that
    # the original is always read from its own file, so re-running cannot align an
    # already-aligned file a second time.
    if not os.path.exists(p["original"]):
        if os.path.exists(p["report"]):
            sys.exit(f"{p['report']} exists but {p['original']} does not: "
                     f"{p['ground_truth']} may already be aligned. Restore the "
                     "UniProt-numbered file before re-running.")
        shutil.copy2(p["ground_truth"], p["original"])
        print(f"kept the UniProt-numbered original -> {p['original']}")

    sites = json.load(open(p["original"]))
    provenance = json.load(open(p["provenance"]))
    held = json.load(open(p["proteins"]))

    accessions = sorted({entry["uniprot_accession"] for entry in provenance.values()
                         if entry.get("uniprot_accession")})
    sequences = ({} if args.refetch or not os.path.exists(p["sequences"])
                 else json.load(open(p["sequences"])))
    # Only what is missing -- a manual override brings in a new accession.
    missing = [a for a in accessions if a not in sequences]
    if missing:
        print(f"fetching {len(missing)} UniProt sequence(s)")
        sequences.update(fetch_sequences(missing))
        with open(p["sequences"], "w") as f:
            json.dump(sequences, f, indent=1, sort_keys=True)

    aligned, report = {}, {}
    for target, features in sites.items():
        prov = provenance.get(target, {})
        seq = held.get(target)
        if seq is None:
            aligned[target], report[target] = [], {"status": f"not_in_{args.dataset}"}
            continue
        aligned[target], report[target] = align_target(
            features, sequences.get(prov.get("uniprot_accession")), seq,
            prov.get("sequence_length"), dataset=args.dataset)

    with open(p["ground_truth"], "w") as f:
        json.dump(aligned, f, indent=1)
    with open(p["report"], "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)

    from collections import Counter
    counts = Counter(r["status"] for r in report.values())
    before = sum(r.get("residues_before", 0) for r in report.values())
    after = sum(r.get("residues_after", 0) for r in report.values())
    print(f"\n{p['ground_truth']} is now in {args.dataset.upper()}-sequence coordinates")
    for status, n in counts.most_common():
        print(f"  {status:26s} {n}")
    print(f"  annotated residues: {before} -> {after} ({before - after} dropped)")
    print(f"report -> {p['report']}")


if __name__ == "__main__":
    main()
