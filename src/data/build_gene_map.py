"""
Track A (124AD0008) -- build the accession -> gene-symbol map that makes the
kinase-confound control run on KIBA.

Why this file exists
--------------------
`src/evaluation/target_family.py` classifies a target as kinase / non-kinase by
matching its name. DAVIS names targets with gene symbols ('ABL1', 'EGFR'), so it
classifies directly. KIBA names them with UniProt accessions ('O00141'), which
carry no family signal at all -- so every KIBA target reads UNKNOWN, the
non-kinase count reads 0, and `control_is_usable` is False for reasons that have
nothing to do with the data actually being unsuitable.

That matters more than a missing column. The kinase confound is named in
docs/00_MASTER_PLAN_V2.md as "the objection most likely to sink the paper": a
cold-target kinase still shares the ATP pocket of the several hundred kinases
already in training, so a good cold-target precision@k could be family
similarity rather than generalisation. Without stratification there is no way to
tell the two apart, and a stratification that silently covers one dataset out of
two is not a control.

Where the data comes from
-------------------------
Nowhere new. `fetch_binding_sites.py` already downloads the full UniProt entry
for every target and now records `gene_name` and `protein_name` into
`*_provenance.json`. This script just reshapes that file. No second pass over
the API, no extra ~229 requests, and -- the part that actually matters -- the
symbols are guaranteed to come from the same UniProt snapshot as the binding
sites they will be stratifying, rather than from a lookup run on a different day
against a database that had moved.

Usage
-----
    python -m src.data.build_gene_map --dataset kiba
    python -m src.data.build_gene_map --dataset davis
    python -m src.data.build_gene_map --provenance path/to/x_provenance.json \
                                      --out data/x_uniprot_to_gene.json

Then confirm the control:

    python -m src.evaluation.target_family
"""
import argparse
import json
import os


def build_mapping(provenance: dict) -> dict:
    """provenance -> {target_id: {"gene_name": ..., "protein_name": ...}}.

    Targets that resolved to nothing are omitted rather than written with empty
    strings. An empty entry and an absent entry classify identically, but only
    the absent one is honest about coverage -- and the coverage number is what
    tells you whether the control arm is real.
    """
    mapping = {}
    for target_id, record in provenance.items():
        if not isinstance(record, dict):
            continue
        gene = (record.get("gene_name") or "").strip()
        protein = (record.get("protein_name") or "").strip()
        if not gene and not protein:
            continue
        mapping[target_id] = {"gene_name": gene, "protein_name": protein}
    return mapping


def coverage(provenance: dict, mapping: dict) -> dict:
    total = len(provenance)
    resolved = len(mapping)
    with_gene = sum(1 for v in mapping.values() if v["gene_name"])
    return {
        "targets_in_provenance": total,
        "targets_with_any_name": resolved,
        "targets_with_gene_symbol": with_gene,
        "targets_unresolved": total - resolved,
        "pct_with_gene_symbol": round(100 * with_gene / total, 1) if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build accession -> gene-symbol map from a provenance file")
    parser.add_argument("--dataset", choices=["davis", "kiba"])
    parser.add_argument("--provenance", help="explicit provenance JSON path")
    parser.add_argument("--out", help="output JSON path")
    args = parser.parse_args()

    if args.dataset:
        provenance_path = (args.provenance
                           or f"data/{args.dataset}_ground_truth_sites_provenance.json")
        out_path = args.out or f"data/{args.dataset}_uniprot_to_gene.json"
    elif args.provenance:
        provenance_path = args.provenance
        out_path = args.out or provenance_path.replace(
            "_ground_truth_sites_provenance.json", "_uniprot_to_gene.json")
    else:
        parser.error("give either --dataset or --provenance")

    if not os.path.exists(provenance_path):
        raise SystemExit(
            f"No provenance file at {provenance_path}.\n"
            f"It is written by the fetcher, so run that first:\n"
            f"    python -m src.data.fetch_binding_sites --dataset "
            f"{args.dataset or 'davis'}\n"
            f"If the file exists but predates this change, it has no gene_name "
            f"field and the fetch has to be repeated -- the names were never "
            f"recorded."
        )

    with open(provenance_path) as handle:
        provenance = json.load(handle)

    mapping = build_mapping(provenance)
    if not mapping:
        raise SystemExit(
            f"{provenance_path} contains no gene_name/protein_name fields.\n"
            f"That file was written before the fetcher recorded names. Re-run:\n"
            f"    python -m src.data.fetch_binding_sites --dataset "
            f"{args.dataset or 'davis'}"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)

    stats = coverage(provenance, mapping)
    print(f"Saved -> {out_path}")
    for key, value in stats.items():
        print(f"  {key:28s} {value}")
    print("\nNow check the control arm:  python -m src.evaluation.target_family")


if __name__ == "__main__":
    main()
