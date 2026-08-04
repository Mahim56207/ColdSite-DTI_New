"""
Track A (124AD0008) — collect real binding-site ground truth from UniProt.

This replaces three overlapping scripts (binding_sites.py,
fetch_binding_sites.py, fetch_kiba_binding_sites.py) that had drifted apart and
filtered features differently from each other.

Two things changed versus those versions, and both affect the paper's numbers:

1. The UniProt 'Site' feature type is NO LONGER collected. 'Site' is a
   catch-all that carries protease cleavage points, chromosomal breakpoints and
   interaction residues alongside genuine pockets. Roughly 7% of the previously
   committed annotations were of that kind. Every one of them is a position a
   drug does not bind to, counted as a correct answer -- which inflates
   precision@k and flatters the paper's central claim.

2. Every record now carries its 'type'. Without it, downstream filtering has to
   guess from free-text descriptions, which has known false negatives. Storing
   the type makes src/data/ground_truth.py filter exactly.

Usage
-----
    python -m src.data.fetch_binding_sites --dataset davis
    python -m src.data.fetch_binding_sites --dataset kiba
    python -m src.data.fetch_binding_sites --ids P00533 P04626 --out data/custom.json

Requires network access to rest.uniprot.org.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from src.data.ground_truth import BINDING_FEATURE_TYPES, normalise_target_id

UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{}.json"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search?query={}&size=1"
HUMAN_TAXON = 9606


def _get_json(url: str, timeout: int = 20):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def resolve_entry(target_id: str) -> tuple:
    """Target ID -> (UniProt entry JSON, accession, how it was resolved).

    DAVIS names point mutants like 'ABL1(T315I)p'. UniProt has no entry for
    those, so the mutation is stripped and the wild-type entry is used. That is
    a real approximation, not a formatting detail: every ABL1 variant ends up
    with an identical site list, and the Methods section has to say so. The
    resolution route is recorded per target so the paper can report how many
    entries came in that way.
    """
    base = normalise_target_id(target_id)

    try:
        return _get_json(UNIPROT_ENTRY.format(base)), base, "accession"
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 404):
            raise
    except Exception:
        return None, base, "failed"

    query = urllib.parse.quote(
        f"(gene:{base}) AND (organism_id:{HUMAN_TAXON}) AND (reviewed:true)"
    )
    try:
        results = _get_json(UNIPROT_SEARCH.format(query)).get("results", [])
    except Exception:
        return None, base, "failed"
    if not results:
        return None, base, "not_found"
    entry = results[0]
    return entry, entry.get("primaryAccession", base), "gene_search"


def extract_features(entry: dict) -> list:
    """UniProt entry -> the binding-relevant features, with their types kept."""
    features = []
    for feature in entry.get("features", []):
        ftype = feature.get("type")
        if ftype not in BINDING_FEATURE_TYPES:
            continue
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value", start)
        if start is None:
            continue
        features.append({
            "start": int(start),
            "end": int(end if end is not None else start),
            "type": ftype,
            "description": feature.get("description", "") or "",
        })
    return features


def fetch_for_targets(target_ids, delay: float = 0.2, verbose: bool = True) -> dict:
    """Fetch every target, caching by resolved accession.

    DAVIS has 72 variant IDs collapsing onto ~17 base genes, so caching removes
    a few hundred redundant requests and, more importantly, guarantees the
    variants of one gene get byte-identical annotations rather than two
    different snapshots of a database that changed mid-run.
    """
    ground_truth, provenance, cache = {}, {}, {}
    total = len(target_ids)

    for i, target_id in enumerate(target_ids, start=1):
        base = normalise_target_id(target_id)
        if base in cache:
            features, accession, route = cache[base]
            route = f"{route}(cached)"
        else:
            entry, accession, route = resolve_entry(target_id)
            features = extract_features(entry) if entry else []
            cache[base] = (features, accession, route)
            time.sleep(delay)

        ground_truth[target_id] = features
        provenance[target_id] = {
            "resolved_from": base,
            "uniprot_accession": accession,
            "resolution": route,
            "is_variant": base != str(target_id).strip(),
            "n_features": len(features),
        }
        if verbose:
            status = "ok" if features else "NO SITES"
            print(f"[{i}/{total}] {target_id} -> {accession} ({route}) {status}")

    return {"sites": ground_truth, "provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description="Fetch UniProt binding sites")
    parser.add_argument("--dataset", choices=["davis", "kiba"],
                        help="pull target IDs from this dataset")
    parser.add_argument("--ids", nargs="+", help="explicit UniProt IDs instead")
    parser.add_argument("--out", help="output JSON path")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    if args.dataset:
        from src.data.load_data import load_deepdta_dataset
        target_ids = list(load_deepdta_dataset(args.dataset)["Target_ID"].unique())
        out_path = args.out or f"data/{args.dataset}_ground_truth_sites.json"
    elif args.ids:
        target_ids = args.ids
        out_path = args.out or "data/custom_ground_truth_sites.json"
    else:
        parser.error("give either --dataset or --ids")

    print(f"Fetching binding sites for {len(target_ids)} targets...")
    result = fetch_for_targets(target_ids, delay=args.delay)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result["sites"], f, indent=2)
    provenance_path = out_path.replace(".json", "_provenance.json")
    with open(provenance_path, "w") as f:
        json.dump(result["provenance"], f, indent=2)

    with_sites = sum(1 for v in result["sites"].values() if v)
    variants = sum(1 for p in result["provenance"].values() if p["is_variant"])
    print(f"\n{with_sites}/{len(target_ids)} targets have annotated sites")
    print(f"{variants} were variant IDs resolved to a wild-type entry")
    print(f"Saved -> {out_path}")
    print(f"Saved -> {provenance_path}")
    print("\nNow verify with:  python -m src.data.ground_truth")


if __name__ == "__main__":
    main()
