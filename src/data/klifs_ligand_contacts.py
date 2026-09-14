"""
Drug-specific ground truth: the pocket residues that touch *this* drug.

Why
---
Both ground truths so far are per protein: UniProt's annotated residues, and the
85-residue KLIFS ATP pocket (`src/data/klifs_pocket.py`). Neither knows which drug the
model was asked about, so a reviewer can fairly object that "attention marks the binding
site" was never tested against the site of *the* binding event -- the model sees a pair.

KLIFS publishes, for every co-crystal structure, an **interaction fingerprint**: 85 pocket
positions x 7 interaction types (apolar, aromatic face-to-face and edge-to-face, hydrogen
bond donor and acceptor, and the two ionic directions). A position with any bit set is a
residue that contacts the bound ligand in that structure. Where a DAVIS drug is that
ligand, the fingerprint is a per-pair ground truth measured from a crystal structure.

How a pair is built
-------------------
1. DAVIS drugs are matched to KLIFS ligands by **InChIKey** (RDKit, from the SMILES in the
   split files; DAVIS's own drug ids are PubChem CIDs and are kept for provenance).
2. The protein side reuses `klifs_pocket` exactly -- same KLIFS entry by UniProt
   accession, same `place_pocket`, same `choose_domain`, same remap to DAVIS numbering --
   so drug sites and pocket sites live in one coordinate frame and can be compared.
3. Every structure of that kinase with that ligand contributes its contacts. A position
   counts when it is contacted in at least `MIN_FRACTION` of them: one crystal is one
   observation, and imatinib has 18 ABL1 structures that do not agree residue for residue.
   The union is kept too, for a sensitivity check.

Contacts come from the structure, so a residue KLIFS could not resolve is absent rather
than negative; `report` records how many structures each pair rests on.

    python -m src.data.klifs_ligand_contacts --dataset davis
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

import pandas as pd

from src.data.align_ground_truth import paths, residue_map
from src.data.ground_truth import PAIR_SEPARATOR
from src.data.klifs_pocket import (FEATURE_TYPE, GAPS, KLIFS_API, MIN_IDENTITY,
                                   POCKET_LENGTH, choose_domain, fetch_klifs, out_paths,
                                   place_pocket, usable_pocket)

N_TYPES = 7                    # apolar, 2 aromatic, 2 hydrogen bond, 2 ionic
MIN_FRACTION = 0.5             # a contact must hold in half the structures of the pair
SPLIT_PARTS = ("train", "valid", "test")


def features(davis_positions: list[int]) -> list[dict]:
    """0-based DAVIS positions -> 1-based features, in the ground-truth shape.

    Same shape as the pocket ground truth so the ladder can read either, but the
    description says what these residues are: contacts with one particular drug.
    """
    out = []
    for position in sorted(set(davis_positions)):
        if out and position + 1 == out[-1]["end"] + 1:
            out[-1]["end"] = position + 1
        else:
            out.append({"start": position + 1, "end": position + 1, "type": FEATURE_TYPE,
                        "description": "KLIFS ligand contact (co-crystal)",
                        "source": "KLIFS-IFP"})
    return out


def out_files(dataset: str = "davis") -> dict:
    return {"sites": f"data/{dataset}_drug_sites.json",
            "report": f"data/{dataset}_drug_sites_report.json",
            "ligands": f"data/klifs_ligands.json",
            "swapped": f"data/{dataset}_drug_sites_swapped.json",
            "paired": f"data/{dataset}_drug_sites_paired.json",
            "structures": f"data/klifs_structures.json"}


def get(path: str, timeout: int = 60):
    """One KLIFS call. A kinase with no structures answers 4xx, which is not an error."""
    try:
        with urllib.request.urlopen(f"{KLIFS_API}/{path}", timeout=timeout) as handle:
            return json.load(handle)
    except urllib.error.HTTPError:
        return []


def contacts(ifp: str) -> list[int]:
    """0-based KLIFS positions whose fingerprint has any interaction bit set."""
    if len(ifp) != POCKET_LENGTH * N_TYPES:
        raise ValueError(f"fingerprint is {len(ifp)} long, expected "
                         f"{POCKET_LENGTH * N_TYPES}")
    return [i for i in range(POCKET_LENGTH) if "1" in ifp[i * N_TYPES:(i + 1) * N_TYPES]]


def agreed(per_structure: list[list[int]], min_fraction: float = MIN_FRACTION) -> list[int]:
    """Positions contacted in at least `min_fraction` of a pair's structures."""
    if not per_structure:
        return []
    need = max(1, int(len(per_structure) * min_fraction + 0.9999))   # ceil
    counts = {}
    for positions in per_structure:
        for p in set(positions):
            counts[p] = counts.get(p, 0) + 1
    return sorted(p for p, c in counts.items() if c >= need)


def drug_inchikeys(split_root: str = "data/splits", dataset: str = "davis") -> dict:
    """{InChIKey: Drug_ID} for every drug in the dataset, from the SMILES it was trained on."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    frames = [pd.read_csv(os.path.join(split_root, dataset, "random", f"{p}.csv"))
              for p in SPLIT_PARTS]
    drugs = pd.concat(frames).drop_duplicates("Drug_ID")[["Drug_ID", "Drug"]]
    keys = {}
    for drug_id, smiles in zip(drugs.Drug_ID.astype(str), drugs.Drug):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            keys[Chem.MolToInchiKey(mol)] = drug_id
    return keys


def placement(entries: list, uniprot: str, davis: str, target: str):
    """KLIFS position -> DAVIS position, through the same steps as the pocket ground truth.

    Returns ({klifs position: davis position}, report fields). A KLIFS position the kinase
    lacks (a gap in its pocket), or one whose residue is not in the DAVIS sequence, is
    simply absent.
    """
    placed, rejected = [], []
    for entry in entries:
        positions, identity = place_pocket(entry["pocket"], uniprot)
        if identity >= MIN_IDENTITY:
            placed.append((entry, positions))
        else:
            rejected.append({"klifs": entry["name"], "identity": round(identity, 3)})
    placed, domain_rule = choose_domain(target, placed)
    if not placed:
        return {}, {"status": "pocket_not_placed", "rejected": rejected}
    mapping = ({i: i for i in range(len(uniprot))} if uniprot == davis
               else residue_map(uniprot, davis)[0])
    klifs_to_davis = {}
    for entry, positions in placed:
        columns = [i for i, c in enumerate(entry["pocket"]) if c not in GAPS]
        for column, uniprot_position in zip(columns, positions):
            davis_position = mapping.get(uniprot_position)
            if davis_position is not None:
                klifs_to_davis.setdefault(column, davis_position)
    return klifs_to_davis, {"status": "placed", "klifs": [e["name"] for e, _ in placed],
                            "domain_rule": domain_rule, "rejected": rejected,
                            "positions_placed": len(klifs_to_davis)}


def pair_key(drug_id: str, target: str) -> str:
    return f"{drug_id}|{target}"


def build(dataset: str = "davis", split_root: str = "data/splits",
          min_fraction: float = MIN_FRACTION, limit: int | None = None) -> tuple[dict, dict]:
    p, o, f = paths(dataset), out_paths(dataset), out_files(dataset)
    klifs = fetch_klifs(o["klifs"])
    by_accession = {}
    for entry in klifs:
        if usable_pocket(entry.get("pocket")):
            by_accession.setdefault(entry["uniprot"], []).append(entry)

    drugs = drug_inchikeys(split_root, dataset)
    provenance = json.load(open(p["provenance"]))
    uniprot_sequences = json.load(open(p["sequences"]))
    held = json.load(open(p["proteins"]))

    # DAVIS holds many mutants of one kinase (ABL1 appears four times), all sharing an
    # accession and therefore the same structures -- fetch each fingerprint once.
    ligand_cache, structure_cache, ifp_cache = {}, {}, {}
    sites, report = {}, {}
    targets = sorted(held)[:limit] if limit else sorted(held)
    for n, target in enumerate(targets, 1):
        accession = provenance.get(target, {}).get("uniprot_accession")
        entries = by_accession.get(accession, [])
        uniprot = uniprot_sequences.get(accession)
        if not entries or uniprot is None:
            report[target] = {"status": "not_in_klifs" if not entries
                              else "no_uniprot_sequence", "accession": accession}
            continue
        klifs_to_davis, fields = placement(entries, uniprot, held[target], target)
        fields["accession"] = accession
        if not klifs_to_davis:
            report[target] = fields
            continue

        pairs = {}
        for entry in entries:
            kinase_id = entry["kinase_ID"]
            if kinase_id not in ligand_cache:
                ligand_cache[kinase_id] = get(f"ligands_list?kinase_ID={kinase_id}")
                structure_cache[kinase_id] = get(f"structures_list?kinase_ID={kinase_id}")
            # the drugs of this dataset that KLIFS has crystallised with this kinase
            wanted = {lig["PDB-code"]: drugs[lig["InChIKey"]]
                      for lig in ligand_cache[kinase_id]
                      if lig.get("InChIKey") in drugs and lig.get("PDB-code")}
            for structure in structure_cache[kinase_id]:
                drug_id = wanted.get(structure.get("ligand"))
                if drug_id is None:
                    continue
                pairs.setdefault(drug_id, []).append(structure)

        fields["drugs_crystallised"] = len(pairs)
        for drug_id, structures in sorted(pairs.items()):
            per_structure, used = [], []
            for structure in structures:
                structure_id = structure["structure_ID"]
                if structure_id not in ifp_cache:
                    answer = get(f"interactions_get_IFP?structure_ID={structure_id}")
                    fingerprint = answer[0].get("IFP") if answer else None
                    try:
                        ifp_cache[structure_id] = contacts(fingerprint) if fingerprint else None
                    except ValueError:
                        ifp_cache[structure_id] = None
                positions = ifp_cache[structure_id]
                if positions is None:
                    continue
                per_structure.append(positions)
                used.append(structure["pdb"])
            if not per_structure:
                continue
            columns = agreed(per_structure, min_fraction)
            davis_positions = sorted({klifs_to_davis[c] for c in columns
                                      if c in klifs_to_davis})
            union = sorted({klifs_to_davis[c] for ps in per_structure for c in ps
                            if c in klifs_to_davis})
            if not davis_positions:
                continue
            key = pair_key(drug_id, target)
            sites[key] = features(davis_positions)
            report[key] = {"status": "pair", "target": target, "drug": drug_id,
                           "structures": len(per_structure),
                           "pdb_entries": sorted(set(used)), "pdb": sorted(used),
                           "klifs_positions_contacted": len(columns),
                           "residues": len(davis_positions),
                           "residues_union": len(union),
                           "first_residue": davis_positions[0] + 1,
                           "last_residue": davis_positions[-1] + 1}
        report[target] = fields
        if n % 25 == 0 or n == len(targets):
            print(f"   {n}/{len(targets)} targets, {sum(1 for k in report.values() if k.get('status') == 'pair')} pairs so far",
                  flush=True)
    with open(f["ligands"], "w") as handle:
        json.dump({str(k): v for k, v in ligand_cache.items()}, handle)
    with open(f["structures"], "w") as handle:
        json.dump({str(k): v for k, v in structure_cache.items()}, handle)
    return sites, report


def swap_drugs(sites: dict, seed: int = 0) -> dict:
    """The control arm: every pair scored against ANOTHER drug's contacts on the SAME protein.

    Drug-specific sites sit inside one pocket, so a model that merely finds the pocket
    scores well against any drug's contacts there. This arm holds the protein and the
    number of sites fixed and changes only *which* drug the sites belong to: precision
    above this null is the part of the signal that is specific to the drug in the pair.

    Only proteins with at least two crystallised drugs can be swapped; pairs on a protein
    with one are dropped, so the two arms are compared on the same keys. Deterministic:
    the rotation is by sorted drug id.
    """
    by_target = {}
    for key in sites:
        drug, target = key.split(PAIR_SEPARATOR, 1)
        by_target.setdefault(target, []).append(drug)
    out = {}
    for target, drugs in by_target.items():
        if len(drugs) < 2:
            continue
        order = sorted(drugs)
        for i, drug in enumerate(order):
            other = order[(i + 1) % len(order)]           # a rotation: never itself
            out[f"{drug}{PAIR_SEPARATOR}{target}"] = sites[f"{other}{PAIR_SEPARATOR}{target}"]
    return out


def coverage(sites: dict, split_root: str = "data/splits", dataset: str = "davis",
             swapped: dict | None = None) -> list[dict]:
    """Per split level: how many test pairs this ground truth can score at all.

    The honest limit on the measurement. DAVIS holds 68 drugs and only some are
    crystallised, so a cold level -- 13 held-out drugs -- can be very thin.
    """
    out = []
    for level in ("random", "cold_drug", "cold_target", "cold_pair"):
        path = os.path.join(split_root, dataset, level, "test.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        keys = [pair_key(str(d), str(t))
                for d, t in zip(frame.Drug_ID, frame.Target_ID)]
        mine = [k for k in keys if k in sites]
        out.append({"level": level, "test_rows": len(frame), "scorable": len(mine),
                    "proteins": len({k.split(PAIR_SEPARATOR)[1] for k in mine}),
                    "drugs": len({k.split(PAIR_SEPARATOR)[0] for k in mine}),
                    "swappable": sum(1 for k in mine if swapped and k in swapped)})
    return out


def summary(sites: dict, report: dict, rows: list[dict] | None = None) -> str:
    pairs = [v for v in report.values() if v.get("status") == "pair"]
    proteins = sorted({v["target"] for v in pairs})
    drugs = sorted({v["drug"] for v in pairs})
    residues = [v["residues"] for v in pairs]
    structures = [v["structures"] for v in pairs]
    lines = ["# Drug-specific ground truth — KLIFS ligand contacts", "",
             f"{len(pairs)} drug-target pairs with a co-crystal structure, covering "
             f"{len(proteins)} proteins and {len(drugs)} drugs "
             f"(`src/data/klifs_ligand_contacts.py`).", "",
             f"- contacted residues per pair: median {int(pd.Series(residues).median())}, "
             f"range {min(residues)}-{max(residues)}" if residues else "- no pairs",
             f"- structures per pair: median {int(pd.Series(structures).median())}, "
             f"range {min(structures)}-{max(structures)}" if structures else "",
             f"- a residue counts when it is contacted in at least "
             f"{int(MIN_FRACTION * 100)}% of the pair's structures", ""]
    multi = [v for v in pairs if v["structures"] > 1]
    if multi:
        shrink = sum(v["residues"] for v in multi) / sum(v["residues_union"] for v in multi)
        lines += [f"- of the {len(multi)} pairs with more than one structure, the agreed "
                  f"set is {shrink:.0%} the size of the union — crystals of one pair do "
                  f"not contact identical residues", ""]
    if rows:
        lines += ["## What each split can support", "",
                  "| level | test rows | scorable pairs | proteins | drugs | swappable |",
                  "|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['level'].replace('_', '-')} | {row['test_rows']} | "
                         f"**{row['scorable']}** | {row['proteins']} | {row['drugs']} | "
                         f"{row['swappable']} |")
        thin = [r["level"] for r in rows if r["scorable"] < 20]
        lines += ["", "A pair is scorable only where that exact drug has been crystallised "
                  "with that exact kinase. "
                  + (f"At {', '.join(t.replace('_', '-') for t in thin)} there are too few "
                     "pairs for a precision@k test to be worth much, which is a fact about "
                     "what DAVIS supports, not a result." if thin else ""), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", default="davis", choices=("davis", "kiba"))
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--min-fraction", type=float, default=MIN_FRACTION)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N targets (for a quick check)")
    args = parser.parse_args()
    sites, report = build(args.dataset, args.split_root, args.min_fraction, args.limit)
    f = out_files(args.dataset)
    with open(f["sites"], "w") as handle:
        json.dump(sites, handle, indent=1)
    with open(f["report"], "w") as handle:
        json.dump(report, handle, indent=1)
    swapped = swap_drugs(sites)
    with open(f["swapped"], "w") as handle:
        json.dump(swapped, handle, indent=1)
    # The swapped arm can only cover proteins with at least two crystallised drugs, so
    # the full drug arm has more pairs than it does (36 vs 29 at DAVIS random). Comparing
    # the two then compares different pair sets. This file is the drug arm restricted to
    # exactly the swappable pairs: same keys, same n, so the difference between the arms
    # is the drug and nothing else.
    paired = {key: sites[key] for key in swapped}
    with open(f["paired"], "w") as handle:
        json.dump(paired, handle, indent=1)
    print(f"{len(swapped)} pairs written as the swapped-drug control, and the same "
          f"{len(paired)} as the paired drug arm")
    text = summary(sites, report,
                   coverage(sites, args.split_root, args.dataset, swapped))
    print("\n" + text)
    with open(f"results/drug_sites_{args.dataset}.md", "w") as handle:
        handle.write(text)
    print(f"Saved -> {f['sites']}, {f['report']}, results/drug_sites_{args.dataset}.md")


if __name__ == "__main__":
    main()
