"""
Track A (124AD0008) -- build the non-kinase control panel.

Why the antiviral five are not enough
-------------------------------------
`src/evaluation/target_family.py` gates the confound control at **>=20 distinct
non-kinase targets**, and below that it says so plainly: fewer than 20 and "the
comparison has no power, and the honest move is to state the limitation rather
than report a stratified number a reviewer will not believe."

The v2 master plan's control arm is five proteins. It cannot clear its own gate,
and that was true before any of the extraction problems. Two further findings
from the 2026-07-31 BindingDB release make the five worse than they look:

* All 18,149 SARS-CoV-2 rows are filed under "Replicase polyprotein 1ab" with
  the full **7,096-residue** polyprotein sequence. Mpro is residues 3264-3569 of
  that chain and only 3 rows say so. Separating Mpro from RdRp by target name is
  not possible, and 7,096 residues sits almost entirely outside the model's
  1,000-residue window regardless.

* That leaves HIV-1 protease, HIV-1 RT and influenza neuraminidase: three
  distinct proteins against a threshold of twenty.

So the arm is widened. The question the control has to answer is not "do viral
proteins behave differently" -- it is "does the cold-target result survive on
proteins that do not share the ATP pocket the model saw several hundred times in
training". Any well-annotated non-kinase answers that. The antiviral five stay
in as a named case study inside the panel, which is what the master plan wanted
them for; they simply stop being the whole of it.

Where the sequence comes from, and why it matters
-------------------------------------------------
Sequences are taken from **UniProt**, not from BindingDB's target-chain column,
even though BindingDB ships one. The binding-site coordinates come from UniProt,
and a coordinate is meaningless against a different sequence. BindingDB chains
are frequently constructs -- tagged, truncated, or a single domain -- so pairing
its sequence with UniProt's numbering would silently offset every index. That is
the same failure as the 1-indexed/0-indexed bug in Section 6 of the handover:
no crash, plausible output, every number wrong.

Usage
-----
    # 1. one pass over the bulk TSV, counting pairs per UniProt accession
    python -m src.data.build_nonkinase_panel --scan --source data/raw/BindingDB_All.tsv

    # 2. fetch UniProt for the busiest accessions, keep the non-kinase ones
    #    that have real binding-site annotation (needs rest.uniprot.org)
    python -m src.data.build_nonkinase_panel --select

    # 3. write the panel CSV + its ground truth
    python -m src.data.build_nonkinase_panel --build

Then confirm the arm is real:

    python -m src.evaluation.target_family --panel
"""
import argparse
import collections
import json
import os
import re

import numpy as np
import pandas as pd

from src.data.extract_antiviral import (
    AFFINITY_COLUMNS,
    _parse_affinity,
    find_column,
    to_p_scale,
)

PAIRS_PATH = "data/raw/_nonkinase_pairs.tsv"
COUNTS_PATH = "data/raw/_nonkinase_accession_counts.tsv"
SELECTED_PATH = "data/nonkinase_panel_targets.json"
PANEL_PATH = "data/processed/nonkinase_panel.csv"
GROUND_TRUTH_PATH = "data/nonkinase_ground_truth_sites.json"
PROVENANCE_PATH = "data/nonkinase_ground_truth_sites_provenance.json"
FAMILY_MAP_PATH = "data/nonkinase_panel_families.json"

# Per-accession cap on rows carried out of the scan. Without it a handful of
# heavily-assayed targets (PARP1 alone has ~8,000 rows) would dominate both the
# file size and, worse, the panel's average -- precision@k averaged over pairs
# rather than proteins would then be a report on PARP1 wearing a panel's name.
MAX_ROWS_PER_ACCESSION = 400

# Below this a target has too little data to say anything about.
MIN_PAIRS_PER_ACCESSION = 40

# The model reads at most this many residues, so a site past it can never be
# retrieved and the protein scores a guaranteed zero for reasons that have
# nothing to do with explanation quality. Keep the panel inside the window.
MAX_SEQUENCE_LENGTH = 1000

# How many accessions to spend UniProt requests on in --select.
CANDIDATE_LIMIT = 400

# Target size for the final panel. Comfortably above the gate of 20 so that
# losing some to missing annotation still leaves a usable arm.
PANEL_TARGET_SIZE = 60


# The confound is not "kinase" as a word. It is the ATP pocket: DAVIS and KIBA
# train the model on several hundred proteins that all bind a nucleotide in a
# structurally similar cleft, and the question is whether cold-target success is
# that pocket being recognised rather than an explanation generalising.
#
# So a control target must not bind a nucleotide either. Several non-kinases do
# -- HSP90, ATPases, helicases, DNA gyrase B, myosin. Admitting them would make
# the control arm quietly contain the very thing it is controlling for, and the
# stratified panel would show "no difference" for a reason that has nothing to
# do with the paper's claim.
#
# Checked against UniProt's annotated ligand rather than the protein's name,
# because the name does not reliably say ("Heat shock protein HSP 90-alpha"
# mentions no nucleotide at all).
#
# `\b` is wrong on the left here. UniProt writes deoxy and modified forms as
# "dGTP", "8-oxo-dGTP", "2'-deoxy-ATP" -- and `\bGTP\b` cannot match inside
# "dGTP" because `d` is a word character, so a Nudix hydrolase whose annotated
# ligand is 8-oxo-dGTP would sail into the control arm as a clean non-binder.
# Match on a non-letter boundary instead, with an optional deoxy prefix, and
# order the alternatives longest-first so NADP is not consumed as NAD.
_NUCLEOTIDE_LIGAND = re.compile(
    r"(?:^|[^A-Za-z])(?:d|dd|2'-deoxy-)?"
    # reduced forms spelled out: the trailing-letter guard below rejects
    # "NADPH" if only "NADP" is listed, and NADPH-dependent enzymes are
    # exactly the kind of nucleotide-cofactor binder this filter is for
    r"(?:NADPH|NADH|NADP|NAD|FADH2|FADH|FAD|FMNH2|FMN|"
    r"ATP|ADP|AMP|GTP|GDP|GMP|UTP|UDP|UMP|"
    r"CTP|CDP|CMP|TTP|TDP|TMP|ITP|IDP|IMP|SAM|SAH|CoA)"
    r"(?![A-Za-z])"
    r"|S-adenosyl|nucleotide|nucleoside|adenosine|guanosine|inosine|"
    r"deoxyribonucle|ribonucle|purine",
    re.IGNORECASE)

_KINASE_EVIDENCE = re.compile(
    r"kinase|phosphotransferase|\bPI3K\b|\bPI4K\b", re.IGNORECASE)


def binding_ligands(entry: dict) -> list:
    """Ligand names attached to this entry's binding-site features.

    UniProt records the ligand on the feature itself; `extract_features` in the
    fetcher drops it because DAVIS and KIBA do not need it. The panel does --
    it is the only reliable way to tell an ATP binder from a non-binder.
    """
    ligands = []
    for feature in (entry or {}).get("features", []):
        if feature.get("type") not in ("Binding site", "Active site",
                                       "Nucleotide binding"):
            continue
        ligand = feature.get("ligand") or {}
        name = ligand.get("name") or ""
        part = (feature.get("ligandPart") or {}).get("name") or ""
        text = f"{name} {part} {feature.get('description', '') or ''}".strip()
        if text:
            ligands.append(text)
    return ligands


def features_with_ligands(entry: dict) -> list:
    """Binding-site features, keeping the ligand name on each one.

    `extract_features` in the fetcher drops the ligand because DAVIS and KIBA do
    not need it. The panel does, and not only for the nucleotide filter: the
    panel's targets are annotated for a much wider variety of ligands than a
    kinase panel is, and not all of them are places a drug goes.

    A sodium site in a serotonin transporter is a real UniProt binding site and
    a real coordination residue -- and no SSRI binds there. Counting it as a
    correct answer for precision@k inflates the metric exactly the way the
    `Site` catch-all did before it was removed. Carrying the ligand through to
    the ground truth lets Track C see that distinction instead of having to
    assume it away.

    Note this cuts both ways, which is why nothing is filtered here: the zinc in
    carbonic anhydrase and in the HDACs *is* the drug site -- inhibitors of both
    chelate that zinc directly. A blanket "drop the metal sites" rule would be
    as wrong as counting them all.
    """
    out = []
    for feature in (entry or {}).get("features", []):
        ftype = feature.get("type")
        if ftype not in ("Binding site", "Active site", "Nucleotide binding"):
            continue
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value", start)
        if start is None:
            continue
        out.append({
            "start": int(start),
            "end": int(end if end is not None else start),
            "type": ftype,
            "description": feature.get("description", "") or "",
            "ligand": (feature.get("ligand") or {}).get("name", "") or "",
        })
    return out


def panel_verdict(gene: str, protein: str, entry: dict) -> tuple:
    """(is_usable_control, reason).

    Rejects in order of how badly the target would damage the arm:
    a kinase outright, then anything binding a nucleotide.
    """
    from src.evaluation.target_family import KINASE, classify_target

    name_text = f"{gene} {protein}"

    # Two independent kinase checks, because neither alone is sufficient.
    #
    # The name often does not say it: UniProt's recommended name for EGFR is
    # "Epidermal growth factor receptor", with no "kinase" anywhere. In real
    # data EGFR is caught by its ATP ligand, but relying on that means a kinase
    # with sparse ligand annotation walks in.
    #
    # So the curated gene-symbol families in target_family.py are consulted as
    # well. That regex is deliberately eager -- METAP2 collides with the "MET"
    # family prefix and gets rejected as a kinase despite being a perfectly good
    # non-kinase drug target. That is the right direction to be wrong in: it
    # costs one candidate out of a few hundred, where the opposite error puts a
    # kinase in the arm built to exclude them.
    if _KINASE_EVIDENCE.search(name_text):
        return False, "kinase"
    if classify_target(gene or "?", gene_map={gene or "?": name_text}) == KINASE:
        return False, "kinase"

    ligands = binding_ligands(entry)
    for ligand in ligands:
        if _NUCLEOTIDE_LIGAND.search(ligand):
            return False, f"binds_nucleotide({ligand[:28]})"

    if not protein.strip():
        # No recommended name means no evidence either way, and a control arm
        # is the wrong place to guess.
        return False, "unnamed_entry"

    return True, "ok"


def scan(source_path: str, chunksize: int = 100_000, verbose: bool = True):
    """One pass over the bulk TSV: pair counts per UniProt accession.

    Deliberately keeps only accession, SMILES and affinity -- no sequences. The
    sequence is fetched from UniProt in --select, so carrying BindingDB's here
    would cost ~500 bytes a row for a column that must not be used anyway.
    """
    counts = collections.Counter()
    kept, scanned = [], 0

    reader = pd.read_csv(source_path, sep="\t", chunksize=chunksize,
                         low_memory=False, on_bad_lines="skip")
    for chunk in reader:
        scanned += len(chunk)

        accession_column = (find_column(chunk.columns, "uniprot", "swissprot", "primary")
                            or find_column(chunk.columns, "uniprot", "primary"))
        smiles_column = find_column(chunk.columns, "ligand", "smiles")
        if accession_column is None or smiles_column is None:
            raise KeyError(
                f"need a UniProt-accession column and a SMILES column; got "
                f"{list(chunk.columns)[:14]}..."
            )

        values = pd.Series(np.nan, index=chunk.index)
        kinds = pd.Series(pd.NA, index=chunk.index, dtype="object")
        for kind, column_name in AFFINITY_COLUMNS.items():
            column = find_column(chunk.columns, column_name.split()[0])
            if column is None:
                continue
            parsed = chunk[column].map(_parse_affinity)
            take = values.isna() & parsed.notna()
            values = values.where(~take, parsed)
            kinds = kinds.where(~take, kind)

        frame = pd.DataFrame({
            "accession": chunk[accession_column],
            "Drug": chunk[smiles_column],
            "affinity_nM": values,
            "affinity_type": kinds,
        })
        frame = frame.dropna(subset=["accession", "Drug", "affinity_nM"])
        frame = frame[frame["accession"].astype(str).str.strip().str.len() >= 6]

        if len(frame):
            counts.update(frame["accession"].astype(str))
            kept.append(frame)

        if verbose:
            print(f"  scanned {scanned:,} rows, "
                  f"{len(counts):,} distinct accessions", end="\r")

    if verbose:
        print()
    if not kept:
        raise RuntimeError("no rows with a UniProt accession and an affinity")

    pairs = pd.concat(kept, ignore_index=True)
    pairs["accession"] = pairs["accession"].astype(str).str.strip()

    # cap per accession, and drop the thinly-measured ones
    frequent = {a for a, n in counts.items() if n >= MIN_PAIRS_PER_ACCESSION}
    pairs = pairs[pairs["accession"].isin(frequent)]
    pairs = pairs.groupby("accession", group_keys=False).head(MAX_ROWS_PER_ACCESSION)

    os.makedirs(os.path.dirname(PAIRS_PATH) or ".", exist_ok=True)
    pairs.to_csv(PAIRS_PATH, sep="\t", index=False)
    (pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]),
                  columns=["accession", "n_pairs"])
       .to_csv(COUNTS_PATH, sep="\t", index=False))

    print(f"  {len(counts):,} distinct accessions, "
          f"{len(frequent):,} with >= {MIN_PAIRS_PER_ACCESSION} pairs")
    print(f"  saved -> {PAIRS_PATH} ({len(pairs):,} rows, capped at "
          f"{MAX_ROWS_PER_ACCESSION}/accession)")
    print(f"  saved -> {COUNTS_PATH}")
    print("\nNext:  python -m src.data.build_nonkinase_panel --select")


def select(limit: int = CANDIDATE_LIMIT, panel_size: int = PANEL_TARGET_SIZE,
           delay: float = 0.2):
    """Fetch UniProt for the busiest accessions; keep usable non-kinase ones.

    Four conditions, all necessary:

      classified non-kinase   -- the whole point of the arm
      has binding-site annotation -- no annotation, no precision@k, at all
      sequence within the window  -- otherwise a guaranteed zero
      enough measured pairs       -- enforced already in --scan
    """
    import time

    from src.data.fetch_binding_sites import (
        UNIPROT_ENTRY,
        _get_json,
        _entry_sequence_length,
        extract_features,
        extract_names,
    )

    if not os.path.exists(COUNTS_PATH):
        raise SystemExit(
            f"No {COUNTS_PATH}. Run the scan first:\n"
            f"    python -m src.data.build_nonkinase_panel --scan "
            f"--source data\\raw\\BindingDB_All.tsv"
        )

    counts = pd.read_csv(COUNTS_PATH, sep="\t")
    counts = counts[counts["n_pairs"] >= MIN_PAIRS_PER_ACCESSION]
    candidates = counts["accession"].astype(str).tolist()[:limit]
    print(f"Fetching UniProt for {len(candidates)} candidate accessions...")

    selected, rejected = {}, collections.Counter()
    by_gene = {}   # upper-case gene symbol -> accession currently holding it

    for i, accession in enumerate(candidates, start=1):
        if len(selected) >= panel_size:
            print(f"\n  reached the panel target of {panel_size}; stopping early")
            break
        try:
            entry = _get_json(UNIPROT_ENTRY.format(accession))
        except Exception:
            rejected["fetch_failed"] += 1
            continue
        time.sleep(delay)

        gene, protein = extract_names(entry)
        length = _entry_sequence_length(entry)
        features = features_with_ligands(entry)
        sequence = ((entry or {}).get("sequence") or {}).get("value", "")
        organism = (entry or {}).get("organism") or {}
        taxon = organism.get("taxonId")
        organism_name = organism.get("scientificName", "") or ""

        usable, reason = panel_verdict(gene, protein, entry)
        if not usable:
            rejected[reason.split("(")[0]] += 1
            continue
        if not features:
            rejected["no_binding_annotation"] += 1
            continue
        if not sequence or length > MAX_SEQUENCE_LENGTH:
            rejected["sequence_too_long_or_missing"] += 1
            continue

        # One accession per gene. BindingDB carries rat and human orthologues of
        # the same target under separate accessions -- Drd2/DRD2, Htr1a/HTR1A,
        # Slc6a4/SLC6A4 -- and they are ~90% identical with the same binding
        # residues. Counting both would inflate the panel's apparent size
        # without adding an independent test of anything, and the >=20 gate is a
        # statement about independent targets. Human wins where available: it is
        # what the drug programmes behind these measurements were aimed at.
        key = (gene or accession).upper()
        incumbent = by_gene.get(key)
        if incumbent:
            incumbent_is_human = selected[incumbent]["organism_taxon"] == 9606
            if incumbent_is_human or taxon != 9606:
                rejected["orthologue_of_a_target_already_in"] += 1
                continue
            del selected[incumbent]
            rejected["orthologue_replaced_by_human"] += 1

        by_gene[key] = accession
        selected[accession] = {
            "accession": accession,
            "gene_name": gene,
            "protein_name": protein,
            "organism_taxon": taxon,
            "organism_name": organism_name,
            "sequence": sequence,
            "sequence_length": length,
            "features": features,
            "n_features": len(features),
            "n_pairs": int(counts.loc[counts["accession"] == accession,
                                      "n_pairs"].iloc[0]),
            # kept so the non-kinase call is auditable rather than asserted
            "ligands": binding_ligands(entry),
        }
        print(f"  [{i}/{len(candidates)}] {accession} {gene:12s} "
              f"{len(features):>2} sites, {length:>4} aa  "
              f"{organism_name[:18]:20s} {protein[:40]}")

    if not selected:
        raise SystemExit(
            "no accession passed all four conditions. Loosen "
            "MAX_SEQUENCE_LENGTH or raise CANDIDATE_LIMIT and retry -- but "
            "check the rejection counts below before changing anything."
        )

    with open(SELECTED_PATH, "w") as handle:
        json.dump(selected, handle, indent=2)

    print(f"\nselected {len(selected)} non-kinase targets -> {SELECTED_PATH}")
    print("rejected:")
    for reason, n in rejected.most_common():
        print(f"  {reason:36s} {n}")
    print(f"\ngate is >= 20 distinct non-kinase targets. "
          f"{'PASS' if len(selected) >= 20 else 'STILL SHORT'}")
    print("\nNext:  python -m src.data.build_nonkinase_panel --build")


def build():
    """Join the selected targets to their measured pairs and write the panel."""
    if not os.path.exists(SELECTED_PATH):
        raise SystemExit(
            f"No {SELECTED_PATH}. Run --select first."
        )
    if not os.path.exists(PAIRS_PATH):
        raise SystemExit(f"No {PAIRS_PATH}. Run --scan first.")

    with open(SELECTED_PATH) as handle:
        selected = json.load(handle)

    pairs = pd.read_csv(PAIRS_PATH, sep="\t", low_memory=False)
    pairs = pairs[pairs["accession"].astype(str).isin(selected)].copy()

    pairs["Target_ID"] = pairs["accession"].astype(str)
    pairs["Target"] = pairs["Target_ID"].map(lambda a: selected[a]["sequence"])
    pairs["gene_name"] = pairs["Target_ID"].map(lambda a: selected[a]["gene_name"])
    pairs["protein_name"] = pairs["Target_ID"].map(lambda a: selected[a]["protein_name"])
    pairs["Drug_ID"] = pairs["Drug"].astype(str).str.strip()
    pairs["Y"] = [to_p_scale(v) for v in pairs["affinity_nM"].values]

    panel = (pairs.dropna(subset=["Drug", "Target", "Y"])
                  .drop_duplicates(subset=["Drug_ID", "Target_ID", "affinity_type"])
                  [["Drug_ID", "Drug", "Target_ID", "Target", "Y",
                    "affinity_nM", "affinity_type", "gene_name", "protein_name"]]
                  .reset_index(drop=True))

    ground_truth = {a: record["features"] for a, record in selected.items()}
    provenance = {
        a: {
            "resolved_from": a,
            "uniprot_accession": a,
            "resolution": "nonkinase_panel",
            "is_variant": False,
            "n_features": record["n_features"],
            "gene_name": record["gene_name"],
            "protein_name": record["protein_name"],
            "sequence_length": record["sequence_length"],
        }
        for a, record in selected.items()
    }

    # The family assignment is written out explicitly rather than left for the
    # regexes in target_family.py to re-derive. Those regexes recognise kinases
    # by naming convention and a short list of antiviral enzymes; a prostaglandin
    # synthase or a bromodomain comes back UNKNOWN, and an UNKNOWN target counts
    # toward neither arm. Writing the answer down -- next to the ligand evidence
    # it was based on, in nonkinase_panel_targets.json -- is both correct and
    # auditable.
    families = {a: "non_kinase" for a in selected}

    os.makedirs(os.path.dirname(PANEL_PATH) or ".", exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False)
    with open(GROUND_TRUTH_PATH, "w") as handle:
        json.dump(ground_truth, handle, indent=2)
    with open(PROVENANCE_PATH, "w") as handle:
        json.dump(provenance, handle, indent=2)
    with open(FAMILY_MAP_PATH, "w") as handle:
        json.dump(families, handle, indent=2, sort_keys=True)

    print(f"Saved {len(panel):,} pairs across "
          f"{panel['Target_ID'].nunique()} non-kinase targets -> {PANEL_PATH}")
    print(f"Saved ground truth -> {GROUND_TRUTH_PATH}")
    print(f"Saved provenance   -> {PROVENANCE_PATH}")
    print(f"Saved family map   -> {FAMILY_MAP_PATH}")

    from src.evaluation.target_family import confound_report, set_family_map
    set_family_map(families)
    report = confound_report(panel["Target_ID"].tolist())
    print("\nconfound report on the panel:")
    for key, value in report.items():
        print(f"  {key:24s} {value}")

    n = report["distinct_non_kinase"]
    print(f"\ngate is >= 20 distinct non-kinase targets: "
          f"{'PASS' if n >= 20 else f'SHORT by {20 - n}'}")

    ligand_summary(selected)


# Ions that coordinate a transported substrate rather than sit in a drug pocket.
# Deliberately NOT filtered anywhere -- only counted. Zinc is on the opposite
# side of the same question: carbonic anhydrase and HDAC inhibitors chelate the
# catalytic zinc directly, so a zinc site in those targets IS the drug site.
# The distinction is per-protein, not per-ion, which is why it is reported for a
# human to judge instead of being decided by a regex.
_COTRANSPORT_ION = ("Na(+)", "Cl(-)", "K(+)", "chloride")


def ligand_summary(selected: dict) -> None:
    """Report what the panel's binding sites are actually annotated for.

    precision@k counts a hit when attention lands on any annotated position. If
    a large share of those positions are sodium-coordination residues in a
    transporter, the metric is partly scoring "did attention find the sodium
    site", which no drug binds. That is the same inflation the `Site` catch-all
    caused in the DAVIS ground truth, and it is worth knowing about before the
    numbers are produced rather than after.
    """
    per_ligand = collections.Counter()
    ion_heavy = []
    for accession, record in selected.items():
        ligands = [f.get("ligand", "") for f in record["features"]]
        per_ligand.update(l for l in ligands if l)
        if not ligands:
            continue
        n_ion = sum(1 for l in ligands if l in _COTRANSPORT_ION)
        if n_ion and n_ion / len(ligands) >= 0.5:
            ion_heavy.append((accession, record["gene_name"], n_ion, len(ligands),
                              record["protein_name"]))

    print("\nmost common annotated ligands across the panel:")
    for ligand, count in per_ligand.most_common(12):
        print(f"  {count:>4}  {ligand[:52]}")

    if ion_heavy:
        print(f"\n{len(ion_heavy)} target(s) where at least half the annotated "
              f"sites are cotransport ions:")
        for accession, gene, n_ion, total, protein in sorted(ion_heavy):
            print(f"  {accession} {gene:10s} {n_ion}/{total} ion sites   "
                  f"{protein[:44]}")
        print("  No drug binds a sodium-coordination residue, so these sites "
              "inflate precision@k.\n  Raise with 124AD0067 before the audit "
              "runs -- the ligand is on every feature in\n  the ground-truth "
              "JSON, so they can be excluded or reported separately.")


def main():
    parser = argparse.ArgumentParser(
        description="Build the non-kinase control panel")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--source", default="data/raw/BindingDB_All.tsv")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=CANDIDATE_LIMIT)
    parser.add_argument("--panel-size", type=int, default=PANEL_TARGET_SIZE)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    if not (args.scan or args.select or args.build):
        parser.error("give at least one of --scan / --select / --build")

    if args.scan:
        if not os.path.exists(args.source):
            raise SystemExit(f"{args.source} not found.")
        print(f"Scanning {args.source} for per-accession pair counts...")
        scan(args.source, chunksize=args.chunksize)
    if args.select:
        select(limit=args.limit, panel_size=args.panel_size, delay=args.delay)
    if args.build:
        build()


if __name__ == "__main__":
    main()
