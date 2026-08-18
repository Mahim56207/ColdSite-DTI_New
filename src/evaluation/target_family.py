"""
Target-family stratification — the audit's main confound control.

The problem this exists to solve
-------------------------------
DAVIS and KIBA are kinase panels. A "cold-target" kinase is not an unfamiliar
protein: it shares the ATP-binding pocket architecture of the several hundred
kinases already in training, and UniProt annotates that pocket in nearly all of
them. A model can therefore score well on cold-target precision@k by learning
"ATP pockets look roughly like this" -- which is exactly the generic
pattern-matching the paper claims to be testing against.

Left uncontrolled, the headline ladder may be measuring kinase-family
similarity rather than explanation robustness, and that is the objection most
likely to sink the paper in review.

The control
-----------
Stratify every evaluated target into KINASE / NON_KINASE / UNKNOWN and report
the ladder separately for each. The antiviral subset (HIV-1 protease, HIV-1 RT,
SARS-CoV-2 Mpro and RdRp, influenza neuraminidase) is entirely non-kinase, which
is what turns it from a decorative case study into the control arm.

Interpreting the result
-----------------------
  fidelity holds on kinases, collapses on non-kinases
      -> the kinase result was family similarity. Report both; this IS the
         finding, and it is a stronger one than the original framing.
  fidelity degrades similarly on both
      -> the degradation is real and not a family artefact. The claim survives.
  fidelity holds on both
      -> an honest negative result. Report it.

All three outcomes are publishable. What is not publishable is reporting the
kinase number alone and hoping nobody asks.
"""
import json
import os
import re

KINASE = "kinase"
NON_KINASE = "non_kinase"
UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# accession -> gene symbol mapping
# ---------------------------------------------------------------------------
# DAVIS names its targets with gene symbols, so the regexes below classify it
# directly. KIBA names its targets with UniProt accessions ('O00141'), which
# carry no family information whatsoever -- every KIBA target reads UNKNOWN and
# the control arm cannot run on that dataset at all.
#
# The fix is a mapping, registered once and consulted on every lookup. It is a
# module-level registry rather than a parameter threaded through every call
# because the consumers that matter (run_audit's stratified_indices, the
# stratified figure) sit several frames deep, and a mapping that has to be
# passed by hand is a mapping somebody will forget to pass -- which fails
# silently as "no non-kinase targets found" rather than as an error.
#
# Nothing is auto-loaded from disk. Classification must stay a pure function of
# its inputs plus an explicitly registered map, or the same target would
# classify differently depending on which files happen to exist.
_GENE_MAP: dict = {}


# Explicit family assignments, keyed by target id. Separate from the gene map
# because it answers a different question: the gene map supplies a *name* for
# the regexes to read, this supplies the *answer* directly.
#
# It exists because the regexes below cannot classify a general protein. They
# recognise kinases by naming convention and recognise a short list of antiviral
# enzymes, and everything else -- a prostaglandin synthase, a bromodomain, a
# nuclear receptor -- comes back UNKNOWN. That is the right default for DAVIS
# and KIBA, where the only evidence is a bare identifier. It is the wrong
# default for the non-kinase control panel, where the builder has the full
# UniProt entry in hand and has already checked the protein against its
# annotated binding ligands.
#
# Assignments are written to disk by src/data/build_nonkinase_panel.py alongside
# the evidence for each one, so a reviewer can audit the call rather than trust
# a regex to have re-derived it.
_FAMILY_MAP: dict = {}


def set_family_map(mapping: dict) -> None:
    """Register target_id -> KINASE / NON_KINASE / UNKNOWN, decided elsewhere."""
    global _FAMILY_MAP
    _FAMILY_MAP = dict(mapping or {})


def clear_family_map() -> None:
    set_family_map({})


def load_family_map(path: str, register: bool = True) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No family map at {path}. Build it with:\n"
            f"    python -m src.data.build_nonkinase_panel --build"
        )
    with open(path) as handle:
        raw = json.load(handle)
    mapping = {k: str(v) for k, v in raw.items() if v}
    if register:
        set_family_map(mapping)
    return mapping


def set_gene_map(mapping: dict) -> None:
    """Register accession -> gene symbol. Replaces any previous registration."""
    global _GENE_MAP
    _GENE_MAP = dict(mapping or {})


def clear_gene_map() -> None:
    set_gene_map({})


def get_gene_map() -> dict:
    return dict(_GENE_MAP)


def load_gene_map(path: str, register: bool = True) -> dict:
    """Load a mapping written by `python -m src.data.build_gene_map`.

    Accepts either {accession: "SYMBOL"} or the richer
    {accession: {"gene_name": ..., "protein_name": ...}} form, so the raw
    provenance file can be handed straight in during a one-off check.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No gene map at {path}. Build it with:\n"
            f"    python -m src.data.build_gene_map --dataset kiba"
        )
    with open(path) as handle:
        raw = json.load(handle)

    mapping = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            symbol = value.get("gene_name") or ""
            protein = value.get("protein_name") or ""
            mapping[key] = f"{symbol} {protein}".strip()
        else:
            mapping[key] = str(value or "")

    mapping = {k: v for k, v in mapping.items() if v}
    if register:
        set_gene_map(mapping)
    return mapping

# DAVIS/KIBA target IDs are gene symbols. Kinase gene families follow strong
# naming conventions, which covers most of the panel without a lookup table.
_KINASE_NAME = re.compile(
    r"^(ABL|AKT|ALK|AMPK|AURK|AXL|BRAF|BTK|CAMK|CDK|CHEK|CSNK|CSF1R|DDR|"
    r"EGFR|EPH|ERBB|ERK|FGFR|FLT|FYN|GSK3|HCK|IGF1R|IKB|INSR|IRAK|ITK|JAK|"
    r"KDR|KIT|LCK|LYN|MAP\d?K|MAPK|MELK|MET|MTOR|MYLK|NEK|NTRK|PAK|PDGFR|"
    r"PDPK|PIK3|PIM|PKN|PLK|PRKA|PRKC|PRKD|PTK|RAF1|RET|ROCK|ROS1|RPS6K|"
    r"SGK|SRC|STK|SYK|TEK|TGFBR|TIE|TNK|TYK|VEGFR|WEE1|YES1|ZAP70)",
    re.IGNORECASE)

_KINASE_WORD = re.compile(r"kinase|\bCDK\b|\bMAPK\b", re.IGNORECASE)

# The antiviral control arm. None of these are kinases.
_NON_KINASE_WORD = re.compile(
    r"protease|proteinase|reverse transcriptase|neuraminidase|polymerase|"
    r"nsp5|nsp12|rdrp|mpro|3c[- ]?like|integrase|hydrolase|isomerase|"
    r"carbonic anhydrase|secretase",
    re.IGNORECASE)


def classify_target(target_id: str, description: str = "", gene_map: dict = None) -> str:
    """Target identifier -> KINASE / NON_KINASE / UNKNOWN.

    Non-kinase patterns are checked FIRST. Several viral polymerases and
    proteases carry gene-symbol prefixes that collide with kinase families, and
    a false KINASE label would contaminate the control arm -- the one place in
    the audit where contamination is unrecoverable.

    `gene_map` (or, if omitted, whatever was registered with set_gene_map) adds
    the UniProt gene symbol and protein name to the text being matched. It is
    additive, never a replacement: an accession that resolves to nothing still
    gets classified on its own name, so registering a partial map can only
    improve coverage, never change an already-confident answer.
    """
    key = str(target_id).strip()

    # An explicit assignment wins. It was made with the full UniProt entry
    # available, which is strictly more evidence than the regexes below have.
    assigned = _FAMILY_MAP.get(key)
    if assigned in (KINASE, NON_KINASE, UNKNOWN):
        return assigned

    mapping = _GENE_MAP if gene_map is None else gene_map
    resolved = mapping.get(key, "") if mapping else ""

    text = f"{target_id} {resolved} {description}".strip()
    if not text:
        return UNKNOWN

    if _NON_KINASE_WORD.search(text):
        return NON_KINASE
    if _KINASE_WORD.search(text):
        return KINASE

    # Try the raw identifier first, then the mapped gene symbol. A KIBA target
    # like 'O00141' carries no family signal on its own; 'SGK3' does, and some
    # kinase entries name the family only in the symbol and never spell the word
    # "kinase" in the description.
    candidates = [str(target_id).split("(")[0].strip()]
    if resolved:
        candidates.append(resolved.split()[0].strip())
    for base in candidates:
        if base and _KINASE_NAME.match(base):
            return KINASE
    return UNKNOWN


def stratify(target_ids, descriptions: dict = None, gene_map: dict = None) -> dict:
    """Group target IDs by family. Returns {family: [target_id, ...]}."""
    descriptions = descriptions or {}
    groups = {KINASE: [], NON_KINASE: [], UNKNOWN: []}
    for target_id in target_ids:
        family = classify_target(
            target_id, descriptions.get(target_id, ""), gene_map=gene_map)
        groups[family].append(target_id)
    return groups


def stratified_indices(target_ids, gene_map: dict = None) -> dict:
    """{family: [positions in the input list]} -- for slicing aligned results."""
    out = {KINASE: [], NON_KINASE: [], UNKNOWN: []}
    for i, target_id in enumerate(target_ids):
        out[classify_target(target_id, gene_map=gene_map)].append(i)
    return out


def confound_report(target_ids, gene_map: dict = None) -> dict:
    """Is there enough non-kinase data for the control to mean anything?

    `control_is_usable` is the gate. Below ~20 distinct non-kinase targets the
    comparison has no power, and the honest move is to state the limitation
    rather than report a stratified number a reviewer will not believe.
    """
    groups = stratify(target_ids, gene_map=gene_map)
    total = len(target_ids)
    n_non_kinase = len(set(groups[NON_KINASE]))

    return {
        "n_total": total,
        "n_kinase": len(groups[KINASE]),
        "n_non_kinase": len(groups[NON_KINASE]),
        "n_unknown": len(groups[UNKNOWN]),
        "distinct_non_kinase": n_non_kinase,
        "kinase_fraction": len(groups[KINASE]) / total if total else float("nan"),
        "control_is_usable": n_non_kinase >= 20,
        "note": (
            "Enough non-kinase targets for a stratified comparison."
            if n_non_kinase >= 20 else
            "Too few non-kinase targets to stratify. Either enlarge the "
            "antiviral/BindingDB subset or state the kinase confound as an "
            "explicit limitation in the Discussion -- do not report the "
            "unstratified ladder as if the confound were absent."
        ),
    }


def _report_panel() -> int:
    """The non-kinase control arm, reported against its own family assignments.

    Separate from the DAVIS/KIBA report because the panel is a different kind
    of object: its targets are BindingDB proteins that no DAVIS/KIBA-trained
    model has ever seen, and their family is decided by the builder from
    UniProt's annotated ligand rather than guessed from a gene symbol. Reading
    the panel through `classify_target` would return UNKNOWN for all 60 and
    report `control_is_usable: False` on an arm that is in fact usable.
    """
    sites_path = "data/nonkinase_ground_truth_sites.json"
    families_path = "data/nonkinase_panel_families.json"

    for path in (sites_path, families_path):
        if not os.path.exists(path):
            print(f"\n{path} -- missing. Build the panel first:\n"
                  f"    python -m src.data.build_nonkinase_panel --build")
            return 1

    with open(sites_path) as handle:
        targets = list(json.load(handle))
    # register: classify_target reads the family map before it tries to guess
    # from the gene symbol, and an accession like P14416 is unguessable
    mapping = load_family_map(families_path)

    report = confound_report(targets)
    print(f"\n{sites_path}")
    print(f"  family map: {len(mapping)} assignments from {families_path}")
    for key, value in report.items():
        print(f"  {key:24s} {value}")
    return 0


if __name__ == "__main__":
    import sys

    if "--panel" in sys.argv[1:]:
        raise SystemExit(_report_panel())

    DATASETS = {
        "davis": ("data/davis_ground_truth_sites.json",
                  "data/davis_uniprot_to_gene.json"),
        "kiba": ("data/kiba_ground_truth_sites.json",
                 "data/kiba_uniprot_to_gene.json"),
    }

    for dataset, (sites_path, map_path) in DATASETS.items():
        if not os.path.exists(sites_path):
            print(f"\n{sites_path}  -- missing, skipping. Fetch it with: "
                  f"python -m src.data.fetch_binding_sites --dataset {dataset}")
            continue

        with open(sites_path) as f:
            targets = list(json.load(f))

        mapping = {}
        if os.path.exists(map_path):
            mapping = load_gene_map(map_path, register=False)
            note = f"gene map: {len(mapping)} entries from {map_path}"
        else:
            note = (f"gene map: NONE ({map_path} not built -- run "
                    f"`python -m src.data.build_gene_map --dataset {dataset}`)")

        report = confound_report(targets, gene_map=mapping)
        print(f"\n{sites_path}")
        print(f"  {note}")
        for key, value in report.items():
            print(f"  {key:24s} {value}")
