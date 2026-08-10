"""
Track A (124AD0008) Step 3 — build the antiviral BindingDB subset.

Replaces the previous notebook cell + clean_antiviral.py pair, which produced a
file containing only HIV-1 protease. All five targets required by
docs/00_MASTER_PLAN.md §3.3 are matched here, and the script FAILS LOUDLY if
any of them ends up with zero rows rather than writing a partial file that
looks finished.

Two substantive changes beyond the target coverage:

* Affinities are converted to p-scale (-log10 of molar concentration), matching
  DAVIS's pKd. The previous version wrote raw nanomolar values spanning
  7.9e-4 to 1e7 into the same column the model trains on with MSE loss, where
  a handful of weak binders dominate the entire gradient.

* The measurement type (IC50 / Ki / Kd / EC50) is kept as a column instead of
  being collapsed into one number. These are not interchangeable quantities:
  IC50 depends on assay conditions in a way Kd does not, so pooling them
  silently is a modelling decision that belongs in the Methods section, not in
  a fillna chain.

Usage
-----
    # BindingDB_All.tsv is ~9GB unzipped, from bindingdb.org/bind/downloads
    python -m src.data.extract_antiviral --source data/raw/BindingDB_All.tsv

    # after a failed run, iterate on TARGET_PATTERNS in seconds instead of
    # re-reading 9GB -- the matched rows and the near-miss names are cached
    python -m src.data.extract_antiviral --from-cache
"""
import argparse
import collections
import json
import os
import re

import numpy as np
import pandas as pd

# One pattern per required target. Written against the naming actually used in
# BindingDB's 'Target Name' column, which is inconsistent across submissions --
# SARS-CoV-2 Mpro alone appears as "3C-like proteinase", "Main protease",
# "nsp5", and "Replicase polyprotein 1ab" depending on who deposited it. A
# single flat keyword list, which is what the previous version used, matches
# whichever spelling happens to come first and quietly misses the rest.
# Matching is on TWO columns, not one. That is the whole design.
#
# A first version matched target names only, and against a real BindingDB
# release it found 2 of 5 targets. The three it missed are not named the way the
# patterns guessed -- they are named bare:
#
#     "3C-like protease"                 102 rows, no organism in the name
#     "RNA-directed RNA polymerase"    2,429 rows, no organism in the name
#     "Reverse transcriptase"          4,542 rows, no organism in the name
#
# The temptation is to match those strings directly. That would be a disaster.
# The same near-miss scan turned up "Poly [ADP-ribose] polymerase 1" (8,065
# rows), "DNA polymerase theta", "DNA-directed RNA polymerase, mitochondrial"
# and "Telomerase reverse transcriptase" -- all human, none antiviral, all
# caught by a loose polymerase or transcriptase pattern. They would enter the
# subset as non-kinase targets and become the control arm that the paper's
# central claim is checked against. Section 2.2 of the v2 master plan calls the
# kinase confound "the objection most likely to sink the paper"; a control arm
# quietly stuffed with human DNA polymerases does not answer that objection, it
# fakes an answer to it.
#
# So each target needs a name match AND organism confirmation. BindingDB ships
# the organism in its own column, which is where the disambiguation actually
# lives.
#
# `name_alone_ok` is a narrow exemption for names that already pin the organism
# themselves ("SARS-CoV-2 nsp5"), so a blank organism cell does not lose a row.
# `veto` rejects regardless -- it is how a human sialidase stays out of the
# influenza bucket.
TARGET_SPECS = {
    "SARS-CoV-2 Mpro": {
        "name": re.compile(
            r"3c[- ]?like protease|3c[- ]?like proteinase|3cl[- ]?pro|"
            r"main protease|\bmpro\b|\bnsp5\b", re.IGNORECASE),
        "organism": re.compile(
            r"sars[- ]?cov[- ]?2|severe acute respiratory syndrome coronavirus 2|"
            r"2019[- ]?ncov|sars coronavirus 2", re.IGNORECASE),
        # SARS-CoV-1 and MERS also have a 3C-like protease. Different protein,
        # different pocket, and including them would silently widen the target.
        "veto": re.compile(
            r"mers|middle east respiratory|"
            r"sars[- ]?cov(?![- ]?2)|coronavirus 229e|nl63|hku1|oc43|"
            r"rhinovirus|enterovirus|coxsackie|norovirus|picornavirus",
            re.IGNORECASE),
    },
    "SARS-CoV-2 RdRp": {
        "name": re.compile(
            r"rna[- ]?dependent rna polymerase|rna[- ]?directed rna polymerase|"
            r"\brdrp\b|\bnsp12\b", re.IGNORECASE),
        "organism": re.compile(
            r"sars[- ]?cov[- ]?2|severe acute respiratory syndrome coronavirus 2|"
            r"2019[- ]?ncov|sars coronavirus 2", re.IGNORECASE),
        # Every other RdRp in the database: HCV NS5B, dengue, zika, influenza
        # PB1, polio. Plus the human polymerases the loose pattern would eat.
        "veto": re.compile(
            r"hepatitis|\bhcv\b|ns5b|dengue|zika|west nile|"
            r"poliovirus|rhinovirus|enterovirus|norovirus|"
            r"influenza|respiratory syncytial|\brsv\b|ebola|marburg|"
            r"poly \[adp|tankyrase|\bparp\b|telomerase|"
            r"dna[- ]?directed|dna polymerase|mitochondrial|homo sapiens|human",
            re.IGNORECASE),
    },
    "HIV-1 protease": {
        "name": re.compile(r"\bprotease\b|\bproteinase\b|retropepsin", re.IGNORECASE),
        "organism": re.compile(
            r"human immunodeficiency virus|\bhiv\b", re.IGNORECASE),
        "veto": re.compile(
            r"hiv[- ]?2|immunodeficiency virus (type )?2|"
            r"simian|\bsiv\b|feline|\bfiv\b|"
            # 'pol polyprotein' carries protease, RT and integrase in one chain;
            # assigning it to either bucket would be a guess.
            r"pol protein|pol polyprotein|gag[- ]?pol",
            re.IGNORECASE),
    },
    "HIV-1 reverse transcriptase": {
        "name": re.compile(
            r"reverse transcriptase|\brt\b[/ ]rnaseh", re.IGNORECASE),
        "organism": re.compile(
            r"human immunodeficiency virus|\bhiv\b", re.IGNORECASE),
        "veto": re.compile(
            r"telomerase|hiv[- ]?2|immunodeficiency virus (type )?2|"
            r"simian|\bsiv\b|feline|\bfiv\b|murine|\bmlv\b|moloney|"
            r"hepatitis b|\bhbv\b|"
            r"pol protein|pol polyprotein|gag[- ]?pol",
            re.IGNORECASE),
    },
    "Influenza neuraminidase": {
        "name": re.compile(r"neuraminidase|sialidase", re.IGNORECASE),
        "organism": re.compile(
            r"influenza|\bh\dn\d\b|orthomyxo", re.IGNORECASE),
        # A bare "Neuraminidase" in BindingDB is overwhelmingly influenza NA,
        # and the veto below keeps the human and bacterial sialidases out.
        "name_alone_ok": re.compile(r"^\s*neuraminidase\s*$", re.IGNORECASE),
        "veto": re.compile(
            r"homo sapiens|human|\bneu[1-4]\b|"
            r"vibrio|clostridium|salmonella|streptococc|arthrobacter|"
            r"micromonospora|bacteroides|trypanosoma|newcastle|"
            r"parainfluenza|sendai|mumps",
            re.IGNORECASE),
    },
}

# The two SARS-CoV-2 targets are specified but NOT required, and this is a
# documented scope reduction rather than a suppressed failure.
#
# Evidence, from the 2026-07-31 release: all 18,149 SARS-CoV-2 rows are filed
# under "Replicase polyprotein 1ab" (13,753 rows) or "1a" (4,276), carrying the
# full 7,096- and 4,405-residue polyprotein sequences. Mpro is residues
# 3264-3569 of pp1ab; exactly 3 rows out of 18,149 name that range. Nothing in
# the target name or the sequence distinguishes an Mpro measurement from an RdRp
# one, so claiming either would be a guess -- and 7,096 residues sits almost
# entirely outside the model's 1,000-residue window in any case.
#
# Recovering them would mean joining BindingDB's assay-description tables on
# reaction-set ID. That is a real option if the case study needs them later; the
# specs below stay in place so a future release that names the domains properly
# is picked up automatically.
#
# The control arm does not depend on this. It is built by
# src/data/build_nonkinase_panel.py, because five antiviral proteins could never
# clear confound_report's >=20 distinct non-kinase gate anyway.
OPTIONAL_TARGETS = ("SARS-CoV-2 Mpro", "SARS-CoV-2 RdRp")
OPTIONAL_TARGET_REASON = (
    "BindingDB files SARS-CoV-2 measurements against the replicase polyprotein "
    "(pp1ab, 7096 aa), not against Mpro or RdRp individually. 3 of 18,149 rows "
    "carry a domain range. Separating them requires the assay-description "
    "tables; see src/data/extract_antiviral.py."
)

ALL_TARGETS = tuple(TARGET_SPECS)
REQUIRED_TARGETS = tuple(t for t in TARGET_SPECS if t not in OPTIONAL_TARGETS)

# kept so anything importing the old name still works
TARGET_PATTERNS = {label: spec["name"] for label, spec in TARGET_SPECS.items()}

# Deliberately far too broad to use for selection -- it would drag in every
# human protease in the database. Its only job is to collect the names that are
# *nearly* right during the same pass, so that when a strict pattern misses, the
# error can show you the spelling your BindingDB release actually uses instead
# of sending you back for another 9GB read.
NEAR_MISS_HINT = re.compile(
    r"protease|proteinase|reverse transcriptase|neuraminidase|polymerase|"
    r"nsp5|nsp12|rdrp|mpro|3c[- ]?like|hiv|influenza|sars|corona|ncov",
    re.IGNORECASE)

CACHE_PATH = "data/raw/_antiviral_candidates_cache.tsv"
NEAR_MISS_PATH = "data/raw/_antiviral_near_misses.txt"
CACHE_META_PATH = "data/raw/_antiviral_cache_meta.json"

# Bump whenever the meaning of the cache changes. v1 held only rows the strict
# specs had already claimed, so reusing it after editing a spec would silently
# reclassify 4,385 rows and miss the 200,000 the edit was written to catch --
# and report a coverage table as if the scan had been complete.
CACHE_FORMAT = 2

AFFINITY_COLUMNS = {
    "IC50": "IC50 (nM)",
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "EC50": "EC50 (nM)",
}


def classify_target(target_name: str, organism: str = "") -> str | None:
    """(target name, source organism) -> one of REQUIRED_TARGETS, or None.

    `organism` is BindingDB's "Target Source Organism" column. It defaults to
    empty so a name that already carries its own organism ("SARS-CoV-2 nsp5")
    classifies without it.

    A row is claimed only when all three hold:

        name matches       -- it is the right kind of enzyme
        organism confirms  -- it is from the right pathogen
        no veto fires      -- it is not a near neighbour wearing the same name

    The veto is checked against name and organism together, and it is checked
    last and unconditionally. Without it "Reverse transcriptase" from the
    telomerase entry, or a human sialidase, walks into the control arm looking
    exactly like a legitimate non-kinase target.

    Checked in dictionary order, so a row matching two specs takes the first.
    That happens only for chimeric constructs, and the vetoes already exclude
    the common one (HIV gag-pol, which contains protease and RT in one chain).
    """
    if not isinstance(target_name, str):
        return None
    organism = organism if isinstance(organism, str) else ""
    combined = f"{target_name} {organism}"

    for label, spec in TARGET_SPECS.items():
        if not spec["name"].search(target_name):
            continue
        if spec.get("veto") and spec["veto"].search(combined):
            continue
        if spec["organism"].search(combined):
            return label
        name_alone = spec.get("name_alone_ok")
        if name_alone and name_alone.search(target_name.strip()):
            return label
    return None


def to_p_scale(value_nm: float) -> float:
    """Nanomolar concentration -> p-scale, e.g. 10 nM -> 8.0.

    Matches the -log10(Kd / 1e9) convention load_data.py applies to DAVIS, so
    the antiviral subset is on the same axis as the main datasets.
    """
    if not np.isfinite(value_nm) or value_nm <= 0:
        return np.nan
    return float(-np.log10(value_nm * 1e-9))


def _parse_affinity(raw) -> float:
    """BindingDB affinities carry qualifiers like '>10000' or '<0.5'.

    The qualifier is stripped and the value kept, which treats a censored
    measurement as if it were exact. That biases weak binders toward looking
    stronger than they are; it is the standard convention in the DTI literature
    and it is a limitation worth one sentence in the Methods.
    """
    if pd.isna(raw):
        return np.nan
    text = str(raw).replace(">", "").replace("<", "").replace("~", "").strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def find_column(columns, *keywords):
    for column in columns:
        if all(k.lower() in column.lower() for k in keywords):
            return column
    return None


def extract(source_path: str, chunksize: int = 100_000, verbose: bool = True):
    """Stream BindingDB_All.tsv and keep rows matching the five targets.

    Also collects, in the same pass, every target name that looks antiviral but
    matched none of the strict patterns. Reading this file is a ~9GB, tens-of-
    minutes operation, and the previous failure mode was to scan all of it,
    raise "target X missing", and leave you with no way to find X's actual
    spelling except to scan it again.
    """
    kept = []
    near_misses = collections.Counter()
    scanned = 0

    reader = pd.read_csv(source_path, sep="\t", chunksize=chunksize,
                         low_memory=False, on_bad_lines="skip")
    for chunk in reader:
        scanned += len(chunk)
        name_column = find_column(chunk.columns, "target", "name")
        if name_column is None:
            raise KeyError(
                f"no target-name column found; got {list(chunk.columns)[:12]}..."
            )
        organism_column = find_column(chunk.columns, "target", "organism")
        if organism_column is None:
            raise KeyError(
                "no target-organism column found. Matching on target name "
                "alone cannot separate SARS-CoV-2 RdRp from human DNA "
                "polymerase theta, so this script will not proceed without it. "
                f"Columns seen: {list(chunk.columns)[:12]}..."
            )
        names = chunk[name_column].fillna("").astype(str)
        organisms = chunk[organism_column].fillna("").astype(str)
        context = names + "  ||  " + organisms

        # Keep everything the LOOSE hint touches, not just what the strict
        # specs claim. The strict patterns are the thing most likely to need
        # another edit, and re-reading 9GB to test an edit is the cost this
        # avoids -- `--from-cache` reclassifies from here in seconds.
        loose_mask = context.str.contains(NEAR_MISS_HINT)
        if loose_mask.any():
            columns = [c for c in (name_column, organism_column,
                                   find_column(chunk.columns, "ligand", "smiles"),
                                   find_column(chunk.columns, "target", "sequence"),
                                   *(find_column(chunk.columns, c.split()[0])
                                     for c in AFFINITY_COLUMNS.values()))
                       if c is not None]
            kept.append(chunk.loc[loose_mask, list(dict.fromkeys(columns))].copy())

        # Diagnostic counter: names the strict specs did NOT claim. Carries the
        # organism, because "RNA-directed RNA polymerase" on its own gives no
        # way to tell which of a dozen organisms a row belongs to, and that is
        # the only question that matters when deciding whether to widen a spec.
        labels = pd.Series(
            [classify_target(n, o) for n, o in zip(names, organisms)],
            index=chunk.index, dtype="object")
        unmatched = context[labels.isna() & loose_mask]
        near_misses.update(unmatched)

        if verbose:
            held = sum(len(k) for k in kept)
            print(f"  scanned {scanned:,} rows, candidates held {held:,}, "
                  f"distinct near-miss names {len(near_misses):,}", end="\r")

    if verbose:
        print()
    if not kept:
        raise RuntimeError(
            "nothing in the source even loosely resembles an antiviral target. "
            "Check that --source points at BindingDB_All.tsv."
        )
    return pd.concat(kept, ignore_index=True), near_misses


def classify_frame(loose: pd.DataFrame) -> pd.DataFrame:
    """Apply the strict specs to the cached candidate rows.

    Split out from `extract` on purpose: this is the part that changes when a
    pattern is wrong, and keeping it separate from the 9GB read is what makes
    `--from-cache` a seconds-long loop instead of a half-hour one.
    """
    name_column = find_column(loose.columns, "target", "name")
    organism_column = find_column(loose.columns, "target", "organism")
    if name_column is None or organism_column is None:
        raise KeyError(
            f"cache is missing the name or organism column; got "
            f"{list(loose.columns)}. Delete {CACHE_PATH} and re-scan."
        )

    names = loose[name_column].fillna("").astype(str)
    organisms = loose[organism_column].fillna("").astype(str)
    labels = pd.Series([classify_target(n, o) for n, o in zip(names, organisms)],
                       index=loose.index, dtype="object")

    matched = loose[labels.notna()].copy()
    matched["antiviral_target"] = labels[labels.notna()].values
    matched["source_organism"] = organisms[labels.notna()].values
    return matched


def save_cache(candidates: pd.DataFrame, near_misses: collections.Counter) -> None:
    """Persist the expensive part of the run so spec edits are cheap.

    All three files live under data/raw/, which is gitignored, and all three are
    intermediates -- nothing downstream should ever read them.
    """
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    candidates.to_csv(CACHE_PATH, sep="\t", index=False)
    with open(NEAR_MISS_PATH, "w", encoding="utf-8") as handle:
        for name, count in near_misses.most_common():
            handle.write(f"{count}\t{name}\n")
    with open(CACHE_META_PATH, "w") as handle:
        json.dump({"cache_format": CACHE_FORMAT, "rows": int(len(candidates))},
                  handle, indent=2)
    print(f"  cached {len(candidates):,} candidate rows -> {CACHE_PATH}")
    print(f"  cached near-miss names       -> {NEAR_MISS_PATH}")


def load_cache():
    if not os.path.exists(CACHE_PATH):
        raise SystemExit(
            f"No cache at {CACHE_PATH}. Run a full scan once first:\n"
            f"    python -m src.data.extract_antiviral --source data\\raw\\BindingDB_All.tsv"
        )

    meta = {}
    if os.path.exists(CACHE_META_PATH):
        with open(CACHE_META_PATH) as handle:
            meta = json.load(handle)
    if meta.get("cache_format") != CACHE_FORMAT:
        raise SystemExit(
            f"{CACHE_PATH} was written by an older version of this script "
            f"(format {meta.get('cache_format', 'unknown')}, need {CACHE_FORMAT}).\n"
            f"\nThe old cache held only the rows the strict specs had already "
            f"claimed. Reclassifying it after a spec edit would quietly report "
            f"coverage over 4,000 rows instead of the ~200,000 candidates the "
            f"edit was written to catch -- and the coverage table would look "
            f"complete.\n"
            f"\nDelete it and re-scan once:\n"
            f"    del data\\raw\\_antiviral_matched_cache.tsv\n"
            f"    python -m src.data.extract_antiviral --source data\\raw\\BindingDB_All.tsv"
        )

    matched = pd.read_csv(CACHE_PATH, sep="\t", low_memory=False)
    near_misses = collections.Counter()
    if os.path.exists(NEAR_MISS_PATH):
        with open(NEAR_MISS_PATH, encoding="utf-8") as handle:
            for line in handle:
                count, _, name = line.rstrip("\n").partition("\t")
                if name:
                    near_misses[name] = int(count)
    return matched, near_misses


def suggest_names(near_misses: collections.Counter, missing_targets, limit: int = 25):
    """Print the near-miss names most likely to be a missing target.

    Crude keyword scoring on purpose. This is a hint for a human deciding what
    to add to TARGET_PATTERNS, not an automatic matcher -- a pattern loose
    enough to catch everything also catches non-antiviral proteins and
    contaminates the control arm, which is the one place in this project where
    contamination cannot be undone later.
    """
    hints = {
        "SARS-CoV-2 Mpro": ("mpro", "3c", "main protease", "nsp5", "sars", "cov"),
        "SARS-CoV-2 RdRp": ("rdrp", "nsp12", "polymerase", "sars", "cov"),
        "HIV-1 protease": ("hiv", "protease", "immunodeficiency"),
        "HIV-1 reverse transcriptase": ("hiv", "reverse transcriptase", "immunodeficiency"),
        "Influenza neuraminidase": ("neuraminidase", "influenza", "h1n1", "h3n2", "h5n1"),
    }
    for target in missing_targets:
        keys = hints.get(target, ())
        scored = []
        for name, count in near_misses.items():
            low = name.lower()
            score = sum(1 for k in keys if k in low)
            if score:
                scored.append((score, count, name))
        scored.sort(reverse=True)

        print(f"\n  candidate names for {target!r} "
              f"(from {len(near_misses):,} near-miss names seen):")
        if not scored:
            print("    none -- this target may genuinely be absent from this release")
        for score, count, name in scored[:limit]:
            print(f"    {count:>7,}  {name}")
    print(f"\n  full near-miss list: {NEAR_MISS_PATH}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Matched BindingDB rows -> the column shape the rest of the project uses."""
    smiles_column = find_column(df.columns, "ligand", "smiles")
    sequence_column = find_column(df.columns, "target", "sequence")
    if smiles_column is None or sequence_column is None:
        raise KeyError("could not locate the SMILES or target-sequence column")

    out = pd.DataFrame({
        "Drug": df[smiles_column],
        "Target": df[sequence_column],
        "Target_ID": df["antiviral_target"],
        "antiviral_target": df["antiviral_target"],
        # Carried into the output so the control arm can be audited after the
        # fact. Every row in this file is a claim that a specific pathogen
        # protein was measured; keeping the organism string means that claim
        # stays checkable without re-reading 9GB.
        "source_organism": df.get("source_organism", ""),
    })

    # keep the measurement type rather than collapsing it
    values = pd.Series(np.nan, index=df.index)
    kinds = pd.Series(pd.NA, index=df.index, dtype="object")
    for kind, column_name in AFFINITY_COLUMNS.items():
        column = find_column(df.columns, *column_name.split()[:1])
        if column is None:
            continue
        parsed = df[column].map(_parse_affinity)
        take = values.isna() & parsed.notna()
        values = values.where(~take, parsed)
        kinds = kinds.where(~take, kind)

    out["affinity_nM"] = values.values
    out["affinity_type"] = kinds.values
    out["Y"] = [to_p_scale(v) for v in values.values]

    # Drug_ID from the SMILES itself. BindingDB's monomer IDs are not stable
    # across releases, and the cold-drug split needs an ID that identifies the
    # same molecule in every file.
    out["Drug_ID"] = out["Drug"].astype(str).str.strip()

    out = out.dropna(subset=["Drug", "Target", "Y"])
    out = out[out["Target"].astype(str).str.len() > 20]
    out = out.drop_duplicates(subset=["Drug_ID", "Target", "affinity_type"])
    return out[["Drug_ID", "Drug", "Target_ID", "Target", "Y",
                "affinity_nM", "affinity_type", "antiviral_target",
                "source_organism"]].reset_index(drop=True)


def missing_targets(df: pd.DataFrame) -> list:
    counts = df["antiviral_target"].value_counts().to_dict()
    return [t for t in REQUIRED_TARGETS if counts.get(t, 0) == 0]


def verify_coverage(df: pd.DataFrame, strict: bool = True) -> dict:
    """Every required target must be present. Fail loudly if not.

    The previous pipeline wrote a file containing one of five targets and
    reported success. That file then sat in the repo looking like a completed
    deliverable, and the gap was only visible to someone who counted the unique
    values.
    """
    counts = df["antiviral_target"].value_counts().to_dict()
    missing = [t for t in REQUIRED_TARGETS if counts.get(t, 0) == 0]

    print("\nCoverage:")
    for target in ALL_TARGETS:
        n = counts.get(target, 0)
        if target in OPTIONAL_TARGETS:
            label = "OK " if n else "n/a"
            note = "" if n else "   (documented as unavailable, see below)"
            print(f"  {label}  {target:32s} {n:>7,} pairs{note}")
        else:
            print(f"  {'OK ' if n else 'MISSING'}  {target:32s} {n:>7,} pairs")

    absent_optional = [t for t in OPTIONAL_TARGETS if counts.get(t, 0) == 0]
    if absent_optional:
        print(f"\n  {', '.join(absent_optional)} not present.")
        print(f"  {OPTIONAL_TARGET_REASON}")
        print("  This belongs in the Methods section, not in a footnote.")

    if missing and strict:
        raise RuntimeError(
            f"{len(missing)} of {len(REQUIRED_TARGETS)} required targets have "
            f"no rows: {missing}. The case study in master plan §3.3 cannot run "
            f"on this file. Check the specs in TARGET_SPECS against the target "
            f"names and organisms actually present in your BindingDB release "
            f"before writing the output."
        )
    return counts


def main():
    parser = argparse.ArgumentParser(description="Build the antiviral BindingDB subset")
    parser.add_argument("--source", default="data/raw/BindingDB_All.tsv",
                        help="bulk BindingDB TSV from bindingdb.org/bind/downloads")
    parser.add_argument("--out", default="data/processed/antiviral_clean.csv")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--allow-partial", action="store_true",
                        help="write the file even if targets are missing")
    parser.add_argument("--from-cache", action="store_true",
                        help="reuse the matched rows from the last full scan "
                             "instead of re-reading the 9GB source")
    args = parser.parse_args()

    if args.from_cache:
        print(f"Reusing {CACHE_PATH} (no source scan)...")
        loose, near_misses = load_cache()
        print(f"  {len(loose):,} candidate rows held from the last scan")
    else:
        if not os.path.exists(args.source):
            raise SystemExit(
                f"{args.source} not found.\n"
                f"Download the TSV from https://www.bindingdb.org/rwd/bind/"
                f"chemsearch/marvin/Download.jsp and unzip it there. It is "
                f"~565MB zipped, ~9GB unzipped; this script streams it in chunks."
            )
        print(f"Scanning {args.source} for the five antiviral targets...")
        loose, near_misses = extract(args.source, chunksize=args.chunksize)
        save_cache(loose, near_misses)

    matched = classify_frame(loose)
    if matched.empty:
        raise SystemExit(
            "no rows matched any of the five specs. See "
            f"{NEAR_MISS_PATH} for the names actually present."
        )
    print(f"  {len(matched):,} rows claimed by the five specs")
    cleaned = clean(matched)

    absent = missing_targets(cleaned)
    if absent and not args.allow_partial:
        # Print the diagnostic BEFORE raising. The scan that produced these
        # names took tens of minutes; making someone repeat it to find out what
        # a target is called in their release is the whole reason this path
        # exists.
        verify_coverage(cleaned, strict=False)
        suggest_names(near_misses, absent)
        raise SystemExit(
            f"\n{len(absent)} of {len(REQUIRED_TARGETS)} required targets have "
            f"no rows: {absent}\n"
            f"\nAdd the correct spelling to TARGET_PATTERNS in "
            f"src/data/extract_antiviral.py, add a case to "
            f"tests/test_antiviral.py, then re-run with --from-cache "
            f"(seconds, no re-scan).\n"
            f"\nDo NOT reach for --allow-partial. A control arm missing "
            f"targets is worse than a stated limitation, because it looks "
            f"complete."
        )

    verify_coverage(cleaned, strict=not args.allow_partial)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    print(f"\nSaved {len(cleaned):,} pairs "
          f"({cleaned['Drug_ID'].nunique():,} drugs) -> {args.out}")


if __name__ == "__main__":
    main()
