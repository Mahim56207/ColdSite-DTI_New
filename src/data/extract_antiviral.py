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
    # BindingDB_All.tsv is a ~3GB bulk download from bindingdb.org/bind/downloads
    python -m src.data.extract_antiviral --source data/raw/BindingDB_All.tsv
"""
import argparse
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
TARGET_PATTERNS = {
    "SARS-CoV-2 Mpro": re.compile(
        r"(sars[- ]?cov[- ]?2|2019[- ]?ncov|sars coronavirus 2).{0,40}"
        r"(main protease|3c[- ]?like|mpro|nsp5)"
        r"|3c[- ]?like proteinase.{0,40}(sars[- ]?cov[- ]?2|2019[- ]?ncov)"
        r"|^mpro$|sars-cov-2 3cl", re.IGNORECASE),
    "SARS-CoV-2 RdRp": re.compile(
        r"(sars[- ]?cov[- ]?2|2019[- ]?ncov).{0,40}"
        r"(rna[- ]?dependent rna polymerase|rdrp|nsp12)"
        r"|rna[- ]?directed rna polymerase.{0,40}(sars[- ]?cov[- ]?2|2019[- ]?ncov)",
        re.IGNORECASE),
    "HIV-1 protease": re.compile(
        r"hiv[- ]?1?.{0,20}protease|protease.{0,20}hiv[- ]?1"
        r"|human immunodeficiency virus.{0,30}protease", re.IGNORECASE),
    "HIV-1 reverse transcriptase": re.compile(
        r"hiv[- ]?1?.{0,20}reverse transcriptase"
        r"|reverse transcriptase.{0,20}hiv[- ]?1"
        r"|human immunodeficiency virus.{0,30}reverse transcriptase", re.IGNORECASE),
    "Influenza neuraminidase": re.compile(
        r"(influenza|h1n1|h3n2|h5n1).{0,40}neuraminidase"
        r"|neuraminidase.{0,40}(influenza|h1n1|h3n2|h5n1)"
        r"|^neuraminidase$", re.IGNORECASE),
}

REQUIRED_TARGETS = tuple(TARGET_PATTERNS)

AFFINITY_COLUMNS = {
    "IC50": "IC50 (nM)",
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "EC50": "EC50 (nM)",
}


def classify_target(target_name: str) -> str | None:
    """Target name -> one of REQUIRED_TARGETS, or None.

    Checked in dictionary order, so a name matching two patterns takes the
    first. That only happens for chimeric constructs, which are rare enough to
    accept and frequent enough that raising would abort a 3GB pass.
    """
    if not isinstance(target_name, str):
        return None
    for label, pattern in TARGET_PATTERNS.items():
        if pattern.search(target_name):
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
    """Stream BindingDB_All.tsv and keep rows matching the five targets."""
    kept = []
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
        labels = chunk[name_column].map(classify_target)
        matched = chunk[labels.notna()].copy()
        if len(matched):
            matched["antiviral_target"] = labels[labels.notna()].values
            kept.append(matched)
        if verbose:
            found = sum(len(k) for k in kept)
            print(f"  scanned {scanned:,} rows, kept {found:,}", end="\r")

    if verbose:
        print()
    if not kept:
        raise RuntimeError("no rows matched any of the five targets")
    return pd.concat(kept, ignore_index=True)


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
                "affinity_nM", "affinity_type", "antiviral_target"]].reset_index(drop=True)


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
    for target in REQUIRED_TARGETS:
        n = counts.get(target, 0)
        print(f"  {'OK ' if n else 'MISSING'}  {target:32s} {n:>7,} pairs")

    if missing and strict:
        raise RuntimeError(
            f"{len(missing)} of 5 required targets have no rows: {missing}. "
            f"The case study in master plan §3.3 cannot run on this file. Check "
            f"the patterns in TARGET_PATTERNS against the target names actually "
            f"present in your BindingDB release before writing the output."
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
    args = parser.parse_args()

    if not os.path.exists(args.source):
        raise SystemExit(
            f"{args.source} not found.\n"
            f"Download BindingDB_All.tsv from https://www.bindingdb.org/bind/downloads "
            f"and place it there. It is ~3GB; this script streams it in chunks."
        )

    print(f"Scanning {args.source} for the five antiviral targets...")
    matched = extract(args.source, chunksize=args.chunksize)
    cleaned = clean(matched)
    verify_coverage(cleaned, strict=not args.allow_partial)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    print(f"\nSaved {len(cleaned):,} pairs "
          f"({cleaned['Drug_ID'].nunique():,} drugs) -> {args.out}")


if __name__ == "__main__":
    main()
