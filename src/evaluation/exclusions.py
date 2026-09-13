"""
Which test proteins the explanation metrics may score, and what counts as one protein.

Option A, decided 2026-09-13 after `src/data/sequence_audit.py` found three problems in
DeepDTA's DAVIS sequences (`results/sequence_audit_davis.md`). Nothing is retrained; the
evaluation stops counting rows it should never have counted:

1. **Seen by sequence.** At cold-target 12 of 88 test targets and at cold-pair 11 of 88
   are unseen by name but identical in sequence to a training target (a mutant whose
   sequence is the wild type's). They are not cold, so they are dropped at the cold
   levels. Nothing is dropped at random or cold-drug, where targets are shared by design.
2. **No kinase pocket.** Ten targets' sequences hold fewer than
   `sequence_audit.POCKET_MINIMUM` of the 85 KLIFS pocket residues -- DAVIS's RET is
   residues 1-430, the extracellular part. The model was never shown where a drug binds,
   and five of them were being scored against non-drug sites (RET's calcium sites).
3. **One sequence, many names.** 442 DAVIS targets are 379 distinct sequences (17 ABL1
   entries are one sequence). precision@k is a per-protein quantity, so a protein is a
   distinct *sequence*, not a name -- otherwise ABL1 enters the mean seventeen times and
   `n` counts correlated rows, the same mistake as averaging every pair of a protein.

KIBA has none of the three (`results/sequence_audit_kiba.md`), so the policy changes
nothing there, and it is written so that it cannot silently do nothing: `describe`
reports what it dropped, and `policy=False` reproduces the pre-decision numbers.
"""
from __future__ import annotations

import functools
import os

LEAK_LEVELS = ("cold_target", "cold_pair")


def dataset_level_from_split_dir(split_dir: str) -> tuple[str | None, str | None]:
    """'data/splits/davis/cold_pair' -> ('davis', 'cold_pair'); unknown shapes -> Nones."""
    parts = os.path.normpath(split_dir).split(os.sep)
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, None


@functools.lru_cache(maxsize=8)
def _seen_by_sequence(dataset: str) -> dict:
    from src.data.sequence_audit import sequence_leakage
    return {level: frozenset(entry["test"]["seen_by_sequence"])
            for level, entry in sequence_leakage(dataset).items()}


@functools.lru_cache(maxsize=8)
def _pocketless(dataset: str) -> frozenset:
    from src.data.sequence_audit import missing_pocket
    found = missing_pocket(dataset)
    if found is None:              # the KLIFS report has not been built for this dataset
        return frozenset()
    return frozenset(target for target, _n in found)


@functools.lru_cache(maxsize=32)
def excluded_target_ids(dataset: str | None, level: str | None,
                        policy: bool = True) -> frozenset:
    """Test targets the explanation metrics must not score."""
    if not policy or not dataset or not level:
        return frozenset()
    # Only a dataset whose splits exist has leakage or pocket reports to consult; a
    # split directory elsewhere (a test fixture, a panel) is left alone.
    if not os.path.isdir(os.path.join("data/splits", dataset, level)):
        return frozenset()
    out = set(_pocketless(dataset))
    if level in LEAK_LEVELS:
        out |= set(_seen_by_sequence(dataset).get(level, frozenset()))
    return frozenset(out)


def protein_key(target_id: str, sequence: str, policy: bool = True) -> str:
    """What counts as one protein: its sequence under the policy, else its name."""
    return str(sequence).upper() if policy else str(target_id)


def describe(dataset: str, level: str, policy: bool = True) -> dict:
    """What the policy does to this level -- for the report, so it is never invisible."""
    import pandas as pd

    path = os.path.join("data/splits", dataset, level, "test.csv")
    frame = pd.read_csv(path, usecols=["Target_ID", "Target"])
    targets = frame.drop_duplicates("Target_ID")
    excluded = excluded_target_ids(dataset, level, policy)
    kept = targets[~targets.Target_ID.astype(str).isin(excluded)]
    return {"policy": bool(policy), "level": level,
            "target_names": len(targets),
            "excluded_names": int(targets.Target_ID.astype(str).isin(excluded).sum()),
            "excluded": sorted(excluded & set(targets.Target_ID.astype(str))),
            "distinct_sequences_kept": int(kept.Target.nunique()),
            "rows_dropped": int(frame.Target_ID.astype(str).isin(excluded).sum()),
            "rows": len(frame)}
