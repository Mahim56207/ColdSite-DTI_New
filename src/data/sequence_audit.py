"""
What sequence does the model actually read for each target?

Three checks on the protein sequences a dataset ships (DAVIS: DeepDTA's
`proteins.txt`), found on 2026-09-13 while building the KLIFS ground truth:

1. **Variants that are not variants.** A mutant or phospho-form whose sequence is
   identical to its wild type: the model cannot tell ABL1(T315I) from ABL1.
2. **Leakage by sequence.** At cold-target and cold-pair a test target is "unseen"
   by name; it is not unseen if a training target has the identical sequence.
3. **No kinase domain.** A sequence that holds few or none of the 85 KLIFS pocket
   residues (`src.data.klifs_pocket`): the model never saw where the drugs bind.

    python -m src.data.sequence_audit --dataset davis     # -> results/sequence_audit_davis.md

Nothing here changes a split or a ground truth; it reports, so the decision of
what to exclude is made once, explicitly, in Methods.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
POCKET_MINIMUM = 40          # of 85 KLIFS pocket residues present in the model's sequence


def load_rows(dataset: str, level: str) -> dict:
    return {p: pd.read_csv(f"data/splits/{dataset}/{level}/{p}.csv")
            for p in ("train", "valid", "test")}


def identical_variants(dataset: str) -> dict:
    held = json.load(open(f"src/data/baselines/deepdta/data/{dataset}/proteins.txt"))
    provenance_path = f"data/{dataset}_ground_truth_sites_provenance.json"
    if not os.path.exists(provenance_path):
        return {"variants": 0, "identical": [], "different": [], "no_wild_type": []}
    provenance = json.load(open(provenance_path))
    out = {"identical": [], "different": [], "no_wild_type": []}
    for target, entry in provenance.items():
        if not entry.get("is_variant"):
            continue
        base = entry.get("resolved_from")
        if base not in held or target not in held:
            out["no_wild_type"].append(target)
        elif held[target] == held[base]:
            out["identical"].append(target)
        else:
            out["different"].append(target)
    out["variants"] = sum(len(v) for v in out.values())
    return out


def sequence_leakage(dataset: str) -> dict:
    """Per level: test (and validation) targets unseen by name but seen by sequence."""
    out = {}
    for level in LEVELS:
        rows = load_rows(dataset, level)
        train_ids, train_seqs = set(rows["train"].Target_ID), set(rows["train"].Target)
        entry = {}
        for part in ("valid", "test"):
            targets = rows[part].drop_duplicates("Target_ID")
            unseen = targets[~targets.Target_ID.isin(train_ids)]
            leaked = sorted(unseen[unseen.Target.isin(train_seqs)].Target_ID)
            entry[part] = {"targets": len(targets), "unseen_by_name": len(unseen),
                           "seen_by_sequence": leaked,
                           "rows": int(rows[part].Target_ID.isin(leaked).sum()),
                           "of_rows": len(rows[part])}
        out[level] = entry
    return out


def missing_pocket(dataset: str) -> list:
    path = f"data/{dataset}_klifs_pocket_report.json"
    if not os.path.exists(path):
        return None                      # not computed -- never report it as "none"
    report = json.load(open(path))
    return sorted((t, r["residues_in_davis"]) for t, r in report.items()
                  if "residues_in_davis" in r and r["residues_in_davis"] < POCKET_MINIMUM)


def report(dataset: str) -> str:
    variants, leakage, pocket = (identical_variants(dataset), sequence_leakage(dataset),
                                 missing_pocket(dataset))
    all_rows = pd.concat(load_rows(dataset, "random").values()).drop_duplicates("Target_ID")
    lines = [f"# Sequence audit — {dataset}\n",
             "What the model reads for each target. Produced by `src/data/sequence_audit.py`; "
             "changes nothing, reports only.\n",
             f"**{all_rows.Target.nunique()} distinct sequences for {len(all_rows)} targets.**\n",
             "## 1. Variants identical to their wild type\n",
             f"{variants['variants']} variant targets: **{len(variants['identical'])} carry exactly "
             f"the wild-type sequence**, {len(variants['different'])} differ from it, "
             f"{len(variants['no_wild_type'])} have no wild-type entry to compare.\n",
             ("Identical: " + ", ".join(variants["identical"]) + "\n") if variants["identical"] else "",
             "## 2. Test targets unseen by name but seen by sequence\n",
             "| level | part | targets | unseen by name | identical sequence in training | rows affected |",
             "|---|---|---|---|---|---|"]
    for level, entry in leakage.items():
        for part in ("test", "valid"):
            e = entry[part]
            share = f"{e['rows']} of {e['of_rows']} ({100 * e['rows'] / e['of_rows']:.1f}%)"
            lines.append(f"| {level} | {part} | {e['targets']} | {e['unseen_by_name']} | "
                         f"{len(e['seen_by_sequence'])} | {share} |")
    for level in ("cold_target", "cold_pair"):
        leaked = leakage[level]["test"]["seen_by_sequence"]
        if leaked:
            lines.append(f"\n{level} test targets seen by sequence: " + ", ".join(leaked))
    lines.append("\n## 3. Sequences without the kinase pocket\n")
    if pocket is None:
        lines.append(f"Not computed: run `python -m src.data.klifs_pocket --dataset {dataset}` first.")
    else:
        lines.append(f"Targets whose sequence holds fewer than {POCKET_MINIMUM} of the 85 KLIFS "
                     f"pocket residues (`data/{dataset}_klifs_pocket_report.json`): "
                     f"**{len(pocket)}**\n")
        lines += [f"- {t}: {n} of 85" for t, n in pocket]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", default="davis", choices=("davis", "kiba"))
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    text = report(args.dataset)
    path = os.path.join(args.out_dir, f"sequence_audit_{args.dataset}.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
