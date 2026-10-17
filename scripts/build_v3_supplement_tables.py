#!/usr/bin/env python3
"""Generate Supplementary Tables S6 (per-cell seed agreement) and S7 (enrichment with intervals) and insert them into paper/v3_ismb/supplement.md
between <!-- BEGIN GENERATED TABLES --> and <!-- END GENERATED TABLES -->. Every row carries a source tag; no value is typed by hand.

S6 sources: results/certification/step1_disagreement.txt (per-seed precision@10, exact chance, exact per-seed p, one line per cell) and the two audit files
(median-over-seeds p and the Holm decision per cell). S7 source: results/effects_v2_2d/enrichment.csv (two-way bootstrap: proteins and seeds resampled).
Run, then: python scripts/check_number_provenance.py paper/v3_ismb/supplement.md
"""
import csv, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUP = os.path.join(ROOT, "paper", "v3_ismb", "supplement.md")
NAME = {"coldsite_dti": "XAttn-Ref", "hyperattentiondti": "HyperAttentionDTI", "moltrans": "MolTrans", "drugban": "DrugBAN"}
INV = {v: k for k, v in NAME.items()}
ORDER = ["moltrans", "hyperattentiondti", "drugban", "coldsite_dti"]
LEVELS = ["random", "cold_drug", "cold_target", "cold_pair"]
lv = lambda s: s.replace("_", "-")

# ---- S6 ----------------------------------------------------------------------------------------------------------------------------------------
STEP1 = "results/certification/step1_disagreement.txt"
rows = {}
for i, line in enumerate(open(os.path.join(ROOT, STEP1), encoding="utf-8"), 1):
    m = re.match(r"^(DAVIS|KIBA)\s+(\S+)\s+(\S+)\s+P@10 (\S+) (\S+) (\S+) \| chance (\S+) \| exact p (\S+) (\S+) (\S+) \| marks (\S+)", line)
    if m:
        ds, mod, lev, p1, p2, p3, ch, e1, e2, e3, marks = m.groups()
        rows[(ds.lower(), INV[mod], lev)] = dict(line=i, p=(p1, p2, p3), chance=ch, e=(e1, e2, e3), marks=marks, disagree="DISAGREE" in line)
assert len(rows) == 22, len(rows)
AUD = {"davis": "results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md", "kiba": "results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md"}
aud = {}
for ds, path in AUD.items():
    for i, line in enumerate(open(os.path.join(ROOT, path), encoding="utf-8"), 1):
        m = re.match(r"^- `(\w+)\|(\w+)\|(\w+)` p=(\S+) \(threshold (\S+)\) -> (yes|no)", line)
        if m and m.group(1) != "uniform_control":
            aud[(m.group(2), m.group(1), m.group(3))] = dict(path=path, line=i, p=m.group(4), yes=m.group(6) == "yes")
assert all(k in aud for k in rows), [k for k in rows if k not in aud]

def fmtp(tok):                       # 9.999e-05 -> 0.0001 in prose is NOT used: print the audit's own token so it matches the line
    return tok

s6 = ["**Supplementary Table S6.** Seed agreement for each of the 22 model–dataset–level cells scored against UniProt residues. Precision@10 of the three seeds in order, the cell's exact chance level, "
      "and each seed's exact permutation p-value; the marks give, for each seed in order, whether its exact p-value is below 0.05 (\\* yes, . no); \"disagree\" means some but not all seeds are below it. "
      "The last two columns are the audit's median-over-seeds p-value and whether the cell survives Holm correction over its dataset's family (uniform-map cells are in the family but not listed).\n"
      "<!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->\n<!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->\n",
      "| dataset | model | level | precision@10, per seed | chance | exact p, per seed | marks | disagree | audit p (median) | Holm |",
      "|" + "|".join("-" * w for w in (7, 14, 10, 20, 8, 22, 7, 8, 10, 6)) + "|"]
for ds in ("davis", "kiba"):
    for mod in ORDER:
        for lev in LEVELS:
            r = rows.get((ds, mod, lev))
            if not r: continue
            a = aud[(ds, mod, lev)]
            tag = (f"<!-- src: {STEP1}:{r['line']} = {', '.join(r['p'])}, {r['chance']}, {', '.join(r['e'])} --> "
                   f"<!-- src: {a['path']}:{a['line']} = {a['p']} -->")
            s6.append(f"| {ds.upper()} | {NAME[mod]} | {lv(lev)} | {' / '.join(r['p'])} | {r['chance']} | {' / '.join(r['e'])} | {r['marks']} | {'yes' if r['disagree'] else 'no'} | {a['p']} | {'yes' if a['yes'] else 'no'} | {tag}")

# ---- S7 ----------------------------------------------------------------------------------------------------------------------------------------
CSV = "results/effects_v2_2d/enrichment.csv"
E = {(x["family"], x["model"], x["level"]): x for x in csv.DictReader(open(os.path.join(ROOT, CSV), encoding="utf-8"))}
s7 = ["**Supplementary Table S7.** Enrichment over chance (mean precision@10 divided by mean chance, over the same proteins) for every model, level and ground truth, with 95% percentile intervals "
      "from a bootstrap that resamples both proteins and seeds (10,000 resamples); n is the number of proteins scored after the sequence policy; an interval that excludes one lies entirely on one side of chance. "
      "This is the table behind Figure 2.\n"
      "<!-- src: results/effects_v2_2d/enrichment.md:3 = 95, 10000 -->\n",
      "| ground truth | dataset | model | level | n | precision@10 | chance | enrichment | interval |",
      "|" + "|".join("-" * w for w in (11, 7, 14, 11, 5, 9, 8, 9, 14)) + "|"]
for fam, gt, ds in (("P1", "UniProt", "DAVIS"), ("P2", "UniProt", "KIBA"), ("S3-D", "KLIFS pocket", "DAVIS"), ("S3-K", "KLIFS pocket", "KIBA")):
    for mod in ORDER:
        for lev in LEVELS:
            x = E.get((fam, mod, lev))
            if not x: continue
            n, pr, ch, en, lo, hi = int(x["n"]), float(x["precision"]), float(x["chance"]), float(x["enrichment"]), float(x["enrichment_low"]), float(x["enrichment_high"])
            vals = f"{n}, {pr:.3f}, {ch:.3f}, {en:.2f}, {lo:.2f}, {hi:.2f}"
            tag = f"<!-- src: {CSV}#family={fam}&model={mod}&level={lev}->n,precision,chance,enrichment,enrichment_low,enrichment_high = {vals} -->"
            s7.append(f"| {gt} | {ds} | {NAME[mod]} | {lv(lev)} | {n} | {pr:.3f} | {ch:.3f} | {en:.2f} | {lo:.2f} to {hi:.2f} | {tag}")

block = ("<!-- BEGIN GENERATED TABLES (scripts/build_v3_supplement_tables.py; do not edit by hand) -->\n\n## S8 Per-cell tables\n\n" + "\n".join(s6) + "\n\n" + "\n".join(s7) + "\n\n<!-- END GENERATED TABLES -->\n")
sup = open(SUP, encoding="utf-8").read()
if "<!-- BEGIN GENERATED TABLES" in sup:
    sup = re.sub(r"<!-- BEGIN GENERATED TABLES.*?<!-- END GENERATED TABLES -->\n", lambda m: block, sup, flags=re.S)
else:
    assert "## S8 Supplementary figures" in sup
    sup = sup.replace("## S8 Supplementary figures", block + "\n## S9 Supplementary figures")
    sup = sup.replace("are placed in S8 as Supplementary Figures", "are placed in S9 as Supplementary Figures")
open(SUP, "w", encoding="utf-8").write(sup)
print(f"S6: {len(s6)-3} rows, S7: {len(s7)-3} rows")
