"""PHASE 2 STEP 3 -- the DAVIS leak, independently.
A. sequence audit from proteins.txt + split CSVs (names -> sequences), no use of sequence_audit.py
B. AUROC (sklearn.roc_auc_score) on leaked vs unleaked test rows from the RAW saved predictions of the ORIGINAL grid models;
   labels are re-derived from raw pKd (Y >= 7.0) and 'seen by sequence' is re-derived from sequences (both repo columns ignored, then compared)
C. retrained arms (seqclean / seqmatched): raw predictions are NOT in the repo -> verify summary math + split structure only."""
import sys, csv, gzip, json, re, os, itertools, numpy as np, collections
from sklearn.metrics import roc_auc_score
from scipy import stats
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
seqs = json.load(open("src/data/baselines/deepdta/data/davis/proteins.txt"))
print("A. SEQUENCE AUDIT")
print("  targets:", len(seqs), "| distinct sequences:", len(set(seqs.values())), "   [repo: 442 / 379]")
mut = re.compile(r"^(?:[A-Z]\d+[A-Z]|[A-Z]\d+[A-Z][A-Z]\d+[A-Z]|[A-Z]\d+[A-Z]-[A-Z]\d+[A-Z]|[A-Z]\d+[A-Z]\d+del|[A-Z]?\d*[A-Z]?\d+del|ITD|E746A750del|L747E749del|L747S752del|L747T751del|S752I759del|L747[A-Z]\d+del)$")
variants = {}
for name in seqs:
    m = re.match(r"^(.+?)(\((.+?)\))?(p)?$", name)
    base, inner, phos = m.group(1), m.group(3), m.group(4)
    is_var = bool(phos and m.group(2) is None and base + "p" == name and base in seqs) or bool(inner and mut.match(inner)) or bool(inner and phos)
    if is_var:
        wt = base if base in seqs else None
        variants[name] = wt
domain_like = [n for n in seqs if "(" in n and n not in variants]
print("  '(...)' names NOT counted as mutants (domain annotations):", domain_like)
ident = [n for n, w in variants.items() if w and seqs[n] == seqs[w]]; diff = [n for n, w in variants.items() if w and seqs[n] != seqs[w]]; nowt = [n for n, w in variants.items() if not w]
print(f"  variants: {len(variants)} | identical to WT: {len(ident)} | differ: {len(diff)} | no WT entry: {len(nowt)}   [repo: 74 / 54 / 0 / 20]")
print("  differ:", diff, "| no WT:", nowt)
# sequence-only cross-check that does not depend on naming rules: names sharing a sequence
groups = collections.defaultdict(list)
for n, s in seqs.items(): groups[s].append(n)
dup = [g for g in groups.values() if len(g) > 1]
print("  naming-free check: %d sequence groups hold >1 target name; names sitting in such groups = %d; removing one representative per group leaves %d redundant names (= 442 - 379 = 63)" % (len(dup), sum(map(len, dup)), sum(len(g) - 1 for g in dup)))
def read(path): return list(csv.DictReader(open(path, encoding="utf-8")))
def seen_sets(level, part="test"):
    tr = read(f"data/splits/davis/{level}/train.csv"); te = read(f"data/splits/davis/{level}/{part}.csv")
    trseq = {r["Target"] for r in tr}; trname = {r["Target_ID"] for r in tr}
    names = sorted({r["Target_ID"] for r in te}); unseen_name = [n for n in names if n not in trname]
    leaked = sorted({n for n in unseen_name if seqs[n] in trseq}); rows_aff = sum(1 for r in te if r["Target_ID"] in set(leaked))
    return te, trseq, names, unseen_name, leaked, rows_aff
print("\n  level        part  test targets  unseen-by-name  seen-by-sequence  rows affected      [repo table: cold_target 88/88/12, 816 of 5984 ; cold_pair 88/88/11, 143 of 1144]")
for level in ("cold_target", "cold_pair"):
    for part in ("test", "valid"):
        te, trseq, names, un, lk, ra = seen_sets(level, part)
        print("  %-12s %-5s %6d %14d %17d %9d of %d (%.1f%%)" % (level, part, len(names), len(un), len(lk), ra, len(te), 100 * ra / len(te)))
te, trseq, names, un, lk_ct, ra = seen_sets("cold_target"); _, _, _, _, lk_cp, _ = seen_sets("cold_pair")
print("  cold_target leaked targets:", lk_ct); print("  cold_pair leaked targets:  ", lk_cp)
# ------------------------------------------------------------------ B
print("\nB. AUROC leaked vs unleaked, recomputed with sklearn from RAW predictions of the original grid models")
print("   (label from raw pKd >= 7.0; leak flag from sequences; repo's `label`/`seen_by_sequence` columns compared afterwards)")
def load_pred(ds, level, model, seed):
    p = f"results/accuracy_v2/predictions/{ds}_{level}_{model}_seed{seed}"
    rows = list(csv.DictReader(gzip.open(p + ".csv.gz", "rt"))); meta = json.load(open(p + ".meta.json")); return rows, meta
out = {}
col_mismatch = 0; lab_mismatch = 0; auc_gap = 0.0; total_files = 0
for level, lk in (("cold_target", set(lk_ct)), ("cold_pair", set(lk_cp))):
    test = read(f"data/splits/davis/{level}/test.csv")
    key = {(r["Drug_ID"], r["Target_ID"]): float(r["Y"]) for r in test}
    for model in ("deepdta", "moltrans", "hyperattentiondti", "coldsite_dti"):
        res = []
        for seed in (1, 2, 3):
            rows, meta = load_pred("davis", level, model, seed); total_files += 1
            y = np.array([1 if key[(r["Drug_ID"], r["Target_ID"])] >= 7.0 else 0 for r in rows]); s = np.array([float(r["logit"]) for r in rows])
            leak = np.array([r["Target_ID"] in lk for r in rows])
            lab_mismatch += int((y != np.array([int(float(r["label"])) for r in rows])).sum())
            col_mismatch += int((leak != np.array([r["seen_by_sequence"] == "True" for r in rows])).sum())
            a_all, a_lk, a_un = roc_auc_score(y, s), roc_auc_score(y[leak], s[leak]), roc_auc_score(y[~leak], s[~leak])
            auc_gap = max(auc_gap, abs(a_all - meta["recorded_test_metrics"]["auroc"]))
            res.append((a_all, a_lk, a_un, int(leak.sum()), len(y)))
        out[(level, model)] = res
print("   files scored: %d | rows where my label != repo label: %d | rows where my leak flag != repo seen_by_sequence: %d | max |my all-rows AUROC - recorded AUROC| = %.2e" % (total_files, lab_mismatch, col_mismatch, auc_gap))
print("   %-11s %-18s %-24s %-24s %-24s  leaked rows / all" % ("level", "model", "AUROC all (3 seeds)", "leaked rows", "unleaked rows"))
for (level, model), res in out.items():
    f = lambda i: " ".join("%.3f" % r[i] for r in res) + " (m=%.3f)" % np.mean([r[i] for r in res])
    print("   %-11s %-18s %-24s %-24s %-24s  %d / %d" % (level, model, f(0), f(1), f(2), res[0][3], res[0][4]))
d = out[("cold_target", "deepdta")]
print("\n   DeepDTA cold_target ORIGINAL arm, mine vs leakage_retrain_davis.md: all %.3f (md 0.907)  leaked %.3f (md 0.950)  unleaked %.3f (md 0.884)" % tuple(np.mean([r[i] for r in d]) for i in (0, 1, 2)))
print("   leaked-minus-unleaked AUROC gap, per model, cold_target (mean of 3 seeds): " + ", ".join("%s %+.3f" % (m, np.mean([r[1] - r[2] for r in out[('cold_target', m)]])) for m in ("deepdta", "moltrans", "hyperattentiondti", "coldsite_dti")))
# ------------------------------------------------------------------ C
print("\nC. RETRAINED ARMS -- raw predictions of the seqclean/seqmatched retrains are NOT in the repo (no .pt, no prediction files).")
print("   Full AUROC recomputation from scratch requires the raw prediction logs, which are not currently in the tracked directory;")
print("   we verified the summary math and the split structure instead.")
J = json.load(open("results/leakage_retrain_davis.json"))
for level in ("cold_target", "cold_pair"):
    tab = {(r["arm"], r["seed"]): r["subsets"] for r in J if r["level"] == level}
    print(f"\n   [{level}] per-seed test AUROC  (arm: all / leaked / unleaked)")
    for arm in ("original", "seqmatched", "seqclean"):
        print("     %-10s " % arm + " | ".join("s%d: %.3f/%.3f/%.3f" % (s, *[tab[(arm, s)][k]["auroc"] for k in ("all", "leaked", "unleaked")]) for s in (1, 2, 3)))
    for k in ("all", "leaked", "unleaked"):
        leak = np.array([tab[("seqmatched", s)][k]["auroc"] - tab[("seqclean", s)][k]["auroc"] for s in (1, 2, 3)])
        size = np.array([tab[("original", s)][k]["auroc"] - tab[("seqmatched", s)][k]["auroc"] for s in (1, 2, 3)])
        t = stats.ttest_1samp(leak, 0.0); w = np.sqrt(len(leak))
        ci = stats.t.interval(0.95, 2, loc=leak.mean(), scale=leak.std(ddof=1) / w)
        print("     leak (seqmatched - seqclean) on %-8s rows: per seed %s  mean %+.4f  sd %.4f  95%% t-CI [%+.3f, %+.3f]  one-sample t p = %.4f | fewer-rows effect (original - seqmatched) mean %+.4f" % (k, np.round(leak, 3), leak.mean(), leak.std(ddof=1), *ci, t.pvalue, size.mean()))
    lk = np.array([tab[("seqmatched", s)]["leaked"]["rows"] for s in (1,2,3)]); print("     leaked-row counts per seed:", lk, "(should be constant)")
# structure of the arms from the split files
print("\n   split structure of the arms (from data/splits, hash-verified in Phase 1):")
for level in ("cold_target", "cold_pair"):
    te_o = read(f"data/splits/davis/{level}/test.csv"); ident = True
    for arm in ("seqclean", "seqmatched"):
        tr = read(f"data/splits/davis/{level}_{arm}/train.csv"); te = read(f"data/splits/davis/{level}_{arm}/test.csv")
        ident = ident and [ (r["Drug_ID"], r["Target_ID"], r["Y"]) for r in te] == [(r["Drug_ID"], r["Target_ID"], r["Y"]) for r in te_o]
        trseq = {r["Target"] for r in tr}; ov = sorted({r["Target_ID"] for r in te if r["Target"] in trseq and r["Target_ID"] not in {x["Target_ID"] for x in tr}})
        pos = sum(float(r["Y"]) >= 7.0 for r in tr)
        print(f"     {level:11} {arm:10} train rows {len(tr):5d}  positives {pos:5d}  test targets seen-by-sequence in this train set: {len(ov)}")
    print(f"     {level:11} test set identical across original/seqclean/seqmatched: {ident}")
