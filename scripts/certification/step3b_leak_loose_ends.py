import sys, csv, gzip, json, re, os, numpy as np, collections
from sklearn.metrics import roc_auc_score
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
seqs = json.load(open("src/data/baselines/deepdta/data/davis/proteins.txt"))
# (1) diff my mutant list against the repo audit's identical list
md = open("results/sequence_audit_davis.md").read()
repo_ident = set(re.search(r"Identical: (.+)", md).group(1).split(", "))
mine = set(n for n in seqs if re.search(r"\([A-Z]\d+[A-Z]|del\)|ITD|\)p$|^ABL1p$", n) and not re.search(r"KinDom|JH\d|Pfalc|Mtub", n))
print("(1) repo identical-list size %d ; mine %d" % (len(repo_ident), len(mine)))
print("    in mine, not in repo list:", sorted(mine - repo_ident)); print("    in repo list, not in mine:", sorted(repo_ident - mine))
odd = [n for n in mine - repo_ident]
for n in odd:
    base = re.sub(r"\(.*\)p?$", "", n); print("    ", n, "| base", base, "in names:", base in seqs, "| same seq as base:", base in seqs and seqs[n] == seqs[base], "| same seq as ANY other name:", [m for m in seqs if m != n and seqs[m] == seqs[n]][:5])
print("    repo 'no wild-type' 20 = names the repo treats as variants but that have no WT entry; my 21 domain-like names (incl. GCN2(KinDom2S808G)):")
print("    domain-like names sharing a sequence with another name:", {n: [m for m in seqs if m != n and seqs[m] == seqs[n]][:2] for n in seqs if re.search(r"KinDom|JH\d|Pfalc|Mtub", n) and any(m != n and seqs[m] == seqs[n] for m in seqs)})
# (2) are the test sets the same rows?
def read(p): return list(csv.DictReader(open(p, encoding="utf-8")))
print("\n(2) test-set identity across arms")
for level in ("cold_target", "cold_pair"):
    o = read(f"data/splits/davis/{level}/test.csv"); key = lambda rows: collections.Counter((r["Drug_ID"], r["Target_ID"], r["Y"]) for r in rows)
    for arm in ("seqclean", "seqmatched"):
        a = read(f"data/splits/davis/{level}_{arm}/test.csv")
        print("   %-11s %-10s rows %d vs %d | same multiset: %s | same order: %s | differing rows: %d" % (level, arm, len(o), len(a), key(o) == key(a), [(r['Drug_ID'], r['Target_ID']) for r in o] == [(r['Drug_ID'], r['Target_ID']) for r in a], sum((key(o) - key(a)).values())))
    s = read(f"data/splits/davis/{level}_seqclean/test.csv"); m = read(f"data/splits/davis/{level}_seqmatched/test.csv")
    print("   %-11s seqclean vs seqmatched test: identical order+content: %s" % (level, [(r['Drug_ID'], r['Target_ID'], r['Y']) for r in s] == [(r['Drug_ID'], r['Target_ID'], r['Y']) for r in m]))
# (3) AUROC gap vs recorded, by cell
print("\n(3) |my AUROC from saved predictions - recorded test AUROC| by cell (only > 1e-4 shown)")
rows = []
for f in sorted(os.listdir("results/accuracy_v2/predictions")):
    if not f.endswith(".csv.gz") or not f.startswith("davis_"): continue
    base = f[:-7]; meta = json.load(open(f"results/accuracy_v2/predictions/{base}.meta.json"))
    r = list(csv.DictReader(gzip.open(f"results/accuracy_v2/predictions/{f}", "rt")))
    y = np.array([int(float(x["label"])) for x in r]); s = np.array([float(x["logit"]) for x in r])
    gap = roc_auc_score(y, s) - meta["recorded_test_metrics"]["auroc"]; rows.append((abs(gap), base, gap))
rows.sort(reverse=True); print("   davis prediction files:", len(rows), "| files with gap > 1e-4:", sum(r[0] > 1e-4 for r in rows), "| > 1e-3:", sum(r[0] > 1e-3 for r in rows), "| max:", "%.5f" % rows[0][0])
for g, b, gap in rows[:8]: print("   %-48s %+.5f" % (b, gap))
by = collections.defaultdict(list)
for g, b, gap in rows: by[b.split("_")[-2] if "coldsite" not in b else "coldsite_dti"].append(g)
print("   by model, max gap:", {("coldsite_dti" if "coldsite" in b else b.split("_")[-2]): 0 for _, b, _ in rows[:0]})
mm = collections.defaultdict(float)
for g, b, gap in rows:
    m = [x for x in ("deepdta", "moltrans", "hyperattentiondti", "coldsite_dti") if x in b][0]; mm[m] = max(mm[m], g)
print("   ", {k: round(v, 5) for k, v in mm.items()})
