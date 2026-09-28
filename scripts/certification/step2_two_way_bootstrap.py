"""PHASE 2 STEP 2 -- independent two-way (proteins x seeds) bootstrap of enrichment = P@10 / chance.
Inputs: per-protein P@10 from ladder_*.json (3 seeds, same proteins), chance_i = S_i/L_i from raw ground truth + sequences.
Not used: bootstrap_ci.py, effects_v2.py.   Enrichment of a resample = mean_i,s(P) / mean_i(chance_i)   ('ratio of means')."""
import sys, numpy as np, csv; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from cert_common import *
K = 10
def load_cell(ds, model, level):
    seqs = load_seqs(ds); gt = load_sites(ds); Xs = []; ids0 = None
    for s in (1, 2, 3):
        e = ladder(ds, model, s)[level]; c = e["by_k"][str(K)]
        ls = LS(ds, e["ids"], seqs=seqs, sites=gt); keep = [i for i, (L, S) in enumerate(ls) if S > 0 and L >= K]
        ids = [e["ids"][i] for i in keep]; assert ids0 is None or ids == ids0, "proteins differ across seeds"; ids0 = ids
        assert len(keep) == len(c["per_protein"]); Xs.append(np.array(c["per_protein"]))
        chance = np.array([ls[i][1] / ls[i][0] for i in keep])
    return np.vstack(Xs).T, chance          # (n_prot, 3), (n_prot,)
def boot(X, ch, B, rng, two_way=True, chunk=1000, fixed_denominator=False):
    n = X.shape[0]; out = np.empty(B); done = 0
    while done < B:
        m = min(chunk, B - done); idx = rng.integers(0, n, size=(m, n))
        if two_way:
            sidx = rng.integers(0, 3, size=(m, 3)); W = np.stack([(sidx == j).sum(1) for j in range(3)], 1) / 3.0   # seed weights
        else:
            W = np.full((m, 3), 1 / 3.0)
        prec = np.einsum("mnj,mj->m", X[idx], W) / n
        den = ch.mean() if fixed_denominator else ch[idx].mean(1)
        out[done:done + m] = prec / den; done += m
    return out
def ci(v): return np.percentile(v, [2.5, 97.5])
# ---------------- headline cell
X, ch = load_cell("davis", "HyperAttentionDTI", "cold_drug")
point = X.mean() / ch.mean()
print("HAT DAVIS cold-drug: n_proteins=%d, seed means of P@10 = %s, chance = %.6f, point enrichment = %.4f" % (X.shape[0], np.round(X.mean(0), 4), ch.mean(), point))
print("draft/enrichment.csv: point 1.98, proteins-only [1.81, 2.15], proteins+seeds [0.94, 3.65]\n")
print("%-34s %-8s %-18s %s" % ("variant", "B", "95% percentile CI", "crosses 1.0?"))
for label, tw, fixed in (("proteins only (ratio of means)", False, False), ("proteins only (fixed denominator)", False, True),
                         ("proteins + seeds (ratio of means)", True, False), ("proteins + seeds (fixed denominator)", True, True)):
    v = boot(X, ch, 200_000, np.random.default_rng(7), tw, fixed_denominator=fixed); lo, hi = ci(v)
    print("%-34s %-8d [%.3f, %.3f]        %s   P(resample<1)=%.4f" % (label, 200_000, lo, hi, "YES" if lo <= 1 <= hi else "no", (v < 1).mean()))
print("\nMonte-Carlo spread at the draft's B = 10,000 (10 independent RNG seeds), proteins + seeds, ratio of means:")
los, his = [], []
for sd in range(10):
    lo, hi = ci(boot(X, ch, 10_000, np.random.default_rng(1000 + sd), True)); los.append(lo); his.append(hi)
print("  lower bound: min %.3f  max %.3f  mean %.3f   | upper bound: min %.3f  max %.3f  mean %.3f" % (min(los), max(los), np.mean(los), min(his), max(his), np.mean(his)))
print("  draft [0.94, 3.65] lies within the observed range of my bounds: lower %s, upper %s" % (min(los) - .05 <= 0.9361 <= max(los) + .05, min(his) - .15 <= 3.6487 <= max(his) + .15))
# enumerate the 3^3 seed-resample structure: how much of the width is seed variance?
print("\nWhy the interval widens: seed-level P@10 for this cell = %s (mean %.4f); between-seed sd = %.4f vs SE from proteins alone = %.4f" % (
    np.round(X.mean(0), 4), X.mean(), X.mean(0).std(ddof=1), X.mean(1).std(ddof=1) / np.sqrt(X.shape[0])))
# ---------------- all 22 cells
print("\n" + "=" * 30, "survivor counts, all 22 cells (B = 50,000 each, RNG seed fixed)")
rows = []; B = 50_000
print("%-5s %-18s %-11s %7s %8s | %-16s | %-16s" % ("ds", "model", "level", "point", "chance", "proteins-only", "proteins+seeds"))
for ds in ("davis", "kiba"):
    for model, level in cells(ds):
        X, ch = load_cell(ds, model, level); pt = X.mean() / ch.mean()
        a = ci(boot(X, ch, B, np.random.default_rng(11), False)); b = ci(boot(X, ch, B, np.random.default_rng(12), True))
        rows.append((ds, model, level, pt, a, b))
        print("%-5s %-18s %-11s %7.3f %8.4f | [%5.2f, %5.2f] %s | [%5.2f, %5.2f] %s" % (ds.upper(), model, level, pt, ch.mean(), a[0], a[1], "ABOVE" if a[0] > 1 else "     ", b[0], b[1], "ABOVE" if b[0] > 1 else "     "))
n1 = sum(r[4][0] > 1 for r in rows); n2 = sum(r[5][0] > 1 for r in rows)
print(f"\nRESIDUE-LEVEL (UniProt) cells with CI lower bound > 1:   proteins-only {n1} of {len(rows)}   |   proteins+seeds {n2} of {len(rows)}      [draft: 7/22 -> 1/22]")
print("cells whose lower bound is within 0.03 of 1 (verdict could flip with more resamples / a different RNG):")
for r in rows:
    for nm, lb in (("proteins-only", r[4][0]), ("proteins+seeds", r[5][0])):
        if abs(lb - 1) < 0.03: print("   ", r[0], r[1], r[2], nm, "lower=%.3f" % lb)
print("which cells survive proteins+seeds:", [(r[0], r[1], r[2], round(r[5][0], 3)) for r in rows if r[5][0] > 1])
# compare to the repo's numbers cell-by-cell (post-hoc, after computing mine)
rep = {(x["dataset"], x["model"], x["level"]): x for x in csv.DictReader(open("results/effects_v2_2d/enrichment.csv")) if x["family"] in ("P1", "P2")}
rep1 = {(x["dataset"], x["model"], x["level"]): x for x in csv.DictReader(open("results/effects_v2/enrichment.csv")) if x["family"] in ("P1", "P2")}
alias = {"XAttn-Ref": "coldsite_dti", "HyperAttentionDTI": "hyperattentiondti", "MolTrans": "moltrans", "DrugBAN": "drugban"}
lvl = lambda l: l
worst = 0; worst_cell = None; flips = 0
for ds, model, level, pt, a, b in rows:
    r2 = rep[(ds, alias[model], level)]; r1 = rep1[(ds, alias[model], level)]
    d = max(abs(float(r2["enrichment_low"]) - b[0]), abs(float(r2["enrichment_high"]) - b[1]), abs(float(r1["enrichment_low"]) - a[0]), abs(float(r1["enrichment_high"]) - a[1]))
    dp = abs(float(r2["enrichment"]) - pt)
    if d > worst: worst, worst_cell = d, (ds, model, level)
    flips += ((float(r2["enrichment_low"]) > 1) != (b[0] > 1)) + ((float(r1["enrichment_low"]) > 1) != (a[0] > 1))
    assert dp < 1e-6, (ds, model, level, dp)
print(f"\nvs repo CSVs: point estimates identical to 1e-6 in all {len(rows)} cells; largest CI-bound difference = {worst:.3f} at {worst_cell}; verdict (lower>1) disagreements = {flips}")
