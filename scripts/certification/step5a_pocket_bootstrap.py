"""PHASE 3 support -- pocket-level (KLIFS 85-residue) two-way bootstrap, independent of effects_v2.py.
Same estimator as step2 (ratio of means, targets and seeds resampled), ground truth = KLIFS pocket, per-protein
chance = pocket residues inside the scored window / window length. Reports how many of the 22 cells have a
95% lower bound above chance, for proteins-only and proteins+seeds resampling."""
import sys, numpy as np, csv
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import cert_common as C
from cert_common import *
C.DIRS = {"davis": "results/analysis_davis_policyA_klifs", "kiba": "results/analysis_kiba_policyA_klifs"}
K = 10
def load_cell(ds, model, level):
    seqs = load_seqs(ds); gt = load_sites(ds, "klifs"); Xs = []; ids0 = None
    for s in (1, 2, 3):
        e = ladder(ds, model, s)[level]; c = e["by_k"][str(K)]
        ls = LS(ds, e["ids"], kind="klifs", seqs=seqs, sites=gt); keep = [i for i, (L, S) in enumerate(ls) if S > 0 and L >= K]
        ids = [e["ids"][i] for i in keep]; assert ids0 is None or ids == ids0; ids0 = ids
        assert len(keep) == len(c["per_protein"]), (ds, model, level, s, len(keep), len(c["per_protein"])); Xs.append(np.array(c["per_protein"]))
        chance = np.array([ls[i][1] / ls[i][0] for i in keep])
    return np.vstack(Xs).T, chance
def boot(X, ch, B, rng, two_way, chunk=1000):
    n = X.shape[0]; out = np.empty(B); done = 0
    while done < B:
        m = min(chunk, B - done); idx = rng.integers(0, n, size=(m, n))
        if two_way: sidx = rng.integers(0, 3, size=(m, 3)); W = np.stack([(sidx == j).sum(1) for j in range(3)], 1) / 3.0
        else: W = np.full((m, 3), 1 / 3.0)
        out[done:done + m] = np.einsum("mnj,mj->m", X[idx], W) / n / ch[idx].mean(1); done += m
    return out
rows = []; B = 50_000
print("%-5s %-18s %-11s %7s %8s | %-16s | %-16s" % ("ds", "model", "level", "point", "chance", "proteins-only", "proteins+seeds"))
for ds in ("davis", "kiba"):
    for model, level in cells(ds):
        X, ch = load_cell(ds, model, level); pt = X.mean() / ch.mean()
        a = np.percentile(boot(X, ch, B, np.random.default_rng(21), False), [2.5, 97.5]); b = np.percentile(boot(X, ch, B, np.random.default_rng(22), True), [2.5, 97.5])
        rows.append((ds, model, level, pt, a, b))
        print("%-5s %-18s %-11s %7.3f %8.4f | [%5.2f, %5.2f] %s | [%5.2f, %5.2f] %s" % (ds.upper(), model, level, pt, ch.mean(), a[0], a[1], "ABOVE" if a[0] > 1 else "     ", b[0], b[1], "ABOVE" if b[0] > 1 else "     "))
n1 = sum(r[4][0] > 1 for r in rows); n2 = sum(r[5][0] > 1 for r in rows)
print(f"\nPOCKET-LEVEL (KLIFS) cells with CI lower bound > 1:  proteins-only {n1} of {len(rows)} | proteins+seeds {n2} of {len(rows)}")
for m in ("XAttn-Ref", "HyperAttentionDTI", "MolTrans", "DrugBAN"):
    lv = [r[2] for r in rows if r[0] == "davis" and r[1] == m and r[5][0] > 1]
    print(f"  DAVIS {m:18s} levels above chance (proteins+seeds): {len(lv)} of 4 {lv}")
print("cells with lower bound within 0.03 of 1:", [(r[0], r[1], r[2], round(r[4][0], 3), round(r[5][0], 3)) for r in rows if min(abs(r[4][0] - 1), abs(r[5][0] - 1)) < 0.03])
print("pocket-level range of point enrichment among surviving DAVIS cells: %.2f to %.2f" % (min(r[3] for r in rows if r[0] == "davis" and r[5][0] > 1), max(r[3] for r in rows if r[0] == "davis" and r[5][0] > 1)))
