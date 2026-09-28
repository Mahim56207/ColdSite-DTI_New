import sys, numpy as np; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from cert_common import *
from scipy import stats
K = 10
def exact_p(ls, obs_total):
    pmf = np.array([1.0])
    for L, S in ls: pmf = np.convolve(pmf, stats.hypergeom.pmf(np.arange(K + 1), L, S, K))
    return float(pmf[obs_total:].sum())
data = []
for ds in ("davis", "kiba"):
    seqs = load_seqs(ds); gt = load_sites(ds)
    for model, level in cells(ds):
        ps = []; rec = []; tots = []; lss = []
        for s in (1, 2, 3):
            e = ladder(ds, model, s)[level]; c = e["by_k"][str(K)]; ls = LS(ds, e["ids"], seqs=seqs, sites=gt)
            ls = [x for x in ls if x[1] > 0 and x[0] >= K]; tot = int(round(sum(c["per_protein"]) * K))
            ps.append(exact_p(ls, tot)); rec.append(c["p_value"]); tots.append(tot); lss.append(ls)
        data.append((ds, model, level, ps, rec, tots, lss))
# (1) correct diagnostic: recorded p is an add-one MC estimate (1+x)/(1+1000); its expectation is (1+1000*p)/1001
zs = []
for ds, model, level, ps, rec, tots, lss in data:
    for p, r in zip(ps, rec):
        exp = (1 + 1000 * p) / 1001; se = np.sqrt(exp * (1 - exp) / 1000); zs.append((abs(r - exp) / se, ds, model, level, p, r))
zs.sort(reverse=True)
print("(1) recorded MC p vs exact p, z = |recorded - E[recorded]| / MC s.e.   (66 seed-cells)")
print("    max |z| = %.2f ; #|z|>3 = %d ; mean z^2 = %.2f (expect ~1)" % (zs[0][0], sum(z[0] > 3 for z in zs), np.mean([z[0] ** 2 for z in zs])))
for z in zs[:4]: print("    top:", "z=%.2f %s %s %s exact=%.4f recorded=%.4f" % z)
# (2) validate the convolution by brute force on the borderline cell
ds, model, level, ps, rec, tots, lss = [d for d in data if d[:3] == ("davis", "MolTrans", "random")][0]
rng = np.random.default_rng(1); N = 400_000; L = np.array([x[0] for x in lss[0]]); S = np.array([x[1] for x in lss[0]])
ge = 0; done = 0
while done < N:
    n = min(20000, N - done); H = rng.hypergeometric(S, L - S, K, size=(n, len(L))).sum(1); ge += (H >= tots[0]).sum(); done += n
print("\n(2) brute-force check, DAVIS MolTrans random seed 1: observed hits=%d of %d proteins" % (tots[0], len(L)))
print("    exact convolution p = %.5f | Monte-Carlo (400,000 draws) p = %.5f +/- %.5f | repo's 1,000-perm p = %.5f" % (ps[0], ge / N, np.sqrt(ps[0] * (1 - ps[0]) / N), rec[0]))
print("    repo p is %.2f MC-standard-errors below the exact value: consistent with sampling noise, but it flips the verdict at alpha=0.05" % ((ps[0] - rec[0]) / np.sqrt(ps[0] * (1 - ps[0]) / 1000)))
# (3) sensitivity of the headline count
print("\n(3) headline counts vs alpha (exact p, uncorrected), 22 cells")
print("    alpha   disagree  all-3-pass  none-pass")
for a in (0.01, 0.025, 0.05, 0.10, 0.15):
    dis = sum(0 < sum(p < a for p in d[3]) < 3 for d in data); a3 = sum(all(p < a for p in d[3]) for d in data); nn = sum(not any(p < a for p in d[3]) for d in data)
    print(f"    {a:5.3f}   {dis:5d}     {a3:5d}       {nn:5d}")
dis_rec = sum(0 < sum(p < 0.05 for p in d[4]) < 3 for d in data)
print(f"    recorded (1,000-perm) p at 0.05: disagree = {dis_rec}")
# (4) how many distinct seeds' verdict hinge on being within one MC s.e. of alpha
close = [(d[0], d[1], d[2], s + 1, d[3][s], d[4][s]) for d in data for s in range(3) if abs(d[3][s] - 0.05) < 0.0146]
print("\n(4) seed-verdicts within 2 MC s.e. (~0.0146) of alpha=0.05 in the repo's 1,000-permutation p:")
for c in close: print("    %s %s %s seed%d exact=%.4f recorded=%.4f" % c)
