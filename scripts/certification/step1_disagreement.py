"""PHASE 2 STEP 1 -- the disagreement claim (12 of 22), recomputed from raw per-protein precision@10.
Inputs used: per-protein hit fractions in ladder_*.json (earliest recorded artifact), DAVIS/KIBA sequences,
ground-truth sites. NOT used: seed_agreement.py, run_ladder.py, significance_test.py.
Null = k=10 uniformly random positions per protein => hits_i ~ Hypergeometric(L_i, S_i, 10), independent.
p = P(sum hits >= observed sum) computed EXACTLY by convolving the per-protein pmfs (no Monte Carlo)."""
import sys, numpy as np; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from cert_common import *
from scipy import stats
K = 10; ALPHA = 0.05
def exact_p(ls, obs_total, k=K):
    pmf = np.array([1.0])
    for L, S in ls:
        pmf = np.convolve(pmf, stats.hypergeom.pmf(np.arange(k + 1), L, S, k))
    tail = pmf[obs_total:].sum() if obs_total < len(pmf) else 0.0
    return float(tail), float(sum(S / L for L, S in ls) / len(ls))
def holm(pvals, alpha=ALPHA):
    m = len(pvals); order = np.argsort(pvals); rej = [False] * m; adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i])); adj[i] = run
        if (m - r) * pvals[i] <= alpha and (r == 0 or rej[order[r - 1]]): rej[i] = True
    return adj, rej
rows = []; worst_z = 0; mismatched_class = []
for ds in ("davis", "kiba"):
    seqs = load_seqs(ds); gt = load_sites(ds)
    for model, level in cells(ds):
        per = []
        for s in (1, 2, 3):
            e = ladder(ds, model, s)[level]; cell = e["by_k"][str(K)]
            ids = e["ids"]; ls = LS(ds, ids, seqs=seqs, sites=gt)
            pp = np.array(cell["per_protein"])
            keep = [i for i, (L, S) in enumerate(ls) if S > 0 and L >= K]       # proteins a precision@k test can use
            if len(pp) != len(keep): raise SystemExit(f"per-protein length mismatch {ds} {model} {level} s{s}: {len(pp)} vs {len(keep)}")
            ls = [ls[i] for i in keep]
            hits = pp * K; assert np.allclose(hits, np.round(hits), atol=1e-9), "non-integer hits"
            tot = int(round(hits.sum())); p_ex, chance = exact_p(ls, tot)
            p_rec = cell["p_value"]; se = np.sqrt(max(p_ex * (1 - p_ex), 1e-9) / 1000)
            worst_z = max(worst_z, abs(p_rec - p_ex) / se)
            per.append(dict(prec=pp.mean(), chance=chance, chance_rec=cell["chance"], p=p_ex, p_rec=p_rec, hits=hits, n=len(pp)))
            if (p_ex < ALPHA) != (p_rec < ALPHA): mismatched_class.append((ds, model, level, s, p_ex, p_rec))
        rows.append((ds, model, level, per))
# ---- (a) the draft's definition: seeds disagree about their own uncorrected verdict
n = len(rows); dis = all3 = none = wide_mine = wide_rec = 0; lines = []
for ds, model, level, per in rows:
    sig = [x["p"] < ALPHA for x in per]; sig_rec = [x["p_rec"] < ALPHA for x in per]
    prec = [x["prec"] for x in per]; ch = per[0]["chance"]; ch_rec = per[0]["chance_rec"]
    d = 0 < sum(sig) < 3; dis += d; all3 += sum(sig) == 3; none += sum(sig) == 0
    wide_mine += (max(prec) - min(prec)) > abs(np.mean(prec) - ch)
    wide_rec += (max(prec) - min(prec)) > abs(np.mean(prec) - ch_rec)
    lines.append(f"{ds.upper():5} {model:18} {level:11} P@10 " + " ".join(f"{x:.3f}" for x in prec) + f" | chance {ch:.4f} | exact p " +
                 " ".join(f"{x['p']:.4f}" for x in per) + " | marks " + "".join("*" if s else "." for s in sig) + (" | recorded-p marks " + "".join("*" if p['p_rec'] < ALPHA else "." for p in per)) + ("  <-- DISAGREE" if d else ""))
print("=" * 30, "STEP 1(a): per-seed uncorrected permutation verdicts (draft's definition)")
print("\n".join(lines))
print(f"\nCELLS: {n}   (DAVIS {sum(r[0]=='davis' for r in rows)}, KIBA {sum(r[0]=='kiba' for r in rows)})")
print(f"SEEDS DISAGREE (0 < #seeds with p<0.05 < 3):  {dis} of {n}      [draft: 12 of 22]")
print(f"ALL THREE pass: {all3}   [draft: 1]     NONE pass: {none}   [draft: 9]")
print(f"spread(max-min P@10) > |mean - chance|:  exact-chance {wide_mine} of {n} ; recorded-chance {wide_rec} of {n}   [draft: 21 of 22]")
print(f"exact p vs recorded (1000-permutation MC) p: max |z| = {worst_z:.2f} ; classification flips at alpha=0.05: {len(mismatched_class)}")
for m in mismatched_class: print("   FLIP", m)
davis_dis = sum(0 < sum(x['p'] < ALPHA for x in per) < 3 for ds, _, _, per in rows if ds == 'davis')
print(f"DAVIS cells that disagree: {davis_dis}  KIBA: {dis - davis_dis}    [draft: 9 DAVIS, 3 KIBA]")
# ---- (b) STRICTER reading of the request: is between-seed variation itself statistically significant?
rng = np.random.default_rng(20260929); print("\n" + "=" * 30, "STEP 1(b): formal heterogeneity test (extension; NOT the draft's claim)")
print("H0: the three seeds have the same expected P@10 on the same proteins. Friedman (scipy) + within-protein seed-label permutation (20,000)")
fp, pp_ = [], []
for ds, model, level, per in rows:
    H = np.vstack([x["hits"] for x in per]).T                      # proteins x 3 seeds
    try: fr = stats.friedmanchisquare(*H.T).pvalue
    except Exception: fr = 1.0
    obs = H.mean(0).var(); cnt = 0; N = 20000
    for _ in range(N):
        idx = rng.random(H.shape).argsort(1); Hs = np.take_along_axis(H, idx, 1); cnt += Hs.mean(0).var() >= obs - 1e-15
    pperm = (1 + cnt) / (1 + N); fp.append(fr); pp_.append(pperm)
print(f"{'cell':44} Friedman p   perm p")
for (ds, model, level, per), a, b in zip(rows, fp, pp_): print(f"{ds.upper():5} {model:18} {level:11}   {a:9.4f}  {b:9.4f}")
for name, arr in (("Friedman", fp), ("permutation", pp_)):
    adj, rej = holm(arr)
    print(f"{name:12}: cells with p<0.05 (uncorrected) = {sum(p<ALPHA for p in arr)} of {n};  after Holm over {n} = {sum(rej)} of {n}")
