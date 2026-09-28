"""PHASE 2 STEP 4 -- Holm-Bonferroni, independently.
Raw p per cell in the audit = median over the 3 seeds of a per-seed permutation p (uniform_control cells are extra cells whose p comes from a random map).
Recomputation: per-seed p is exact (convolution of hypergeometrics, see Step 1); median over seeds; the 4 (DAVIS) / 2 (KIBA) uniform-control p-values cannot be
regenerated (they are a single random draw each) so they are taken from the audit JSON and used as-is. Holm is implemented here from its definition."""
import sys, json, numpy as np; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from cert_common import *
from scipy import stats
K = 10; ALPHA = 0.05
def exact_p(ls, tot):
    pmf = np.array([1.0])
    for L, S in ls: pmf = np.convolve(pmf, stats.hypergeom.pmf(np.arange(K + 1), L, S, K))
    return float(pmf[tot:].sum())
def holm(named, alpha=ALPHA):
    items = sorted(named.items(), key=lambda kv: kv[1]); m = len(items); res = {}; alive = True; run = 0.0
    for r, (name, p) in enumerate(items, 1):
        thr = alpha / (m - r + 1); alive = alive and p <= thr; run = max(run, min(1.0, (m - r + 1) * p))
        res[name] = dict(rank=r, p=p, threshold=thr, adj=run, reject=alive)
    return res
def bh(named, alpha=ALPHA):
    items = sorted(named.items(), key=lambda kv: kv[1]); m = len(items); k = 0
    for r, (n, p) in enumerate(items, 1):
        if p <= alpha * r / m: k = r
    return {n for r, (n, p) in enumerate(items, 1) if r <= k}
ALIAS = {"XAttn-Ref": "coldsite_dti", "HyperAttentionDTI": "hyperattentiondti", "MolTrans": "moltrans", "DrugBAN": "drugban"}
summary = {}
for ds, audit_file in (("davis", "results/analysis_davis_policyA/audit_davis_binary_10k_permutations.json"), ("kiba", "results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.json")):
    A = json.load(open(audit_file)); rec = A["p_values_raw"]; seqs = load_seqs(ds); gt = load_sites(ds)
    mine = {}; perseed = {}
    for model, level in cells(ds):
        ps = []
        for s in (1, 2, 3):
            e = ladder(ds, model, s)[level]; c = e["by_k"][str(K)]; ls = LS(ds, e["ids"], seqs=seqs, sites=gt); ls = [x for x in ls if x[1] > 0 and x[0] >= K]
            ps.append(exact_p(ls, int(round(sum(c["per_protein"]) * K))))
        key = f"{ALIAS[model]}|{ds}|{level}"; mine[key] = float(np.median(ps)); perseed[key] = ps
    unif = {k: v for k, v in rec.items() if k.startswith("uniform_control")}
    fam_rec = dict(rec); fam_mine = {**mine, **unif}
    print("=" * 30, ds.upper(), f"family size: recorded {len(fam_rec)} | recomputed {len(fam_mine)}  (model cells {len(mine)} + uniform-control cells {len(unif)})")
    print("  %-40s %12s %12s %9s" % ("cell", "recorded p", "exact-median p", "Δ"))
    for k in sorted(fam_rec, key=lambda k: fam_rec[k]): print("  %-40s %12.6f %12.6f %+9.4f" % (k, fam_rec[k], fam_mine.get(k, float('nan')), fam_mine.get(k, float('nan')) - fam_rec[k]))
    H1 = holm(fam_rec); H2 = holm(fam_mine)
    surv1 = [k for k, v in H1.items() if v["reject"]]; surv2 = [k for k, v in H2.items() if v["reject"]]
    print(f"\n  HOLM on the audit's recorded raw p (my implementation):      survivors {len(surv1)} of {len(fam_rec)}: {surv1}")
    print(f"  HOLM on my exact median-of-seeds p (+ recorded uniform p):     survivors {len(surv2)} of {len(fam_mine)}: {surv2}")
    audit_says = [k for k, v in A["p_values_corrected"].items() if v["significant"]]
    print(f"  audit's own corrected verdict: {audit_says}   | thresholds agree with mine: {all(abs(A['p_values_corrected'][k]['adjusted_alpha'] - H1[k]['threshold']) < 1e-12 for k in A['p_values_corrected'])}")
    top = sorted(H2.items(), key=lambda kv: kv[1]["rank"])[:3]
    for k, v in top: print("     rank %d %-36s p=%.3e  Holm threshold=%.5f  adj p=%.4f  %s" % (v["rank"], k, v["p"], v["threshold"], v["adj"], "REJECT" if v["reject"] else "keep H0"))
    try:
        from statsmodels.stats.multitest import multipletests
        ks = list(fam_mine); rj = multipletests([fam_mine[k] for k in ks], alpha=ALPHA, method="holm")[0]
        print("  statsmodels.multipletests(holm) cross-check: survivors", [k for k, r in zip(ks, rj) if r], "| equals mine:", set(k for k, r in zip(ks, rj) if r) == set(surv2))
    except Exception as ex: print("  statsmodels not available for cross-check:", type(ex).__name__)
    # robustness to the family definition and to the aggregation
    only_models = {k: v for k, v in fam_mine.items() if not k.startswith("uniform")}
    print(f"  ROBUSTNESS  Holm over the {len(only_models)} model cells only: survivors {[k for k, v in holm(only_models).items() if v['reject']]}")
    print(f"              Bonferroni over {len(fam_mine)}: survivors {[k for k, p in fam_mine.items() if p <= ALPHA / len(fam_mine)]}")
    print(f"              Benjamini-Hochberg (FDR 5%) over {len(fam_mine)}: {sorted(bh(fam_mine))}")
    allseed = {f"{k}|s{i+1}": p for k, ps in perseed.items() for i, p in enumerate(ps)}
    hs = holm(allseed); sv = [k for k, v in hs.items() if v["reject"]]
    print(f"              per-seed tests (Holm over {len(allseed)} seed-level exact p): {len(sv)} seed-runs survive, in cells {sorted({k.rsplit('|s',1)[0] for k in sv})}")
    summary[ds] = (len(surv1), len(surv2), len(fam_mine))
print("\nSUMMARY  DAVIS survivors (recorded, recomputed):", summary["davis"][:2], "of", summary["davis"][2], "   [draft: 1 of 20]   |   KIBA:", summary["kiba"][:2], "of", summary["kiba"][2], "   [draft: 0 of 8]")
