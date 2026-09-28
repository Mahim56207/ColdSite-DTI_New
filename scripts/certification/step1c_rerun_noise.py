"""Is between-seed spread bigger than the pipeline's own re-run noise?  Same checkpoints, two independent collection passes:
   pass A = ladder_*.json per-protein hits (used by seed_agreement / draft headline); pass B = audit_*_10k grid per-seed precision (fresh collection)."""
import sys, json, numpy as np; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from cert_common import *
ALIAS = {"XAttn-Ref": "coldsite_dti", "HyperAttentionDTI": "hyperattentiondti", "MolTrans": "moltrans", "DrugBAN": "drugban"}
rows = []
for ds, f in (("davis", "results/analysis_davis_policyA/audit_davis_binary_10k_permutations.json"), ("kiba", "results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.json")):
    A = json.load(open(f))["grid"]
    for model, level in cells(ds):
        a = [ladder(ds, model, s)[level]["by_k"]["10"]["precision_at_k"] for s in (1, 2, 3)]
        b = A[ALIAS[model]][level]["precision_at_k"]["values"]
        rows.append((ds, model, level, np.array(a), np.array(b)))
d = np.concatenate([r[3] - r[4] for r in rows]); ad = np.abs(d)
print("66 seed-cells: |pass A - pass B| P@10 : median %.4f  mean %.4f  max %.4f   (identical checkpoint, only the collection pass differs)" % (np.median(ad), ad.mean(), ad.max()))
sp_a = np.array([r[3].max() - r[3].min() for r in rows]); sp_b = np.array([r[4].max() - r[4].min() for r in rows])
print("between-seed spread (max-min) per cell: pass A median %.4f, pass B median %.4f" % (np.median(sp_a), np.median(sp_b)))
print("re-run noise sd (per seed-cell, from A-B differences / sqrt2) = %.4f ; between-seed sd (pooled within-cell, pass A) = %.4f ; ratio = %.2f" % (d.std(ddof=1) / np.sqrt(2), np.sqrt(np.mean([r[3].var(ddof=1) for r in rows])), np.sqrt(np.mean([r[3].var(ddof=1) for r in rows])) / (d.std(ddof=1) / np.sqrt(2))))
# variance decomposition: is spread across seeds larger than re-run noise, cell by cell?
print("\n%-5s %-18s %-11s %-22s %-22s %8s %8s" % ("ds", "model", "level", "pass A P@10", "pass B P@10", "spread A", "max|A-B|"))
n_more = 0
for ds, model, level, a, b in rows:
    n_more += (a.max() - a.min()) > np.abs(a - b).max()
    print("%-5s %-18s %-11s %-22s %-22s %8.4f %8.4f" % (ds.upper(), model, level, np.round(a, 3), np.round(b, 3), a.max() - a.min(), np.abs(a - b).max()))
print(f"\ncells where the between-seed spread (pass A) exceeds the largest re-run difference: {n_more} of {len(rows)}")
# does the 'seeds disagree' count survive the re-run?  Use pass-B precision with the same exact null (per-protein details not stored in B) -> use p-values recorded in the 10k audit is aggregated only; so compare the sign of (P@10 - chance) instead
print("\nverdict-free check: rank order of the three seeds within each cell agrees between the two passes in %d of %d cells" % (sum(np.array_equal(np.argsort(a), np.argsort(b)) for *_, a, b in rows), len(rows)))
