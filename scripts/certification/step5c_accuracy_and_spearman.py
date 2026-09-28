"""PHASE 3 support -- (1) DAVIS random-split AUROC per model, recomputed with sklearn from the saved predictions (DrugBAN: its accuracy JSON);
(2) Spearman rho between AUROC and localisation over the 48 DAVIS attention cells, with an independent percentile bootstrap over cells."""
import csv, glob, gzip, json, os, numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
print("(1) DAVIS random split, test AUROC per model (mean of 3 seeds), from raw predictions")
mean = {}
for model in ("deepdta", "moltrans", "hyperattentiondti", "coldsite_dti"):
    a = []
    for s in (1, 2, 3):
        r = list(csv.DictReader(gzip.open(f"results/accuracy_v2/predictions/davis_random_{model}_seed{s}.csv.gz", "rt")))
        a.append(roc_auc_score([int(float(x["label"])) for x in r], [float(x["logit"]) for x in r]))
    mean[model] = np.mean(a); print("  %-18s seeds %s  mean %.4f" % (model, np.round(a, 4), mean[model]))
d = []
for s in (1, 2, 3):
    j = json.load(open(f"results/analysis_davis_policyA/accuracy_drugban_davis_seed{s}.json")); d.append(j)
print("  drugban accuracy JSON keys:", list(d[0].keys())[:8])
def find_auroc(j):
    for k in ("random", "warm"):
        if k in j: v = j[k]; return v if isinstance(v, float) else v.get("auroc", v)
    for k, v in j.items():
        if isinstance(v, dict) and "random" in v: return v["random"] if isinstance(v["random"], float) else v["random"].get("auroc")
da = [find_auroc(j) for j in d]; mean["drugban"] = float(np.mean(da)); print("  %-18s seeds %s  mean %.4f" % ("drugban", np.round(da, 4), mean["drugban"]))
print("  range of the five model means: %.3f to %.3f" % (min(mean.values()), max(mean.values())))
print("\n(2) Spearman rho, AUROC vs localisation, 48 DAVIS attention-model cells (rows of localization_cells_davis.csv)")
rows = [r for r in csv.DictReader(open("results/accuracy_v2/localization_cells_davis.csv")) if r["model"] != "deepdta"]
print("  cells:", len(rows))
au = np.array([float(r["auroc"]) for r in rows]); en = np.array([float(r["enrichment"]) for r in rows]); pr = np.array([float(r["precision_at_10"]) for r in rows])
rng = np.random.default_rng(41); n = len(rows)
for name, y in (("enrichment (precision@10 / chance)", en), ("precision@10", pr)):
    rho = stats.spearmanr(au, y).statistic; b = []
    for _ in range(20000):
        i = rng.integers(0, n, n); v = stats.spearmanr(au[i], y[i]).statistic
        if np.isfinite(v): b.append(v)
    lo, hi = np.percentile(b, [2.5, 97.5]); print("  %-36s rho = %+.3f   95%% percentile bootstrap over cells [%+.3f, %+.3f]" % (name, rho, lo, hi))
