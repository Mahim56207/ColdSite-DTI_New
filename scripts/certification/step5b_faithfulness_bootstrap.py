"""PHASE 3 support -- faithfulness (masking attended residues minus size-matched random masking): independent two-way bootstrap.
Reads per-pair deltas from results/effects_v2/faithfulness/*faithfulness_*_seed*.json. Per seed, deltas are averaged within target;
targets present in all three seeds are resampled together with the three seeds (percentile 95% interval). DrugBAN has no per-pair
values, so its seed-level deltas are read from the markdown tables and only counted."""
import glob, json, os, re, sys, numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
PAT = re.compile(r"^(?P<tok>token_)?faithfulness_(?:(?P<model>[a-z_]+?)_)?(?P<ds>davis|kiba)_seed(?P<seed>\d)\.json$")
files = {}
for p in sorted(glob.glob("results/effects_v2/faithfulness/*faithfulness_*_seed*.json")):
    m = PAT.match(os.path.basename(p))
    if m: files.setdefault((m["model"] or "xattn_ref", m["ds"]), {})[int(m["seed"])] = p
rng = np.random.default_rng(31); B = 50_000; out = []
print("%-18s %-5s %-11s %6s %8s  %-18s %s" % ("model", "ds", "level", "n_tgt", "delta", "95% two-way CI", "lower>0"))
for (model, ds), seeds in sorted(files.items()):
    if len(seeds) < 3: continue
    P = {s: json.load(open(p)) for s, p in seeds.items()}
    for level in ("random", "cold_drug", "cold_target", "cold_pair"):
        per = {}
        for s, d in P.items():
            e = (d.get("levels", d)).get(level)
            if not e: continue
            g = {}
            for pr in e["per_pair"]:
                if pr.get("id") is not None and pr["comprehensiveness_delta"] is not None: g.setdefault(pr["id"], []).append(pr["comprehensiveness_delta"])
            per[s] = {t: float(np.mean(v)) for t, v in g.items()}
        if len(per) < 3: continue
        common = sorted(set.intersection(*[set(v) for v in per.values()])); X = np.array([[per[s][t] for s in (1, 2, 3)] for t in common])
        n = len(common); vals = np.empty(B); done = 0
        while done < B:
            m = min(2000, B - done); idx = rng.integers(0, n, size=(m, n)); sidx = rng.integers(0, 3, size=(m, 3))
            W = np.stack([(sidx == j).sum(1) for j in range(3)], 1) / 3.0; vals[done:done + m] = np.einsum("mnj,mj->m", X[idx], W) / n; done += m
        lo, hi = np.percentile(vals, [2.5, 97.5]); out.append((model, ds, level, n, X.mean(), lo, hi))
        print("%-18s %-5s %-11s %6d %8.4f  [%7.4f, %7.4f]   %s" % (model, ds.upper(), level, n, X.mean(), lo, hi, "YES" if lo > 0 else "no"))
print(f"\nCELLS: {len(out)} ; intervals with lower bound above zero: {sum(o[5] > 0 for o in out)} of {len(out)}")
small = min((o for o in out if o[0] == 'hyperattentiondti'), key=lambda o: o[4])
print("smallest HyperAttentionDTI cell: %s %s %s  delta %.3f  [%.3f, %.3f]" % (small[0], small[1], small[2], small[4], small[5], small[6]))
print("models covered:", sorted({o[0] for o in out}))
# DrugBAN seed-level deltas straight from the committed markdown tables (no per-pair data => no interval)
print("\nDrugBAN DAVIS seed-level deltas (comprehensiveness minus random control), from faithfulness_drugban_davis_seed{1,2,3}.md:")
vals = {}
for s in (1, 2, 3):
    for line in open(f"results/analysis_davis_policyA/faithfulness_drugban_davis_seed{s}.md"):
        m = re.match(r"\| (Warm|Cold-Drug|Cold-Target|Cold-Pair) \| [\d.]+ \| [\d.]+ \| \*\*(-?[\d.]+)\*\* ", line)
        if m: vals[(s, m[1])] = float(m[2])
for s in (1, 2, 3): print("  seed %d: " % s + "  ".join(f"{l}={vals[(s, l)]:+.4f}" for l in ("Warm", "Cold-Drug", "Cold-Target", "Cold-Pair")))
print("  negative seed-level values: %d of %d" % (sum(v < 0 for v in vals.values()), len(vals)))
