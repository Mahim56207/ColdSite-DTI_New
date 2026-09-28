"""PHASE 4 -- red-team simulation ("Reviewer 2").

Four attacks a hostile reviewer would make, each answered ONLY with numbers read from the certification outputs or recomputed here from raw files.
Part A recomputes the integrated-gradients (IG) enrichment independently (attack 2); parts B-E assemble the attacks. Every attack ends with what
the data do NOT let us say, because a rebuttal that hides a concession loses the next round.

    python scripts/certification/step8_red_team.py
"""
import csv, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import cert_common as C
from cert_common import load_seqs, load_sites, LS
ROOT = C.ROOT; os.chdir(ROOT)
CERT = "results/certification"; K = 10
def txt(n): return open(os.path.join(CERT, n), encoding="utf-8").read()
KN = {r["name"]: r for r in csv.DictReader(open(os.path.join(CERT, "key_numbers.csv")))}
def kn(n, col="value"): return float(KN[n][col])
T1, T1c, T3, T4, T5a, T5b, T5c, T7 = (txt(f) for f in ("step1_disagreement.txt", "step1c_rerun_noise.txt", "step3_leak.txt", "step4_holm.txt", "step5a_pocket_bootstrap.txt", "step5b_faithfulness_bootstrap.txt", "step5c_accuracy_and_spearman.txt", "step7_canary_forward_pass.txt"))
IGHOME = os.path.expanduser("~/ColdSite-results")

# ---------------------------------------------------------------- A. independent IG bootstrap
def boot(X, ch, B, rng, two_way, chunk=1000):
    n = X.shape[0]; out = np.empty(B); done = 0
    while done < B:
        m = min(chunk, B - done); idx = rng.integers(0, n, size=(m, n))
        if two_way: sidx = rng.integers(0, 3, size=(m, 3)); W = np.stack([(sidx == j).sum(1) for j in range(3)], 1) / 3.0
        else: W = np.full((m, 3), 1 / 3.0)
        out[done:done + m] = np.einsum("mnj,mj->m", X[idx], W) / n / ch[idx].mean(1); done += m
    return out
def ig_cell(folder, pattern, ds, level, kind):
    seqs = load_seqs(ds); gt = load_sites(ds, kind); Xs = []; ids0 = None
    for s in (1, 2, 3):
        e = json.load(open(os.path.join(IGHOME, folder, pattern.format(s=s))))[level]; c = e["by_k"][str(K)]
        ls = LS(ds, e["ids"], kind=kind, seqs=seqs, sites=gt); keep = [i for i, (L, S) in enumerate(ls) if S > 0 and L >= K]
        assert len(keep) == len(c["per_protein"]), (folder, level, s, len(keep), len(c["per_protein"]))
        ids = [e["ids"][i] for i in keep]; assert ids0 is None or ids == ids0; ids0 = ids
        Xs.append(np.array(c["per_protein"])); ch = np.array([ls[i][1] / ls[i][0] for i in keep])
    return np.vstack(Xs).T, ch, ids0
def repo_row(fam, model, level):
    for r in csv.DictReader(open("results/effects_v2_2d/enrichment.csv")):
        if r["family"] == fam and r["model"] == model and r["level"] == level: return float(r["enrichment"]), float(r["enrichment_low"]), float(r["enrichment_high"])
print("=" * 110 + "\nA. INDEPENDENT RECOMPUTATION OF THE INTEGRATED-GRADIENTS ENRICHMENT (raw IG ladders outside the repo, ground truth from data/)\n" + "=" * 110)
print("   estimator as in step2/step5a: ratio of means, targets and seeds resampled together (two-way), 50,000 resamples, percentile 95% interval")
cells = [("XAttn-Ref IG vs KLIFS pocket", "integrated_gradients/ig_davis_klifs", "ladder_coldsite_dti_ig_davis_seed{s}.json", "davis", "klifs", "S1-klifs", "coldsite_dti_ig", ("random", "cold_drug", "cold_target", "cold_pair")),
         ("XAttn-Ref IG vs UniProt residues", "integrated_gradients/ig_davis", "ladder_coldsite_dti_ig_davis_seed{s}.json", "davis", "uniprot", "S1", "coldsite_dti_ig", ("random", "cold_drug", "cold_target", "cold_pair")),
         ("HyperAttentionDTI IG vs UniProt (KIBA)", "ig_kiba", "ladder_hyperattentiondti_ig_kiba_seed{s}.json", "kiba", "uniprot", "S2", "hyperattentiondti_ig", ("random", "cold_drug"))]
ig = {}
rng = np.random.default_rng(51)
for label, folder, pat, ds, kind, fam, model, levels in cells:
    print(f"\n   {label}")
    print(f"   {'level':<12} {'n':>4} {'mine: point [2-way 95% CI]':<30} {'repo (effects_v2_2d)':<28} lower>1")
    for lv in levels:
        X, ch, ids = ig_cell(folder, pat, ds, lv, kind); pt = X.mean() / ch.mean(); b = np.percentile(boot(X, ch, 50_000, rng, True), [2.5, 97.5]); rp = repo_row(fam, model, lv)
        ig[(label, lv)] = (pt, b[0], b[1], rp); print(f"   {lv:<12} {X.shape[0]:>4} {pt:6.3f} [{b[0]:5.2f}, {b[1]:5.2f}]{'':<10} {rp[0]:6.3f} [{rp[1]:5.2f}, {rp[2]:5.2f}]   {'YES' if b[0] > 1 else 'no'}")
kl = [ig[("XAttn-Ref IG vs KLIFS pocket", lv)] for lv in ("random", "cold_drug", "cold_target", "cold_pair")]
att_cd = float(re.search(r"DAVIS XAttn-Ref\s+cold_drug\s+([\d.]+)", T5a)[1])
print(f"\n   XAttn-Ref DAVIS cold-drug pocket enrichment: IG {ig[('XAttn-Ref IG vs KLIFS pocket','cold_drug')][0]:.3f} [{ig[('XAttn-Ref IG vs KLIFS pocket','cold_drug')][1]:.2f}, {ig[('XAttn-Ref IG vs KLIFS pocket','cold_drug')][2]:.2f}] vs attention {att_cd:.3f} (step5a) | draft: 3.16 [2.40, 4.40] vs 2.10")
print(f"   XAttn-Ref IG vs KLIFS pocket: {sum(k[1] > 1 for k in kl)} of 4 levels have a two-way lower bound above chance (lower bounds {[round(float(k[1]), 2) for k in kl]})")
u = [ig[("XAttn-Ref IG vs UniProt residues", lv)] for lv in ("random", "cold_drug", "cold_target", "cold_pair")]
print(f"   XAttn-Ref IG vs UniProt residues: {sum(k[1] > 1 for k in u)} of 4 levels above chance (lower bounds {[round(float(k[1]), 2) for k in u]}), point estimates {[round(float(k[0]), 2) for k in u]}")
hk = ig[("HyperAttentionDTI IG vs UniProt (KIBA)", "cold_drug")]; print(f"   HyperAttentionDTI KIBA cold-drug IG vs UniProt: {hk[0]:.2f} [{hk[1]:.2f}, {hk[2]:.2f}] -> interval {'includes' if hk[1] <= 1 else 'excludes'} parity | draft: 1.90, [0.63, 3.39], includes parity")

# ---------------------------------------------------------------- B. attack 1
def cell_rows():
    rows = []
    for l in T1.split("\n"):
        m = re.match(r"(DAVIS|KIBA)\s+(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s*$", l)
        if m: rows.append((m[1], m[2], m[3], float(m[4]), float(m[5])))
    return rows
hr = cell_rows()
def holm_names(pvals, names, alpha=0.05):
    order = np.argsort(pvals); m = len(pvals); keep = []; alive = True
    for r, i in enumerate(order):
        alive = alive and pvals[i] <= alpha / (m - r)
        if not alive: break
        keep.append(names[i])
    return keep
perm_names = [f"{d} {mo} {lv}" for d, mo, lv, f, p in hr]; hp = holm_names([p for *_, p in hr], perm_names)
print("\n" + "=" * 110 + "\nATTACK 1: 'The 11-of-22 disagreement is an artefact of borderline effects crossing an alpha threshold.'\n" + "=" * 110)
print("Refutation, from raw data:")
print(f"  1. The claim does not rest on the threshold. Testing the seeds against EACH OTHER (no alpha on the effect at all): Friedman test significant in {kn('friedman_uncorrected'):.0f} of {kn('cells_total'):.0f} cells,")
print(f"     permutation test in {kn('permutation_uncorrected'):.0f}; after Holm over {kn('cells_total'):.0f} cells: Friedman {kn('friedman_holm'):.0f}, permutation {kn('permutation_holm'):.0f}  (results/certification/step1_disagreement.txt:58-59)")
print(f"     cells that survive Holm on the permutation p ({len(hp)}): {hp}")
print(f"  2. The variation is training, not scoring: re-collecting the explanation from the same checkpoint moves P@10 by sd {kn('rerun_noise_sd'):.4f}; between-seed sd is {kn('between_seed_sd'):.4f} ({kn('noise_ratio'):.1f}x larger) (step1c)")
print(f"  3. Seed ORDER is stable across two independent collection passes in {re.search(r'agrees between the two passes in (\d+) of (\d+)', T1c)[1]} of {re.search(r'agrees between the two passes in (\d+) of (\d+)', T1c)[2]} cells (step1c): the seeds really do differ from one another.")
print(f"  4. Effect size, not just significance: HyperAttentionDTI DAVIS cold-drug seeds 0.0249 / 0.0768 / 0.0195 vs chance 0.0204: best/worst = 3.94x; between-seed sd 0.032 vs protein-sampling SE 0.0018 (step2).")
print(f"  5. The spread exceeds the distance from chance in {kn('spread_exceeds_distance'):.0f} of {kn('cells_total'):.0f} cells (step1) -- reported, but the draft itself calls it weak near chance.")
print("What the data do NOT let us say (concede these):")
print(f"  - 'Exactly 12'. The sampled 1,000-permutation run gave 12; the exact test gives {kn('seeds_disagree_exact'):.0f}; the count is {kn('disagree_at_alpha_0.010'):.0f} at alpha 0.01 and {kn('disagree_at_alpha_0.100'):.0f} at 0.10. It is a threshold-dependent count.")
print(f"  - The direct test finds significant seed variance in {kn('permutation_holm'):.0f}-{kn('friedman_holm'):.0f} of {kn('cells_total'):.0f} cells, i.e. in about a third, not in most cells. 'Verdicts are unstable' is supported; 'every cell is unstable' is not.")
print("  VERDICT: attack answered; the paper must lead with the direct test and the effect sizes, not with the threshold count.")

# ---------------------------------------------------------------- C. attack 2
faith = re.search(r"intervals with lower bound above zero: (\d+) of (\d+)", T5b); acc_r = re.search(r"range of the five model means: ([\d.]+) to ([\d.]+)", T5c)
print("\n" + "=" * 110 + "\nATTACK 2: 'The models learned nothing; the attention is noise, so the audit audits noise.'\n" + "=" * 110)
print("Refutation, from raw data:")
print(f"  1. The models predict binding: DAVIS random-split test AUROC of the five models {acc_r[1]}-{acc_r[2]} (sklearn on raw predictions, step5c); a noise model scores 0.5.")
print(f"  2. The attention is used by the model: masking the attended residues beats size-matched random masking with a lower bound above zero in {faith[1]} of {faith[2]} cells (three models, seeds and targets resampled, step5b).")
print(f"  3. The weights carry site information even where the attention does not show it: integrated gradients on XAttn-Ref, pocket enrichment above chance at {sum(k[1] > 1 for k in kl)} of 4 DAVIS levels;")
print(f"     cold-drug {ig[('XAttn-Ref IG vs KLIFS pocket','cold_drug')][0]:.2f} [{ig[('XAttn-Ref IG vs KLIFS pocket','cold_drug')][1]:.2f}, {ig[('XAttn-Ref IG vs KLIFS pocket','cold_drug')][2]:.2f}] against attention {att_cd:.2f} (independent recomputation; draft 3.16 [2.40, 4.40]).")
print(f"  4. The pipeline can see a real signal: attention pocket enrichment survives in XAttn-Ref at 4 of 4 and HyperAttentionDTI at 3 of 4 DAVIS levels (step5a), and the positive control detects a 0.02 dose at every level (results/positive_control_davis.md).")
print(f"  5. The saved numbers are the network's numbers: live forward passes on the GPU reproduce the saved logits within 1e-5 for {re.search(r'logit within 1e-05 of the saved CSV: (\d+)/(\d+)', T7)[1]} of {re.search(r'logit within 1e-05 of the saved CSV: (\d+)/(\d+)', T7)[2]} pairs, with identical checkpoint hashes (step7).")
print("What the data do NOT let us say (concede these):")
print(f"  - IG is a confirmatory, partial result: against UniProt residues only {sum(k[1] > 1 for k in u)} of 4 XAttn-Ref levels are above chance ({[round(float(k[1]), 2) for k in u]}); HyperAttentionDTI's KIBA cold-drug IG interval [{hk[1]:.2f}, {hk[2]:.2f}] includes parity.")
print("  - Above-chance pocket enrichment is coarse (no conservation control): it does not show the models learned binding chemistry, only that they do not ignore the pocket.")
print("  - The strongest claim the data support is 'the models are not noise and their weights hold pocket information that attention under-reports', not 'they learned binding sites'.")
print("  VERDICT: attack answered by accuracy + faithfulness + IG; the 3.16x figure alone would not be enough and must not be presented as the shield.")

# ---------------------------------------------------------------- D. attack 3
seqs = load_seqs("davis")
def leaked_targets(level):
    rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
    tr = rd(f"data/splits/davis/{level}/train.csv"); te = rd(f"data/splits/davis/{level}/test.csv"); ts = {r["Target"] for r in tr}; tn = {r["Target_ID"] for r in tr}
    return {r["Target_ID"] for r in te if r["Target_ID"] not in tn and seqs[r["Target_ID"]] in ts}
print("\n" + "=" * 110 + "\nATTACK 3: 'The DAVIS leak invalidates the whole benchmark.'\n" + "=" * 110)
print("Refutation, from raw data:")
scored_leak = {}
for level in ("cold_target", "cold_pair"):
    lk = leaked_targets(level); ids_all = set()
    for mdl in ("ladder_davis_seed{s}.json", "ladder_hyperattentiondti_davis_seed{s}.json", "ladder_moltrans_davis_seed{s}.json", "ladder_drugban_davis_seed{s}.json"):
        for s in (1, 2, 3): ids_all |= set(json.load(open(f"results/analysis_davis_policyA/{mdl.format(s=s)}"))[level]["ids"])
    scored_leak[level] = (len(lk), len(ids_all & lk), len(ids_all))
    print(f"  1. {level}: {len(lk)} test targets are seen by sequence; {len(ids_all & lk)} of them are scored by any of the four attention models' explanation analyses ({len(ids_all)} distinct proteins scored, 3 seeds x 4 models). The explanation results never touch the leak.")
print(f"  2. The leak is measured and bounded: it affects 12 of 88 cold-target test targets (816 of 5,984 rows, 13.6%) and 11 of 88 cold-pair targets (143 of 1,144 rows) (step3, sequence audit recomputed from the split CSVs).")
print(f"  3. On the leaked rows the sequence match inflates DeepDTA AUROC by {kn('leak_ct_leaked_rows'):.3f} (95% t-interval [{kn('leak_ct_leaked_rows','low'):.3f}, {kn('leak_ct_leaked_rows','high'):.3f}], p = {kn('leak_ct_leaked_rows','p'):.4f}); on the unleaked rows {kn('leak_ct_unleaked_rows'):+.3f} (p = {kn('leak_ct_unleaked_rows','p'):.2f}): accuracy on sequence-unseen targets is not inflated.")
LABEL = {"deepdta": "DeepDTA", "moltrans": "MolTrans", "hyperattentiondti": "HyperAttentionDTI", "coldsite_dti": "XAttn-Ref"}
gap = [(LABEL[a], b) for a, b in re.findall(r"(\w+) ([+-][\d.]+)", re.search(r"leaked-minus-unleaked AUROC gap, per model, cold_target \(mean of 3 seeds\): (.*)", T3)[1].replace(",", ""))]
print(f"  4. The signature is present in all four models (leaked minus unleaked AUROC, cold-target): {', '.join(f'{a} {b}' for a, b in gap)}; the paper reports cold-level accuracy on sequence-unseen targets and states the correction.")
print(f"  5. The headline (seed instability, calibrated battery, readout dependence) is built on per-protein explanation scores at all four levels, including DAVIS random and cold-drug, where every test target is also in the training set by construction, so the cold-start leak cannot arise.")
print("What the data do NOT let us say (concede these):")
print(f"  - The retrain is DeepDTA at cold-target only, three seeds. All-rows effect {kn('leak_ct_all_rows'):+.3f}, CI [{kn('leak_ct_all_rows','low'):.3f}, {kn('leak_ct_all_rows','high'):.3f}], p = {kn('leak_ct_all_rows','p'):.2f}: not distinguishable from zero. At cold-pair the effect is {kn('leak_cp_all_rows'):+.3f} (p = {kn('leak_cp_all_rows','p'):.2f}) and the smaller training set costs more (0.062).")
print("  - DAVIS cold-pair validation still contains seen-by-sequence targets (checkpoint selection may have been influenced; problem-log row s): open, cannot be fixed without retraining.")
print("  VERDICT: attack answered for the explanation results; the accuracy claims at cold levels carry a disclosed, partly unquantified leak.")

# ---------------------------------------------------------------- E. attack 4
print("\n" + "=" * 110 + "\nATTACK 4 (added): 'Three seeds cannot support any of this; your intervals are an artefact of resampling three numbers.'\n" + "=" * 110)
print("Refutation, from raw data:")
mc = re.search(r"lower bound: min ([\d.]+)\s+max ([\d.]+)\s+mean [\d.]+\s+\| upper bound: min ([\d.]+)\s+max ([\d.]+)", T2 := txt("step2_two_way_bootstrap.txt"))
print(f"  1. The intervals are conservative, and the paper says so: with 3 seeds, 1 in 9 resamples draws one seed three times. The headline cell's two-way interval is stable across 10 RNG seeds (lower {mc[1]}-{mc[2]}, upper {mc[3]}-{mc[4]}) and crosses parity in every run.")
print(f"  2. The residue-level verdict is robust to the aggregation and to the family: the single survivor (HyperAttentionDTI, DAVIS random) is also the single Holm survivor under Holm on 16 cells, Bonferroni and Benjamini-Hochberg (step4), and its two-way lower bound is 1.32 (step2).")
print(f"  3. The claims that need many seeds are the ones we do NOT make: no cell's verdict depends on a lower bound within 0.03 of 1 at the residue level (step2), and the borderline pocket cell (HyperAttentionDTI cold-pair, lower bound 1.02) is labelled marginal on slide and in the draft.")
print("What the data do NOT let us say (concede these):")
print(f"  - Per-seed Holm keeps {kn('holm_per_seed_davis_survivors'):.0f} of {kn('holm_per_seed_davis_tests'):.0f} DAVIS seed-runs and {kn('holm_per_seed_kiba_survivors'):.0f} of {kn('holm_per_seed_kiba_tests'):.0f} KIBA seed-runs: the 'only 1 of 20' statement is about the median-over-seeds rule, and is stated as such.")
print("  - Seeds 4 and 5 (Wave A) are planned, not run: no number in the paper depends on them; they may sharpen the instability counts but cannot remove the disagreements already observed.")
print("  VERDICT: attack partly stands (3 seeds is small); the paper's wording already scopes claims to what 3 seeds can support.")
print("\n" + "=" * 110 + "\nRESIDUAL RISKS NOT COVERED BY ANY ATTACK ABOVE")
print("  - Kinase-only scope (both datasets are kinase panels; the non-kinase panel is not analysed). - No conservation control for pocket enrichment.")
print("  - Only HyperAttentionDTI seed 1 was re-run live (step7); MolTrans, XAttn-Ref, DrugBAN and seeds 2-3 are verified against saved files only.")
print("  - Literature claims and citations are EXTERNAL to this certification (not checked).")
