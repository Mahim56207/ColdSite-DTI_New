"""Ironclad diagnostics for HyperAttentionDTI, seeds 1-3, DAVIS (all four levels). DIAGNOSTICS, NOT PROOFS.

Why these are diagnostics. Seed-to-seed variation is a statement about SEPARATELY TRAINED networks; nothing computed on one
fixed network (a Jacobian, a perturbation) can prove it, and independently initialised networks are not close in weight
space, so "the weights are nearly identical" cannot be the premise. What CAN be measured, and is measured here:

  D1a weight space: flattened and per-tensor cosine, relative distance, and two permutation-tolerant comparisons
      (best-match cosine of first-layer filters; residue-embedding geometry), against untrained networks;
  D1b function space: agreement of the predictions (saved logits, all test rows) for four models;
  D1c attention space: Jensen-Shannon divergence, rank correlation and top-10 overlap of the per-residue attention
      between seeds on every scored protein (live GPU forward passes), with references: same seed / different drug,
      untrained networks, chance;
  D1d mechanism (found while checking D2): the trained protein-convolution features are almost entirely zero, the top
      attended positions are the few active ones, and different seeds activate different positions;
  D2  local conditioning of the attention gate. HyperAttentionDTI's gate is sigmoid(mean over drug positions of
      Linear(ReLU(drug_att + Linear(protein features)))): no softmax (a softmax Jacobian has a zero singular value by
      construction, so its condition number is infinite and says nothing). Per protein position the gate is a 160 -> 160
      map: exact 160 x 160 Jacobian, numerical rank and condition number over its range, at ACTIVE positions (nonzero
      features) and INACTIVE ones, against untrained networks, and against how much the seeds disagree on the protein;
  D3  how far must the weights move? Gaussian perturbations of a trained checkpoint against attention and prediction,
      beside the actual seed-to-seed distance;
  D4  cross-optimizer: not run (every checkpoint is AdamW, the published recipe; see the report).

    python scripts/certification/step9_ironclad_diagnostics.py [--device mps|cpu] [--quick]
"""
import argparse, csv, gzip, itertools, json, os, sys, time
import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines", "HpyerAttentionDTI"))
from hyperparameter import hyperparameter
from model import AttentionDTI
from dataset import CHARISOSMISET, CHARPROTSET, label_sequence, label_smiles

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
DEV = args.device; CK = os.path.expanduser("~/ColdSite-results/davis_binary"); T_START = time.time()
LEVELS = ("random", "cold_drug", "cold_target", "cold_pair"); SEEDS = (1, 2, 3); UNTRAINED = (101, 102, 103)
OFFSET, CENTRE, K = 21, 10, 10
NCAP = 40 if args.quick else 10_000            # proteins per level for D1c/D1d
NJAC, NPOS = (10, 4) if args.quick else (50, 8)  # D2: proteins, positions per category
NPERT, NDRAW = (20, 3) if args.quick else (100, 5)
print(f"torch {torch.__version__} | device {DEV} | mps {torch.backends.mps.is_available()} | quick {args.quick}")

def read(p): return list(csv.DictReader(open(p, encoding="utf-8")))
def load_trained(seed, level, device):
    m = AttentionDTI(hyperparameter()); st = torch.load(f"{CK}/coldsite_dti_davis_{level}_binary_seed{seed}_hyperattentiondti.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(st.get("model_state", st)); return m.to(device).eval()
def untrained(seed, device):
    torch.manual_seed(seed); return AttentionDTI(hyperparameter()).to(device).eval()

@torch.no_grad()
def pack(model, device, smiles, seq, want_jac=False):
    """One live forward pass: per-residue attention (channel-mean gate at the centre residue), the logit, the per-position
    protein-feature norm, and (want_jac) the tensors the gate Jacobian needs. Everything is read from the REAL forward pass."""
    d = torch.from_numpy(label_smiles(smiles, CHARISOSMISET, 100)).unsqueeze(0).to(device)
    p = torch.from_numpy(label_sequence(seq, CHARPROTSET, 1000)).unsqueeze(0).to(device); cap = {}
    hs = [model.attention_layer.register_forward_hook(lambda m, i, o: cap.__setitem__("A", o)),
          model.Protein_CNNs.register_forward_hook(lambda m, i, o: cap.__setitem__("pc", o)),
          model.drug_attention_layer.register_forward_hook(lambda m, i, o: cap.__setitem__("da", o))]
    logits = model(d, p).reshape(-1).float().cpu()
    for h in hs: h.remove()
    L = min(len(seq), 1000); nv = L - OFFSET
    gate = torch.sigmoid(cap["A"].mean(dim=1))[0, :nv]                     # (nv, 160): the gate the real forward used
    w = np.zeros(L); w[np.arange(nv) + CENTRE] = gate.mean(-1).cpu().double().numpy()
    out = dict(w=w, logit=float(logits[1] - logits[0]), nv=nv, fnorm=cap["pc"][0, :, :nv].norm(dim=0).cpu().double().numpy(),
               dead=int((cap["pc"][0, :, :nv] == 0).all(1).sum()))
    if want_jac:
        out.update(pc=cap["pc"][0, :, :nv].cpu(), da=cap["da"][0].cpu(), gate=gate.cpu())
    return out
def interior(w, nv): return w[CENTRE:CENTRE + nv]
def js_div(a, b):
    p, q = a / a.sum(), b / b.sum(); m = 0.5 * (p + q)
    def kl(x, y):
        nz = x > 0; return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
def top(x): return set(np.argsort(-x, kind="stable")[:K])
def compare(a, b):
    ia, ib = interior(a["w"], a["nv"]), interior(b["w"], b["nv"]); common = (a["fnorm"] > 0) & (b["fnorm"] > 0)
    rc = float(stats.spearmanr(ia[common], ib[common]).statistic) if common.sum() >= 10 else float("nan")
    return dict(js=js_div(ia, ib), rho=float(stats.spearmanr(ia, ib).statistic), top=len(top(ia) & top(ib)) / K, rho_common=rc,
                jacc=len(set(np.where(a["fnorm"] > 0)[0]) & set(np.where(b["fnorm"] > 0)[0])) / max(len(set(np.where(a["fnorm"] > 0)[0]) | set(np.where(b["fnorm"] > 0)[0])), 1))
def summ(rows, key, fmt="{:.3f}"):
    v = np.array([r[key] for r in rows], dtype=float); v = v[np.isfinite(v)]
    return f"{fmt.format(v.mean())} (median {fmt.format(np.median(v))}, IQR {fmt.format(np.percentile(v,25))} to {fmt.format(np.percentile(v,75))})"

ladder = json.load(open("results/analysis_davis_policyA/ladder_hyperattentiondti_davis_seed1.json"))
def pairs_for(level):
    test = read(f"data/splits/davis/{level}/test.csv"); first, second = {}, {}
    for i, r in enumerate(test): first.setdefault(r["Target_ID"], i)
    for i, r in enumerate(test):
        t = r["Target_ID"]
        if t in first and t not in second and r["Drug_ID"] != test[first[t]]["Drug_ID"]: second[t] = i
    return [dict(tid=t, smiles=test[first[t]]["Drug"], seq=test[first[t]]["Target"], alt=(test[second[t]]["Drug"] if t in second else None)) for t in ladder[level]["ids"][:NCAP]]

# ============================================================================================ D1a
print("\n" + "=" * 112 + "\nD1a. WEIGHT SPACE: seed-to-seed similarity of the trained weights (and of two untrained networks)\n" + "=" * 112)
def state(m): return {k: v.detach().cpu() for k, v in m.state_dict().items()}
def flat(sd): return torch.cat([v.flatten().double() for v in sd.values()])
def cos(a, b): return float(torch.dot(a, b) / (a.norm() * b.norm()))
un = {s: state(untrained(s, "cpu")) for s in UNTRAINED}
tr = {(s, l): state(load_trained(s, l, "cpu")) for s in SEEDS for l in LEVELS}
def wrep(sa, sb):
    fa, fb = flat(sa), flat(sb); wt = [k for k, v in sa.items() if v.ndim >= 2]; per = [cos(sa[k].flatten().double(), sb[k].flatten().double()) for k in wt]
    return cos(fa, fb), float((fa - fb).norm() / fa.norm()), float(np.median(per)), float(np.max(np.abs(per)))
print(f"{'pair':<36} {'global cos':>10} {'rel. distance':>14} {'median tensor cos':>18} {'max |tensor cos|':>17}")
tr_rows, un_rows = [], []
for l in LEVELS:
    for a, b in itertools.combinations(SEEDS, 2):
        r = wrep(tr[(a, l)], tr[(b, l)]); tr_rows.append(r); print(f"trained seeds {a} vs {b}, {l:<11}        {r[0]:10.4f} {r[1]:14.3f} {r[2]:18.4f} {r[3]:17.4f}")
for a, b in itertools.combinations(UNTRAINED, 2):
    r = wrep(un[a], un[b]); un_rows.append(r); print(f"UNTRAINED inits {a} vs {b}                 {r[0]:10.4f} {r[1]:14.3f} {r[2]:18.4f} {r[3]:17.4f}")
print(f"-> trained pairs: global cosine {min(r[0] for r in tr_rows):.4f} to {max(r[0] for r in tr_rows):.4f}; relative distance {min(r[1] for r in tr_rows):.2f} to {max(r[1] for r in tr_rows):.2f}. Two untrained inits: cosine {min(r[0] for r in un_rows):.4f} to {max(r[0] for r in un_rows):.4f}.")
print("   Independently initialised networks are FAR apart in weight space (as far as two untrained ones): the premise 'the seeds learned nearly identical weights' is false here.")
print("\nPermutation-tolerant comparisons (a hidden unit's index means nothing, so raw cosine can hide a shared solution):")
def match_cos(A, B):
    A = A.reshape(A.shape[0], -1).double(); B = B.reshape(B.shape[0], -1).double()
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12); B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12); C = (A @ B.T).numpy(); r, c = linear_sum_assignment(-C); return float(C[r, c].mean())
def rsa(E, F):
    keep = (E.norm(dim=1) > 0) & (F.norm(dim=1) > 0)                     # drop the padding row
    def gram(X): X = X[keep].double(); X = X / X.norm(dim=1, keepdim=True); return (X @ X.T).numpy()[np.triu_indices(int(keep.sum()), 1)]
    return float(stats.spearmanr(gram(E), gram(F)).statistic)
print(f"{'pair':<36} {'conv1 filters: best-match cos':>30} {'residue-embedding geometry (RSA rho)':>38}")
for l in LEVELS:
    for a, b in itertools.combinations(SEEDS, 2):
        sa, sb = tr[(a, l)], tr[(b, l)]
        print(f"trained seeds {a} vs {b}, {l:<11}        {match_cos(sa['Protein_CNNs.0.weight'], sb['Protein_CNNs.0.weight']):30.3f} {rsa(sa['protein_embed.weight'], sb['protein_embed.weight']):38.3f}")
for a, b in itertools.combinations(UNTRAINED, 2):
    print(f"UNTRAINED inits {a} vs {b}                 {match_cos(un[a]['Protein_CNNs.0.weight'], un[b]['Protein_CNNs.0.weight']):30.3f} {rsa(un[a]['protein_embed.weight'], un[b]['protein_embed.weight']):38.3f}")

# ============================================================================================ D1b
print("\n" + "=" * 112 + "\nD1b. FUNCTION SPACE: do the seeds make the same predictions? (saved logits, every test row; the three seed pairs)\n" + "=" * 112)
def logits(model, level, seed): return np.array([float(r["logit"]) for r in csv.DictReader(gzip.open(f"results/accuracy_v2/predictions/davis_{level}_{model}_seed{seed}.csv.gz", "rt"))])
print(f"{'model':<20} {'level':<12} {'Pearson r':>28} {'Spearman rho':>28}")
func = {}
for model in ("hyperattentiondti", "moltrans", "coldsite_dti", "deepdta"):
    for l in LEVELS:
        Lg = {s: logits(model, l, s) for s in SEEDS}; pr = [stats.pearsonr(Lg[a], Lg[b]).statistic for a, b in itertools.combinations(SEEDS, 2)]
        sp = [stats.spearmanr(Lg[a], Lg[b]).statistic for a, b in itertools.combinations(SEEDS, 2)]; func[(model, l)] = (float(np.mean(pr)), float(np.mean(sp)))
        print(f"{model:<20} {l:<12} {' '.join(f'{x:.3f}' for x in pr):>28} {' '.join(f'{x:.3f}' for x in sp):>28}")

# ============================================================================================ D1c / D1d
print("\n" + "=" * 112 + "\nD1c. ATTENTION SPACE: do the seeds attend to the same residues? (live GPU forward passes, every scored protein)\n" + "=" * 112)
P, packs = {}, {}
for l in LEVELS:
    P[l] = pairs_for(l)
    for s in SEEDS:
        m = load_trained(s, l, DEV); packs[(l, s)] = [pack(m, DEV, p["smiles"], p["seq"]) for p in P[l]]; del m
    print(f"  {l}: {len(P[l])} proteins x 3 seeds done ({time.time()-T_START:.0f} s elapsed)", flush=True)
print(f"\n{'level':<12} {'n':>4} {'pair':>5} {'Spearman rho over residues':<44} {'top-10 overlap':>14} {'JS divergence (bits)':>22}")
cmp_rows = {}
for l in LEVELS:
    for a, b in itertools.combinations(SEEDS, 2):
        rows = [compare(packs[(l, a)][i], packs[(l, b)][i]) for i in range(len(P[l]))]; cmp_rows[(l, a, b)] = rows
        print(f"{l:<12} {len(rows):>4} {a} v {b} {summ(rows,'rho'):<44} {np.mean([r['top'] for r in rows]):14.3f} {np.mean([r['js'] for r in rows]):22.2e}")
prs = P["random"]; m0 = None
alt_rows = []
for s in SEEDS:
    m = load_trained(s, "random", DEV)
    for i, p in enumerate(prs):
        if p["alt"]: alt_rows.append(compare(packs[("random", s)][i], pack(m, DEV, p["alt"], p["seq"])))
    del m
un_pk = {}
for s in UNTRAINED:
    m = untrained(s, DEV); un_pk[s] = [pack(m, DEV, p["smiles"], p["seq"]) for p in prs]; del m
un_rows_att = [compare(un_pk[a][i], un_pk[b][i]) for a, b in itertools.combinations(UNTRAINED, 2) for i in range(len(prs))]
chance_top = float(np.mean([K / pk["nv"] for pk in packs[("random", 1)]]))
uni_js = float(np.mean([js_div(interior(pk["w"], pk["nv"]), np.ones(pk["nv"])) for pk in packs[("random", 1)]]))
print("\nREFERENCES (level 'random'):")
print(f"  same seed, same protein, DIFFERENT DRUG  ({len(alt_rows)} comparisons): rho {summ(alt_rows,'rho')} | top-10 {np.mean([r['top'] for r in alt_rows]):.3f} | JS {np.mean([r['js'] for r in alt_rows]):.2e}")
print(f"  two UNTRAINED networks (3 pairs x {len(prs)} proteins): rho {summ(un_rows_att,'rho')} | top-10 {np.mean([r['top'] for r in un_rows_att]):.3f} | JS {np.mean([r['js'] for r in un_rows_att]):.2e}")
print(f"  chance top-10 overlap of two random maps: {chance_top:.3f} | JS between a trained map and the UNIFORM map: {uni_js:.2e} bits (maps are nearly flat: read JS relative to that, and prefer the rank statistics)")
gm = [float(np.mean([pk["w"][CENTRE:CENTRE + pk["nv"]].max() - pk["w"][CENTRE:CENTRE + pk["nv"]].min() for pk in packs[("random", s)]])) for s in SEEDS]
print(f"  dynamic range of the trained attention map (max - min of the channel-mean gate, mean over proteins) per seed: {', '.join(f'{x:.4f}' for x in gm)} (the gate sits near 0.5)")
rho_prot = np.array([[cmp_rows[("random", a, b)][i]["rho"] for a, b in itertools.combinations(SEEDS, 2)] for i in range(len(prs))]).mean(1)
print(f"  per-protein mean cross-seed rho, level random: percentiles 5/25/50/75/95 = {np.percentile(rho_prot,[5,25,50,75,95]).round(3)}; share of proteins with rho < 0.5: {(rho_prot<0.5).mean():.2f}")
w_rho = float(np.mean([r["rho"] for r in itertools.chain(*[cmp_rows[("random", a, b)] for a, b in itertools.combinations(SEEDS, 2)])]))
print(f"\nSIDE BY SIDE (HyperAttentionDTI, random level): weight cosine {np.mean([r[0] for r in tr_rows[:3]]):.3f} | prediction Spearman {func[('hyperattentiondti','random')][1]:.3f} | attention Spearman {w_rho:.3f} (same seed, other drug: {np.mean([r['rho'] for r in alt_rows]):.3f})")

print("\n" + "=" * 112 + "\nD1d. MECHANISM: how sparse are the trained protein features, and do the seeds use the same positions?\n" + "=" * 112)
print(f"{'level':<12} {'seed':>4} {'share of valid positions with any nonzero feature':>52} {'channels dead everywhere (of 160)':>34} {'top-10 attended that are active':>32} {'rho(attention, feature norm)':>29}")
for l in LEVELS:
    for s in SEEDS:
        pk = packs[(l, s)]
        act = np.mean([(x["fnorm"] > 0).mean() for x in pk]); dead = np.mean([x["dead"] for x in pk])
        top_act = np.mean([np.mean([x["fnorm"][i] > 0 for i in top(interior(x["w"], x["nv"]))]) for x in pk])
        rr = np.nanmean([stats.spearmanr(interior(x["w"], x["nv"]), x["fnorm"]).statistic for x in pk if (x["fnorm"] > 0).sum() >= 3])
        print(f"{l:<12} {s:>4} {act:52.3f} {dead:34.1f} {top_act:32.3f} {rr:29.3f}")
act_un = np.mean([(x["fnorm"] > 0).mean() for s in UNTRAINED for x in un_pk[s]])
print(f"UNTRAINED networks: share of positions with any nonzero feature {act_un:.3f}; channels dead everywhere {np.mean([x['dead'] for s in UNTRAINED for x in un_pk[s]]):.1f}")
dens = {s: np.array([(x["fnorm"] > 0).mean() for x in packs[("random", s)]]) for s in SEEDS}
print("\nDo different seeds activate the same positions? (level random; Jaccard of the active-position sets, vs what two independent random sets of the same densities would give)")
for a, b in itertools.combinations(SEEDS, 2):
    exp = np.mean([(da * db) / (da + db - da * db) for da, db in zip(dens[a], dens[b])]); obs = np.mean([r["jacc"] for r in cmp_rows[("random", a, b)]])
    print(f"  seeds {a} vs {b}: observed Jaccard {obs:.3f}  | expected for independent random sets {exp:.3f}  | attention rho over ALL positions {np.mean([r['rho'] for r in cmp_rows[('random', a, b)]]):.3f}, over positions active in BOTH seeds {summ(cmp_rows[('random', a, b)], 'rho_common')}")

# ============================================================================================ D2
print("\n" + "=" * 112 + "\nD2. LOCAL CONDITIONING OF THE ATTENTION GATE (exact 160 x 160 Jacobian per protein position, float64)\n" + "=" * 112)
rng = np.random.default_rng(9)
def jac_model(model, pks):
    Wp, bp, Wa, ba = [t.detach().cpu().double() for t in (model.protein_attention_layer.weight, model.protein_attention_layer.bias, model.attention_layer.weight, model.attention_layer.bias)]
    per = []
    for pk in pks:
        pc, da, tg = pk["pc"].double(), pk["da"].double(), pk["gate"].double(); fn = pc.norm(dim=0)
        def gate(pf): return torch.sigmoid((torch.relu(da + (Wp @ pf + bp)) @ Wa.T + ba).mean(0))
        rec = {}
        for cat, pool in (("active", np.where(fn.numpy() > 0)[0]), ("inactive", np.where(fn.numpy() == 0)[0])):
            vals = []
            for j in (rng.choice(pool, size=min(NPOS, len(pool)), replace=False) if len(pool) else []):
                pf = pc[:, j].clone(); g = gate(pf); assert float((g - tg[j]).abs().max()) < 1e-4, "gate re-derivation does not match the real forward pass"
                J = torch.func.jacrev(gate)(pf); s = torch.linalg.svdvals(J).detach(); rank = int((s > 1e-10 * s[0]).sum())
                vals.append((rank, float(s[0] / s[rank - 1]), float(s[0]), float((J.T @ torch.full((160,), 1 / 160, dtype=torch.float64)).norm() * pf.norm() / g.mean())))
            rec[cat] = vals
        per.append(rec)
    return per
def agg(per, cat, idx): return np.array([np.median([v[idx] for v in r[cat]]) for r in per if r.get(cat)])
jac = {}
for s in SEEDS:
    m = load_trained(s, "random", DEV); jp = [pack(m, DEV, p["smiles"], p["seq"], want_jac=True) for p in prs[:NJAC]]; jac[("trained", s)] = jac_model(m, jp); del m
for s in UNTRAINED:
    m = untrained(s, DEV); jp = [pack(m, DEV, p["smiles"], p["seq"], want_jac=True) for p in prs[:NJAC]]; jac[("untrained", s)] = jac_model(m, jp); del m
def q(x): return f"median {np.median(x):.3g} (IQR {np.percentile(x,25):.3g} to {np.percentile(x,75):.3g}, max {np.max(x):.3g})" if len(x) else "n/a"
print(f"Gate re-derivation matches the real forward pass at every sampled position (|diff| < 1e-4, float32 features). {NJAC} proteins x up to {NPOS} positions per category.")
for kind in ("trained", "untrained"):
    for s in (SEEDS if kind == "trained" else UNTRAINED):
        per = jac[(kind, s)]; allv = [v for r in per for c in r.values() for v in c]
        print(f"  {kind:9s} {s}: rank<160 in {np.mean([v[0] < 160 for v in allv]):.0%} of Jacobians (median rank {np.median([v[0] for v in allv]):.0f}); "
              f"cond over range, ACTIVE positions: {q(agg(per,'active',1))}; INACTIVE: {q(agg(per,'inactive',1))}")
        if kind == "trained": print(f"{'':16s}readout elasticity at ACTIVE positions {q(agg(per,'active',3))}; at INACTIVE positions (features are zero): {q(agg(per,'inactive',3))}")
for name, Wname in (("attention_layer", "attention_layer.weight"), ("protein_attention_layer", "protein_attention_layer.weight")):
    ct = [float(np.linalg.cond(tr[(s, 'random')][Wname].double().numpy())) for s in SEEDS]; cu = [float(np.linalg.cond(un[s][Wname].double().numpy())) for s in UNTRAINED]
    print(f"  weight matrix {name}: cond trained {', '.join(f'{c:.0f}' for c in ct)} | untrained {', '.join(f'{c:.0f}' for c in cu)}   (orthogonal = 1)")
ca = agg(jac[("trained", 1)], "active", 1); n_ = min(len(ca), NJAC); r1 = stats.spearmanr(np.log10(ca[:n_]), rho_prot[:n_])
print(f"\nDoes local conditioning explain WHICH proteins the seeds disagree on? Spearman across {n_} proteins, seed-1 median log10 condition number at active positions vs cross-seed attention rho: {r1.statistic:+.3f} (p = {r1.pvalue:.2f})")
ct_med = float(np.median(np.concatenate([agg(jac[("trained", s)], "active", 1) for s in SEEDS]))); cu_med = float(np.median(np.concatenate([agg(jac[("untrained", s)], "active", 1) for s in UNTRAINED])))
el_med = float(np.median(np.concatenate([agg(jac[("trained", s)], "active", 3) for s in SEEDS])))
print(f"Reading: every Jacobian is rank-deficient (dead hidden channels), so the unrestricted condition number is infinite by construction. Over its range the trained gate is {ct_med/cu_med:.0f}x more anisotropic than an untrained one (median {ct_med:.3g} vs {cu_med:.3g}), "
      f"but the readout is not amplifying: its elasticity is {el_med:.3f}, i.e. a 100% relative change in a position's features moves the channel-mean gate by about {100*el_med:.1f}%.")

# ============================================================================================ D3
print("\n" + "=" * 112 + "\nD3. HOW FAR MUST THE WEIGHTS MOVE? Gaussian perturbations of trained seed 1 (random level), relative to each tensor's RMS\n" + "=" * 112)
sd0 = {k: v.detach().cpu().clone() for k, v in load_trained(1, "random", "cpu").state_dict().items()}
base = load_trained(1, "random", DEV); PP = prs[:NPERT]; b_out = [pack(base, DEV, p["smiles"], p["seq"]) for p in PP]; b_logit = np.array([o["logit"] for o in b_out]); del base
seed_rho = w_rho; seed_pred = func[("hyperattentiondti", "random")][1]
print(f"{'eps (relative)':>14} {'attention rho vs original':>26} {'top-10 overlap':>15} {'logit rank rho':>15} {'|dlogit| / sd(logit)':>21} {'share active positions kept':>28}")
curve = []; g = torch.Generator().manual_seed(5)
for eps in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
    rr, tt, lr_, dl, ka = [], [], [], [], []
    for draw in range(NDRAW):
        m = AttentionDTI(hyperparameter())
        m.load_state_dict({k: (v + eps * v.float().pow(2).mean().sqrt() * torch.randn(v.shape, generator=g)) if v.is_floating_point() else v for k, v in sd0.items()}); m = m.to(DEV).eval()
        outs = [pack(m, DEV, p["smiles"], p["seq"]) for p in PP]; del m
        rr.append(np.mean([stats.spearmanr(interior(o["w"], o["nv"]), interior(b["w"], b["nv"])).statistic for o, b in zip(outs, b_out)]))
        tt.append(np.mean([len(top(interior(o["w"], o["nv"])) & top(interior(b["w"], b["nv"]))) / K for o, b in zip(outs, b_out)]))
        lg = np.array([o["logit"] for o in outs]); lr_.append(stats.spearmanr(lg, b_logit).statistic); dl.append(np.mean(np.abs(lg - b_logit)) / b_logit.std())
        ka.append(np.mean([((o["fnorm"] > 0) & (b["fnorm"] > 0)).sum() / max((b["fnorm"] > 0).sum(), 1) for o, b in zip(outs, b_out)]))
    curve.append((eps, np.mean(rr), np.mean(tt), np.mean(lr_), np.mean(dl), np.mean(ka)))
    print(f"{eps:14.3f} {np.mean(rr):26.3f} {np.mean(tt):15.3f} {np.mean(lr_):15.3f} {np.mean(dl):21.3f} {np.mean(ka):28.3f}", flush=True)
rel = float(np.mean([r[1] for r in tr_rows]))
print(f"\nFor scale: two trained seeds differ by relative weight distance {rel:.2f}; their attention rho is {seed_rho:.3f} and their prediction Spearman {seed_pred:.3f}.")
below = [c for c in curve if c[1] <= seed_rho + 0.05]
print("Smallest tested perturbation at which the attention correlation reaches the seed-to-seed level (within 0.05): " + (f"eps = {below[0][0]}, where the logit rank correlation has fallen to {below[0][3]:.3f} (independently trained seeds keep {seed_pred:.3f})" if below else "none of the tested sizes"))
print("\nD4 (not run): the trainer hard-codes AdamW + cyclic LR (the published recipe); no checkpoint used another optimizer.")
print(f"\nelapsed {time.time()-T_START:.0f} s")
