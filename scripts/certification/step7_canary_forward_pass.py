"""PHASE 2.5 -- canary forward pass: does a LIVE forward pass through a saved checkpoint reproduce the saved result files?

Model: HyperAttentionDTI, DAVIS, seed 1 (checkpoint hash-checked against the hash recorded in the predictions metadata).
For each split level, 10 pairs are drawn at random (recorded seed) from the pairs the analysis actually scored (the first test row of each
scored protein: only those pairs have a saved per-protein precision). For each pair a live forward pass on the GPU (Apple MPS; there is no CUDA
on this machine) and on the CPU produces
   * the log-odds prediction         -> compared with the `logit` column of results/accuracy_v2/predictions/davis_<level>_hyperattentiondti_seed1.csv.gz
   * the raw attention tensor (captured by a forward hook on the real forward pass, not by re-implementing it)
        -> sigmoid(mean over drug axis) -> mean over 160 channels -> one weight per convolution position -> centre residue (offset 10)
        -> top-10 residues -> precision@10 against UniProt sites and against the KLIFS pocket, and enrichment over exact chance
   * those precisions are compared with the per-protein values saved in
        results/analysis_davis_policyA/ladder_hyperattentiondti_davis_seed1.json (UniProt) and results/analysis_davis_policyA_klifs/... (KLIFS)
Independent of: src/evaluation/{attention_projection,collect,precision_at_k,run_ladder}.py. Uses: the vendored model class and tokenisers (they define the input).

    python scripts/certification/step7_canary_forward_pass.py [--device mps|cpu] [--n 10]
"""
import argparse, csv, gzip, hashlib, json, os, sys, time
import numpy as np, torch

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=20260929)
args = ap.parse_args()
CKPT_DIR = os.path.expanduser("~/ColdSite-results/davis_binary")
sys.path.insert(0, os.path.join(ROOT, "baselines", "HpyerAttentionDTI"))
from hyperparameter import hyperparameter            # vendored
from model import AttentionDTI                        # vendored
from dataset import CHARISOSMISET, CHARPROTSET, label_sequence, label_smiles   # vendored tokenisers
TOL = 1e-5; K = 10; OFFSET = 4 - 1 + 8 - 1 + 12 - 1; CENTRE = OFFSET // 2      # kernels [4, 8, 12] -> 21, 10
dev = args.device
print(f"torch {torch.__version__} | requested device: {dev} | mps available: {torch.backends.mps.is_available()} | cuda available: {torch.cuda.is_available()}")
if dev == "mps" and not torch.backends.mps.is_available(): raise SystemExit("MPS not available")
seqs = json.load(open("src/data/baselines/deepdta/data/davis/proteins.txt"))
uni = json.load(open("data/davis_ground_truth_sites.json")); klifs = json.load(open("data/davis_klifs_pocket_sites.json"))
def zero_indexed_sites(entry, L):
    pos = set()
    if isinstance(entry, dict): entry = entry.get("residues", entry.get("sites", []))
    for e in entry:
        if isinstance(e, dict): pos.update(range(int(e["start"]) - 1, int(e["end"])))     # 1-based inclusive -> 0-based
        else: pos.add(int(e) - 1)
    return {p for p in pos if 0 <= p < L}
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def load_model(path, device):
    m = AttentionDTI(hyperparameter()); state = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(state.get("model_state", state)); return m.to(device).eval()
def live(model, device, smiles, sequence):
    drug = torch.from_numpy(label_smiles(smiles, CHARISOSMISET, 100)).unsqueeze(0).to(device)
    prot_np = label_sequence(sequence, CHARPROTSET, 1000); prot = torch.from_numpy(prot_np).unsqueeze(0).to(device)
    cap = []; h = model.attention_layer.register_forward_hook(lambda mod, inp, out: cap.append(out.detach()))
    with torch.no_grad(): logits = model(drug, prot).reshape(-1).float().cpu()
    h.remove()
    assert len(cap) == 1 and cap[0].shape[0] == 1 and cap[0].shape[-1] == 160, cap[0].shape
    A = cap[0]                                                              # (1, drug_pos, prot_pos, 160): the tensor the real forward used
    per_pos = torch.sigmoid(A.mean(dim=1)).mean(dim=-1)[0].cpu().double().numpy()   # sigmoid(mean over drug axis), then mean over channels -> (979,)
    L = min(len(sequence), 1000); nonzero = int((prot_np != 0).sum())
    n_valid = L - OFFSET; w = np.zeros(L); w[np.arange(n_valid) + CENTRE] = per_pos[:n_valid]
    return dict(logit=float(logits[1] - logits[0]), attn=w, L=L, nonzero=nonzero, tensor_shape=tuple(A.shape))
def top10_precision(w, sites):
    """precision@10 with the tie ambiguity made explicit: returns (hits_min, hits_max, hits_strict_order, tie_at_boundary)."""
    order = np.lexsort((np.arange(w.size), -w)); thr = w[order[K - 1]]
    sure = [i for i in range(w.size) if w[i] > thr]; tied = [i for i in range(w.size) if w[i] == thr]; need = K - len(sure)
    sure_hits = sum(i in sites for i in sure); tied_sites = sum(i in sites for i in tied)
    lo = sure_hits + max(0, need - (len(tied) - tied_sites)); hi = sure_hits + min(need, tied_sites)
    strict = sum(int(i) in sites for i in order[:K]); return lo, hi, strict, len(tied) > need, [int(i) for i in order[:K]]
def read(p): return list(csv.DictReader(open(p, encoding="utf-8")))

grand = dict(pairs=0, logit_ok=0, prec_ok=0, prec_total=0, klifs_ok=0, klifs_total=0, max_logit=0.0, max_logit_cpu=0.0, ties=0)
for level in ("random", "cold_drug", "cold_target", "cold_pair"):
    primary = level == "random"
    print("\n" + "=" * 118 + f"\nLEVEL: {level}" + ("   (PRIMARY, printed in full)" if primary else ""))
    ck = os.path.join(CKPT_DIR, f"coldsite_dti_davis_{level}_binary_seed1_hyperattentiondti.pt")
    meta = json.load(open(f"results/accuracy_v2/predictions/davis_{level}_hyperattentiondti_seed1.meta.json"))
    digest = sha256(ck); print(f"checkpoint {os.path.basename(ck)}\n  sha256 live   {digest}\n  sha256 saved  {meta['checkpoint_sha256']}   -> {'IDENTICAL' if digest == meta['checkpoint_sha256'] else 'DIFFERENT !!'}")
    if digest != meta["checkpoint_sha256"]: raise SystemExit("checkpoint is not the one the saved files were made from")
    test = read(f"data/splits/davis/{level}/test.csv"); preds = list(csv.DictReader(gzip.open(f"results/accuracy_v2/predictions/davis_{level}_hyperattentiondti_seed1.csv.gz", "rt")))
    ladder_u = json.load(open("results/analysis_davis_policyA/ladder_hyperattentiondti_davis_seed1.json"))[level]
    ladder_k = json.load(open("results/analysis_davis_policyA_klifs/ladder_hyperattentiondti_davis_seed1.json"))[level]
    ids = ladder_u["ids"]; saved_u = dict(zip(ids, ladder_u["by_k"]["10"]["per_protein"])); saved_k = dict(zip(ladder_k["ids"], ladder_k["by_k"]["10"]["per_protein"]))
    first_row = {}
    for i, r in enumerate(test): first_row.setdefault(r["Target_ID"], i)
    pool = [first_row[t] for t in ids if t in first_row]; assert len(pool) == len(ids)
    rng = np.random.default_rng(args.seed + ["random", "cold_drug", "cold_target", "cold_pair"].index(level))
    pick = sorted(rng.choice(len(ids), size=args.n, replace=False).tolist())
    model_dev = load_model(ck, dev); model_cpu = load_model(ck, "cpu")
    rows = []
    for j in pick:
        ridx = pool[j]; r = test[ridx]; tid = r["Target_ID"]; seq = r["Target"]; assert seqs[tid] == seq
        p = preds[ridx]; assert (p["Drug_ID"], p["Target_ID"]) == (r["Drug_ID"], r["Target_ID"]), "prediction row is not this pair"
        a = live(model_dev, dev, r["Drug"], seq); c = live(model_cpu, "cpu", r["Drug"], seq)
        sites_u = zero_indexed_sites(uni[tid], a["L"]); lo, hi, strict, tie, top = top10_precision(a["attn"], sites_u)
        rows.append(dict(ridx=ridx, tid=tid, drug=r["Drug_ID"], saved_logit=float(p["logit"]), live=a, cpu=c, sites_u=sites_u, lo=lo, hi=hi, strict=strict, tie=tie, top=top, saved_u=saved_u[tid],
                         k=(top10_precision(a["attn"], zero_indexed_sites(klifs[tid], a["L"])) if tid in saved_k else None), saved_k=saved_k.get(tid), sites_k=zero_indexed_sites(klifs[tid], a["L"]) if tid in saved_k else None))
    if primary:
        print(f"\n  attention tensor captured from the real forward pass: shape {rows[0]['live']['tensor_shape']} (batch, drug positions, protein positions, channels)")
        print(f"\n  {'row':>5} {'target':<14} {'drug':>9} {'saved logit':>13} {'live GPU logit':>15} {'|GPU-saved|':>11} {'live CPU logit':>15} {'|CPU-saved|':>11}")
        for x in rows:
            print(f"  {x['ridx']:>5} {x['tid']:<14} {x['drug']:>9} {x['saved_logit']:>13.8f} {x['live']['logit']:>15.8f} {abs(x['live']['logit']-x['saved_logit']):>11.2e} {x['cpu']['logit']:>15.8f} {abs(x['cpu']['logit']-x['saved_logit']):>11.2e}")
        print(f"\n  {'target':<14} {'L':>4} {'nz':>4} {'#UniProt':>8} {'top-10 residues (0-indexed, live GPU)':<48} {'P@10 live':>9} {'P@10 saved':>10} {'tie?':>5} {'P@10 pocket live':>17} {'saved':>7}")
        for x in rows:
            kp = "n/a" if x["k"] is None else (f"{x['k'][2]/K:.1f}" if x["k"][0] == x["k"][1] else f"{x['k'][0]/K:.1f}-{x['k'][1]/K:.1f}")
            up = f"{x['strict']/K:.1f}" if x["lo"] == x["hi"] else f"{x['lo']/K:.1f}-{x['hi']/K:.1f}"
            print(f"  {x['tid']:<14} {x['live']['L']:>4} {x['live']['nonzero']:>4} {len(x['sites_u']):>8} {str(x['top']):<48} {up:>9} {x['saved_u']:>10.1f} {'TIE' if x['tie'] else 'no':>5} {kp:>17} {('n/a' if x['saved_k'] is None else format(x['saved_k'], '.1f')):>7}")
    # ---- level summary
    d_gpu = np.array([abs(x["live"]["logit"] - x["saved_logit"]) for x in rows]); d_cpu = np.array([abs(x["cpu"]["logit"] - x["saved_logit"]) for x in rows])
    d_dev = np.array([abs(x["live"]["logit"] - x["cpu"]["logit"]) for x in rows])
    okp = sum(x["lo"] / K - 1e-12 <= x["saved_u"] <= x["hi"] / K + 1e-12 for x in rows); exact_u = sum((x["lo"] == x["hi"]) and abs(x["strict"] / K - x["saved_u"]) < 1e-12 for x in rows)
    kk = [x for x in rows if x["k"] is not None]; okk = sum(x["k"][0] / K - 1e-12 <= x["saved_k"] <= x["k"][1] / K + 1e-12 for x in kk)
    same_top = sum(1 for x in rows if x["live"]["attn"].shape == x["cpu"]["attn"].shape and set(np.argsort(-x["live"]["attn"])[:K]) == set(np.argsort(-x["cpu"]["attn"])[:K]))
    print(f"\n  SUMMARY {level}: pairs {len(rows)} | logit: max |GPU-saved| {d_gpu.max():.2e}, max |CPU-saved| {d_cpu.max():.2e}, max |GPU-CPU| {d_dev.max():.2e}  (tolerance {TOL:g}) -> {'PASS' if d_gpu.max() <= TOL else 'FAIL'}")
    print(f"           UniProt P@10: saved value inside the live GPU range for {okp}/{len(rows)} pairs ({exact_u} exact with no tie); ties at the top-10 boundary: {sum(x['tie'] for x in rows)}")
    print(f"           KLIFS   P@10: saved value inside the live GPU range for {okk}/{len(kk)} pairs scored in the KLIFS ladder")
    print(f"           same top-10 residue set on GPU and CPU: {same_top}/{len(rows)}")
    # how informative is the agreement, and what breaks the GPU/CPU top-10 identity?
    nz_u = sum(x["saved_u"] > 0 for x in rows); nz_k = sum((x["saved_k"] or 0) > 0 for x in kk)
    print(f"           informativeness: pairs with saved UniProt P@10 > 0: {nz_u}/{len(rows)}; pocket P@10 > 0: {nz_k}/{len(kk)}; distinct pocket values: {sorted({x['saved_k'] for x in kk})}")
    for x in rows:
        a_, c_ = x["live"]["attn"], x["cpu"]["attn"]; sa, sc = set(np.argsort(-a_)[:K]), set(np.argsort(-c_)[:K])
        if sa != sc:
            srt = np.sort(a_)[::-1]; diff = sorted(sa ^ sc)
            print(f"           GPU/CPU top-10 differ for {x['tid']}: positions {diff} | GPU weights at those positions {[round(float(a_[i]), 7) for i in diff]} vs CPU {[round(float(c_[i]), 7) for i in diff]} | GPU rank-10 minus rank-11 gap {srt[K-1] - srt[K]:.2e} | max |GPU-CPU weight| {np.abs(a_ - c_).max():.2e} | precision changes: {sum(int(i) in x['sites_u'] for i in sa) != sum(int(i) in x['sites_u'] for i in sc)}")
    # enrichment for the 10 pairs
    chance_u = np.mean([len(x["sites_u"]) / x["live"]["L"] for x in rows]); lo_p = np.mean([x["lo"] for x in rows]) / K; hi_p = np.mean([x["hi"] for x in rows]) / K; sv = np.mean([x["saved_u"] for x in rows])
    print(f"           UniProt enrichment over exact chance for these {len(rows)} pairs: live {lo_p / chance_u:.3f}" + (f"-{hi_p / chance_u:.3f}" if hi_p != lo_p else "") + f" | from saved per-protein values {sv / chance_u:.3f} | mean chance {chance_u:.4f}")
    if kk:
        chance_k = np.mean([len(x["sites_k"]) / x["live"]["L"] for x in kk]); lk = np.mean([x["k"][0] for x in kk]) / K; hk = np.mean([x["k"][1] for x in kk]) / K; sk = np.mean([x["saved_k"] for x in kk])
        print(f"           pocket enrichment over exact chance for the {len(kk)} scored pairs: live {lk / chance_k:.3f}" + (f"-{hk / chance_k:.3f}" if hk != lk else "") + f" | from saved {sk / chance_k:.3f} | mean chance {chance_k:.4f}")
    grand["pairs"] += len(rows); grand["logit_ok"] += int((d_gpu <= TOL).sum()); grand["prec_ok"] += okp; grand["prec_total"] += len(rows); grand["klifs_ok"] += okk; grand["klifs_total"] += len(kk)
    grand["max_logit"] = max(grand["max_logit"], d_gpu.max()); grand["max_logit_cpu"] = max(grand["max_logit_cpu"], d_cpu.max()); grand["ties"] += sum(x["tie"] for x in rows)
    del model_dev, model_cpu
print("\n" + "=" * 118)
print(f"OVERALL ({args.n} random scored pairs at each of 4 levels, HyperAttentionDTI seed 1, device {dev}): pairs {grand['pairs']}")
print(f"  logit within {TOL:g} of the saved CSV: {grand['logit_ok']}/{grand['pairs']}   (max |live GPU - saved| = {grand['max_logit']:.2e}; max |live CPU - saved| = {grand['max_logit_cpu']:.2e})")
print(f"  UniProt precision@10 consistent with the saved per-protein value: {grand['prec_ok']}/{grand['prec_total']};  KLIFS: {grand['klifs_ok']}/{grand['klifs_total']};  boundary ties: {grand['ties']}")
