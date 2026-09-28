import json, glob, os, re, numpy as np
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
MAXLEN = 1000
def load_seqs(ds):
    p = f"src/data/baselines/deepdta/data/{ds}/proteins.txt"
    return json.load(open(p))
def load_sites(ds, kind="uniprot"):
    f = f"data/{ds}_ground_truth_sites.json" if kind == "uniprot" else f"data/{ds}_klifs_pocket_sites.json"
    return json.load(open(f))
def site_positions(entry):
    """entry: list of {start,end,...} (UniProt) or list/dict of residues (KLIFS). 1-based inclusive."""
    pos = set()
    if isinstance(entry, dict): entry = entry.get("residues", entry.get("sites", []))
    for e in entry:
        if isinstance(e, dict): pos.update(range(int(e["start"]), int(e["end"]) + 1))
        else: pos.add(int(e))
    return pos
def LS(ds, ids, kind="uniprot", seqs=None, sites=None):
    seqs = seqs or load_seqs(ds); sites = sites or load_sites(ds, kind)
    out = []
    for i in ids:
        L = min(len(seqs[i]), MAXLEN)
        S = len([p for p in site_positions(sites[i]) if 1 <= p <= L])
        out.append((L, S))
    return out
MODELS_DAVIS = {"XAttn-Ref": "ladder_davis_seed{s}.json", "HyperAttentionDTI": "ladder_hyperattentiondti_davis_seed{s}.json",
                "MolTrans": "ladder_moltrans_davis_seed{s}.json", "DrugBAN": "ladder_drugban_davis_seed{s}.json"}
MODELS_KIBA = {"XAttn-Ref": "ladder_kiba_seed{s}.json", "HyperAttentionDTI": "ladder_hyperattentiondti_kiba_seed{s}.json",
               "MolTrans": "ladder_moltrans_kiba_seed{s}.json"}
DIRS = {"davis": "results/analysis_davis_policyA", "kiba": "results/analysis_kiba_policyA"}
def ladder(ds, model, seed):
    tbl = MODELS_DAVIS if ds == "davis" else MODELS_KIBA
    return json.load(open(f"{DIRS[ds]}/{tbl[model].format(s=seed)}"))
def cells(ds):
    tbl = MODELS_DAVIS if ds == "davis" else MODELS_KIBA
    for m in tbl:
        d1 = ladder(ds, m, 1)
        for level in d1:
            if isinstance(d1[level], dict) and "by_k" in d1[level]:
                yield m, level
