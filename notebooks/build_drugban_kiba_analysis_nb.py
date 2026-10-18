"""Build notebooks/kaggle_drugban_kiba_analysis.ipynb: the declared family E2 (docs/PROTOCOL_AMENDMENT_v2.md section 3): DrugBAN on KIBA, {random, cold-drug} x seeds 1-3, scored
against UniProt residues, Holm over its own two cells (the uniform control is not re-counted, it is already in P2).

    python notebooks/build_drugban_kiba_analysis_nb.py [--commit <40-hex>]       # default commit: the one the Wave A notebooks pin

Why a notebook: DrugBAN needs DGL, which the analysis Mac does not have. The notebook reuses the harness notebook's own clone / DGL / data cells (so it is the environment ACC3 trained in:
torch 2.6.0 + DGL), verifies the six checkpoints with the harness's marker check, runs `run_all --extension --steps ladder,faithfulness` and computes the E2 statistics.
The statistics cell (STATS_CODE) is tested against known KIBA numbers by tests/test_drugban_kiba_notebook.py.
"""
import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_COMMIT = "5d55c63d7e4991a714fa8c2608e6c5242e751f4c"          # the commit the Wave A notebooks pin (it holds both verdicts)

STATS_CODE = r'''
import json, os, glob, numpy as np
from scipy import stats

K, ALPHA, MAXLEN = 10, 0.05, 1000


def load_seqs(ds):
    return json.load(open(f'src/data/baselines/deepdta/data/{ds}/proteins.txt'))


def load_sites(ds):
    return json.load(open(f'data/{ds}_ground_truth_sites.json'))


def site_positions(entry):
    pos = set()
    if isinstance(entry, dict):
        entry = entry.get('residues', entry.get('sites', []))
    for e in entry:
        if isinstance(e, dict):
            pos.update(range(int(e['start']), int(e['end']) + 1))
        else:
            pos.add(int(e))
    return pos


def LS(ids, seqs, sites):
    out = []
    for i in ids:
        L = min(len(seqs[i]), MAXLEN)
        out.append((L, len([p for p in site_positions(sites[i]) if 1 <= p <= L])))
    return out


def exact_p(ls, total, k=K):
    """Hits of the k most-attended residues are, under the null, hypergeometric and independent per protein: the exact upper tail of their convolution."""
    pmf = np.array([1.0])
    for L, S in ls:
        pmf = np.convolve(pmf, stats.hypergeom.pmf(np.arange(k + 1), L, S, k))
    return float(pmf[total:].sum()) if total < len(pmf) else 0.0


def holm(pvals, alpha=ALPHA):
    m = len(pvals)
    order = np.argsort(pvals)
    adj, rej, run = [0.0] * m, [False] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
        rej[i] = (m - rank) * pvals[i] <= alpha and (rank == 0 or rej[order[rank - 1]])
    return adj, rej


def e2_report(ladder_dir, pattern, dataset='kiba', seeds=(1, 2, 3), levels=('random', 'cold_drug'), label='DrugBAN', out_dir=None):
    """pattern e.g. 'ladder_drugban_kiba_seed{s}.json'. Per seed: precision@10, exact chance, exact p; per cell: disagreement and spread; Holm over the cells' median p."""
    seqs, sites = load_seqs(dataset), load_sites(dataset)
    rows = []
    for level in levels:
        per = []
        for s in seeds:
            entry = json.load(open(os.path.join(ladder_dir, pattern.format(s=s))))[level]
            cell = entry['by_k'][str(K)]
            ls = LS(entry['ids'], seqs, sites)
            pp = np.array(cell['per_protein'])
            keep = [i for i, (L, S) in enumerate(ls) if S > 0 and L >= K]       # the proteins a precision@k test can use
            assert len(pp) == len(keep), f'per-protein length mismatch {level} seed {s}: {len(pp)} vs {len(keep)}'
            ls = [ls[i] for i in keep]
            hits = pp * K
            assert np.allclose(hits, np.round(hits), atol=1e-9), 'non-integer hits'
            per.append({'prec': float(pp.mean()), 'chance': float(sum(S / L for L, S in ls) / len(ls)),
                        'p': exact_p(ls, int(round(hits.sum()))), 'n': len(pp)})
        rows.append((level, per))
    lines = [f'# E2: {label} on {dataset.upper()}, seeds {list(seeds)}, UniProt residues (exact per-seed p; Holm over the cells)', '']
    for level, per in rows:
        marks = ''.join('*' if x['p'] < ALPHA else '.' for x in per)
        sig = sum(x['p'] < ALPHA for x in per)
        prec = [x['prec'] for x in per]
        lines.append(f"{label:10s}{level:11s} P@10 " + ' '.join(f"{x:.3f}" for x in prec) + f" | chance {per[0]['chance']:.4f} | exact p " +
                     ' '.join(f"{x['p']:.4f}" for x in per) + f' | marks {marks} | n proteins {per[0]["n"]}' +
                     ('  <-- seeds DISAGREE' if 0 < sig < len(per) else '') +
                     ('  spread exceeds signal' if (max(prec) - min(prec)) > abs(float(np.mean(prec)) - per[0]['chance']) else ''))
    med = [float(np.median([x['p'] for x in per])) for _l, per in rows]
    adj, rej = holm(med)
    lines += ['', f'Holm over the {len(rows)} cells of the family, on the median of the seeds\' exact p:']
    for (level, _per), p, a, r in zip(rows, med, adj, rej):
        lines.append(f'  {label} {level:11s} median p {p:.6g}   Holm-adjusted {a:.4f}   {"REJECT" if r else "keep"}')
    text = '\n'.join(lines) + '\n'
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, 'E2_report.md'), 'w').write(text)
        json.dump({'cells': [{'level': l, 'seeds': per} for l, per in rows], 'median_p': med, 'holm_adjusted': adj, 'rejected': rej},
                  open(os.path.join(out_dir, 'E2_summary.json'), 'w'), indent=1)
    return text
'''


def _harness_cells(commit: str) -> dict:
    sys.path.insert(0, ROOT)
    spec = importlib.util.spec_from_file_location("bhn", os.path.join(ROOT, "notebooks", "build_harness_notebook.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb = mod.build("ACC3", "config/waves.json", commit, False)
    src = ["".join(c["source"]) for c in nb["cells"]]
    pick = {}
    for i, text in enumerate(src):
        if nb["cells"][i]["cell_type"] != "code":
            continue
        if "git', 'clone'" in text:
            pick["clone"] = text
        elif "TORCH_PIN" in text:
            pick["dgl"] = text
        elif "BASE = 'https://raw.githubusercontent.com/hkmztrk" in text:
            pick["data"] = text
    assert set(pick) == {"clone", "dgl", "data"}, sorted(pick)
    return pick


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def build(commit: str) -> dict:
    h = _harness_cells(commit)
    cells = [
        md(f"""
# DrugBAN on KIBA -- the attention analysis (declared family E2)

Scores the six DrugBAN KIBA checkpoints that ACC3 trained (random and cold-drug x seeds 1 to 3) against UniProt's annotated binding residues:
precision@10 per seed against its exact chance level, an exact per-seed p-value, whether the seeds agree, and Holm correction over the family's two cells
(`docs/PROTOCOL_AMENDMENT_v2.md` section 3, row E2; the uniform control is not re-counted). Also DrugBAN's faithfulness (masking) for each seed.

Generated by `notebooks/build_drugban_kiba_analysis_nb.py`; do not hand-edit.

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x1 or x2** (a GPU is only for speed; the analysis also runs on CPU, more slowly) |
| Internet | **On** (clone, DGL install, raw data) |
| Input | the private dataset made from `drugban_kiba_for_kaggle.zip` (the six checkpoints and their markers, 25 MB): **Add Input** before running |
| How to run | **Run All** is enough; use **Save Version -> Save & Run All (Commit)** to keep the outputs |

## How long this takes
Setup and the DGL install about 5 minutes; the ladder and faithfulness runs about 15 to 40 minutes on a T4 (an estimate, not measured: DrugBAN's forward pass is light); the statistics a few seconds.
It spends well under 1 GPU-hour.

## What you will see, and what is saved
Time stamps ("[time] ... since the session began") after each section. Each finished job writes its own file into `/kaggle/working/analysis_kiba_E2`, and a re-run skips jobs whose file exists,
so a stopped session loses at most the job in progress. The last cell zips the folder: download it and give it to Claude.
"""),
        md("## 1. Settings"),
        code(f"""
COMMIT = {commit!r}        # the exact commit this notebook was generated for (the commit the Wave A notebooks pin)
REPO = 'https://github.com/Mahim56207/ColdSite-DTI_New.git'
SEEDS = '1,2,3'
RUN_FAITHFULNESS = True   # False: only the precision@10 ladder (faster)
"""),
        md("## 2. Session start and the clock"),
        code("""
import os, subprocess, sys, time
START = time.time()
def since_start():
    m = int((time.time() - START) // 60)
    return f'[time] {m // 60}h{m % 60:02d}m since the session began'

print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout or 'nvidia-smi: none (CPU run)')
"""),
        md("## 3. Clone the repo at the pinned commit"),
        code(h["clone"]),
        md("## 3b. DGL for DrugBAN (the same pinned environment ACC3 trained in: torch 2.6.0)"),
        code(h["dgl"]),
        md("## 4. Raw data and splits, then check the KIBA splits against the project's manifest"),
        code(h["data"]),
        code("""
import hashlib, json
man = json.load(open('data/splits/MANIFEST.json'))['files']
need = [p for p in man if '/kiba/random/' in p or '/kiba/cold_drug/' in p] + ['data/kiba_ground_truth_sites.json']
bad = [p for p in need if hashlib.sha256(open(p, 'rb').read()).hexdigest() != man[p]['sha256']]
print(f'KIBA random and cold_drug splits: {len(need)} files checked against data/splits/MANIFEST.json, mismatches: {len(bad)}')
assert need and not bad, bad
print(since_start())
"""),
        md("## 5. Collect the six checkpoints and verify every one with the harness's own marker check"),
        code("""
import glob, json, shutil
from src.cloud import markers
from src.cloud.config import Cell
RESULTS = '/kaggle/working/results'
folders = {os.path.dirname(p) for p in glob.glob('/kaggle/input/**/*_drugban.pt', recursive=True)}
assert len(folders) == 1, f'expected exactly one input folder holding the DrugBAN checkpoints, found {sorted(folders)}: attach the dataset made from drugban_kiba_for_kaggle.zip'
src_dir = folders.pop()
os.makedirs(f'{RESULTS}/kiba_binary', exist_ok=True)
for f in glob.glob(f'{src_dir}/*'):
    shutil.copy(f, f'{RESULTS}/kiba_binary/')
cells = [Cell('kiba', 'drugban', lv, s) for lv in ('random', 'cold_drug') for s in (1, 2, 3)]
for c in cells:
    state, why = markers.state(RESULTS, c)
    print(f'{state:10s} {c.id}' + ('' if state == 'complete' else f'   {why}'))
assert all(markers.state(RESULTS, c)[0] == 'complete' for c in cells), 'a checkpoint failed the marker check: do not analyse it'
print(since_start())
"""),
        md("## 6. The plan (changes nothing)"),
        code("""
OUT = '/kaggle/working/analysis_kiba_E2'
steps = 'ladder,faithfulness' if RUN_FAITHFULNESS else 'ladder'
base = [sys.executable, '-u', '-m', 'src.evaluation.run_all', '--dataset', 'kiba', '--models', 'drugban', '--seeds', SEEDS,
        '--checkpoint-dir', f'{RESULTS}/kiba_binary', '--extension', '--out-dir', OUT, '--steps', steps]
subprocess.run(base + ['--dry-run'])
"""),
        md("## 7. Run it (each job writes its own file; a re-run skips finished ones)"),
        code("""
if os.path.exists(OUT) and os.listdir(OUT):
    print('NOTE: the output folder is not empty; --extension needs an empty one. Delete it only if you want a clean start.')
code_ = subprocess.run(base).returncode
print('run_all exit code', code_, '(0 = every job finished)')
print(since_start())
"""),
        md("## 8. The E2 statistics"),
        code(STATS_CODE),
        code("""
print(e2_report(OUT, 'ladder_drugban_kiba_seed{s}.json', out_dir=OUT))
print(since_start())
"""),
        md("## 9. What to take with you"),
        code("""
ZIP = 'drugban_kiba_E2_results.zip'
subprocess.run(f'cd /kaggle/working && rm -f {ZIP} && zip -qr {ZIP} analysis_kiba_E2', shell=True)
print('/kaggle/working/' + ZIP + ' -- download it from the Output panel and give it to Claude')
print(since_start())
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--out", default="notebooks/kaggle_drugban_kiba_analysis.ipynb")
    args = ap.parse_args(argv)
    nb = build(args.commit)
    with open(os.path.join(ROOT, args.out), "w") as handle:
        json.dump(nb, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
