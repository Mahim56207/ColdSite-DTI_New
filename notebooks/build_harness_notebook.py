"""Generate a Kaggle notebook for the cloud harness (`src/cloud`) from a wave manifest.

    python notebooks/build_harness_notebook.py --waves config/canary_wave.json --account CANARY \
        --out notebooks/kaggle_canary.ipynb --canary
    python notebooks/build_harness_notebook.py --waves config/waves.json --account ACC1 \
        --out notebooks/kaggle_wave_a_acc1.ipynb

One notebook per account: it clones the repo at a PINNED commit, rebuilds the splits and the raw
DAVIS/KIBA files the way every earlier notebook did, then hands everything to
`python -m src.cloud.runner`, which does the pre-flight, restore, one-process-per-GPU launch,
self-stop and complete-markers. The notebook holds no credentials and no training logic.

`DRY_RUN = True` is the shipped default: running it as-is does the pre-flight and prints the plan,
then stops before any GPU work. Flipping it to False is the user's decision (it spends quota).

Edit this builder, rebuild, commit both; never hand-edit the .ipynb.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPO = "https://github.com/Mahim56207/ColdSite-DTI_New.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def expected_time_md(account: str, waves: dict, canary: bool) -> str:
    """Approximate wall-clock on Kaggle 2 x T4, from config/wave_budget.json (the canary: from the two recorded attempts)."""
    if canary:
        return """## How long this takes (Kaggle 2 x T4)

| part | approx. wall-clock | where the number comes from |
|---|---|---|
| setup: clone, data, splits | 5-10 min | the Kaggle logs of earlier runs |
| the canary cell (one GPU; the other waits) | **about 1.0 to 1.3 h** (27 to 38 epochs) | the two recorded attempts: 3,529 s and 4,727 s (`docs/canary_attempts/README.md`) |
| the epoch-1 gate (3 one-epoch runs on the two GPUs) | about 5 to 10 min | one recorded gate run |
| **whole notebook** | **about 1.3 to 1.7 h** | |

It spends about 1 to 1.5 GPU-hours of the account's weekly quota (the quota line in the monitor shows it live)."""
    budget = json.load(open(os.path.join(ROOT, "config", "wave_budget.json")))
    pa = budget["per_account"][account]
    per_commit = round(budget["gpu_hours_per_commit_per_account"] / 2, 2)       # hours of training one GPU gets in one commit
    by_gpu = pa["gpu_hours_mean_by_gpu"]
    worst_gpu = max(by_gpu.values())
    scale = pa["gpu_hours_high"] / pa["gpu_hours_mean"]
    commits = math.ceil(worst_gpu * scale / per_commit)
    notes = ("These are ESTIMATES, not measurements, for DrugBAN on KIBA: the seconds per epoch were measured once "
             "(`results/drugban_smoke_run_kiba_t4.md`); the number of epochs is taken from DrugBAN's DAVIS runs plus its patience." if account == "ACC3"
             else "Per-epoch times are the measured ones in `results/speed_test_kiba_t4.md` and the Kaggle logs; epochs are the committed seeds' best epoch plus patience.")
    return f"""## How long this takes (Kaggle 2 x T4, both GPUs working in parallel)

| | estimate | range (the three committed seeds' epoch counts) |
|---|---|---|
| training time per GPU | **about {worst_gpu:.1f} h** | {pa['gpu_hours_low'] / 2:.1f} to {pa['gpu_hours_high'] / 2:.1f} h |
| GPU-hours of weekly quota (a 2-GPU hour counts twice) | **{pa['gpu_hours_mean']:.1f}** | {pa['gpu_hours_low']:.1f} to {pa['gpu_hours_high']:.1f} |
| commits (sessions) needed | **{commits}** | each session self-stops after {per_commit:.2f} h of training per GPU |
| plus setup per session | 10 to 15 min | clone, data, splits{" and the DGL install" if account == "ACC3" else ""} |

{"The last commit is short: it only finishes what the earlier one left." if commits > 1 else "One session should be enough; if it is cut short, nothing is lost: the unfinished cell resumes in the next session."}
{notes}"""


def live_help_md() -> str:
    return """## What you will see while it runs, and what is saved

**Time checker.** Every 5 minutes the runner prints a block like

```
[monitor] elapsed 2h14m | self-stop in 8h01m (session limit 11 h) | cells complete 2/12 | quota used 4.6 of 25.5 gpu-hours
  gpu0: davis_random_hyperattentiondti_seed4 epoch 7, 361.2 s/epoch | this cell finishes within 3h20m (upper bound; ...) | 5 cell(s) left in its queue
  gpu1: ...
```

`self-stop in` is the time left before the harness stops cleanly (the session limit minus a 45-minute margin); a time estimate for a cell is an
UPPER bound because early stopping can end it sooner. Sections 2 to 4 also print the time since the session began.

**Progress saver.** Nothing is saved only at the end:
* every epoch each cell writes its resume file (atomically) and a status file, so a stop at any moment loses at most one epoch;
* a cell is marked complete only when its checkpoint, results and predictions hash are written and agree;
* every 5 minutes `results/progress.json` is rewritten (time used, cells complete, what each GPU is on);
* section 6 zips `results/` (including `progress.json` and the quota ledger) into the file you download.
If the session dies, download whatever Kaggle kept, add it as a dataset and run again with `RESTORE_FROM = '/kaggle/input'`."""


def head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()


def build(account: str, waves_path: str, commit: str, canary: bool) -> dict:
    from src.cloud.config import load_waves
    waves = load_waves(os.path.join(ROOT, waves_path))
    if account not in waves["accounts"]:
        raise SystemExit(f"{account!r} is not in {waves_path}: {sorted(waves['accounts'])}")
    cells = [c for q in waves["accounts"][account].values() for c in q]
    models = sorted({c.model for c in cells})
    needs_dgl = "drugban" in models
    title = ("CANARY -- does the harness reproduce a committed cell?" if canary
             else f"Wave {waves['wave']} -- account {account}")
    plan_lines = "\n".join(f"| {gpu} | {c.id} |" for gpu, q in waves["accounts"][account].items()
                           for c in q) or "| - | - |"

    out = [md(f"""
# ColdSite-DTI cloud harness -- {title}

Generated by `notebooks/build_harness_notebook.py` from `{waves_path}`; do not hand-edit.

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** (the pre-flight refuses anything else) |
| Internet | **On** |
| Environment | **Pin to original** |
| How to run | **Save Version -> Save & Run All (Commit)** |
| First run | leave `DRY_RUN = True`: it does the pre-flight and prints the plan, then stops |

## What this account trains ({len(cells)} cell(s))

| GPU queue | cell |
|---|---|
{plan_lines}

Every cell of the plan appears once in `{waves_path}` across all accounts. Nothing here trains a
cell another account owns.

## The loop (each commit starts empty)

Kaggle output is per version. After a commit ends, download `results/` (the runner zips it, section 6),
add it as a private dataset, set `RESTORE_FROM = '/kaggle/input'` and run again. The runner copies back only
this account's cells, never over a file already there, verifies each finished cell's marker, and continues
partial cells from their last finished epoch. A commit self-stops before the session limit
(`config/harness.json`).
"""),
        md(expected_time_md(account, waves, canary)),
        md(live_help_md()),
        md("## 1. Settings"),
        code(f"""
ACCOUNT = {account!r}
WAVES = {waves_path!r}
COMMIT = {commit!r}        # the exact commit this notebook was generated for
REPO = {REPO!r}
DRY_RUN = True            # True: pre-flight + plan, then stop. False: train (spends GPU quota).
RESTORE_FROM = None       # None on the first commit; '/kaggle/input' after attaching the previous output
"""),
        md("## 2. Session start and GPUs"),
        code("""
import os, subprocess, sys, time
START = time.time()          # the runner's self-stop is measured from here
def since_start():
    m = int((time.time() - START) // 60)
    return f'[time] {m // 60}h{m % 60:02d}m since the session began'

print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout or 'nvidia-smi: none')
"""),
        md("## 3. Clone the repo at the pinned commit"),
        code("""
WORK = '/kaggle/working'
SRC = f'{WORK}/ColdSite-DTI_New'
if not os.path.exists(SRC):
    subprocess.run(['git', 'clone', REPO, SRC], check=True)
os.chdir(SRC)
subprocess.run(['git', 'fetch', '--all', '-q'], check=True)
subprocess.run(['git', 'checkout', '-q', COMMIT], check=True)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert head == COMMIT, f'checked out {head}, expected {COMMIT}'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tabulate', 'subword-nmt'], check=True)
print('repo at', head)
print(since_start())
"""),
    ]
    if needs_dgl:
        out += [md("## 3b. DGL for DrugBAN (this account trains it)"),
                code("""
# DGL is built against a specific torch; this account trains DrugBAN only, so pinning torch
# here cannot move any other model's numbers (the other accounts do not run this cell).
TORCH_PIN, TORCHVISION_PIN = 'torch==2.6.0', 'torchvision==0.21.0'
DGL_INDEX = 'https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html'
pip = [sys.executable, '-m', 'pip', 'install', '-q']
subprocess.run(pip + [TORCH_PIN, TORCHVISION_PIN, '--index-url', 'https://download.pytorch.org/whl/cu124'], check=True)
subprocess.run(pip + ['dgl', '-f', DGL_INDEX], check=True)
subprocess.run(pip + ['dgllife', 'rdkit', 'yacs', 'prettytable'], check=True)
probe = subprocess.run([sys.executable, '-c', 'import dgl'], capture_output=True, text=True,
                       env={**os.environ, 'DGLBACKEND': 'pytorch'})
if 'torchdata.datapipes' in probe.stderr:
    subprocess.run(pip + ['torchdata==0.7.1'], check=True)
    probe = subprocess.run([sys.executable, '-c', 'import dgl'], capture_output=True, text=True,
                           env={**os.environ, 'DGLBACKEND': 'pytorch'})
assert probe.returncode == 0, 'DGL does not import: ' + probe.stderr[-800:]
print('dgl imports cleanly')
print(since_start())
""")]
    out += [
        md("""
## 4. Raw data and splits

The same steps every earlier notebook ran. The runner's pre-flight then checks the result against
`data/splits/MANIFEST.json` and `config/cloud_manifest.json` (SHA-256), which is stricter than the row counts
the older notebooks compared.
"""),
        code("""
BASE = 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data'
for ds in ('davis', 'kiba'):
    os.makedirs(f'src/data/baselines/deepdta/data/{ds}', exist_ok=True)
    for fname in ('ligands_can.txt', 'proteins.txt', 'Y'):
        target = f'src/data/baselines/deepdta/data/{ds}/{fname}'
        if not os.path.exists(target):
            subprocess.run(['curl', '-sL', f'{BASE}/{ds}/{fname}', '-o', target], check=True)
subprocess.run([sys.executable, '-m', 'src.data.load_data'], check=True)
subprocess.run([sys.executable, '-m', 'src.data.build_splits'], check=True)
print(since_start())
"""),
        md("## 5. Run: pre-flight, restore, one process per GPU"),
        code("""
RESULTS = f'{WORK}/results'
cmd = [sys.executable, '-u', '-m', 'src.cloud.runner', '--account', ACCOUNT, '--waves', WAVES,
       '--results-root', RESULTS, '--session-start', str(START)]
if RESTORE_FROM:
    cmd += ['--restore-from', RESTORE_FROM]
if DRY_RUN:
    cmd += ['--dry-run']
print(' '.join(cmd))
code_ = subprocess.run(cmd).returncode
print('runner exit code', code_, '(0 ok, 1 a cell needs attention, 2 refused by the pre-flight)')
"""),
    ]
    if canary:
        out += [md("## 5b. The verdict\n\nPass rule: `docs/PROTOCOL_AMENDMENT_v2.md` section 6 with the tolerance declared before this re-run: both metrics within **4** committed-seed SDs, same seed (`src/cloud/canary.py`, rule history in its docstring)."),
                code("""
if not DRY_RUN:
    verdict = subprocess.run([sys.executable, '-m', 'src.cloud.canary', '--new', RESULTS,
                              '--out', f'{RESULTS}/canary_verdict.json'])
    print('canary exit code', verdict.returncode, '(0 pass, 1 fail, 3 inconclusive: go back to the user)')
else:
    print('DRY_RUN: no canary was trained.')
"""),
                md("""## 5c. The epoch-1 gate

One epoch each of the canary cell at the committed seed (1, the replay) and at the new wave seeds (4 and 5). Two verdicts are written from
the SAME three runs: `epoch_gate_verdict_original_rule.json` (the rule as first written; it can fail a healthy harness) and
**`epoch_gate_verdict.json`, the official one**, from `src/cloud/gate_rule_v2.py` (Gate B is the family-wise prediction interval of a new draw; Gate A,
finiteness and distinctness unchanged; `docs/canary_attempts/README.md` says why). About 5 to 10 minutes on the two T4s."""),
                code("""
if not DRY_RUN:
    gate = subprocess.run([sys.executable, '-m', 'src.cloud.epoch_gate', '--run', '--seeds', '1', '4', '5',
                           '--root', f'{RESULTS}/epoch_gate', '--out', f'{RESULTS}/epoch_gate_verdict_original_rule.json'])
    print('original-rule exit code', gate.returncode, '(0 pass, 1 fail, 3 inconclusive); the runs are on disk either way')
    official = subprocess.run([sys.executable, '-m', 'src.cloud.gate_rule_v2', '--evaluate', '--seeds', '1', '4', '5',
                               '--root', f'{RESULTS}/epoch_gate', '--out', f'{RESULTS}/epoch_gate_verdict.json'])
    print('OFFICIAL epoch gate exit code', official.returncode, '(0 pass, 1 fail, 3 inconclusive: go back to the user)')
    used_h = (time.time() - START) / 3600
    print(f'This account has now spent up to {2 * used_h:.2f} GPU-hours (2 GPUs x {used_h:.2f} h) outside the wave. Record it:')
    print(f"  python -m src.cloud.quota --root results --account <ACCOUNT> --add-external {2 * used_h:.2f} "
          f"--external-id canary-{time.strftime('%Y-%m-%d')} --note 'canary + epoch gate'")
    print(since_start())
else:
    print('DRY_RUN: the epoch gate was not run.')
""")]
    out += [
        md("## 6. What landed, and the file to take with you"),
        code("""
import glob, json
for path in sorted(glob.glob(f'{RESULTS}/runner_summary_*.json')):
    for row in json.load(open(path))['cells']:
        print(f"{row['outcome']:24s} {row['cell']}  {row['detail']}")
print()
print('complete-markers:', len(glob.glob(f'{RESULTS}/**/*_complete.json', recursive=True)))
prog = f'{RESULTS}/progress.json'
if os.path.exists(prog):
    pr = json.load(open(prog))
    print(f"progress.json: {pr['cells_complete']} of {pr['cells_total']} cells complete; session {pr['elapsed_s'] / 3600:.2f} h long")
    if pr['cells_complete'] < pr['cells_total'] and not DRY_RUN:
        print("NOT FINISHED: download the zip below, add it as a private dataset, set RESTORE_FROM = '/kaggle/input' and run this notebook again (the next session continues each cell from its last epoch).")
    elif pr['cells_total'] and not DRY_RUN:
        print('ALL CELLS COMPLETE for this account: keep the zip; analysis needs it.')
print(since_start())
ZIP = f'{ACCOUNT.lower()}_results.zip'
subprocess.run(f'cd {WORK} && rm -f {ZIP} && zip -qr {ZIP} results', shell=True)
print(f'{WORK}/{ZIP} -- download from the Output panel, add it to your restore dataset')
"""),
    ]
    return {"cells": out, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--waves", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default=None, help="default: the current HEAD")
    ap.add_argument("--canary", action="store_true")
    args = ap.parse_args(argv)
    nb = build(args.account, args.waves, args.commit or head_commit(), args.canary)
    with open(os.path.join(ROOT, args.out), "w") as handle:
        json.dump(nb, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
