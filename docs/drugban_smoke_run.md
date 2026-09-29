# DrugBAN smoke run (ACC3) — checklist

**Why.** DrugBAN has never been timed on KIBA. Its notebook carries a 0.6 min/epoch placeholder that its own comment calls "known to be wrong", and
`config/wave_budget.json` has `hours: null` for all six ACC3 cells. This run measures seconds per epoch in full precision (fp32) and in mixed precision
(`--amp`) so ACC3's hours and required quota can be written, and it gives the evidence for one open decision: whether DrugBAN on KIBA trains with `--amp`
(every other KIBA cell except MolTrans did; DrugBAN under `--amp` has never been validated).

**It does not need the canary.** It trains nothing that counts: two epochs, into scratch folders, no real cell touched. Run it on **a different account from the
one running the canary**, so the two runs do not use the same weekly quota.

## Steps

1. **Open the notebook.** Kaggle → Create → New Notebook → File → Import Notebook → **Upload** `notebooks/kaggle_wave_a_acc3.ipynb`
   (https://github.com/Mahim56207/ColdSite-DTI_New/blob/remediation/rigor-fixes/notebooks/kaggle_wave_a_acc3.ipynb ; raw:
   https://raw.githubusercontent.com/Mahim56207/ColdSite-DTI_New/remediation/rigor-fixes/notebooks/kaggle_wave_a_acc3.ipynb).
2. **Check the version.** In the settings cell, `COMMIT` must **start with** `490fa6f` (the full value is `490fa6fced52ec6382dc7e4270c534bc2f99b27f`) and `DRY_RUN` must say `True`. Leave `DRY_RUN = True`: this notebook's runner is never used here.
3. **Settings (right-hand panel).** Accelerator **GPU T4 x2**, Internet **On**, Environment **Pin to original**.
4. **Run cells 1 to 4 only**, one after another: Settings, Session start and GPUs, Clone the repo, **3b DGL**, Raw data and splits. Do **not** run section 5 (the runner) or 6.
   3b installs the torch/DGL versions the real ACC3 run will use, which is the point: time it in the environment it will run in. Wait for `dgl imports cleanly`.
5. **Add one new code cell** after section 4 and paste this whole cell, then run it:

```python
import json, os, subprocess, sys, time
t0 = time.time()

def smoke(tag, amp):
    out = f'results/smoke_{tag}'
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, '-u', '-m', 'src.model.train_drugban',
           '--split-dir', 'data/splits/kiba/random', '--dataset', 'kiba', '--split', 'random', '--seed', '1',
           '--batch-size', '64', '--lr', '5e-05', '--min-epochs', '10', '--epochs', '100', '--patience', '15',
           '--log-every', '100', *(['--amp'] if amp else []),
           '--checkpoint-dir', out, '--results-dir', out, '--stop-after-epoch', '2']
    env = {**os.environ, 'COLDSITE_HARNESS': '1', 'COLDSITE_STATUS_JSON': f'{out}/status.json', 'DGLBACKEND': 'pytorch'}
    return subprocess.run(cmd, env=env).returncode

print('torch', subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'], capture_output=True, text=True).stdout.strip())
print('fp32 exit code', smoke('fp32', False))
print('amp  exit code', smoke('amp', True))
for tag in ('fp32', 'amp'):
    s = json.load(open(f'results/smoke_{tag}/status.json'))
    print(f"{tag}: epochs done {s['epoch']}, mean_epoch_s {s['mean_epoch_s']}, last_epoch_s {s['last_epoch_s']}")
hours = (time.time() - t0) / 3600
print(f'elapsed {hours:.2f} h; GPU-hours to record against this account: {hours:.2f} (one GPU) to {2 * hours:.2f} (both counted)')
```

6. **Watch the first epoch.** How long it takes is the thing being measured, so there is no expected value. Each command stops itself after 2 epochs. If one single epoch
   runs past about 45 minutes, interrupt the cell and send back what the log shows; a slow answer is still an answer.
7. **Copy the whole printed output** of that cell, plus, from the log above it, the lines that start with `Epoch 1/` and `Epoch 2/` for **both** runs (they hold the validation numbers), and any
   line mentioning `nan`, `inf`, `non-finite` or `skipped step`.

## What to send back

| Item | Where it is |
|---|---|
| `mean_epoch_s` for fp32 and for amp | printed at the end of the cell |
| Both exit codes (must be 0) and epochs done (must be 2) | printed |
| Epoch 1 and 2 validation AUROC / loss for fp32 and for amp | the `Epoch 1/`, `Epoch 2/` lines |
| Any non-finite step message | search the log |
| torch version | first printed line |
| Elapsed hours | last printed line |

**Rough check on the way:** the fp32 and amp epoch-1 validation numbers should be close (within about 0.02 AUROC). A large gap, or any non-finite step under amp, means DrugBAN should stay in fp32 on KIBA.

## What happens next

DrugBAN's hours per cell = seconds per epoch × batches × (the twelve committed DrugBAN DAVIS `best_epoch` values + 15), from your two numbers. That fills `config/wave_budget.json`
(`python -m src.cloud.budget --write`), gives ACC3 its required weekly quota, and settles amp or fp32 for DrugBAN on KIBA. The GPU-hours this run spent are recorded against ACC3's quota
ledger when its wave starts (`python -m src.cloud.quota ... --add-external <hours> --external-id drugban-smoke-<date>`; I will give you the exact line).
