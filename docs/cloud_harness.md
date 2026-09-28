# Cloud harness (T09)

`src/cloud/` wraps the existing trainers. It changes none of their recipes: a cell's command is
`src/cloud/recipes.py`, which a test compares with the notebooks' own flags. The user launches; nothing
here spends quota by itself, and no Kaggle credential exists anywhere in it.

```
python -m src.cloud.runner --account ACC1 --waves config/waves.json --results-root results \
       [--restore-from /kaggle/input] [--session-start <unix>] [--dry-run]
```

## Where each cloud rule lives (docs/REMEDIATION_PLAN.md, cloud_rules)

| rule | implemented in | verified by |
|---|---|---|
| (a) pre-flight, refuse with non-zero exit and no GPU work | `src/cloud/preflight.py`, exit code 2 from `runner.main` | one test per refusal in `tests/test_cloud_harness.py`; four guards mutation-checked |
| 2 × T4 | `check_gpus` (via `nvidia-smi`, so the launcher opens no CUDA context) | wrong count, wrong card, dry-run reported as SKIPPED |
| split / ground-truth / panel hashes | `check_splits` against `data/splits/MANIFEST.json` (T02) | tamper, missing, extra, derived-split scoping |
| raw dataset + vendored hashes | `check_raw_dataset`, `check_vendored` against `config/cloud_manifest.json` | tamper tests; the manifest matches this repo's tracked files exactly |
| RNG-import guard | `check_rng_import`: each vendored trainer module imported in a fresh process | a planted seeding import is refused; the real MolTrans and HyperAttentionDTI imports pass |
| disk | `check_disk`: planned bytes = measured checkpoint size × 4 per cell + a declared reserve | too-little-disk refused |
| not already complete / partial detected | `markers.state`: fresh / partial / complete / needs_finalize / inconsistent | orphan checkpoint, foreign results and tampered cells are refused, never retrained over |
| wave manifest lists the cell for this account | `config.load_waves` + `check_wave` | duplicate cell across accounts refused |
| (b) self-stop at limit − margin; no epoch that would cross it | `config.stop_at` → `COLDSITE_STOP_AT` → `Resumable.interrupt_now` (projection = now + longest epoch measured in the process); runner hard-kills only at the true limit | deadline test; end-to-end cut and resume |
| (c) resumable checkpoint, atomic | the trainers' existing `src/model/resume.py` (model, optimiser, scheduler, GradScaler, selector = best-metric and early-stopping counter, epoch, history, every RNG; temp file then rename) | `tests/test_resume.py` (existing) + the runner-level equivalence test |
| … on SIGTERM / SIGINT | `Resumable` handler: finish the epoch in progress, save, exit 0; a second signal is not caught | SIGTERM test on a running process |
| … bit-for-bit resume | CPU only, as before (GPU cuDNN is not deterministic run to run, CLAUDE.md §5) | a cell cut by the deadline and resumed by the next session ends with parameters identical to an uninterrupted run |
| (d) restore from the previous output | `src/cloud/restore.py`: only this account's cells, never over an existing file, ambiguity and marker mismatch are errors | four restore tests |
| (e) status JSON per cell, every epoch | `Resumable._write_status`: cell, account, gpu, epoch, elapsed, projected finish (an upper bound: assumes no early stopping), checkpoint and resume-file paths with SHA-256 | status test |
| (f) complete-marker | `markers.write_marker`: checkpoint hash, results hash, **test-predictions hash** (T04's `predict_cell`, run in a child on the worker's GPU), start record hash, environment (python, torch, CUDA, cuDNN, cudnn flags, vendored content hashes) | marker test; no marker without a predictions hash |
| (g) determinism flags | none set: the trainers set only the seeds; the marker records the cuDNN flags actually in force | code read: `grep` finds no `use_deterministic_algorithms`/`cudnn` setting in `src/model/train*.py` |
| one process per GPU, no DDP | `runner.child_env`: `CUDA_VISIBLE_DEVICES=<i>`, one worker thread and one subprocess per GPU queue | env test |
| every run records seed, RNG start state and initial-weight hash | `Resumable._record_start` → `<cell>_start.json` (fresh cells only) | three seeds give three different hashes; resume keeps the original record |

## Deviations from the plan's text (each declared)

1. **SIGTERM/SIGINT finishes the current epoch before saving**, rather than saving mid-epoch. Mid-epoch
   state is not resumable (the batch order lives in the data loader), and a batch-level resume would change
   the trainers. A harder kill still leaves the previous epoch-end file, written atomically.
2. **"Vendored repo commit hashes" are content hashes** of each vendored directory (sorted path + file
   SHA-256). The vendored code is copied in-tree without its upstream `.git`, so an upstream commit id is not
   recoverable from the files; `baselines/*/PROVENANCE.md` says where each came from.
3. **`session_limit_hours` is 11**, the self-stop every earlier grid ran under (`notebooks/kaggle_binary_grid.ipynb`,
   `DEADLINE = START + 11 * 3600`); Kaggle's own cap is not asserted. With the 45-minute margin a session stops
   training at 10 h 15 min. The user states each account's session limit in T10 and this value is then set.
4. **Trainers gained inert hooks** in `src/model/resume.py` (`COLDSITE_HARNESS`, `COLDSITE_STOP_AT`, `COLDSITE_STATUS_JSON`,
   `COLDSITE_CELL_CONTEXT`). With the variables unset — every earlier notebook — behaviour is unchanged (a test
   pins it). They read state and write side files; they touch no weight, optimiser state or random number.

## The canary (amendment §6, decision D7)

Retrain **DAVIS / ColdSite-DTI / cold_target / seed 1** under the harness (`config/canary_wave.json`,
`notebooks/kaggle_canary.ipynb`) and judge it against `config/canary_reference.json`, frozen from the three committed
seeds: pass iff test AUROC **and** test AUPRC each lie within one sample SD (ddof = 1, over the three committed seeds)
of the committed seed-1 value; SD = 0 → inconclusive, back to the user (`src/cloud/canary.py`, exit code 0 / 1 / 3).
ColdSite-DTI is the cheapest attention model (2 min 15 s per epoch on a T4, CLAUDE.md §4). `cold_target` was chosen
after looking at the committed SDs: 0.0109 AUROC, against 0.0012 at `random` (55 epochs, about twice the cost) and
0.0991 at `cold_pair`, where a tolerance that wide would pass a broken harness. Seed 1 ran 26 epochs there.
The canary re-scores no explanation and adds nothing to any family.

### To run it (the user's step; it spends quota)

1. Push the commit the notebook names (`git log -1` on the branch; the notebook pins it) to the fork.
2. Kaggle → Create → New Notebook → File → Import Notebook → **Upload** `notebooks/kaggle_canary.ipynb`.
3. Accelerator **GPU T4 x2**, Internet **On**, Environment **Pin to original**.
4. Run once with `DRY_RUN = True` (the shipped default): it prints `PRE-FLIGHT … RESULT: OK` and stops.
5. Set `DRY_RUN = False`, **Save Version → Save & Run All (Commit)**. Expect about 1 h on one T4 for the canary, then about
   5 minutes for the epoch-1 gate (section 5c of the notebook).
6. The notebook prints two verdicts: `canary_verdict.json` (section 5b) and `epoch_gate_verdict.json` (section 5c). Send both back;
   T11 stays blocked until both pass. Section 5c also prints the exact command that records this session's GPU time in the quota
   ledger (`--add-external`), for the account that ran it.

## The epoch-1 gate (`src/cloud/epoch_gate.py`; requested 2026-09-29)

The canary retrains one committed cell to the end. The gate is the cheap companion that runs beside it: ONE epoch each of the canary
cell (ColdSite-DTI, DAVIS cold_target) at the committed seed 1 and at the new wave seeds 4 and 5, all under the harness, then two rules
fixed in the module before any number exists.

| Gate | Rule |
|---|---|
| A, replay (seed 1) | epoch-1 train loss, validation loss and validation AUROC each within **one sample SD** (over the three committed seeds' epoch-1 values) of the committed seed-1 value; SD = 0 is inconclusive |
| B, each new seed | every metric finite; every metric inside the committed seeds' **envelope widened by one SD**; **distinct**: initial-weight hash differs from every other run in the gate and the three metrics are not identical to another run's |

The original request was that Seed 4's loss should match Seed 1's trajectory. That cannot be asked of a different seed (the paper's
premise is that seeds differ), so the "matches seed 1" check is the replay of seed 1 itself (A), and the new seeds are checked for
being plausible draws (B, inside the envelope) and for really being different (distinct: the MolTrans defect, three seeds that were
one run, is exactly what fails it). The reference is `config/epoch1_reference.json` (epoch-1 values and SHA-256 of the committed
history files). A pass, copied to `config/epoch_gate_verdict.json` and committed, is required by the pre-flight for every wave: for
this cell, this harness (`harness_sha256`, `canary.HARNESS_FILES`) and every new seed (seed > 3) the account trains. The one-epoch
command is `run_grid.train_command` plus `--stop-after-epoch 1`: no training code is touched.

## The weekly quota (`src/cloud/quota.py`; requested 2026-09-29)

Kaggle allows 30 GPU-hours per account per week (stated 2026-09-25). The harness keeps a ledger, `<results>/quota_ledger.json`
(one interval per cell per GPU, a heartbeat every 30 s), that travels with the results zip and is merged back on `--restore-from`.

| Setting (`config/harness.json`) | Value | Meaning |
|---|---|---|
| `weekly_quota_gpu_hours` | 30 | the stated quota |
| `quota_unit` | `gpu_hours` | each GPU worker's wall-clock counts (a two-GPU hour is two GPU-hours). **The unit of a two-GPU session was not stated; this is the conservative reading.** `session_hours` counts a two-GPU session once |
| `quota_reserve_fraction` | 0.15 | usable = 30 × 0.85 = **25.5** (the figure `docs/wave_plan.md` fits the plan against: 23.2 mean, 24.7 high case) |
| `quota_soft_fraction` | 0.95 | past 95 % of usable no FRESH cell starts (a partial cell may resume). Not lower: the plan's high case must be able to finish |
| `quota_window_days` | 7 | rolling window, which can only over-count against Kaggle's fixed week |
| `quota_min_start_hours` | 0.5 | a cell with no measured epoch needs this many usable hours left |

Behaviour: the pre-flight reports usage and refuses a real run once the usable hours are spent; the runner refuses a cell that cannot
pay for one epoch; the trainers' stop time is `min(session deadline, the moment the usable hours run out with both GPUs busy)`, so they
**pause between epochs** with the state saved and outcome `paused (quota)`; a trainer that overshoots gets SIGTERM (finish the epoch,
save) and is killed only past the nominal 30 hours. A session that died leaves its interval closed at its last heartbeat, never later.
Limits: it cannot see Kaggle's own counter, so usage outside the harness (the canary, anything run by hand) must be declared:
`python -m src.cloud.quota --root results --account ACC1 --add-external 2.2 --external-id canary-2026-09-29 --note "canary + epoch gate"`
(idempotent per id; `--status` prints the ledger; usage of one account is never merged into another's).
