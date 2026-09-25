# Wave A — budget and partition across three Kaggle accounts (T10)

Generated numbers: `python -m src.cloud.budget --write` → `config/waves.json` (who trains what) and `config/wave_budget.json`
(hours per cell, per account, inputs, cross-check). A test regenerates both and fails if either file drifts (`tests/test_wave_plan.py`).
**Nothing here has been launched.** T12 (ingest) is skipped by the user's instruction; the canary (T09) has not run.

## What Wave A trains: 30 cells, each exactly once

| account | cells | why |
|---|---|---|
| **ACC1** | DAVIS × seed **4** × {HyperAttentionDTI, MolTrans, ColdSite-DTI} × 4 levels = 12 | extension family E1 (seed instability, headline (a)) |
| **ACC2** | DAVIS × seed **5** × the same 12 | E1 |
| **ACC3** | **DrugBAN** × KIBA × {random, cold_drug} × seeds 1–3 = 6 | family E2 (problem P08: DrugBAN has no KIBA cell) |

DrugBAN has its own account because its notebook pins `torch==2.6.0` for DGL (`notebooks/build_harness_notebook.py`, cell 3b): that pin
must not touch the numbers of any other model. Seeds 1–3 of DAVIS already exist and are not retrained. The two seeds of a model are on
different accounts, so the loss of one account costs one seed, not a model.

Each GPU queue (largest cell first, so the cell most likely to meet the session stop starts earliest) is in `config/waves.json`; the
two GPUs of ACC1 and ACC2 differ by 0.02 GPU-hours out of 11.6 (11.61 and 11.63).

**Decision D2 (amendment §3 row E1), taken here, before any seed-4/5 result exists: DrugBAN seeds 4–5 are NOT in E1.** Reason: no
DrugBAN training time is recorded anywhere in the repository (below), so there is no measured cost to plan its 8 DAVIS cells against, and
the plan rules forbid a guess. E1 therefore has m = 16 (3 models × 4 levels + the uniform control × 4), not 20. If the user wants them, the
smoke run below gives the number, and an addendum (A6) must add them before any seed-4/5 result is seen. Proposed as A6 in
`docs/PROTOCOL_AMENDMENT_v2.md` §10 (not in force until the user replies "approved").

## How the hours were derived (measured inputs only)

Per cell: `epochs × batches-per-epoch × seconds-per-batch × overhead`.

| input | source |
|---|---|
| seconds per training batch, fp32, Tesla T4: ColdSite-DTI 0.387, HyperAttentionDTI 0.552, MolTrans 0.228 | `results/speed_test_kiba_t4.md` (first table); it agrees with the Kaggle grid logs (its own header) |
| batches per epoch = training rows / batch size (64, 32, 16; MolTrans drops the last) | `n_train_rows` in the committed `_results.json`; batch sizes from `src/cloud/recipes.py` (the notebooks') |
| overhead of validation/checkpointing: ColdSite-DTI ×1.06, HyperAttentionDTI ×1.00, MolTrans ×1.00 | ratio of the Kaggle epoch times in `CLAUDE.md` §4 (2 min 15 s, ~6 min, ~5 min on DAVIS `random`) to batches × s/batch, floored at 1 |
| epochs a cell runs = `best_epoch` of the model's three committed seeds at that level + patience | `best_epoch` in each `_results.json`; early stopping fires when `epoch − best_epoch ≥ patience` (`src/model/early_stopping.py`); ColdSite-DTI's own histories show +16 in 18 of 18 cells, the two baselines use patience 15 (**+15 is read from the rule; no per-epoch log of a baseline exists to confirm it**) |

Range: mean, min and max of the three committed seeds' epoch counts. Cross-check of the method (never an input): the same formula on the
24 KIBA cells that were trained (mixed precision except MolTrans, as they were) gives **107.5 GPU-hours** against the report's "~101
GPU-hours" (+6.5 %; the 101 is a claim `docs/inventory.md` could not find in any log).

## Totals (GPU-hours: one cell on one T4 for one hour is one)

| account | cells | mean | low–high | per GPU (mean) | commits needed at 10.25 h of training per GPU per commit |
|---|---|---|---|---|---|
| ACC1 (seed 4) | 12 | **23.2** (ColdSite-DTI 4.9, HyperAttentionDTI 10.3, MolTrans 8.1) | 21.8–24.7 | 11.6 / 11.6 | 2 (the second is short) |
| ACC2 (seed 5) | 12 | **23.2** | 21.8–24.7 | 11.6 / 11.6 | 2 |
| ACC3 (DrugBAN, KIBA) | 6 | **not measurable yet** | — | — | — |

A commit gives one account `2 × (11 − 0.75) = 20.5` GPU-hours (`config/harness.json`: 11 h session, 45 min margin).

## Quota — stated by the user (2026-09-25)

The user stated: **30 GPU-hours per week per account**; **session limit 12 hours**; the notebooks' **11-hour self-stop is
correct** (so `config/harness.json` keeps `session_limit_hours = 11`, 45 min margin). Each account needs, for the plan to fit
with the ≥ 15 % reserve (used ≤ 85 % of the quota; `required = hours / 0.85`):

| | needed (GPU-hours, a T4 × 2 session counting double) | stated quota | fits with ≥ 15 % reserve? |
|---|---|---|---|
| ACC1, ACC2 — mean | 27.3 | 30 | yes |
| ACC1, ACC2 — high case | **29.1** | 30 | yes, by 0.9 GPU-hours |
| ACC3 | after the smoke run | 30 | not verifiable: DrugBAN hours unmeasured |

Numbers from `config/wave_budget.json` (`required_weekly_quota_gpu_h_for_mean/high`). The fit holds in the stricter unit
(GPU-hours); if Kaggle counts a two-GPU session once, the need halves (13.7 / 14.6). ACC1/ACC2 each need two commits
(23.2 GPU-hours mean vs 20.5 per commit), both inside one week's 30. **Kaggle runs are deferred by the user (2026-09-25);**
nothing here has been launched.

## ACC3 — DrugBAN on KIBA: measure first, then queue

The last DrugBAN run (12 DAVIS cells, 2026-09-18) left no wall-clock record; its notebook carries a 0.6 min/epoch placeholder that its
own comment calls "known to be wrong". `hours` is `null` for all six cells in `config/wave_budget.json`. Before ACC3's cells are queued, run
this once on ACC3, in a scratch folder so no real cell is touched (the harness variables make the trainer write its measured epoch time):

```bash
mkdir -p results/smoke_fp32 results/smoke_amp
COLDSITE_HARNESS=1 COLDSITE_STATUS_JSON=results/smoke_fp32/status.json python -u -m src.model.train_drugban \
  --split-dir data/splits/kiba/random --dataset kiba --split random --seed 1 --batch-size 64 --lr 5e-05 \
  --min-epochs 10 --epochs 100 --patience 15 --log-every 100 \
  --checkpoint-dir results/smoke_fp32 --results-dir results/smoke_fp32 --stop-after-epoch 2
COLDSITE_HARNESS=1 COLDSITE_STATUS_JSON=results/smoke_amp/status.json python -u -m src.model.train_drugban \
  --split-dir data/splits/kiba/random --dataset kiba --split random --seed 1 --batch-size 64 --lr 5e-05 \
  --min-epochs 10 --epochs 100 --patience 15 --log-every 100 --amp \
  --checkpoint-dir results/smoke_amp --results-dir results/smoke_amp --stop-after-epoch 2
```

Send back `mean_epoch_s` from each `status.json`, the non-finite-step counts if the log prints any, and whether the two runs' epoch-1
validation numbers are close. That gives the KIBA seconds/epoch for DrugBAN (fp32 and mixed precision); epochs come from the twelve
committed DrugBAN DAVIS `best_epoch` values + 15. Only then can ACC3's hours and required quota be written.

**Open decision: mixed precision for DrugBAN on KIBA.** Every other KIBA cell but MolTrans's was trained with `--amp` and
`recipes.uses_amp` follows that, but DrugBAN under `--amp` has never been validated (the speed test and the AMP validation did not include
it). The smoke run above gives the evidence; the default in `waves.json`/`recipes.py` stays "amp on for KIBA" until the user decides.

## Analysis work deferred from T05 (to this phase) — costs unmeasured, not in `waves.json`

| job | needs | cost |
|---|---|---|
| DrugBAN faithfulness CIs: `run_faithfulness --model drugban --record-pairs`, 12 DAVIS cells | DGL (ACC3's environment), the 12 committed DrugBAN checkpoints (4.3 MB each) | unmeasured — time the first one |
| 8 HyperAttentionDTI-IG cells with no per-protein scores: re-run the IG ladders (`run_ladder --model hyperattentiondti_ig`) | `notebooks/kaggle_integrated_gradients_davis.ipynb` exists; HyperAttentionDTI checkpoints (9.2 MB each) | unmeasured — time the first one |
| uniform-control enrichment | a ladder for the uniform control | CPU only, no GPU budget |
| T06 conservation null (E4) | a user-approved conservation source (amendment D6 / A1) | CPU only, no GPU budget |
| T08 explanation panel, E3-D 28 cells + E3-K 12 cells | amendment addendum A2 signed | unmeasured; occlusion is ~L/2 forward passes per protein |

## Order of work

1. The user launches the **canary** (`docs/cloud_harness.md`); T11's notebooks stay unlaunchable until it passes.
2. ACC3: the smoke run above; its numbers close this plan.
3. ACC1 and ACC2: `notebooks/kaggle_wave_a_acc1.ipynb` / `_acc2.ipynb` (T11), each about two commits, restoring the previous output between them.
4. ACC3 after its numbers are in.
Each account owns its cells; the manifest lists each cell once and the runner reads only its own account's queue.
