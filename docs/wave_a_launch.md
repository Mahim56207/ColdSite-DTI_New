# Wave A — launch steps (T11)

**Status 2026-10-17:** the canary and the official epoch-1 gate PASSED on harness `52bd3335…` (`docs/canary_attempts/README.md`, attempt 3) and both verdicts are in `config/`. The three Wave A notebooks were
regenerated and are pinned to commit `5d55c63` (which contains the verdicts); the pre-flight reports `PASS canary` and `PASS epoch_gate` for all three. Decided: ACC1 and ACC2 use Kaggle's current image (torch 2.11.0;
`docs/PROTOCOL_AMENDMENT_v2.md`, Addendum A7 item 6). The account `mahim5` has already used about 1.2 GPU-hours of its weekly quota on the canary and the gate.
**ACC3: always run it once with `DRY_RUN = True` first.** On Kaggle, with DGL installed by its cell 3b, that dry run performs for the first time the `rng_import` check of the vendored DrugBAN (it cannot run on a machine without
DGL); a failure there costs minutes, not GPU hours. Everything below describes how the steps were and are done.

The three notebooks (`notebooks/kaggle_wave_a_acc1.ipynb`, `_acc2`, `_acc3`) are generated from `config/waves.json` by
`notebooks/build_harness_notebook.py` and pinned to a commit. They ship with `DRY_RUN = True`, and **a real run is refused by the
pre-flight until a passing canary verdict AND a passing epoch-1 gate verdict for this exact harness are committed**
(`config/canary_verdict.json`, `config/epoch_gate_verdict.json`), and it stops itself at the weekly GPU-hour quota
(`docs/cloud_harness.md`, "The weekly quota"). Launching spends quota.

## 0. Before anything

1. **Push** the branch's commits to the fork (`git push`; the notebooks `git checkout` a pinned commit that must exist there).
2. State each account's **weekly quota and its unit**, and the **session limit** (`docs/wave_plan.md`). Put the session limit in
   `config/harness.json`, run `python -m src.cloud.budget --write`, commit.

## 1. The canary (gate)

1. Kaggle → Create → New Notebook → File → Import Notebook → **Upload** `notebooks/kaggle_canary.ipynb`.
2. Accelerator **GPU T4 x2**, Internet **On**, Environment **Pin to original**.
3. Run with `DRY_RUN = True` first: it must print `RESULT: OK (skipped in dry run: gpus)`.
4. `DRY_RUN = False` → **Save Version → Save & Run All (Commit)**. About 1.0 to 1.3 h for the canary on one T4 (the two recorded attempts took 3,529 s and
   4,727 s), then 5 to 10 minutes for the epoch-1 gate (section 5c: seeds 1, 4 and 5, one epoch each). Every 5 minutes the notebook prints a `[monitor]` block with the
   elapsed time, the time left before the self-stop and the current epoch, and `results/progress.json` is rewritten.
5. **Status 2026-10-17:** the earlier verdicts are stale and were removed from `config/` (archived in `docs/canary_attempts/`) because `recipes.py`, `runner.py` and
   `canary.py` changed (`docs/PROTOCOL_AMENDMENT_v2.md`, Addendum A7: canary tolerance 4 committed-seed SDs, DrugBAN in full precision). Run the canary notebook again. It
   writes `canary_verdict.json`, `epoch_gate_verdict_original_rule.json` and **`epoch_gate_verdict.json`, the official gate verdict (`gate_rule_v2`)**. On **both official verdicts passing**:
   download `results/canary_verdict.json` and `results/epoch_gate_verdict.json` from the Output panel, copy them to `config/`, commit, push. On anything else: stop and send it back.
   Rules: `docs/cloud_harness.md` ("The canary", "The epoch-1 gate"); history and caveats: `docs/canary_attempts/README.md`.
6. Section 5c prints a `python -m src.cloud.quota … --add-external …` command with this session's GPU time. Run it (it is idempotent) in
   the account's next notebook or locally against its results folder, so the quota ledger counts the canary against the account that
   ran it. `quota_unit` defaults to the conservative reading (a two-GPU hour counts twice); change it in `config/harness.json` if
   Kaggle is found to count a two-GPU session once.
7. **If `src/cloud/*.py` (the files in `canary.HARNESS_FILES`, now including `quota.py` and `epoch_gate.py`) or `src/model/resume.py`
   changes afterwards, both verdicts go stale** and the pre-flight says so; the canary and the epoch gate must be run again on the new
   harness. (Editing `budget.py`, tests, docs or notebooks does not.)

## 2. Regenerate the wave notebooks at the commit that holds the verdict

```bash
for a in acc1 acc2 acc3; do A=$(echo $a | tr a-z A-Z)
  python notebooks/build_harness_notebook.py --waves config/waves.json --account $A --out notebooks/kaggle_wave_a_$a.ipynb
done
git add notebooks config && git commit -m "Wave A notebooks pinned to the canaried harness" && git push
```
(The notebooks pin `HEAD`; a notebook pinned to a commit without the verdict would be refused.)

## 3. ACC3 first: the DrugBAN smoke run

**Done.** The smoke run was run and recorded in `results/drugban_smoke_run_kiba_t4.md` (fp32 320.4 s per epoch; `--amp` gives a NaN first batch); the plan has DrugBAN's
hours (an estimate) and DrugBAN is full precision on KIBA (`docs/PROTOCOL_AMENDMENT_v2.md`, Addendum A7).

## 4. Each account (ACC1: seed 4, ACC2: seed 5, ACC3: DrugBAN on KIBA)

1. Import that account's notebook by **upload**. Accelerator **GPU T4 x2**, Internet **On**, Environment **Pin to original**.
2. Run once with `DRY_RUN = True`. Expect `PASS` on everything except `gpus`/`canary` reported as `SKIPPED` — on a real T4 x2 the `gpus`
   line reads `PASS Tesla T4, Tesla T4`. Any `FAIL` (splits, hashes, RNG import, disk, wave, cells) means stop and send it back.
3. `DRY_RUN = False` → **Save Version → Save & Run All (Commit)**. The runner starts one process per GPU, resumes any partial cell, and
   stops itself before the session limit minus 45 minutes.
4. When the commit ends: Output → download `<account>_results.zip` → add it as a **private** Kaggle Dataset.
5. Next commit: open the notebook again, set `RESTORE_FROM = '/kaggle/input'`, attach that dataset, run. Repeat until section 6 shows
   every cell `complete` (about two commits for ACC1 and ACC2, by the plan's estimate).
6. Copy the zip to Google Drive.

Two accounts must never run the same account name: each notebook trains only the cells the manifest gives its account.

## 5. After all three are complete

T12 (ingest into `_v2` directories, verify markers, seed-disagreement counts) — skipped for now by the user's decision.

## 5. How long each notebook takes (Kaggle 2 x T4, estimates; the notebook's own header shows the same table)

| notebook | what it trains | wall-clock | sessions | weekly quota used |
|---|---|---|---|---|
| `kaggle_canary.ipynb` | one cell (XAttn-Ref, DAVIS cold-target, seed 1) + the epoch-1 gate | about 1.3 to 1.7 h | 1 | about 1 to 1.5 GPU-h |
| `kaggle_wave_a_acc1.ipynb` | DAVIS seed 4: 12 cells (HyperAttentionDTI, MolTrans, XAttn-Ref x 4 levels) | about 11.6 h per GPU (10.9 to 12.3) | 2 (the second is short) | 23.2 GPU-h (21.8 to 24.7) |
| `kaggle_wave_a_acc2.ipynb` | DAVIS seed 5: the same 12 cells | about 11.6 h per GPU (10.9 to 12.3) | 2 | 23.2 GPU-h |
| `kaggle_wave_a_acc3.ipynb` | DrugBAN on KIBA: random and cold-drug x seeds 1 to 3 = 6 cells (full precision) | about 8.8 h per GPU (8.1 to 9.2; an estimate) | 1 | 17.4 GPU-h (16.1 to 18.3) |

Each session also needs 10 to 15 minutes of setup (ACC3 more for the DGL install). The three accounts run in parallel; the whole wave is about 12 to 13 h of wall-clock
if the second ACC1/ACC2 sessions follow straight after the first.

