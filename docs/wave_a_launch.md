# Wave A — launch steps (T11). Nothing below has been done; each step is the user's.

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
4. `DRY_RUN = False` → **Save Version → Save & Run All (Commit)**. About an hour on one T4 for the canary (estimated: 26 epochs × the
   2 min 15 s the Kaggle logs gave), then about five minutes for the epoch-1 gate (section 5c: seeds 1, 4 and 5, one epoch each).
5. The notebook prints two verdicts (`pass` / `fail` / `inconclusive`). On **both pass**: download `results/canary_verdict.json` **and**
   `results/epoch_gate_verdict.json` from the Output panel, copy them to `config/`, commit, push. On anything else: stop and send it
   back. Rules: `docs/cloud_harness.md` ("The canary", "The epoch-1 gate").
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

The commands are in `docs/wave_plan.md` ("ACC3 — DrugBAN on KIBA: measure first"). Send back `mean_epoch_s` for fp32 and `--amp`; the
plan then gets DrugBAN's hours and ACC3's required quota, and you decide mixed precision for DrugBAN on KIBA.

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
