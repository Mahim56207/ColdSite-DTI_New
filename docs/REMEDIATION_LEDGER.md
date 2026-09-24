# Remediation ledger

Plan: `docs/REMEDIATION_PLAN.md` (saved verbatim, T00). One task per session; a task is DONE
only when its verification criteria pass, the ledger is updated, and the HALT REPORT is printed.

## Task status

| Task | Title | Status | Branch | Notes |
|---|---|---|---|---|
| T00 | Bootstrap | **DONE** (2026-09-24) | remediation/T00 | plan + ledger written; suite run; layout verified; 5 discrepancies recorded below |
| T01 | Inventory | **DONE** (2026-09-24) | remediation/T01 | `docs/inventory.md` written; checkpoint dir `~/ColdSite-results/` confirmed by the user; 15 gaps recorded below |
| T02 | Integrity guards | **DONE** (2026-09-24) | remediation/T02 | 91 new tests (guards a–f) + `data/splits/MANIFEST.json` (64 files); 5 guards mutation-checked; one code deviation and one rule-vs-plan gap recorded below (D6) |
| T03 | Protocol amendment | **DRAFTED — awaiting user "approved"** (2026-09-24) | remediation/T03 | `docs/PROTOCOL_AMENDMENT_v2.md` written, no analysis run; not in force until the user replies "approved"; 8 decisions (D1–D8) and 5 code gaps (G1–G5) recorded |
| T04 | Predictive accuracy table | **DONE** (2026-09-24) | remediation/T04 | all 84 cells tabulated (72 predicted on MPS, 12 DrugBAN from recorded files with MCC/F1 empty); localize run per dataset and pooled; 1044 passed, 5 skipped | KIBA has no DrugBAN cells (P08); DeepDTA is binary-only so no regression metrics (§ T04 findings 1–2) |
| T05 | Effect sizes & CIs | **PARTIAL** (2026-09-25) | remediation/T05 | enrichment CIs (68 cells), seed-spread table, original-verdict reproduction and faithfulness-delta CIs (16 rows: ColdSite-DTI, HyperAttentionDTI, MolTrans) DONE; two declared gaps remain: DrugBAN faithfulness CIs (no DGL here) and 8 HyperAttentionDTI-IG cells (no per-protein scores); the re-run does not reproduce MolTrans's committed faithfulness means exactly (see "Faithfulness re-run" below) |
| T06 | Conservation null + intermediate rung | PENDING | | |
| T07 | Readout primacy | PENDING | | blocked until the user supplies per-model figure references. 9 readout variants are registered; `drugban_maxhead` and `moltrans_interaction_sum` have never been run; `moltrans_interaction` (their Fig. 3 map) exists and must be reused, not re-implemented |
| T08 | Explanation panel | PENDING | | |
| T09 | Cloud harness hardening | PENDING | | |
| T10 | Budget & partition plan | PENDING | | needs quota and session limit from the user |
| T11 | Wave A notebooks | PENDING | | gated on the T09 canary passing |
| T12 | Wave A ingest | PENDING | | |
| T13 | Wave B (optional) | PENDING | | user decides |
| T14 | Modern model feasibility spike | PENDING | | |
| T15 | Modern model integration | PENDING | | |
| T16 | Modern model ingest & audit | PENDING | | |
| T17 | Stricter splits (optional) | PENDING | | |
| T18 | Non-kinase scope | PENDING | | |
| T19 | Citation verification & delta table | PENDING | | |
| T20 | Reframing | PENDING | | |
| T21 | Condense to BiB format | PENDING | | blocked until `docs/bib_guidelines.md` exists |
| T22 | Figures | PENDING | | |
| T23 | Reproducibility release | PENDING | | |
| T24 | Pre-submission audit | PENDING | | |

---

## T00 — Bootstrap (DONE, 2026-09-24)

**Branch:** `remediation/T00`, from `main` at `d5458cd`, working tree clean at start
(`git status --short` printed nothing).

**Files changed:** `docs/REMEDIATION_PLAN.md` (new), `docs/REMEDIATION_LEDGER.md` (new). No other
file created, modified or deleted.

**Commands run**
| command | purpose |
|---|---|
| `git checkout -b remediation/T00` | task branch |
| `python3 -m pytest -p no:warnings` | full suite |
| `ls baselines/`, `ls data/splits/*`, `ls data/*sites*.json` | layout check |
| `python3 -c "... nonkinase_panel.csv"` | panel size |
| `ls ~/ColdSite-results/{davis,kiba}_binary/*.pt \| wc -l`, `du -sh` | cell count, checkpoint size |
| `cat paper/*.md \| wc -w` | draft length |
| `grep -n '"--n-trials"' ...` | permutation defaults |

**Tests:** `937 passed, 4 skipped in 93.42s (0:01:33)` — verbatim summary line, default
environment (`/opt/anaconda3/bin/python3`). 941 collected. The 4 skips are the tests that need
the `drugban` conda environment (DGL / subword_nmt), which the default interpreter lacks.

### Layout verification against `project_facts`

VERIFIED:
- Vendored repos present: `baselines/{DeepDTA, DrugBAN, HpyerAttentionDTI, MolTrans}`.
- DAVIS splits: `random, cold_drug, cold_target, cold_pair` (plus `*_seqclean`, `*_seqmatched`
  variants — see D4). KIBA splits: all four directories exist, though only `random` and
  `cold_drug` are trained (D3).
- Ground truths present: `data/davis_ground_truth_sites.json`, `data/kiba_ground_truth_sites.json`,
  `data/{davis,kiba}_klifs_pocket_sites.json`, `data/davis_drug_sites*.json` (per-pair contacts),
  `data/nonkinase_ground_truth_sites.json`.
- Non-kinase panel: `data/processed/nonkinase_panel.csv` — 21,145 rows over **60** proteins.
- Antiviral subset: module `src/data/extract_antiviral.py` present (no `data/*antiviral*` file; the
  subset was cut from the paper on 2026-09-18 per `paper/results.md`).
- Trained cells outside the repo: `~/ColdSite-results/davis_binary` 60 `.pt` (3.1 GB),
  `~/ColdSite-results/kiba_binary` 24 `.pt` (1.5 GB) = **84 cells**, 4.6 GB.
- DrugBAN: 12 DAVIS checkpoints, **0** KIBA checkpoints — matches "DAVIS 60 incl. DrugBAN;
  KIBA 24 without DrugBAN" and P08.
- Entry point `src/evaluation/run_all.py` present.
- `CLAUDE.md` and `paper/references.md` present.

### Discrepancies (repo wins; recorded per the zero-hallucination rules)

| id | project_fact | repo | impact |
|---|---|---|---|
| D1 | "Tests: 938 claimed" | `937 passed, 4 skipped` (941 collected) in the default env | none scientific; the plan's figure predates the tests added on 2026-09-20. Use the verbatim line above. |
| D2 | cloud pre-flight checks split hashes in `data/splits/MANIFEST.json` | the file does **not** exist | T02 must create it before any pre-flight can enforce it (already scoped in T02). |
| D3 | "DAVIS (random, cold-drug, cold-target, cold-pair) and KIBA (random, cold-drug only)" | KIBA **split directories** exist for all four levels; only random and cold-drug are **trained** | wording only: the restriction is on training, not on the split files. |
| D4 | DAVIS has four split levels | eight DAVIS split directories exist: the four levels plus `cold_target_seqclean`, `cold_target_seqmatched`, `cold_pair_seqclean`, `cold_pair_seqmatched` | T01 must inventory what the `_seqclean` / `_seqmatched` variants are and whether any reported number uses them. |
| D5 | "(5) Permutation resolution capped p at 1/501; **now 10,000 permutations**" | the **defaults remain** 500 (`src/evaluation/run_audit.py:254`) and 1000 (`src/evaluation/run_ladder.py:332`, `src/evaluation/positional_control.py:305`). 10,000-permutation audits exist only as the extra outputs `results/analysis_{davis,kiba}_policyA/audit_*_10k_permutations.{json,md}`; the primary audit files are still the 500-permutation ones | **material.** The statistical-integrity rule (n ≥ 10000) is not yet enforced in code, and the paper's primary audit is the 500-permutation run. T02 must add the enforcement; T03/T05 must decide which run is primary. Flagged to the user. |

Also noted, not a discrepancy: the paper drafts measure **33,142 words** over the six section
files (`introduction`, `methods_data_and_evaluation`, `methods_track_b`, `results`, `discussion`,
`limitations`), against "~27,600" in `project_facts`; that earlier figure excluded table rows.

### Discovered (work belonging to other tasks — not started)

- The `_seqclean` / `_seqmatched` DAVIS split variants need inventorying → T01.
- Permutation defaults below 10,000 need a code-level guard → T02.
- Whether the 500- or the 10,000-permutation audit is the primary analysis is a scientific
  decision for the amendment → T03.


---

## T01 — Inventory (DONE, 2026-09-24)

**Branch:** `remediation/T01`, from `remediation/T00` at `19b9965`, working tree clean at start.

**Checkpoint directory:** `~/ColdSite-results/`, confirmed by the user in the task message.

**Files changed:** `docs/inventory.md` (new), `docs/REMEDIATION_LEDGER.md` (this file). No
other file created, modified or deleted; `git status --short` showed only `?? docs/inventory.md`
before the ledger edit, and `git diff -- results/` was empty.

**Commands run** (all read-only apart from writing the two documents)

| command | purpose |
|---|---|
| `git checkout -b remediation/T01` | task branch |
| `ls`/`find`/`du`/`stat`/`shasum` over `~/ColdSite-results/` | cell census, sizes, duplicate detection |
| `python3 -c "json.load(...)"` over `_results.json`, ladder, audit, control, accuracy and CI outputs | read recorded fields; no metric recomputed |
| `torch.load(..., map_location="cpu")` on one checkpoint | list its four top-level keys |
| `find data/splits -name '*.csv'` + `wc -l` | split row counts |
| `git ls-files`, `git status --ignored` | which splits/results are version-controlled |
| `grep -rn` over `src/`, `paper/` | registries, permutation defaults, claim locations |
| `python3 -m pytest -p no:warnings` | full suite |

**Tests:** `937 passed, 4 skipped in 87.39s (0:01:27)` — verbatim summary line, default
environment. Unchanged from T00's `937 passed, 4 skipped in 93.42s`; T01 added no tests.

**Verification against T01's own criteria**

* every row cites a path — yes, `docs/inventory.md` gives a file path or a command for each claim;
* nothing computed — no metric was recalculated. Permutation counts were *read* from the
  denominators of recorded p-values, and file counts/sizes from the filesystem;
* no files other than `docs/inventory.md` and this ledger changed — confirmed by `git status`.

### Headline findings

* **84 cells confirmed** (DAVIS 60 = 5 models × 4 levels × 3 seeds; KIBA 24 = 4 models ×
  2 levels × 3 seeds, no DrugBAN), 4.6 GB, of which MolTrans's 240 MB-per-cell files are ~97%.
* **D4 resolved:** the four extra DAVIS split directories (`*_seqclean`, `*_seqmatched`)
  belong to 12 further DeepDTA leakage cells in `~/ColdSite-results/leakage_davis/`, which
  are real and are *not* part of the 84.
* **D5 quantified:** primary audits ran at **500** permutations (min p = 1/501 = 0.001996)
  against a smallest Holm threshold of 0.05/20 = 0.0025 — the test's floor is 80% of the
  threshold. The committed 10,000-permutation reruns change **no verdict**: one significant
  DAVIS cell (`hyperattentiondti|davis|random`) in both, none of eight on KIBA in both.
* **Headline contribution (a) is fully evidenced**: `results/seed_agreement.md` — 12 of 22
  cells disagree across seeds, 21 of 22 have a seed spread larger than their distance from chance.
* **No RNG state, initial-weight hash or environment record exists for any of the 84 cells**,
  and none is recoverable: a finished `.pt` holds only `model_state, epoch, task, val_metrics`,
  and `resume.py`'s RNG capture goes to a `_resume.pt` the trainer deletes on completion.
* **`data/splits/` is gitignored** (`.gitignore:13`), so T02's manifest will hash untracked
  local files; nothing today proves they are the files the 84 cells trained on.
* **No vendored-repo commit hashes anywhere**, and **DeepDTA and HpyerAttentionDTI carry no
  licence file** — two pre-flight/release requirements with nothing to check against.

### Discovered (work belonging to other tasks — not started)

1. `data/splits/MANIFEST.json` absent **and** the splits gitignored → T02.
2. No vendored commit hashes; two vendored repos without a licence → T02 pre-flight, T23.
3. No RNG/initial-weight record for the 84 cells; T02(b) can only enforce it going forward.
4. No saved test predictions and no MCC/F1 anywhere → T04 must run inference over all 84.
5. DeepDTA was trained binary; the DeepDTA regression table at `results.md:3–18` has no
   per-cell file on disk → T04 scoping question, T21 provenance risk.
6. No accuracy-vs-localization analysis exists → T04 / P07.
7. The **DAVIS** IG family's Holm correction is not committed (only KIBA's,
   `results/analysis_kiba_policyA/ig_family_kiba.md`); the generator
   `src/evaluation/ladder_family.py` exists → T05 or T08.
8. `~/ColdSite-results/integrated_gradients/ig_vs_attention.csv` leaves ColdSite-DTI's 8 IG
   cells blank although the ladder JSONs hold the values → T22.
9. The "2–12% readout overlap across 25 proteins" claim (`paper/results.md:591`) has **no
   generator and no output file** → T08's top-k IoU subsumes it; otherwise T21 fails on it.
10. `paper/results.md:677` says "one of sixteen" where the committed audit has 20 cells → T20.
11. Permutation defaults are 500 (audit) and 1,000 (everything else) → T02 enforcement,
    T03 primacy decision.
12. `drugban_maxhead` and `moltrans_interaction_sum` are registered but never run → T07.
13. The k=50 faithfulness sensitivity covers only DrugBAN and HyperAttentionDTI → T03.
14. **No training wall-clock logs exist** — the `kaggle_logs/` files are *analysis* logs.
    T10's budget must come from `results/speed_test_kiba_t4.md` (measured s/batch on a T4)
    plus each cell's recorded `best_epoch`, not from the report's 101 GPU-hour aggregate.
15. Much of `results/` is gitignored, so verification criterion 3 ("git diff shows no change
    to original results") cannot detect a change to an untracked output → T02 should extend
    the manifest to the primary analysis outputs.


---

## T02 — Integrity guards (DONE, 2026-09-24)

**Branch:** `remediation/T02`, from `remediation/T01` at `390cd15`, working tree clean at start.

**Tests:** `1027 passed, 5 skipped in 124.82s (0:02:04)` — verbatim summary line, default
environment. Before: `937 passed, 4 skipped` (941 collected). After: 1032 collected = 941 + **91 new**
(`test_data_manifest` 10, `test_integrity_grid` 21, `test_integrity_masking` 12,
`test_integrity_permutations` 33, `test_integrity_rng_seeds` 11, `test_integrity_row_loss` 4). Passed
rose by 90 and skipped by 1 (the new DGL-dependent DrugBAN tokeniser test); no existing test changed
outcome.

**Files changed**

| file | kind | what |
|---|---|---|
| `src/evaluation/integrity.py` | new | permutation floor (`MIN_PERMUTATIONS = 10000`, `check_permutations`, `min_achievable_p`, `permutations_needed`), grid fingerprint + `check_grid_unchanged`, `check_extension_dir`, `is_scratch_dir` |
| `src/model/integrity.py` | new | `rng_state`, `rng_fingerprint`, `rng_delta`, `initial_weight_hash`, `preserve_rng` |
| `src/data/manifest.py` | new | manifest writer/checker (`--write`, `--check`); `--write` refuses to overwrite an existing manifest |
| `data/splits/MANIFEST.json` | new | SHA-256 + size of 64 files: 36 split CSVs, 26 `data/*.json`, 2 `data/processed/*.csv` |
| `.gitignore` | edit | `!data/splits/MANIFEST.json` (splits stay ignored; the manifest is tracked) |
| `src/evaluation/{run_audit,run_ladder,run_control,positional_control,positive_control}.py` | edit | `--n-trials` default 500/1000 → 10,000; `check_permutations` right after `parse_args`; `--allow-low-permutations` escape hatch; `run_audit` also checks its own family size and now **records** `n_trials` and `min_achievable_p` in its JSON |
| `src/evaluation/run_all.py` | edit | writes `grid_state.json` beside outputs; refuses to skip-existing over a changed grid outside a scratch dir (`--allow-partial` overrides); `--extension` refuses an out-dir that already holds outputs |
| `src/model/train_moltrans.py`, `src/evaluation/baseline_adapters.py` | edit | the vendored `from models import BIN_Interaction_Flat` wrapped in `preserve_rng()` (see deviation 1) |
| 6 test files | new | listed above |
| `docs/REMEDIATION_LEDGER.md` | edit | this section |

**Commands run:** `git checkout -b remediation/T02`; `python3 -m pytest -p no:warnings` (full suite);
`python3 -m src.data.manifest --write` → `wrote data/splits/MANIFEST.json: 64 files {'ground_truth': 26,
'processed': 2, 'splits': 36}`, then `--check` → `manifest OK`; `python3 -m src.evaluation.<runner>
--n-trials 500` for each of the five runners (all refuse with "below the floor of 10000"); RNG probe of
each vendored import; MolTrans matched-control probe on 6 real DAVIS proteins; a script reading
`n_train_rows` from all 84 `_results.json` against their split's `train.csv` row count; `git diff --stat
-- results` (empty).

### Verification against T02's own criteria

| criterion | result |
|---|---|
| (a) vendored imports leave RNG unchanged, both orders | pinned for both trainers' `_import_vendored` in both orders, plus "a seed set before the MolTrans import survives it". **Mutation check:** with the two `preserve_rng` wrappers reverted, 4 tests fail (2 order cases, the seed-survives test, the MolTrans initial-weight test); restored, all pass |
| (b) distinct seeds → distinct initial-weight hashes; same seed → same hash | one **fresh interpreter per (model, seed)** for `coldsite_dti`, `hyperattentiondti`, `moltrans`, `deepdta`; a warm process cannot show this because the vendored import has already run. DrugBAN has no such test (needs DGL, see risks) |
| (c) masking arms size-matched in token space per adapter | residue-level: k masked = exactly k tokens changed, attended-shaped and scattered arms, ColdSite-DTI and HyperAttentionDTI on real DAVIS sequences; MolTrans: control lands within `TOLERANCE` on a reachable target, is never farther than a blind draw, and **reports itself unmatched** on an unreachable one; every audited model must have a masking decision. DrugBAN's tokeniser test **skips here** |
| (d) `n_permutations >= 10000` enforced in config; min p < smallest Holm threshold | all five runners default to 10,000 and refuse lower; `run_audit` also refuses a count whose floor is not below `alpha/m` for its own family; DAVIS (20) and KIBA (8) primary families checked, and the committed `*_10k_permutations.json` audits are read as data. **Mutation check:** ladder default reverted to 1000 → `test_every_runner_defaults_to_the_floor…[run_ladder]` fails |
| (e) `run_all` refuses partial grids outside scratch; extensions never target existing dirs | grid fingerprint, refusal, sanctioned `--no-skip-existing` route, scratch exemption, `--allow-partial`, `--extension` refusal, and all five primary analysis dirs refused as extension targets. **Mutation check:** guard call removed → `test_run_all_refuses_to_top_up_a_partial_grid…` fails |
| (f) atom-cap row-loss guard still fails loudly | exactly the limit tolerated, limit + 1 raises `MissingCell` naming the reason; constants pinned. **See D6: the rule is not "≤1%"** |
| manifest + test | 64 files; hash test, coverage test, unlisted-file test, changed/missing/unlisted detection on a toy tree, re-bless refusal |
| full suite passes; count rises only by new tests | 941 → 1032 collected = +91; 937 → 1027 passed, no outcome changed |
| no analysis output changed | no analysis was run; `git diff --stat -- results` empty; the primary 500/1000-permutation outputs are untouched |

### Discrepancy (repo wins)

| id | plan | repo | impact |
|---|---|---|---|
| D6 | (f) "the DrugBAN atom-cap row-loss threshold (<=1%)" and project fact (10) "fails if more than 1% are lost" | `UNENCODABLE_LIMIT_FRACTION = 0.01` but the limit is `max(UNENCODABLE_LIMIT_MIN = 3, int(0.01 × rows))`. Explanation cells hold one row per protein, so the floor of 3 dominates. Measured with `collect._read_test_rows(..., pairs_per_target=1, policy=True)`: **DAVIS random 372 rows → limit 3 = 0.8%; DAVIS cold-target 75 → 3 = 4.0%; KIBA random 228 → 3 = 1.3%; non-kinase panel 60 proteins → 3 = 5.0%** | not a wrong number: no cell is known to have lost more than a row or two (one BindingDB ligand of 322 atoms, per `collect.py`). But the stated protection is weaker than the plan's wording on small cells. Not changed here (existing DrugBAN analyses ran under this rule); the tests pin it as it is |

### Deviations from the task as written

1. **Two small code edits outside "tests + guards".** `train_moltrans.py` and `baseline_adapters.py`
   now wrap the vendored MolTrans import in `preserve_rng()`, because (a) as literally written cannot
   pass otherwise — MolTrans's `models.py` calls `torch.manual_seed(1)` / `np.random.seed(1)` at import
   and the probe showed the trainer path moving both. Effect on numbers: none expected — the trainer
   seeds *after* the import (unchanged), and a grep of `src/evaluation`/`src/data` for global-RNG use
   found only `torch.randint` in `--dummy` paths (`run_ladder.py:291`, `run_faithfulness.py:360`,
   `run_audit.py:243`). Not re-verified by retraining or re-running an analysis.
2. **`--n-trials` defaults changed**, which is the plan's requirement but is a behaviour change:
   `run_all` passes no `--n-trials`, so a fresh `run_all` now runs 10× the permutations of the ladder,
   control and positional steps (and 20× for the audit). Existing outputs are untouched; reproducing them
   now needs `--allow-low-permutations`.

### Risks / limits

* **DrugBAN cannot be tested on this machine.** No conda environment with DGL exists here
  (`conda env list` shows only `amazon_ml`), so the 5 skips are all DGL-gated: two in
  `test_drugban_adapter.py`, two in `test_drugban_import_isolation.py`, and the new DrugBAN tokeniser
  test. DrugBAN's `utils.py` imports `dgl` at the top, so even its pure-numpy tokeniser cannot be
  reached. Guards (a)–(c) for DrugBAN are therefore **unverified locally**; they run wherever DGL is
  installed.
* **The manifest fixes files as they are now.** It cannot show they are the files the 84 cells trained
  on. Indirect corroboration: all **84 of 84** `_results.json` have `n_train_rows` equal to their split's
  `train.csv` row count (checked in this session). That is row counts, not content.
* **The permutation floor cannot be applied retroactively.** The primary DAVIS audit is the
  500-permutation file; it now cannot be regenerated without `--allow-low-permutations`. Which run is
  primary is still T03's decision (ledger D5).
* **The RNG/weight-hash instruments are not yet used by the trainers.** `src/model/integrity.py`
  provides them, and the tests measure with them, but no trainer records a hash, so cells trained from
  now on would still carry no initial-weight record. Wiring that into the trainers is a training-code
  change; not done here.

### Discovered (work belonging to other tasks — not started)

1. **MolTrans's token-matched control fails to match when attention sits at the end of a sequence.**
   Measured on 6 real DAVIS proteins with a synthetic 10-residue block at the sequence end: the
   explanation's arm changes 1.2–3.0% of tokens; the closest random draw after `MAX_TRIES = 200` changes
   38–64% — outside `TOLERANCE = 0.10` in all 6. A mid-sequence block was matched in 4 of 6. This is
   inherent to the tokeniser, not a bug, and `token_matched_control` does report it (`tries == 200`) —
   **but the committed faithfulness JSONs record no match quality at all** (their keys are
   `comprehensiveness*`, `sufficiency*`, `aopc`, `n_pairs`, `k`, `explanation_is_load_bearing`). How many
   real MolTrans cells were unmatched is therefore unknown → T08 / T03 (a per-cell `matched_fraction`
   record, or a stated exclusion). The 6-protein probe used synthetic positions, not MolTrans's attention.
2. `run_faithfulness --n-random-trials` (default 5) and `token_faithfulness` average a masking control;
   they produce no p-value and are outside the permutation floor.
3. Vendored-repo commit hashes and the two missing licences (T01 gaps 2) are still open → T09 / T23.
4. T01 gap 15 (much of `results/` is gitignored, so `git diff` cannot detect a change to an untracked
   output) is **not** addressed: the manifest covers inputs (splits, ground truth), not the primary
   analysis outputs → decision below.

### Decisions needed from the user

1. **D6:** tighten the row-loss limit to a true 1% (which on a 60-protein panel means zero tolerated
   rows, so the one 322-atom BindingDB ligand would fail DrugBAN's control), keep it as is and reword
   the plan, or leave it for T18?
2. Extend the manifest to hash the primary analysis outputs under `results/analysis_*_policyA*` (they
   are untracked, so this is the only way a change to them becomes detectable)? Small, but outside T02's
   written scope.
3. Is there a machine with DGL where the DrugBAN tests can be run before T09, or should they wait for
   the cloud canary?


---

## T03 — Protocol amendment (DRAFTED, 2026-09-24; sign-off pending)

**Branch:** `remediation/T03`, from `remediation/T02` at `299f11b`, working tree clean at start.

**Status:** the document is written and committed but is **not in force**. The task's own gate is
the user's reply "approved"; until then T05 and every analysis that produces a new number stay
blocked. Not marked DONE for that reason.

**Files changed:** `docs/PROTOCOL_AMENDMENT_v2.md` (new), `docs/REMEDIATION_LEDGER.md` (this file).
No code, test, result or data file touched; `git diff --stat -- results` empty.

**Commands run** (read-only apart from writing the two documents): `git checkout -b remediation/T03`;
`cat`/`sed`/`grep -n` over `src/evaluation/{seed_agreement,run_audit,aggregate,significance_test,
bootstrap_ci,ladder_family,faithfulness,token_faithfulness,mask_comparability,run_all,run_ladder,
run_faithfulness,integrity,exclusions,positional_control,run_control}.py`; `python3 -c json.load` over
the primary audit JSONs (cell counts, seeds, k, significant cells, smallest p), one faithfulness JSON and
one control JSON (key structure only); a path-existence check over every path cited in the document;
`python3 -m pytest -p no:warnings`.

**Tests:** `1027 passed, 5 skipped in 125.50s (0:02:05)` — verbatim, default environment. Identical
to T02's `1027 passed, 5 skipped`; T03 adds no tests.

**Verification against T03's own criteria**

| criterion | result |
|---|---|
| declares primary families (unchanged originals) | P1 (DAVIS, 20 cells) and P2 (KIBA, 8), counts read from the committed JSON in this session; S1/S2 (IG, 12 and 4) declared as post-hoc secondary; S3 (KLIFS pocket, 16 and 6) declared |
| each extension family declared | E1 seeds 4–5; E2 DrugBAN-KIBA; E3 new methods; E4 conservation null; E5 modern model; E6 new splits; E7 non-kinase; E8 readouts (via addendum) |
| Holm scope per family | table in §3, every family alone, design `m` fixed |
| pre-specified k | k = 10, with the evidence it predates results (`docs/03_GUIDE_124AD0067.md:44,57,67`, first commit 2026-07-31) |
| effect-size / CI method | §5: 10,000 resamples, unit = target, 95 % percentile, `default_rng(0)`, from `bootstrap_ci.py:43–71` |
| canary tolerance | §6: ± one committed-seed SD, both metrics, same seed (the `amp_validation_davis.md` standard); inconclusive if SD = 0 |
| decision rules copied from code, path cited | §1 table (14 rules with `path:line`), §7 for extension seeds |
| references a code path for every rule | yes; a script confirmed every cited file exists (only the future `docs/method_applicability.md`, `docs/readout_sources.md`, `results/effects_v2/` are absent, by design) |
| committed with a timestamp | drafted 2026-09-24T03:42:16Z; the commit's own time is authoritative |
| no analysis run | none; only descriptive reads of committed outputs |

### Findings recorded in the amendment (repo wins)

* **G1 — `seed_agreement` returns nothing for 5 seeds.** `seed_agreement.py:38` keeps only cells with
  exactly 3 seeds and hard-codes 3 at `:47, :50`; feeding it seeds 1–5 would drop every cell silently.
  T12 must generalise it and reproduce `results/seed_agreement.md` exactly at n = 3 first.
* **G2 — no per-pair faithfulness values exist.** The faithfulness JSONs hold split-level means only, so a
  bootstrap CI for a faithfulness delta needs the faithfulness step re-run into `results/effects_v2/`.
  Pairs are also not one per protein (first 200 from the dataloader), so the CI must group by target.
* **G3 — KLIFS-pocket ladders, positional nulls and the non-kinase control were reported with uncorrected
  `p < 0.05`.** The amendment declares Holm families for the first and third (S3, E7) and keeps the
  positional nulls descriptive.
* **G4 — the "original" primary families grew after results were seen** (DAVIS 16 → 20, KIBA 6 → 8, on
  2026-09-19, superseded files on disk). The amendment freezes 20 and 8.
* **The "load-bearing" verdict is a sign test on a point estimate** (`faithfulness.py:251–253`): no
  interval, no p-value. A CI-qualified verdict is added beside it, not instead of it (D4).
* **No cell-level verdict beyond Holm significance is coded.** "Coarsely plausible" / "about 2× chance"
  are prose; the amendment says they are not decision rules.
* **No conservation data exists in the repo** (`grep` over `src/`, `data/*.json`), so E4 cannot start
  without a user-approved source (D6).
* **`CLAUDE.md`'s "KIBA family of 6" is superseded**; the committed KIBA audit has 8 cells.

### Decisions needed from the user

D1 permutation count of the primary record (default: the 10,000-permutation re-runs, originals kept);
D2 DrugBAN seeds 4–5 in E1 (default: decided by the T10 wave plan before any seed-4/5 result);
D3 Holm on S3 and E7; D4 CI-qualified faithfulness verdict; D5 extend the T02 manifest to primary
outputs (still open from T02); D6 conservation source; D7 canary rule; D8 one joint family for new
methods. Full table in the amendment §11.

### Discovered (work belonging to other tasks — not started)

1. `seed_agreement.py` generalisation to n seeds (G1) → T12.
2. Faithfulness re-run with per-pair values (G2) → T05.
3. The DAVIS IG Holm table is still uncommitted (T01 #7) → `ladder_family.py` over S1's 12 cells, T05/T08.
4. `CLAUDE.md` still says the KIBA Holm family is 6 and describes the older state → documentation, not touched.



---

## T04 — Predictive accuracy table (PARTIAL, 2026-09-24)

**Branch:** `remediation/T04`, from `remediation/T03` at `01712f1`, clean at start.

**Status:** the generator is written, tested and validated against real cells; only 18 of the 84 cells
(DeepDTA) have been predicted. The other 66 need inference that measured out at hours on this Mac, which
`ask-before-long-jobs` requires the user to approve, and 12 of them (DrugBAN) cannot run here at all.
No amendment sign-off gate applies to the accuracy table itself (T05 is the gated task); the
accuracy-vs-localization stage is described in the still-unsigned amendment §5, so it is written but **not
run** until the user replies "approved".

**Files changed:** `src/evaluation/accuracy_table.py` (new), `tests/test_accuracy_table.py` (new, 12 tests),
`.gitignore` (`!results/accuracy_v2/`, predictions dir stays ignored), this file. Uncommitted, untracked
outputs: `results/accuracy_v2/{cells_partial,by_model_partial,predictions_manifest_partial}.csv` and
`results/accuracy_v2/predictions/` (18 files, 2.5 MB) — the `_partial` suffix is deliberate; do not
commit them as the table.

**Design:** three stages. `predict` = one test pass per cell through `clean_accuracy._scores` (each
trainer's own dataset, encoding and `run_epoch`), writing per-row logits + a sidecar with the checkpoint
SHA-256; resumable. `tabulate` reads only the predictions files, verifies each against its recorded
SHA-256, refuses unless all 84 cells are present (`--allow-partial` writes `_partial` files), and writes
`cells.csv`, `by_model.csv` (mean ± sample SD, ddof=1, over 3 seeds), `predictions_manifest.csv`.
`localize` = Spearman(AUROC, precision@10 from the committed ladders) with a 10,000-resample percentile
bootstrap over cells (amendment §5). Cold-target and cold-pair on DAVIS get two rows: `uncorrected` and
`unseen_by_sequence`.

**Validation on real data**
* Decision rule for accuracy/MCC/F1 is the trainers' own (`train.py:107`, probability ≥ 0.5 = logit ≥ 0).
  Probed on DAVIS cold-pair seed 1, four models, accuracy vs recorded `_results.json`: DeepDTA 0.9423/0.9423,
  ColdSite-DTI 0.9371/0.9371, HyperAttentionDTI 0.9449/0.9449, MolTrans 0.9379 (CPU) and 0.9371 (MPS) /
  0.9379 — MolTrans differs by dropout, as `clean_accuracy.py:DROPOUT_DRAWS` documents.
* All 18 DeepDTA cells: AUROC, AUPR and accuracy reproduce the recorded values within 0.005
  (`tabulate` output: `18 cells (66 missing); 18 reproduce their recorded AUROC/AUPR/accuracy, 0 do not`).
* DeepDTA's unseen-by-sequence view matches `results/clean_accuracy_davis.md` exactly (cold-target seed 1:
  0.8801, 5168 of 5984 rows; cold-pair seed 1: 0.7989, 1001 of 1144).
* DeepDTA seed means (`by_model_partial.csv`): DAVIS random 0.9290, cold-drug 0.6915, cold-target 0.9074,
  cold-pair 0.7277 — consistent with the figures in CLAUDE.md §3.

**Tests:** `1039 passed, 5 skipped in 124.52s (0:02:04)` — verbatim. T03: `1027 passed, 5 skipped`;
+12 passed = the 12 new tests; skipped unchanged (DGL-gated).

### Measured cost of the remaining 66 cells (this Mac, MPS; CPU is 2–3× slower)
Per-row rates from two probes (DAVIS cold-pair, 1,144 rows; DAVIS random, 6,011 rows), fixed load time
subtracted: ColdSite-DTI ≈ 0.027, HyperAttentionDTI ≈ 0.025, MolTrans ≈ 0.026 s/row for **one** pass.
Rows to score per model across all its cells: DAVIS 3 × 18,885 = 56,655; KIBA 3 × 46,025 = 138,075 →
194,730 (`wc -l` of the test CSVs). ⇒ **≈ 4 h per model, ≈ 12–13 h for the three, one pass each** (an
extrapolation from those two probes, not a measured full run). MolTrans with `clean_accuracy`'s 5 dropout
draws would be ≈ 5× its share.

### Findings / deviations (repo wins)
1. **DeepDTA has no regression checkpoint.** All 18 DeepDTA cells are binary, so MSE / CI / Pearson /
   Spearman cannot be computed from any file on disk (T01 finding 5). They are reported as classifiers.
   The regression table at `results.md:3–18` remains without a per-cell file.
2. **DrugBAN's 12 DAVIS cells cannot be predicted here** — `import dgl` fails
   (`ModuleNotFoundError: No module named 'dgl'`), as in T02. Their AUROC/AUPR/accuracy exist in
   `_results.json`, but MCC/F1 do not, and no predictions file can be produced without DGL.
3. **The plan's "84 cells" split is 18 DeepDTA + 66 others** (12 of which are the DrugBAN cells above).

### Decisions needed from the user
1. How to run the 54 ColdSite-DTI / HyperAttentionDTI / MolTrans cells' inference — see the HALT REPORT.
2. Where DrugBAN's 12 cells can be scored (a machine with DGL — the T02 question, still open).
3. MolTrans: one seeded pass per cell (default here; reproduces the recorded value to ~0.001) or the
   5-draw mean `clean_accuracy` used.


### T04 addendum — DAVIS run (2026-09-24, after the user's decisions)

**User decisions:** (1) DAVIS only, on this Mac's GPU (MPS); KIBA later. (2) DrugBAN: no DGL install; AUROC / AUPR /
accuracy from the recorded files, MCC and F1 empty for those 12 cells. (3) MolTrans: one pass per cell. (4) T03
amendment approved (`docs/PROTOCOL_AMENDMENT_v2.md` §12) and `localize` cleared.

**Run:** `predict --datasets davis --models coldsite_dti,hyperattentiondti,moltrans --device mps` → 36/36 `ok`, 0 FAIL
(`results/accuracy_v2/predict_davis.log`). `tabulate --datasets davis` → `60 cells (0 missing); 46 reproduce their
recorded AUROC/AUPR/accuracy, 2 do not: ['moltrans cold_pair s1', 'moltrans cold_pair s2']; 18 rows are recorded-only
(no MCC/F1)`. (The 12 DeepDTA cells were predicted earlier; DrugBAN rows have no `reproduces_recorded` — they *are* the
recorded values.)

**The 2 non-reproductions:** MolTrans cold-pair seed 1 (AUROC +0.0064) and seed 2 (−0.0076) against a tolerance of 0.005.
All other MolTrans cells are within 0.0007; ColdSite-DTI, HyperAttentionDTI and DeepDTA reproduce to ≤ 1e-4. Cause
established below (§ "MolTrans cold-pair investigation"); the earlier "consistent with dropout" wording is superseded.

**Localize (amendment §5, description only, 10,000 resamples; 48 cells with a ladder, 12 per attention model):**
Spearman ρ between a cell's AUROC and its precision@10 — pooled 0.033 [−0.276, 0.322]; ColdSite-DTI 0.119
[−0.607, 0.701]; HyperAttentionDTI 0.462 [−0.219, 0.884]; MolTrans −0.081 [−0.663, 0.568]; DrugBAN −0.690
[−0.863, −0.177] (`localization_spearman_davis.csv`). Every interval but DrugBAN's contains 0. Caveats: cells pool the four
levels and three seeds, so the cells are not independent; DrugBAN's AUROC for cold-target/cold-pair is the
unseen-by-sequence value from `clean_accuracy_davis.json`, the others' from their predictions. Not a significance claim.

**Still open:** KIBA's 24 cells (`predict --datasets kiba`; ~3.9× the DAVIS rows); the full 84-cell `cells.csv`;
DrugBAN MCC/F1 (needs DGL; user declined); MolTrans 5-draw check if the two misses matter for the paper.
`_partial` files from the earlier 18-cell probe remain untracked and superseded.


### T04 addendum — MolTrans cold-pair investigation (2026-09-24)

**Question:** why do DAVIS cold-pair seeds 1 and 2 of MolTrans miss their recorded AUROC by 0.0064 / −0.0076?
**Nothing recorded was changed.** Script `results/accuracy_v2/investigate_moltrans_coldpair.py`, raw output
`moltrans_coldpair_investigation.{json,log}`; it scores through the pipeline's own `clean_accuracy._scores`.

**Code facts (read, then checked on a built model):** `train_moltrans.run_epoch` calls `model.train(False)` for any
pass without an optimizer, so eval mode is on for the recorded test pass and for ours: all 14 `nn.Dropout` modules
inactive, both `BatchNorm1d` on running statistics. The one exception is `baselines/MolTrans/models.py:103`,
`F.dropout(i_v, p=self.dropout_rate)` (p = 0.1) — the functional call has no `training=` argument, defaults to
`True`, and is live in eval. The checkpoint config equals the trainer's (`BIN_config_DBPE`, batch 16); test rows are
in file order, no `drop_last`. The trainer does not seed before its test pass, so the recorded draw depends on the RNG
state (Kaggle CUDA) after training and cannot be regenerated from `--seed`.

**Experiment (MPS unless stated; 1,144 test rows; `Δ` = value − recorded):**

| arm | passes | seed 1 AUROC (recorded 0.58988) | seed 2 AUROC (recorded 0.56739) |
|---|---|---|---|
| current pipeline (`manual_seed(cell seed)`) | 2 | 0.59625 both (Δ +0.0064) | 0.55975 both (Δ −0.0076) |
| every dropout forced off (`F.dropout(training=False)`) | 2 MPS + 1 CPU | 0.59247 ×3 (Δ +0.0026) | 0.56320 ×3 (Δ −0.0042) |
| dropout on, seeds 1000–1005 | 6 | 0.5862–0.5954, mean 0.5912, sd 0.0042 | 0.5513–0.5712, mean 0.5616, sd 0.0068 |

* **Determinism / MPS:** the current arm repeats exactly, and matches the saved predictions to |Δlogit| = 5e-9. With dropout
  off, MPS and CPU agree to 1e-6 in logit and to six decimals in AUROC, AUPR and accuracy. So MPS non-determinism, device
  numerics, preprocessing and checkpoint loading are **not** sources here.
* **Dropout:** with the one live `F.dropout` on, AUROC moves by sd 0.004–0.007 between seeds (accuracy by ≤ 0.002); with it
  off there is no variation. The recorded value lies inside the 6-draw range for both cells, which is the repo's own
  reproduction criterion (`clean_accuracy.py`: recorded within the passes' range). The recorded values are 0.3 (seed 1) and
  0.8 (seed 2) sd from the 6-draw means; our two flagged single draws are 1.2 and 0.3 sd from them.
* **Not shown:** which draw the Kaggle run made cannot be reproduced; that the *only* live stochastic op is `models.py:103`
  rests on the code read and the module check, not on an ablation of each layer separately.

**Verdict:** the discrepancy is single-draw dropout noise from MolTrans's functional dropout, as published; there is no
inference bug. **No change to the DAVIS table is required.** Its MolTrans cold-pair AUROCs are single draws with
sd ≈ 0.004–0.007; treat differences of that size between MolTrans cells (or against its recorded value) as noise. The
`reproduces_recorded` flag uses a fixed 0.005 tolerance, which is tighter than this model's own draw-to-draw noise on
1,144 rows; it stays as is (flag = 2 `False` cells, documented here). **No code was changed**; a range-based flag for
MolTrans is a possible later refinement, not made.


### T04 addendum — KIBA run and completion (2026-09-24/25)

**Run:** `predict --datasets kiba --models coldsite_dti,hyperattentiondti,moltrans --device mps` → 18/18 `ok`, 0 FAIL
(`results/accuracy_v2/predict_kiba.log`); DeepDTA's 6 KIBA cells were predicted earlier. Protocol unchanged (one pass per
cell, MolTrans included). The Mac slept between checks (86 min CPU in 6.4 h wall), so wall-clock gaps in the log are not
per-cell costs; `caffeinate -w <pid>` was attached for the last cell.
`tabulate --datasets kiba` → `24 cells (0 missing); 24 reproduce their recorded AUROC/AUPR/accuracy, 0 do not`. Full
`tabulate` → `84 cells (0 missing); 70 reproduce ..., 2 do not: ['moltrans cold_pair s1', 'moltrans cold_pair s2']; 18 rows
are recorded-only (no MCC/F1)` — the same 2 DAVIS cells explained in the MolTrans investigation above; the other 12 of the
84 are the DrugBAN cells, which have no reproduction flag. The DAVIS rows of `cells.csv` equal `cells_davis.csv` exactly.

**KIBA seed means (AUROC ± SD, n = 3):** random — ColdSite-DTI 0.8945 ± 0.0113, HyperAttentionDTI 0.9331 ± 0.0006,
MolTrans 0.9187 ± 0.0040, DeepDTA 0.9178 ± 0.0006; cold-drug — 0.8069 ± 0.0087, 0.8444 ± 0.0039, 0.8120 ± 0.0090,
0.8324 ± 0.0015 (`by_model_kiba.csv`, which also holds AUPR, accuracy, MCC, F1).

**Localize, KIBA (18 cells, 6 per attention model):** Spearman ρ(AUROC, precision@10) pooled 0.064 [−0.466, 0.557];
ColdSite-DTI 0.257 [−1.000, 1.000]; HyperAttentionDTI −0.371 [−1.000, 0.800]; MolTrans −0.371 [−1.000, 0.806]. With 6
cells per model these intervals span nearly the whole range and say nothing about a relationship; only the pooled row is
informative, and it contains 0. Same caveats as DAVIS (levels and seeds pooled, not independent; a description, not a test).
Pooled over both datasets: 66 cells with a ladder (`localization_spearman.csv`).

**Bug found and fixed in `spearman_bootstrap` (`accuracy_table.py`):** the constant-resample guard was `std() > 0`. A
resample that draws one cell six times has a float `std` of 1.1e-16 although its values are identical, so it passed the
guard, `spearmanr` returned NaN, and MolTrans/KIBA's precision@10 interval came out `NaN`. The guard is now the exact range
`np.ptp() > 0`. Regression test `test_spearman_bootstrap_ci_is_finite_when_a_resample_repeats_one_inexact_float` uses the
six real MolTrans/KIBA values and **fails on the old code** (`isfinite(nan)`), passes on the fix. Effect of the fix, checked
by diff against the pre-fix outputs: exactly one number changed (MolTrans KIBA precision@10: NaN → [−1.000, 0.806], one
degenerate resample now counted); `localization_spearman_davis.csv` is byte-identical, and the pooled file has no NaN.

**Tests:** `python3 -m pytest -p no:warnings` → `1044 passed, 5 skipped in 65.22s` (verbatim; the 5 skips are DGL-gated).

**Decisions (user, 2026-09-25):** the three superseded `*_partial.csv` files were deleted (untracked; regenerable). No
range-based reproduction flag for MolTrans will be built; the inference-dropout quirk (`models.py:103`, single-draw test
metrics, the two flagged cold-pair cells) is to be stated in the paper text instead.

**Not done / open:** DrugBAN MCC/F1 (no DGL; user declined). Paper text on MolTrans's inference dropout is still to be
written (Methods, `paper/methods_data_and_evaluation.md`, and a note under the accuracy table).


---

## T05 — Effect sizes & confidence intervals (PARTIAL, 2026-09-25)

**Branch:** `remediation/T05`, from `remediation/T04` at `a936019`, clean at start. **Gate:** T03 amendment approved
(`docs/PROTOCOL_AMENDMENT_v2.md` §12, 2026-09-24); every method below is the amendment's §5.

**Status:** the enrichment CIs, the seed-spread table and the reproduction of the original verdicts are done. The
faithfulness-delta CIs (amendment §5, gap G2) need per-pair values that no file holds, so the faithfulness step must be
re-run; the recorder is written, tested and validated on one real cell, but the run itself is ~3 h on the user's Mac
(`ask-before-long-jobs`) and has **not** been started beyond the probes below.

**Files changed**

| file | kind | what |
|---|---|---|
| `src/evaluation/effects_v2.py` | new | enrichment-over-chance CIs, seed spread, faithfulness-delta CIs by target, and the reproduction checks; writes only to `results/effects_v2/` |
| `tests/test_effects_v2.py` | new | 26 tests (planted enrichment, planted null, planted signal, resampling = `bootstrap_ci`'s, grouping by target, recorder leaves means untouched, verdict reproduction catches a forged flag) |
| `src/evaluation/faithfulness.py`, `run_faithfulness.py`, `token_faithfulness.py` | edit | opt-in `--record-pairs`: adds `per_pair` (target id + deltas) to the JSON. Off by default; with it on, every mean and the verdict are identical (tested, and measured on a real cell below) |
| `.gitignore` | edit | `!results/effects_v2/` (figures stay ignored) |
| `results/effects_v2/` | new | `enrichment.{csv,md}`, `seed_spread.{csv,md}`, `reproduction.json`, `inputs_sha256.json` (SHA-256 of every ladder read), `problems.txt` |

**Method (all from amendment §5):** resampling unit = target; 10,000 resamples (the CLI refuses fewer); 95 % percentile;
`default_rng(0)`; a resampled protein carries all its seeds, averaged. Enrichment = mean precision@10 ÷ mean chance over the
**same** resampled proteins. Per-protein chance = (annotated sites inside the window) ÷ (window = min(sequence length, 1000)),
the exact expectation of the permutation null (`significance_test._chance_precision_sample`; tested). Rebuilt because the
ladders store only the cell mean.

### Verification against T05's own criteria

| criterion | result (source) |
|---|---|
| CIs are computed via the amendment's method | the precision interval reproduces the committed `results/ci_davis.json` for **24 of 24** cells to 1e-12 in n, mean, low and high (script comparison run this session); the resampling is `bootstrap_ci`'s, and a test pins that |
| the rebuilt per-protein chance is right | for **68 of 68** cells the mean of the rebuilt chance is within 5 Monte-Carlo standard errors of the recorded cell chance (tolerance built for 500 permutations, the smallest count any ladder used); largest difference 0.00076 (`enrichment.csv`, `chance_check_*`) |
| original verdicts reproduce exactly | `results/effects_v2/reproduction.json`: **(1)** `seed_agreement.py` regenerated from the ladders is **byte-identical** to `results/seed_agreement.md` — 12 of 22 cells disagree, 21 of 22 spread > signal; my own spread table gives the same 21 of 22 (15 of 16 DAVIS, 6 of 6 KIBA); **(2)** Holm re-applied to the recorded raw p-values of all four audit files (500- and 10,000-permutation, DAVIS 20 cells and KIBA 8) gives the recorded verdicts exactly — one significant cell, `hyperattentiondti|davis|random`, in both DAVIS files, none on KIBA; **(3)** all 84 faithfulness level summaries: recorded `load_bearing` equals the sign of the recorded delta, 63 load-bearing, 0 disagreements |
| no original result changed | SHA-256 of all 774 non-checkpoint files under `results/` (excluding `effects_v2/`) taken before and after: identical (`diff` empty); `git diff --stat -- results` empty |
| permutations ≥ 10,000 / Holm families | no permutation test or Holm family was created; the amendment's P1/S1–S3 are unchanged; nothing here is a p-value |
| tests | `1070 passed, 5 skipped in 131.50s` — before `1044 passed, 5 skipped`; +26 passed = the 26 new tests; skips unchanged (DGL-gated) |

**Mutation checks** (each reverted): independent draws for the chance mean → the exact-2× test fails; grouping by pair instead
of target → the grouping test fails; dropping `abs` in the spread rule → a below-chance test case fails (that case was added
after the first mutation slipped through).

### What the enrichment table says (descriptive; `results/effects_v2/enrichment.md`)

Cells whose 95 % enrichment interval lies entirely above 1 / covers 1 / lies entirely below 1 — P1 (UniProt, DAVIS, 16 model
cells): 4 / 10 / 2; P2 (UniProt, KIBA, 6): 3 / 2 / 1; S3-D (KLIFS pocket, DAVIS, 16): 11 / 5 / 0; S3-K (KIBA, 6): 4 / 2 / 0.
These are counts of intervals, not tests, and no threshold is applied (amendment §5). ColdSite-DTI against the KLIFS pocket:
1.54 [1.48, 1.59] warm, 2.10 [2.02, 2.18] cold-drug, 1.74 [1.58, 1.90] cold-target, 1.99 [1.78, 2.21] cold-pair; DrugBAN's
KLIFS intervals all include 1 (0.98–1.04 point values). Ceiling is ≈ 0.99 (UniProt) and ≈ 1.0 (KLIFS) throughout.

### Faithfulness-delta CIs — what was done and what is needed

* **Recorder.** `--record-pairs` on `run_faithfulness` and `token_faithfulness` stores each scored pair's target id and deltas.
* **Real-cell validation (probe, scratch dir, not in `results/`).** ColdSite-DTI, DAVIS, seed 1, all four levels, CPU: the
  re-run reproduced the committed `faithfulness_davis_seed1.json` with a **difference of exactly 0.0** in all five means at all
  four levels, and the mean of the recorded pairs equals the summary delta exactly. Wall time **15 min 31 s** (1678.7 s user
  CPU; the machine was also running the test suite). HyperAttentionDTI (MPS) and MolTrans token space (MPS) were run on 4
  pairs only, to confirm the ids come through; their reproduction of the committed means is **not yet checked** — the
  effects table records `max_abs_diff_vs_committed` per cell so it will be visible.
* **Grouping matters.** The 200 first pairs cover only 162 / 200 / 76 / 76 distinct targets (warm / cold-drug / cold-target /
  cold-pair, ColdSite-DTI seed 1), so a pair-level interval would be too narrow; the CI resamples targets and reports the
  pair-level mean beside the target-level one (e.g. warm 0.8014 by pair, 0.8513 by target, CI 0.6239–1.1038).
* **Estimated cost of the rest (estimate, not measured here):** file-time gaps between the committed outputs put the DAVIS
  ColdSite-DTI runs at 18–20 min each, MolTrans token space at ≈ 4 min, and the analogous MolTrans residue-space run at
  27–33 min; HyperAttentionDTI's own time is not recoverable (its three files share one timestamp). For 15 files —
  ColdSite-DTI DAVIS ×3, HyperAttentionDTI DAVIS ×3 and KIBA ×3, MolTrans token DAVIS ×3 and KIBA ×3 — that is **≈ 3–3.5 h
  on CPU**, dominated by HyperAttentionDTI. Device would be `cpu`, the device of the original DAVIS run (`run_all.log`).

### Gaps and deviations (repo wins)

1. **DrugBAN faithfulness cannot be re-run here** (no DGL, as in T02/T04): its 12 DAVIS cells get enrichment CIs but no
   faithfulness CI until a machine with DGL runs `run_faithfulness --model drugban --record-pairs`.
2. **The uniform-control arm has no ladder**, so the 20-cell P1 family gives 16 enrichment rows (KIBA 8 → 6). The control's
   enrichment is 1 by construction and is not tabulated.
3. **8 of the 12 S1 integrated-gradient cells (HyperAttentionDTI-IG, DAVIS, UniProt and KLIFS ladders) have no CI:** those
   ladder files predate per-protein scores (`problems.txt`, 24 file-level lines). Re-scoring them means re-running that IG ladder, which is not a T05 step; noted, not done. S1 has 12 declared cells, 8 appear in the table.
4. **MolTrans's faithfulness is taken in token space** (size-matched arms), as `paper/results.md` does; its residue-space arm
   (48 % vs 95 % of tokens changed) is not resampled.
5. The seed-spread `exact` column repeats rule 1.13 with the rebuilt chance; the two agree cell by cell on all 22 UniProt cells (0 differ, `seed_spread.csv`).
6. The faithfulness verdict by interval ("lower bound > 0") is implemented and tested but has **no real value yet** — no file
   contains one until the re-run finishes.

### Commands run

`git checkout -b remediation/T05`; `python3 -m src.evaluation.effects_v2 --out-dir results/effects_v2`; the ci_davis
comparison script; `python3 -m src.evaluation.run_faithfulness --model coldsite_dti --task binary --dataset davis --seed 1 ...
--out-dir <scratch> --device cpu --record-pairs` (15:31); the same for `--model hyperattentiondti --device mps --max-pairs 4`
and `python3 -m src.evaluation.token_faithfulness --model moltrans ... --max-pairs 4 --device mps --record-pairs`; SHA-256
snapshots of `results/` before and after; `python3 -m pytest -p no:warnings`.

### To finish T05 (needs the user's yes — multi-hour on the Mac, ~3–3.5 h estimated)

```bash
cd ~/Documents/ColdSite-DTI && caffeinate -i bash -c '
OUT=results/effects_v2/faithfulness; D=~/ColdSite-results/davis_binary; K=~/ColdSite-results/kiba_binary
for s in 1 2 3; do
  python3 -u -m src.evaluation.run_faithfulness --model coldsite_dti --task binary --dataset davis --seed $s --split-root data/splits --checkpoint-dir $D --results-dir $D --out-dir $OUT --max-pairs 200 --device cpu --record-pairs
  python3 -u -m src.evaluation.run_faithfulness --model hyperattentiondti --task binary --dataset davis --seed $s --split-root data/splits --checkpoint-dir $D --results-dir $D --out-dir $OUT --max-pairs 200 --device cpu --record-pairs
  python3 -u -m src.evaluation.token_faithfulness --model moltrans --dataset davis --seed $s --checkpoint-dir $D --out-dir $OUT --max-pairs 200 --device cpu --record-pairs
  python3 -u -m src.evaluation.run_faithfulness --model hyperattentiondti --task binary --dataset kiba --seed $s --split-root data/splits --checkpoint-dir $K --results-dir $K --out-dir $OUT --max-pairs 200 --device cpu --record-pairs
  python3 -u -m src.evaluation.token_faithfulness --model moltrans --dataset kiba --seed $s --checkpoint-dir $K --out-dir $OUT --max-pairs 200 --device cpu --record-pairs
done'
python3 -m src.evaluation.effects_v2 --out-dir results/effects_v2
```

The last line reads `results/effects_v2/faithfulness/`, writes `faithfulness_effects.{csv,md}` and compares every re-run with
the committed file of the same name (`max_abs_diff_vs_committed`). `run_faithfulness` also writes an `accuracy_*.json` and a
`.png` into that folder; both are harmless (the png is git-ignored).


### Faithfulness re-run (2026-09-25, after the user's go-ahead)

**Run:** the 15 (model, dataset, seed) files of "To finish T05", exactly as written above (CPU, `caffeinate -i`, `--max-pairs 200
--record-pairs`), output `results/effects_v2/faithfulness/`. Driver and log were in the session scratchpad
(`run15.sh`, `run15.log`), not the repo. 15 of 15 `rc=0`; started 02:03:10, ended 04:04:16 (**2 h 1 min**, against the 3–3.5 h estimate).
KIBA HyperAttentionDTI seed 2 took 20 min 39 s (03:14:51–03:35:30) against ~5.5 min for seed 1 and ~3 min for seed 3; the cause was
not investigated (nothing else was using the CPU; the process showed 258 % CPU with 6 min 43 s CPU time at 17 min wall). A first
launch was killed after ~30 s and relaunched detached with the output folder cleared, so every file is from the second launch.
Then `python3 -m src.evaluation.effects_v2 --out-dir results/effects_v2` → `faithfulness_effects.{csv,md}` (16 rows = 4 levels
ColdSite-DTI, HyperAttentionDTI and MolTrans on DAVIS; 2 levels HyperAttentionDTI and MolTrans on KIBA).

**Result (comprehensiveness delta over the random-masking control, mean over seeds by target, 95 % percentile CI, 10,000 resamples,
`faithfulness_effects.csv`):** every one of the 16 intervals lies above 0 (`load_bearing_ci` True in all 16, and the sign test agrees).
ColdSite-DTI DAVIS 1.210 [1.054, 1.378] random, 1.179 [1.054, 1.306] cold-drug, 0.516 [0.442, 0.599] cold-target, 0.661 [0.576, 0.756]
cold-pair; HyperAttentionDTI DAVIS 0.189 [0.149, 0.233], 0.114 [0.087, 0.140], 0.065 [0.046, 0.084], 0.054 [0.036, 0.073], KIBA 0.135
[0.105, 0.166], 0.087 [0.067, 0.110]; MolTrans DAVIS 0.461 [0.409, 0.516], 0.356 [0.316, 0.396], 0.336 [0.274, 0.410], 0.258 [0.212,
0.307], KIBA 0.392 [0.347, 0.441], 0.302 [0.268, 0.339]. Targets resampled: 162 / 200 / 76 / 76 for ColdSite-DTI and HyperAttentionDTI DAVIS,
114 / 112 for HyperAttentionDTI KIBA, 200 / 200 / 75 / 76 and 200 / 200 for MolTrans. Descriptive intervals, not tests.

**Reproduction of the committed faithfulness files (`max_abs_diff_vs_committed`, worst seed):**
| model, dataset | worst difference | reading |
|---|---|---|
| ColdSite-DTI, DAVIS (12 seed-levels) | **0.0** at all four levels | exact, as in the earlier one-cell probe |
| HyperAttentionDTI, KIBA (6) | **0.0** | exact |
| HyperAttentionDTI, DAVIS (12) | 1.2e-7 – 2.6e-3 | not bit-identical; cause not investigated |
| MolTrans, DAVIS (12) | 1.2e-2 – 4.0e-2 | differs in both directions (per seed-level −0.040 to +0.018 against recorded deltas of 0.09–0.77) |
| MolTrans, KIBA (6) | 1.9e-2 – 2.3e-2 | differs in both directions |

Every re-run's own per-pair mean equals its own summary delta (`max_abs_diff_vs_summary` ≤ 2.2e-16 in all 16 rows), so the recorder is
consistent; the difference is between the re-run and the committed file. **Verdicts:** the sign of the delta is the same as the committed
file in all 48 seed-levels compared (0 changes). **Likely cause for MolTrans, not tested here:** the vendored model's functional
dropout (`baselines/MolTrans/models.py:103`) is live at inference (T04 investigation), so each run is a different draw; that would
explain differences that are small, signed both ways and confined to the one model with the quirk. It is not shown that this is the
whole explanation, and the HyperAttentionDTI-DAVIS differences are unexplained. The intervals above are therefore for **this re-run's draw**;
the committed means (`results/analysis_*_policyA/`) stay the primary record and were not changed. The two sets should be reported
together, or the MolTrans row footnoted, until this is settled.

**Checks:** no file outside `results/effects_v2/` is newer than the run start (`find results -newermt '2026-09-25 01:50:00' -not -path
'results/effects_v2/*'` empty); `git diff --stat -- results` shows only `results/effects_v2/inputs_sha256.json` (regenerated: it now also
lists the 15 re-run files). `python3 -m pytest -p no:warnings` → `1070 passed, 5 skipped in 134.61s (0:02:14)`, unchanged from the
earlier T05 run.

**Still open (why T05 stays PARTIAL):** (1) DrugBAN faithfulness CIs — needs DGL (T02/T04 question, unanswered); (2) the 8
HyperAttentionDTI-IG cells; (3) whether to report the MolTrans/HyperAttentionDTI-DAVIS faithfulness intervals as they are, with the
non-reproduction stated, or to investigate first (e.g. a fixed-seed or dropout-off re-run of one MolTrans cell, ~4 min, as T04 did).

**Files added:** `results/effects_v2/faithfulness/` (15 `faithfulness_*`/`token_faithfulness_*` JSON + MD, 9 `accuracy_*.json`; PNGs are
git-ignored), `results/effects_v2/faithfulness_effects.{csv,md}`.
