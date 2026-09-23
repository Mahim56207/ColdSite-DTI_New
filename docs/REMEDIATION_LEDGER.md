# Remediation ledger

Plan: `docs/REMEDIATION_PLAN.md` (saved verbatim, T00). One task per session; a task is DONE
only when its verification criteria pass, the ledger is updated, and the HALT REPORT is printed.

## Task status

| Task | Title | Status | Branch | Notes |
|---|---|---|---|---|
| T00 | Bootstrap | **DONE** (2026-09-24) | remediation/T00 | plan + ledger written; suite run; layout verified; 5 discrepancies recorded below |
| T01 | Inventory | **DONE** (2026-09-24) | remediation/T01 | `docs/inventory.md` written; checkpoint dir `~/ColdSite-results/` confirmed by the user; 15 gaps recorded below |
| T02 | Integrity guards | PENDING | | `data/splits/MANIFEST.json` absent (D2) **and the splits are gitignored**; no vendored commit hashes exist to hash against; no RNG/initial-weight record can be recovered for the existing 84 cells |
| T03 | Protocol amendment | PENDING | | user sign-off required before T05 or any new analysis |
| T04 | Predictive accuracy table | PENDING | | no saved predictions and no MCC/F1 anywhere — inference over all 84 checkpoints is required; DeepDTA was trained **binary**, so the task's regression-metric list needs a user decision |
| T05 | Effect sizes & CIs | PENDING | | requires T03 approval. `src/evaluation/bootstrap_ci.py` + `results/ci_davis.json` already give protein-resampled 10,000-draw CIs for 24 DAVIS cells — T05 extends this, it is not new work |
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
