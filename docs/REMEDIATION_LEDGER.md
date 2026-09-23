# Remediation ledger

Plan: `docs/REMEDIATION_PLAN.md` (saved verbatim, T00). One task per session; a task is DONE
only when its verification criteria pass, the ledger is updated, and the HALT REPORT is printed.

## Task status

| Task | Title | Status | Branch | Notes |
|---|---|---|---|---|
| T00 | Bootstrap | **DONE** (2026-09-24) | remediation/T00 | plan + ledger written; suite run; layout verified; 5 discrepancies recorded below |
| T01 | Inventory | PENDING | | needs the checkpoint dir from the user (default `~/ColdSite-results/`) |
| T02 | Integrity guards | PENDING | | includes `data/splits/MANIFEST.json`, absent today (D2) |
| T03 | Protocol amendment | PENDING | | user sign-off required before T05 or any new analysis |
| T04 | Predictive accuracy table | PENDING | | |
| T05 | Effect sizes & CIs | PENDING | | requires T03 approval |
| T06 | Conservation null + intermediate rung | PENDING | | |
| T07 | Readout primacy | PENDING | | blocked until the user supplies per-model figure references |
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
