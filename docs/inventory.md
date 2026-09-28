# T01 — Inventory: every report claim mapped to the file that evidences it

**Task:** T01 of `docs/REMEDIATION_PLAN.md`. **Branch:** `remediation/T01`.
**Date:** 2026-09-24. **Checkpoint directory (confirmed by the user):** `~/ColdSite-results/`.

**What this document is.** A map from each claim in the author's project report
(`paper/PROJECT_REPORT.md`, dated 20 September 2026 — the source of the plan's
`project_facts`) and from each item the plan's task pipeline presupposes, to the file that
evidences it. Nothing here was computed: every entry is a listing, a file read, or a value
read out of an existing JSON/Markdown output. Where a p-value is quoted the permutation
count is inferred from its denominator (a permutation p is `(hits+1)/(n+1)`), which is a
reading of the file's resolution, not a new statistic.

**Marks.** `VERIFIED` = the file exists and says what the claim says. `PARTIAL` = evidence
exists but does not cover the claim's full scope. `NOT FOUND` = no file in the repo or the
checkpoint directory evidences it. `CONTRADICTED` = a file says something different.

---

## 1. Trained cells and checkpoint locations

`~/ColdSite-results/` holds **5.0 GB**; the 84 cell checkpoints themselves are **4.6 GB**
(`du -ch davis_binary/*.pt kiba_binary/*.pt`).

| claim | evidence | mark |
|---|---|---|
| 84 trained cells | `~/ColdSite-results/davis_binary/` 60 `.pt` + 60 `_results.json`; `~/ColdSite-results/kiba_binary/` 24 `.pt` + 24 `_results.json` | **VERIFIED** |
| DAVIS 60 = 5 models × 4 levels × 3 seeds | file names in `davis_binary/`: `coldsite_dti_davis_<level>_binary_seed<N>[_deepdta\|_drugban\|_hyperattentiondti\|_moltrans].pt`, all 60 present | **VERIFIED** |
| KIBA 24 = 4 models × 2 levels × 3 seeds, no DrugBAN | `kiba_binary/` has `random` and `cold_drug` only, and no `*_drugban.pt` | **VERIFIED** (P08 confirmed: DrugBAN absent from KIBA) |
| checkpoints total 4.6 GB | `du -ch` above | **VERIFIED** |
| checkpoints live outside the repo | both directories are under `~/ColdSite-results/`, not under the working tree | **VERIFIED** |

**Naming.** Checkpoint: `coldsite_dti_<dataset>_<level>_binary_seed<N>[_<model>].pt` — the
`coldsite_dti_` prefix is on *every* model's file, including DeepDTA's. Results JSON:
`<dataset>_<level>_binary_seed<N>[_<model>]_results.json` (no prefix). Resolved by
`src/model/checkpoint_naming.py`.

**Per-model checkpoint size** (DAVIS random seed 1): ColdSite-DTI 2.4 M, DrugBAN 4.1 M,
DeepDTA 7.4 M, HyperAttentionDTI 8.8 M, **MolTrans 240 M**. MolTrans is ~97% of the 4.6 GB
— relevant to T23's hosting plan.

### Cells beyond the 84, in the same directory

| directory | contents | status |
|---|---|---|
| `~/ColdSite-results/leakage_davis/results/` | **12 further DeepDTA cells** on the leakage-variant splits: `cold_target_seqclean`, `cold_target_seqmatched`, `cold_pair_seqclean`, `cold_pair_seqmatched` × 3 seeds, 89 MB | real cells, **not counted in the 84** — resolves ledger D4 |
| `~/ColdSite-results/drugban_davis/` | 12 `.pt` + 12 `_results.json`, byte-identical to the DrugBAN cells in `davis_binary/` (`shasum -a 256` matches on the random-seed-1 file; all 12 match on size) | duplicate staging copy |
| `~/ColdSite-results/staging_acct1_v3/results/` | 36 `.pt`, every one the same size as its `davis_binary/` twin, plus `accuracy_davis_seed{1,2,3}.json` and an `audit_davis_binary.{json,md}` | duplicate staging copy |
| `~/ColdSite-results/partial_do_not_use/` | 1 `.pt` (HyperAttentionDTI DAVIS cold-drug seed 1) with **no** `_results.json` | correctly quarantined per CLAUDE.md §6 |

### What a `_results.json` records — and what it does not

Every one of the 84 carries: `tag, dataset, split, task, seed, checkpoint, best_epoch,
selection{best_epoch,best_val_loss,min_epochs,min_epochs_applied}, n_train_rows,
test_metrics{auroc, auprc, accuracy}`. Model-specific extras appear on some
(`batch_size`, `lr`, `amp`, `resumed_after_epoch`, `test_positive_rate`, `class_weight`,
`accum_steps`, `effective_batch`, `dropout_draws`).

| item | mark |
|---|---|
| a recorded seed per cell | **VERIFIED** (`seed` in all 84) |
| `best_epoch` per cell (T10 needs epoch counts) | **VERIFIED** (all 84; range 10–39) |
| AUROC / AUPRC / accuracy per cell | **VERIFIED** |
| MCC or F1 per cell (T04 asks for them) | **NOT FOUND** — no file holds them |
| DeepDTA regression metrics (MSE, CI, Pearson, Spearman) | **CONTRADICTED as a property of the 84 cells, and PARTIAL overall.** DeepDTA in this grid was trained **binary** and records AUROC/AUPRC/accuracy like the rest — so T04's "DeepDTA: MSE, CI, Pearson" does not describe any of the 84. A DeepDTA regression grid *was* run earlier: `results.md:3–18` tabulates CI / MSE / Pearson for 4 splits × 2 datasets × 3 seeds, mean ± sd. But **no per-cell file backs that table** — `find` over the repo and `~/ColdSite-results` returns no DeepDTA regression checkpoint or `_results.json` anywhere. The only regression cells on disk are ColdSite-DTI's 12 DAVIS ones (`results/coldsite_dti_davis_*_regression_seed*.pt` + their `_results.json`). A T04 scoping question for the user |
| saved test-set predictions for any cell | **NOT FOUND** — `find ~/ColdSite-results -iname '*pred*'` returns nothing. T04's "every value is traceable to a predictions file" requires generating them |
| `_history.json` per cell | **PARTIAL** — 18 exist, all ColdSite-DTI (12 DAVIS + 6 KIBA). The four other models write none |

---

## 2. Split files and hash manifests

| claim | evidence | mark |
|---|---|---|
| DAVIS four levels; KIBA four levels | `data/splits/davis/{random,cold_drug,cold_target,cold_pair}/` and `data/splits/kiba/` likewise, each with `train.csv`, `valid.csv`, `test.csv` | **VERIFIED** |
| four **extra** DAVIS split directories | `cold_target_seqclean`, `cold_target_seqmatched`, `cold_pair_seqclean`, `cold_pair_seqmatched` — the leakage arms trained in `leakage_davis/` (ledger D4) | **VERIFIED**, now explained |
| `data/splits/MANIFEST.json` with SHA-256 hashes | **absent** — `find data -iname '*manifest*' -o -iname '*.sha256'` returns nothing | **NOT FOUND** (ledger D2 stands; T02 must create it) |
| splits are version-controlled | **no** — `.gitignore:13` ignores `data/splits/*` with only `.gitkeep` un-ignored; `git ls-files data/splits` returns 1 file | **CONTRADICTED** as a provenance assumption. A manifest built in T02 will hash *untracked local* files; nothing today proves these are the files the 84 cells trained on |
| ground truths are version-controlled | 26 of 28 `data/*.json` are tracked; only `data/klifs_ligands.json` and `data/klifs_structures.json` are not | **VERIFIED** |

**Row counts** (`wc -l` minus header), the numbers T04/T10 and Methods depend on:

| dataset | level | train | valid | test |
|---|---|---|---|---|
| DAVIS | random | 21,039 | 3,006 | 6,011 |
| DAVIS | cold_drug | 21,658 | 2,652 | 5,746 |
| DAVIS | cold_target | 21,080 | 2,992 | 5,984 |
| DAVIS | cold_pair | 15,190 | 264 | 1,144 |
| DAVIS | cold_target_seqclean / _seqmatched | 17,748 | 2,856 | 5,984 |
| DAVIS | cold_pair_seqclean / _seqmatched | 12,936 | 222 | 1,144 |
| KIBA | random | 82,778 | 11,825 | 23,651 |
| KIBA | cold_drug | 83,807 | 12,073 | 22,374 |
| KIBA | cold_target | 85,452 | 10,701 | 22,101 |
| KIBA | cold_pair | 58,041 | 1,334 | 4,375 |

DAVIS cold-pair's 15,190 train rows match CLAUDE.md §5's volume-control figure; every
`_results.json`'s `n_train_rows` matches its split's train count (spot-checked on
DAVIS random = 21,039 and KIBA random = 82,778).

### Ground-truth files

| file | entries | role |
|---|---|---|
| `data/davis_ground_truth_sites.json` | 442 | UniProt residues, re-numbered to DAVIS sequences (primary) |
| `data/davis_ground_truth_sites_uniprot.json` | 442 | pre-renumbering source |
| `data/davis_klifs_pocket_sites.json` | 442 | 85-residue KLIFS ATP pocket |
| `data/davis_drug_sites.json` / `_paired` / `_swapped` | 217 / 159 / 159 | per-pair KLIFS interaction-fingerprint contacts, plus the swapped-drug control |
| `data/kiba_ground_truth_sites.json` / `_uniprot` / `_klifs_pocket_sites` | 229 each | the same three for KIBA (no per-pair contacts) |
| `data/nonkinase_ground_truth_sites.json` | 60 | BindingDB transfer panel |
| `data/{davis,kiba}_ground_truth_alignment.json`, `*_provenance.json`, `*_klifs_pocket_report.json`, `data/davis_target_overrides.json` | — | re-numbering audit trail |

"Three independent ground truths" (report §3): **VERIFIED** for DAVIS (UniProt residues,
KLIFS pocket, per-pair contacts); **PARTIAL** for KIBA — two only, no per-pair contacts.

### Vendored repositories

`baselines/{DeepDTA, DrugBAN, HpyerAttentionDTI, MolTrans}` (the misspelling is the
directory's real name). **No `.gitmodules`, and no file anywhere records an upstream commit
hash** → the cloud pre-flight's "vendored repo commit hashes match the manifest"
(plan, cloud_rules (a)) has nothing to check against today: **NOT FOUND**.
Licences: `baselines/MolTrans/LICENSE` and `baselines/DrugBAN/LICENSE.md` present;
**DeepDTA and HpyerAttentionDTI carry none** — T23's licence audit has a real gap.

---

## 3. Per-cell predictive metrics

| claim | evidence | mark |
|---|---|---|
| per-cell accuracy exists somewhere | AUROC/AUPRC/accuracy in each of the 84 `_results.json`; AUROC alone, per level, in `results/analysis_{davis,kiba}_policyA/accuracy_[<model>_]<dataset>_seed<N>.json` (these are the ladder's input files, one number per level) | **VERIFIED** for AUROC/AUPRC/accuracy |
| a *table* of per-cell predictive metrics | **NOT FOUND** — no CSV or Markdown table covers all 84 cells. `~/ColdSite-results/davis_binary/grid_status.md` tabulates only ColdSite-DTI's 12 DAVIS cells | **NOT FOUND** (P06 confirmed) |
| a clean-accuracy module exists (T04 says "reuse it if T01 found it") | `src/evaluation/clean_accuracy.py` | **VERIFIED** |
| clean accuracy has been run | `results/clean_accuracy_davis.json` — **30 rows** = 5 models × {cold_target, cold_pair} × 3 seeds. Each row carries `recorded_auroc`, `all_rows{auroc,auprc,rows,positive_rate}`, `unseen_by_sequence{...}`, `reproduces_recorded`, `leaked_targets` (12 named), `rows_dropped` (816). `~/ColdSite-results/clean_acc_drugban/clean_accuracy_davis.json` holds DrugBAN's 6 rows again, byte-equal to the 6 DrugBAN rows already inside `results/clean_accuracy_davis.json` (compared as parsed JSON); it is a duplicate, not extra coverage | **VERIFIED** for the two leaky levels; there is no KIBA equivalent, which is correct — CLAUDE.md §6 records `leaks("kiba", level)` as False everywhere |
| accuracy-vs-localization analysis (T04, P07) | **NOT FOUND** — no module and no output correlates a cell's AUROC with its precision@10 |

---

## 4. Integrated-gradients coverage

Implemented in `src/evaluation/integrated_gradients.py`, which registers exactly three
adapters: `coldsite_dti_ig`, `hyperattentiondti_ig`, `moltrans_ig` (lines 250, 262, 289).
**No `drugban_ig`; no DeepDTA IG.**

| dataset · ground truth | ColdSite-DTI | HyperAttentionDTI | MolTrans | DrugBAN | DeepDTA |
|---|---|---|---|---|---|
| DAVIS · UniProt ladder | ✓ 3 seeds, 4 levels (`~/ColdSite-results/integrated_gradients/ig_davis/ladder_coldsite_dti_ig_davis_seed*.json`) | ✓ (+ `.md`) | ✓ | — | — |
| DAVIS · KLIFS ladder | ✓ (`.../ig_davis_klifs/`) | ✓ | ✓ | — | — |
| DAVIS · faithfulness | — | ✓ 3 seeds | — | — | — |
| KIBA · UniProt ladder | — | ✓ (`~/ColdSite-results/ig_kiba/`) | ✓ | — | — |
| KIBA · KLIFS ladder | — | ✓ (`~/ColdSite-results/ig_kiba_klifs/`) | ✓ | — | — |

| claim | mark |
|---|---|
| "IG survives correction in 7 of 12 DAVIS cells against 1 of 20 for attention" (report §5.5) | **PARTIAL.** The 12 per-cell ladder JSONs exist and hold the numbers, but **no committed file performs the Holm correction over the DAVIS IG family** — the table lives only in `paper/results.md:645–663` (Table R8). KIBA's equivalent *is* committed (`results/analysis_kiba_policyA/ig_family_kiba.md`, 4 cells, "3 of 4 cells survive"), generated by `src/evaluation/ladder_family.py`. Running that module on the DAVIS IG ladders would close the gap |
| `paper/results.md:677` states "one of **sixteen**" attention cells | **CONTRADICTED** by the committed audit: `results/analysis_davis_policyA/audit_davis_binary.json` has **20** cells (DrugBAN was added 18 Sept). `paper/PROJECT_REPORT.md` and `paper/abstract.md` say "1 of 20". `results.md` is stale here |
| the IG-vs-attention summary table | `~/ColdSite-results/integrated_gradients/ig_vs_attention.csv`, 16 rows — but **ColdSite-DTI's 8 IG columns are em-dashes**, although `ladder_coldsite_dti_ig_davis_seed*.json` contain the values (e.g. cold-drug p@10 = 0.0719 at seed 1). The CSV under-reports what was computed | **PARTIAL** |

---

## 5. Readout variants

Registry: `src/evaluation/readout_variants.py`, `READOUTS` dict — **9 variants**, plus the
separately registered `coldsite_dti_selfattn` (line 129).

| variant | outputs present | where |
|---|---|---|
| `coldsite_dti_selfattn` | ✓ 3 seeds, UniProt + KLIFS | `~/ColdSite-results/readouts/readouts_davis[_klifs]/` |
| `hyperattentiondti_maxchannel` | ✓ | same |
| `hyperattentiondti_receptive` | ✓ | same |
| `moltrans_maxhead` | ✓ | same |
| `moltrans_firstlayer` | ✓ | same |
| `moltrans_interaction` | ✓ DAVIS + KIBA, UniProt + KLIFS | `results/readouts_moltrans_interaction/{,klifs,kiba,kiba_klifs}/` |
| `moltrans_interaction_sum` | **none** | — |
| `drugban_maxatom` | ✓ UniProt + KLIFS | `results/readouts_drugban_davis/{,klifs}/` |
| `drugban_receptive` | ✓ | same |
| `drugban_maxhead` | **none** | — |

Summary table: `~/ColdSite-results/readouts/readout_comparison.csv`, 64 rows
(ground truth × model × readout × level).

| claim | mark |
|---|---|
| MolTrans's published interaction map is implemented as a second readout (report problem #4) | **VERIFIED** — `READOUTS["moltrans_interaction"]`, described in the module docstring as "the drug x protein interaction map MolTrans's own paper visualises (Fig. 3)". T07 must reuse, not re-implement, this |
| "top-ten residues of an alternative readout overlap the published readout's by **2–12%** … across 25 proteins" (`paper/results.md:591`) | **NOT FOUND.** No module computes readout-to-readout top-k overlap (`grep -rn "overlap" src/evaluation/` hits only `faithfulness.py` and `drug_dependence.py`), and no output file holds the number. The nearest committed artefact is `results/drug_dependence/*.txt`, which measures overlap *between drugs*, not between readouts. This claim has no provenance and will fail T21's checker |
| a per-model record of which readout the model's own paper displays | **NOT FOUND** — `docs/readout_sources.md` does not exist; T07 is blocked on the user as the plan anticipates |

---

## 6. Non-kinase panel

| claim | evidence | mark |
|---|---|---|
| 60-protein BindingDB panel | `data/processed/nonkinase_panel.csv` — 21,145 rows, 60 unique `Target_ID` and 60 unique `gene_name`; ground truth `data/nonkinase_ground_truth_sites.json` (60 entries) | **VERIFIED** |
| the control was run on all seeds and models | `results/analysis_davis_policyA/control_<model>_davis_seed{1,2,3}[_noions].{json,md}` for **coldsite_dti, drugban, hyperattentiondti, moltrans** — 4 models × 3 seeds × 2 ion settings = 48 files. KIBA: the same for **coldsite_dti, hyperattentiondti, moltrans** (no DrugBAN) = 36 files | **VERIFIED** — CLAUDE.md's "only ever runs on a third of the seeds" is out of date |
| the panel gate is recorded per run | each control JSON opens with `panel_gate{n_total:60, n_kinase:0, n_non_kinase:60, distinct_non_kinase:60, kinase_fraction:0.0, control_is_usable:true}` | **VERIFIED** |
| the primary non-kinase result excludes cotransport ions | both arms exist; `_noions` is the excluded-ion arm (CLAUDE.md §5) | **VERIFIED** |
| effect sizes / CIs for the panel (T18) | **NOT FOUND** — the control JSONs carry `precision_at_k`, `chance`, `p_value`, `n_proteins` and a `gap`, but no interval |
| DrugBAN 290-atom loader cap guard | `src/evaluation/collect.py:257` `UNENCODABLE_LIMIT_FRACTION = 0.01`, with the reason at line 327 | **VERIFIED** (report problem #10) |

---

## 7. Seed and RNG records

| requirement (plan, statistical_integrity_rules) | evidence | mark |
|---|---|---|
| every run records its seed | `seed` in all 84 `_results.json`; `torch.manual_seed(args.seed)` in all five trainers (`train.py:330`, `train_deepdta.py:171`, `train_hyperattentiondti.py:272`, `train_moltrans.py:266`, `train_drugban.py:200`) | **VERIFIED** |
| MolTrans seeds after the vendored import | `src/model/train_moltrans.py:263–266`, with the reason in the comment | **VERIFIED** (report problem #2) |
| every run records torch/numpy/python RNG initial states | **NOT FOUND for the 84 cells.** `src/model/resume.py:58` `capture_rng()` does capture all three plus CUDA, but only into the transient `<checkpoint>_resume.pt`, which the trainer deletes when a cell completes — `find ~/ColdSite-results -name '*_resume.pt'` returns nothing | **NOT FOUND** |
| every run records a hash of its initial weights | **NOT FOUND.** A finished `.pt` holds exactly four keys — `model_state, epoch, task, val_metrics` — no initial weights and no hash. Confirmed programmatically across all 84 results files: none carries `rng_state`, `initial_weight_hash`, `weight_hash` or `env` | **NOT FOUND** |
| distinct seeds produce distinct initial-weight hashes | **cannot be checked retrospectively** for the existing 84 — the initial weights were not kept. T02(b) can only test this going forward, on freshly constructed models |
| an environment record per cell (python/torch/CUDA/cuDNN/vendored commits) | **NOT FOUND** — no `_results.json` carries one, and no complete-marker files exist | **NOT FOUND** |
| seed-disagreement analysis exists | `src/evaluation/seed_agreement.py` → `results/seed_agreement.md`: 22 cells, per-seed precision@10, chance, spread, and a per-seed alpha flag. Its closing line reads: "**12 of 22 cells have seeds that disagree** about their own verdict. In **21 of 22** the spread across seeds is larger than the cell's distance from chance. **1** cell has all three seeds above alpha; **9** have none." | **VERIFIED** — this is headline contribution (a) and it is fully evidenced |

The 22 cells are the four attention models × their trained levels (DAVIS 4 levels × 4
models = 16, KIBA 2 levels × 3 models = 6). DeepDTA has no attention and DrugBAN no KIBA
cells, which is why 22 and not 28.

---

## 8. Permutation counts in existing outputs

Inferred from each recorded `p_value`'s denominator across every JSON in the four primary
analysis directories:

| output family | n_permutations | source |
|---|---|---|
| ladders (UniProt and KLIFS, DAVIS and KIBA) | **1,000** | `run_ladder.py:332` `--n-trials` default 1000 |
| non-kinase controls | **1,000** | `run_control.py:241` default 1000 |
| positional / residue-identity / span nulls | **1,000** | `positional_control.py:305` default 1000 |
| positive control | **1,000** | `positive_control.py:429` default 1000 |
| **audits — primary** (`audit_davis_binary.json`, `audit_kiba_binary.json`) | **500** | `run_audit.py:254` default 500; observed minimum raw p = 0.001996007984031936 = 1/501 |
| audits — extra 10k runs (`audit_*_10k_permutations.json`) | **10,000** | observed minimum raw p = 9.999000099990002e-05 = 1/10001 |

`src/evaluation/run_all.py` passes no `--n-trials`, so every `run_all` output takes the
defaults above.

**Ledger D5 is confirmed and quantified.** The plan's `project_fact` "(5) … now 10,000
permutations" is **CONTRADICTED** as a statement about the code: no default is 10,000, and
the 10k audits are supplementary files beside the 500-permutation primaries.

**Why it matters, read from the files.** The DAVIS family has 20 cells, so Holm's smallest
threshold is 0.05/20 = **0.0025**, and the smallest p a 500-permutation test can produce is
**0.001996**. The one surviving cell clears correction by a margin of 0.0005 — the floor of
the test is 80% of the threshold. The 10,000-permutation rerun moves that floor to 0.0001
and, read from the files, **changes no verdict**: both `audit_davis_binary.json` and
`audit_davis_binary_10k_permutations.json` return exactly one significant cell,
`hyperattentiondti|davis|random`; both KIBA audits return none of eight. The plan's
requirement "min achievable p below every Holm threshold" is satisfied by both DAVIS runs
and by both KIBA runs (threshold 0.05/8 = 0.00625), but only barely for DAVIS at 500.
Which run is primary is a T03 decision, as the ledger already records.

---

## 9. The remaining report claims, checked

| # | report §5 claim | evidence | mark |
|---|---|---|---|
| 1 | "One cell of twenty survives correction (HyperAttentionDTI, random)"; "does not replicate on KIBA, none of eight" | `results/analysis_davis_policyA/audit_davis_binary.json`: 20 cells, `significant` true for `hyperattentiondti\|davis\|random` alone. `results/analysis_kiba_policyA/audit_kiba_binary.json`: 8 cells, none significant | **VERIFIED** |
| 2 | attention is load-bearing; pocket enrichment 1.3–2.1× | faithfulness JSON/MD for 4 models × 3 seeds on DAVIS and 3 models × 3 seeds on KIBA, in the two `analysis_*_policyA` directories; KLIFS ladders in the two `*_klifs` directories | **VERIFIED that the outputs exist**; the 1.3–2.1× range is a summary across those ladders, stated in the manuscript rather than in a committed summary file |
| 3 | 12 of 22 seed-disagreeing cells; 21 of 22 spread > distance from chance | `results/seed_agreement.md` closing paragraph | **VERIFIED** |
| 4 | readouts share 2–12% of their top ten; one choice moves 2.6× chance to below chance | the second half is readable from `readout_comparison.csv` and the variant ladders (HyperAttentionDTI KLIFS cold-target: 0.367 max-channel vs 0.081 receptive vs 0.143 chance). The 2–12% overlap has **no** generator or output file — see §5 | **PARTIAL** |
| 5 | IG 7 of 12 DAVIS vs 1 of 20 attention | see §4 — per-cell ladders exist, the family correction is not committed | **PARTIAL** |
| 6 | DrugBAN at chance in all 12 cells; attention not load-bearing; only model whose map changes with the drug | audit JSON (no DrugBAN cell significant), `faithfulness_drugban_davis_seed*.json`, `results/drug_dependence/drugban.txt` ("mean 4.53/10 … identical top-10 in 0% of 150 drug pairs") beside `hyperattentiondti.txt` (9.72/10, 72%) and `moltrans.txt` (10.00/10, 100%) | **VERIFIED** |
| 7 | a model's residue attention cannot depend on the drug — visible in source | `results/drug_dependence/moltrans.txt`: identical top-10 in **100%** of 150 drug pairs; the interaction readout gives 1.09/10 and 0% | **VERIFIED** |
| 8 | all 54 DAVIS mutant targets carry the wild-type sequence; leakage inflated cold-target accuracy for every model | `results/sequence_audit_davis.md`; `results/clean_accuracy_davis.json` (12 named leaked targets, 816 rows dropped; `all_rows` AUROC exceeds `unseen_by_sequence` in **all 15** cold-target rows — 5 models × 3 seeds, deltas +0.002 to +0.049 — so the inflation claim holds exactly as stated, for cold-target. At cold-pair the sign is mixed: DeepDTA and HyperAttentionDTI are *deflated* in all three seeds) | **VERIFIED** |
| — | problem #3, masking arms not size-matched | `results/mask_comparability_davis.md`: "explanation's residues: **47.9%** of tokens change / random residues: **95.0%** / the random arm is therefore **2.0x** the intervention", token-matched control at 61.6%. Token-space faithfulness exists for MolTrans only — 3 seeds on DAVIS and 3 on KIBA (`token_faithfulness_moltrans_*`), which is the only sub-word model | **VERIFIED** |
| — | problem #8, faithfulness is dose-dependent (k=10 vs k=50) | `results/faithfulness_k50_davis/` — 3 seeds × {DrugBAN, HyperAttentionDTI} only | **PARTIAL** (2 of 4 attention models) |
| — | "938 tests" | `python3 -m pytest -p no:warnings` → `937 passed, 4 skipped in 87.39s (0:01:27)` | **CONTRADICTED** (ledger D1 stands; 941 collected, 4 skips need the `drugban` conda env) |
| — | "67 Python modules, ~17,300 lines, 200 commits" | `find src -name '*.py'` = **68**; `wc -l` over them = **17,357**; `git rev-list --count HEAD` = **202** (two T00/T01 commits since the report) | **VERIFIED** to within the report's own rounding |
| — | "33 analysis modules" | `find src/evaluation -name '*.py'` = 33 | **VERIFIED** |
| — | "20 notebooks" | `ls notebooks/*.ipynb` = 20 | **VERIFIED** |
| — | "the KIBA arm alone took about 101 GPU-hours across four accounts" | **NOT FOUND** — no log in the repo or the checkpoint directory records training wall-clock. `results/analysis_davis_policyA/kaggle_logs/*.log` are *analysis* logs. What does exist is `results/speed_test_kiba_t4.md` (measured s/batch per model on a T4) and `best_epoch` in each `_results.json`. T10 must build its estimate from those two, not from the 101-hour aggregate — which is exactly what the plan says |
| — | "4.6 GB, backed up outside the machine" | 4.6 GB **VERIFIED**; the Drive backup is claimed in CLAUDE.md and not checkable from here | **PARTIAL** |
| — | antiviral subset | `data/processed/antiviral_clean.csv` (10,549 rows) and `src/data/extract_antiviral.py` exist, but `paper/results.md:11` records the case study as **cut on 2026-09-18** | **VERIFIED as cut** |

---

## 10. Analysis outputs, by directory

**Primary (in the repo).** `results/analysis_davis_policyA/` (166 files, 39 git-tracked),
`results/analysis_davis_policyA_klifs/` (27, 21), `results/analysis_kiba_policyA/`
(121, 58), `results/analysis_kiba_policyA_klifs/` (19, 18).

Per directory: `accuracy_*`, `ladder_*`, `faithfulness_*`, `control_*[_noions]`,
`positional_control_*`, `audit_*`, `headline_*.png`, `analysis_summary_<dataset>.md`,
`run_all.log`. Superseded audits are kept under explicit names
(`audit_davis_binary.superseded_16cell_family.*`, `audit_davis_binary_3models_superseded.md`,
`audit_kiba_binary.superseded_6cell_family.*`, `audit_kiba_binary.superseded_uniform_at_4_levels.*`).

**Coverage by model.** DAVIS: faithfulness, ladder, control and positional control for
ColdSite-DTI, HyperAttentionDTI, MolTrans, DrugBAN × 3 seeds. KIBA: the same three minus
DrugBAN; positional control there is KLIFS-only for ColdSite-DTI.

**Other committed outputs.** `results/ci_davis.{json,md}` (bootstrap CIs — see below),
`results/ci_drug_arms_davis.*`, `results/ci_arms/{drug,paired,swapped}/`,
`results/drug_dependence/*.txt` (4), `results/faithfulness_k50_davis/`,
`results/seed_agreement.md`, `results/mask_comparability_davis.md`,
`results/positive_control_{davis,kiba}.{json,md}`, `results/sequence_audit_{davis,kiba}.md`,
`results/leakage_retrain_davis.{json,md}`, `results/volume_control_davis.md`,
`results/speed_test_kiba_t4.md`, `results/figures/` (fig0–fig4, PDF+PNG, `CAPTIONS.md`).

**Outside the repo.** `~/ColdSite-results/{integrated_gradients,ig_kiba,ig_kiba_klifs,
readouts,drug_sites}/` hold the IG, readout and per-pair-contact ladders described above;
`analysis_kaggle/` and `analysis_kaggle_v2/` are earlier DAVIS analysis runs (45 files
each) superseded by `results/analysis_davis_policyA/`; `analysis_kiba*`,
`audit_{davis,kiba}_*cell[_10k]`, `kiba_probe_2026-09-15` are intermediate runs.

**Bootstrap CIs already exist** (bears directly on T05 and P17):
`src/evaluation/bootstrap_ci.py` → `results/ci_davis.json`, **24 rows** = 2 ground truths ×
3 models (ColdSite-DTI, HyperAttentionDTI, MolTrans) × 4 levels, `n_resamples` 10,000,
resampling **proteins** with each protein carried in under all its seeds. Missing: DrugBAN,
the uniform control, all of KIBA, and any CI on a **faithfulness delta**. T05 is therefore
an extension of an existing module, not new work — and the amendment's "unit of resampling
= target" is already this module's choice.

**A git caveat for every later task.** Much of `results/` is gitignored — 239 files are
tracked out of far more on disk (`results/analysis_davis_klifs/` and
`results/analysis_davis_partial/` are entirely untracked). Verification criterion 3 ("no
file under the original results directories changed, git diff shows none") cannot detect a
change to an untracked output. T02's manifest should cover the primary analysis outputs as
well as the splits if that criterion is to mean anything.

---

## 11. Summary of gaps this inventory found

Ordered by which task must close each. Nothing below was acted on; all of it is recorded
in the ledger under "Discovered".

1. **`data/splits/MANIFEST.json` absent, and the splits are gitignored** → T02. A manifest
   over untracked files records what is on this disk today, not what trained the 84 cells.
2. **No vendored-repo commit hashes anywhere; DeepDTA and HpyerAttentionDTI carry no
   licence** → T02 pre-flight, T23.
3. **No RNG-state or initial-weight record for any of the 84 cells**, and none can be
   recovered → T02 can only enforce this going forward.
4. **No saved test predictions and no MCC/F1 anywhere** → T04 must run inference over all
   84 checkpoints, not read existing files.
5. **T04's "DeepDTA: MSE, CI, Pearson, Spearman" does not match the grid**, which trained
   DeepDTA binary; and the DeepDTA regression table at `results.md:3–18` has **no per-cell
   file on disk anywhere** → a T04 scoping question for the user, and a provenance problem
   for T21 if that table enters the manuscript.
6. **No accuracy-vs-localization analysis exists** → T04 / P07, new work.
7. **The DAVIS IG family's Holm correction is not committed** — only KIBA's is. The
   generator (`ladder_family.py`) exists → T08 or T05.
8. **`ig_vs_attention.csv` leaves ColdSite-DTI's IG cells blank** although the JSONs hold
   the values → T22 when the figure is regenerated.
9. **The "2–12% readout overlap" claim has no generator and no output file** → T08 adds
   top-k IoU agreement, which subsumes it; T21's provenance checker would otherwise fail on it.
10. **`paper/results.md:677` says "one of sixteen"** where the committed audit has 20 cells
    → T20.
11. **Permutation defaults are 500/1,000, not 10,000** → T02 enforcement, T03 primacy decision.
12. **Two readout variants are registered but never run** (`drugban_maxhead`,
    `moltrans_interaction_sum`) → T07 decides whether they belong in the sensitivity set.
13. **Faithfulness k=50 sensitivity covers 2 of 4 attention models** → T03 pre-specification.
14. **No training wall-clock logs exist**; T10's budget must come from
    `results/speed_test_kiba_t4.md` plus each cell's `best_epoch`.
15. **Much of `results/` is gitignored**, so "no original result changed" is not
    git-verifiable → T02.
