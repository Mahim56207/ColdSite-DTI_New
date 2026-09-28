# Protocol amendment v2 — pre-specification of the extension analyses

| | |
|---|---|
| **Status** | **APPROVED — in force from 2026-09-24** (user reply "approved" in chat, recorded in §12). Drafted as a draft; rules unchanged by approval. |
| **Drafted** | 2026-09-24T03:42:16Z (UTC), branch `remediation/T03`, on top of `299f11b` |
| **Authoritative timestamp** | the timestamp of the git commit that adds this file; a later edit is a new commit and, after approval, an addendum (§10) |
| **Written by** | the remediation engineer (Claude), from the repository as it stands at `299f11b`; nothing in it comes from a new analysis |
| **Blocks** | T05 and every analysis that produces a new number; until approval, none may run |

## 0. What this document does

The audit's four headline contributions (a seed-inconsistent verdicts, b calibrated battery,
c readout dependence, d leakage impact — `docs/REMEDIATION_PLAN.md`, `strategic_direction`) are
about to be extended with seeds 4–5, DrugBAN on KIBA, new explanation methods, a conservation
null, a modern model and stricter splits. Every one of those extensions adds hypotheses. Holm's
correction is valid only over a family fixed before the p-values exist, so this document fixes
them **now**, before any extension result is seen, and copies the decision rules out of the
code so that a later reading cannot bend them.

Rules that hold for the whole document:

1. **The original analyses are the primary analysis and are not altered.** Seeds 1–3, the
   families in §2, and the rules in §1 stay exactly as they are in the committed outputs.
2. **An extension is a separate family.** It is never added to a primary family, and no
   claim compares a corrected p-value from one family with a corrected p-value from another
   (the same rule Methods already states for the integrated-gradient family,
   `paper/methods_data_and_evaluation.md:439–441`).
3. **Family membership is fixed before scoring.** A family's cells are listed here, or in a
   signed addendum (§10) committed before its first score is computed. A cell, seed, model,
   method or split cannot be added, dropped or swapped after its result is seen. A cell that
   turns out to be impossible (a missing checkpoint, an N/A method) is recorded as missing
   with its reason and **stays counted** in the family size unless the addendum that fixed the
   family named the reason in advance.
4. **New results go to new directories** (`*_v2` or a named extension). Nothing under
   `results/` that exists today is overwritten (`src/evaluation/integrity.py`
   `check_extension_dir`, enforced by `run_all --extension`).

### 0.1 Provenance of the rules

Each rule below quotes what the code does today, with `path:line` at `299f11b`. Where the code
does **less** than the plan requires, that is stated as a gap, and the amendment says what the
extension must do; no rule was invented to make an old result look better.

## 1. Rules copied from the code (unchanged)

| # | Rule | Code |
|---|---|---|
| 1.1 | **Unit of scoring:** one test pair per protein — the first in file order (`--pairs-per-target 1`, default). The audit and the ladder both use it. | `src/evaluation/run_ladder.py:338`, `src/evaluation/run_audit.py:273`, `rows_to_score` `run_ladder.py:56` |
| 1.2 | **Ground truth and window:** explanations are scored over each protein's first 1,000 residues; UniProt binding-site residues re-numbered to the dataset sequence are the primary ground truth; the 85-residue KLIFS ATP pocket is the second. | `src/evaluation/run_audit.py:277`, `src/evaluation/collect.py:265` (`max_protein_len = 1000`) |
| 1.3 | **Sequence policy (option A):** seen-by-sequence and pocketless targets are dropped and one protein is counted per sequence, in every explanation metric. | `src/evaluation/exclusions.py:58` (`excluded_target_ids`), `:73` (`protein_key`) |
| 1.4 | **Statistic:** mean over proteins of precision@k. Ties in the attention map are broken at random, so a tied map behaves like chance. | `src/evaluation/precision_at_k.py:49` (`top_k_positions`), `:126` (`batch_precision_at_k`) |
| 1.5 | **Null and p-value:** for every protein draw k random positions without replacement, take the mean over proteins, repeat `n_trials` times; one-sided, add-one estimator `p = (1 + #{null ≥ observed}) / (1 + n_trials)`. | `src/evaluation/significance_test.py:87` (`permutation_test_batch`), `:119–120` |
| 1.6 | **Permutations:** `n_trials ≥ 10,000` in every runner, refused below that; `run_audit` also refuses a count whose floor `1/(n+1)` is not below `α/m` for its own family. | `src/evaluation/integrity.py:39` (`MIN_PERMUTATIONS`), `:78` (`check_permutations`); `src/evaluation/run_audit.py:280, 341` |
| 1.7 | **Across seeds:** a cell's p is the **median** of its seeds' p-values (not the smallest); its precision is the mean, its spread the sample SD (ddof = 1); fewer than 3 seeds is flagged and not quoted. | `src/evaluation/run_audit.py:129–133`, `src/evaluation/aggregate.py:25, 28–60` |
| 1.8 | **Correction:** Holm–Bonferroni at α = 0.05, sorted ascending, threshold `α/(m − i)`, stop at the first failure (everything after is not rejected). | `src/evaluation/aggregate.py:63–95` |
| 1.9 | **Cell verdict ("survives correction"):** `significant = True` from that Holm step. **No other cell-level verdict is coded.** Statements such as "coarsely plausible" or "about 2× chance" in the draft are prose readings of the effect size; they are not rules and are not to be presented as decision rules. | `src/evaluation/aggregate.py:84–89` |
| 1.10 | **Faithfulness verdict ("load-bearing"):** the split-level mean of `comprehensiveness_delta` (attended-masking effect minus the size-matched random-masking control) is finite and **> 0**. A sign test on a point estimate — no interval, no p-value. Token-space (MolTrans) uses the same sign rule. | `src/evaluation/faithfulness.py:215, 251–253`; `src/evaluation/token_faithfulness.py:160` |
| 1.11 | **Masking size-matching:** residue-level models change exactly k tokens in both arms; for MolTrans the control is matched in token space to within **10 %** of the tokens changed, after at most 200 tries, and reports itself unmatched otherwise. | `src/evaluation/mask_comparability.py:37–39`; test pins in `tests/test_integrity_masking.py` |
| 1.12 | **Faithfulness sampling:** the first 200 pairs a level's dataloader yields (`--max-pairs 200`), 5 random-control draws per pair, k = 10. | `src/evaluation/run_all.py:442` (`--max-pairs`), `src/evaluation/run_faithfulness.py:412–414` (`--k`, `--n-random-trials`, `--max-pairs`) |
| 1.13 | **Seed disagreement:** a cell's seed with `p < 0.05` **uncorrected** counts as "significant on its own" (deliberately uncorrected: it is what a single-seed report would quote). The cell **disagrees** iff `0 < #significant < n_seeds` (coded for `n_seeds = 3`). The cell's **spread exceeds its signal** iff `max(precision) − min(precision) > |mean(precision) − chance|`. Cells are read from precision@10 at UniProt ground truth. | `src/evaluation/seed_agreement.py:21, 44–50` |
| 1.14 | **Confidence intervals (existing):** resample **proteins** with replacement (10,000 resamples), each protein carried in with all its seeds averaged, 95 % **percentile** interval, `default_rng(0)`. | `src/evaluation/bootstrap_ci.py:43–44, 52–71` |

### 1.1 Gaps the code has against the plan (recorded, not hidden)

* **G1 — `seed_agreement` silently drops any cell that does not have exactly 3 seeds**
  (`seed_agreement.py:38`, `len(value) == 3`), and the `disagree`/`wide` tests hard-code 3
  (`:47, :50`). Feeding it seeds 1–5 would return **zero** cells with no error. The extension
  (§3, E1) therefore needs the rule generalised as in §7; the seeds 1–3 statistic must be
  reproduced exactly by the generalised code before any 1–5 number is quoted.
* **G2 — the faithfulness JSONs hold split-level means only** (keys read in this session:
  `comprehensiveness*`, `sufficiency*`, `aopc`, `n_pairs`, `k`, `explanation_is_load_bearing`).
  A bootstrap CI over targets for a faithfulness delta cannot be computed from any existing
  file; it needs the faithfulness step re-run into a `_v2` directory with per-pair values kept.
  Pairs are also not one-per-protein (§1.12), so the CI must group pairs by target.
* **G3 — Holm is applied only to the attention audit and to ladder families**
  (`run_audit.py`, `ladder_family.py`; the only Holm call sites in `src/` are those two plus
  `run_all.py`, `integrity.py` and `paper_figures.py`). The **KLIFS pocket ladder, the
  positional nulls and the non-kinase control** are reported with **uncorrected** `p < 0.05`
  stars (`positional_control.py:278, 289`; `run_control.py:193, 205`). §2.3 and E7 declare
  families for them; this makes claims stricter, not looser.
* **G4 — the "original" primary family has already grown once, after results were seen.**
  DAVIS: 16 cells → 20 when DrugBAN was added on 2026-09-19; KIBA: 6 → 8 when ColdSite-DTI's
  KIBA cells were added the same day (`paper/methods_data_and_evaluation.md:432–436, 525–528`;
  superseded files `audit_davis_binary.superseded_16cell_family.*`,
  `audit_kiba_binary.superseded_6cell_family.*` are on disk). The paper records that the
  earlier p-values did not change. This amendment freezes the current sizes **so that it cannot
  happen again**; the history stays in the Methods and the limitations.
* **G5 — the family sizes are pinned in tests as constants** (`DAVIS_AUDIT_CELLS = 20`,
  `KIBA_AUDIT_CELLS = 8`, `tests/test_integrity_permutations.py:30–31`), so a change to P1 or P2
  fails a test rather than passing silently.

## 2. Primary and secondary families (existing analyses)

### 2.1 Primary families — unchanged

| id | dataset | cells | m | seeds | ground truth | p per cell | files (original, byte-preserved) |
|---|---|---|---|---|---|---|---|
| **P1** | DAVIS | {ColdSite-DTI, HyperAttentionDTI, MolTrans, DrugBAN, uniform control} × {random, cold-drug, cold-target, cold-pair} | **20** | 1, 2, 3 | UniProt | median over seeds of the split-level permutation p (§1.5, 1.7) | `results/analysis_davis_policyA/audit_davis_binary.{json,md}` (500 permutations) |
| **P2** | KIBA | {ColdSite-DTI, HyperAttentionDTI, MolTrans, uniform control} × {random, cold-drug} | **8** | 1, 2, 3 | UniProt | same | `results/analysis_kiba_policyA/audit_kiba_binary.{json,md}` (500 permutations) |

The uniform control is a checkpoint-free arm scored only where a trained model produced a cell
(`run_audit.py:88–100`, `CHECKPOINT_FREE`); it is part of each family's `m`, as it has been.
P1 contains DeepDTA nowhere (no attention). Verified in this session by reading the committed
JSON: `p_values_corrected` has 20 keys (DAVIS) and 8 keys (KIBA); one significant cell on DAVIS
(`hyperattentiondti|davis|random`) and none on KIBA, in both the 500- and the 10,000-permutation
files.

**Which permutation count is primary (Decision D1).** Two audits of the same 20-cell (8-cell)
family exist: the 500-permutation originals (smallest reportable p = 0.001996, read from the
file in this session) and the 10,000-permutation re-runs `audit_*_10k_permutations.{json,md}`
(smallest p = 9.999e-5). The smallest DAVIS Holm threshold is 0.05/20 = 0.0025, so the
500-permutation floor is 80 % of it — that is why the floor of §1.6 exists. **Default proposed:**
the 10,000-permutation audit is the primary *inferential* record (it is the only one that meets
the rule of §1.6); the 500-permutation files stay byte-preserved and are reported alongside as the
record the draft was first written from. Both give the same verdicts (above), so no claim in the
draft changes; if a future re-run ever disagrees the 10,000-permutation result governs.

### 2.2 Existing secondary family, labelled post hoc

| id | cells | m | note |
|---|---|---|---|
| **S1** | integrated-gradient cells, DAVIS: ColdSite-DTI-IG, HyperAttentionDTI-IG, MolTrans-IG × 4 levels (DrugBAN has no IG arm, `paper/results.md:1304`) | 12 | added *after* the attention results were seen (`paper/results.md:712–716`); reported as secondary and never compared with P1. Its DAVIS Holm table is **not committed** (T01 gap 7); it is to be produced by `src/evaluation/ladder_family.py` from the existing ladder files, over exactly these 12 cells. |
| **S2** | integrated-gradient cells, KIBA: HyperAttentionDTI-IG, MolTrans-IG × 2 levels | 4 | committed: `results/analysis_kiba_policyA/ig_family_kiba.md` (3 of 4 survive). |

### 2.3 Existing analyses that were reported uncorrected — families declared now

| id | analysis | cells | m | rule |
|---|---|---|---|---|
| **S3-D** | precision@10 against the 85-residue KLIFS pocket, DAVIS | {ColdSite-DTI, HyperAttentionDTI, MolTrans, DrugBAN} × 4 levels | 16 | median-over-seeds permutation p, Holm within the family. Files: `results/analysis_davis_policyA_klifs/ladder_*.json` |
| **S3-K** | the same, KIBA | {ColdSite-DTI, HyperAttentionDTI, MolTrans} × 2 levels | 6 | files: `results/analysis_kiba_policyA_klifs/ladder_*.json` |
| **S4** | positional / same-residue / within-span nulls (`positional_control.py`) | descriptive only | — | remain uncorrected and **descriptive**; no count of "significant" cells may be quoted from them. A corrected claim needs a family declared in an addendum before it is computed. |

These are declared now, before T05 computes any corrected value, so the correction cannot be
chosen after seeing which cells it would keep.

## 3. Extension families

Every row below is a family **of its own**; `m` is the design size. A family whose realised size
differs from the design (a cell lost) is analysed at the **design** `m` (the missing cell counts as
a non-rejection) and the loss is reported.

| id | task | cells (design) | m | seeds | ground truth | Holm scope |
|---|---|---|---|---|---|---|
| **E1** | T11–T12: seeds 4–5, DAVIS | {HyperAttentionDTI, MolTrans, ColdSite-DTI} × 4 levels, **+ DrugBAN × 4 levels iff the wave plan (T10) lists it** — decided and committed before any seed-4/5 result exists (D2) — **+ the uniform control at the same levels** | 16 without DrugBAN; 20 with | **1–5** pooled: a cell's p is the median over its five seeds (rule 1.7) | UniProt | the family alone. P1 is untouched and remains the primary statement about seeds 1–3. |
| **E2** | T11–T12: DrugBAN on KIBA | DrugBAN × {random, cold-drug} | 2 | 1, 2, 3 | UniProt | the family alone; the uniform control is not re-counted (it is already in P2). The DrugBAN atom-cap row-loss guard (ledger D6) applies. |
| **E3-D** | T08: new explanation methods, DAVIS — attention×gradient, occlusion, attention rollout where architecturally valid | (method × model × level) for every (method, model) marked *applicable* in `docs/method_applicability.md` **committed before any T08 score is computed** | = number of applicable (method, model) pairs × 4 | 1, 2, 3 | UniProt | one joint family over all new methods — deliberately the conservative choice (larger m) rather than one family per method. N/A methods are excluded with their written reason and are not counted. Existing IG stays in S1. |
| **E3-K** | T08, KIBA | same, on the KIBA-trained models × {random, cold-drug} | applicable pairs × 2 | 1, 2, 3 | UniProt | its own joint family. |
| **E4-D / E4-K** | T06: conservation null of the pocket enrichment | one cell per S3 cell: (model × level), DAVIS 16, KIBA 6 (KIBA only if the conservation source covers KIBA sequences) | 16 / 6 | 1, 2, 3 | KLIFS pocket | each dataset its own family. **Test:** the real attention's mean precision@10 against the pocket vs a null that keeps the conservation profile (residues drawn within conservation-matched bins); one-sided, 10,000 permutations, add-one p. The conservation **source and the bin definition** are not in the repository (no conservation data was found by `grep` over `src/` and `data/*.json` in this session) and are sealed by an addendum, approved by the user, before any conservation value is computed. |
| **E5** | T14–T16: the modern model (identity not yet chosen) | the model × 4 DAVIS levels | 4 | 1, 2, 3 | UniProt | one family per integrated model; the model, its readout and its seeds are written into an addendum **before its first checkpoint is scored**. |
| **E6-C / E6-S** | T17: cold kinase-group split and drug-scaffold split (if the compute is approved) | (models trained on that split) × 1 level | number of models | 1, 2, 3 | UniProt | one family per new split. Group labels and the scaffold definition are sealed by addendum before any cell is scored; explanation metrics exclude seen-by-sequence targets (§1.3). |
| **E7** | T18: non-kinase panel (60 proteins) | (audited model × DAVIS training level), arm = non-kinase, **cotransport ions excluded** (`_noions`, the recorded primary) | DAVIS 16; KIBA 6 as its own family | 1, 2, 3 | panel ground truth | with-ions outputs are sensitivity, outside the family. p per cell: median over seeds of the arm's permutation p (`run_control.py:73–90`). |
| **E8** | T07: readout variants | not fixed here | — | 1, 2, 3 | UniProt | the primary readout per model is defined by `docs/readout_sources.md`, which needs the user's figure references. That document plus its family (cells, m) is an addendum, signed **before** any readout variant not already run is scored. The existing P1/P2 keep the readouts they used; a change of primary readout creates a new family, it does not rewrite P1. |

**Descriptive-only quantities (no test, no Holm):** top-k IoU agreement between explanation
methods; the leakage-impact quantification (clean vs uncorrected accuracy, `clean_accuracy.py`,
`leakage_retrain.py`); accuracy-vs-localization Spearman correlations (reported with a bootstrap
CI, §5, and never converted into a significance claim); seeds 4–5 read *alone* as a replication of
the seeds 1–3 disagreement pattern (§7).

## 4. Pre-specified k

* **k = 10** for precision@k, and for the number of residues masked in faithfulness. It is the
  default in every runner (`run_all.py:62` `K = 10`, `run_ladder.py:331`,
  `run_faithfulness.py:412`, `precision_at_k.py:49`) and **was fixed before any result existed**:
  `docs/03_GUIDE_124AD0067.md:44, 57, 67` (the guide added on 2026-07-31 in the first commit of
  `precision_at_k.py`) specify `k=10`. Every extension family is evaluated at k = 10.
* Other k values are **sensitivity analyses only**: precision@k at k = 5 and 20 (the ladder's
  `k_values`, `run_ladder.py:144`) and the k = 50 faithfulness check. Existing k = 50
  faithfulness covers only DrugBAN and HyperAttentionDTI (T01 gap 13); for any model added in an
  extension, k = 50 faithfulness is run alongside k = 10 and is never a verdict.
* "The verdict at k" always means k = 10. A statement that holds at k = 50 but not k = 10 is
  written as exactly that (as `paper/results.md:335–341, 1265–1272` already do), never as the
  finding.

## 5. Effect sizes and confidence intervals

All new intervals use the existing method (§1.14) unless a constraint below applies.

| item | method |
|---|---|
| resampling unit | the **target** (protein); never the pair, never the seed |
| resamples | **10,000** (`bootstrap_ci.N_RESAMPLES`) |
| interval | 95 %, percentile; RNG `default_rng(0)` |
| seeds | a resampled protein carries all its seeds, averaged. Primary-family intervals use seeds 1–3; extension E1 intervals use seeds 1–5, reported separately |
| precision@10 | mean over resampled proteins (existing `bootstrap_mean`) |
| **enrichment over chance** | ratio of two means over the **same** resampled proteins: mean precision@10 ÷ mean chance, with chance per protein = |sites within the scored window| ÷ length (the exact expectation of k uniformly random positions, `significance_test.py:23–28`). Reported with the point value and CI, next to the ceiling. A cell whose mean chance is 0 is not given a ratio. |
| **faithfulness delta** | mean `comprehensiveness_delta`, pairs grouped by target then resampled by target; requires per-pair values (gap G2), so the faithfulness step is re-run into `results/effects_v2/` with per-pair values kept. The existing sign verdict (rule 1.10) stays the primary statement; a **CI-qualified verdict** ("load-bearing iff the lower 95 % bound > 0") is reported **beside** it for every cell and never replaces it. |
| seed spread vs distance from chance | as rule 1.13; for 5 seeds see §7 |
| accuracy-vs-localization | Spearman ρ across cells, unit the cell (model × level × seed), 95 % percentile CI over 10,000 resamples of cells; a *description*, not a test |

No effect-size threshold ("meaningful", "≥ 2×") is defined; effect sizes are reported, not
classified.

## 6. Canary tolerance (T09)

The canary retrains one existing DAVIS cell of the cheapest attention model with the new harness
(`docs/REMEDIATION_PLAN.md`, T09). The pre-specified pass rule:

* **Compared quantities:** test AUROC and test AUPR (the binary-task metrics the trainer already
  writes to `<cell>_results.json`).
* **Tolerance:** the **sample standard deviation** (ddof = 1) of that metric over the three
  committed seeds of the *same model, dataset and level* — the standard `results/amp_validation_davis.md`
  (lines 5–6) already used for the AMP change ("within the full-precision seed-to-seed spread
  (sample sd over three seeds)"). The canary passes iff the retrained value lies within
  **±(that SD)** of the committed value **for the same seed**, for **both** metrics.
* **Degenerate case:** if that SD is 0, the canary is inconclusive and goes back to the user;
  no floor is invented.
* The canary is a **training-harness** check only. It does not re-score any explanation, so it
  adds nothing to any family.

Why not a tighter bit-for-bit criterion: the recorded practice is that identical seeds still vary
run to run through cuDNN non-determinism (`CLAUDE.md` §5, "same seed varies up to 0.058 CI"), and
a different machine or driver adds to that. Why not looser: one seed SD is the amount of
movement the paper already treats as noise, and a same-seed rerun should move less than a different seed does.

## 7. Decision rules for extension seeds (generalising §1.13)

To be implemented by parametrising `seed_agreement.py` (T12), with a test that the **3-seed output
reproduces `results/seed_agreement.md` exactly** before any 5-seed value is quoted:

* A seed is *significant on its own* iff its uncorrected permutation `p < 0.05` (α unchanged).
* A cell **disagrees** iff `0 < #significant < n_seeds`. (For n = 3 this is the coded rule.)
* A cell's **spread exceeds its signal** iff `max(precision) − min(precision) >
  |mean(precision) − chance|`, with the mean over the n seeds. (For n = 3 this is the coded rule.)
* **Reported separately, never merged:** the seeds 1–3 counts (the primary statement, 12 of 22
  cells disagree and 21 of 22 have spread over signal — `results/seed_agreement.md`, read in T01)
  and the seeds 1–5 counts for the E1 cells. A count is a count of cells, not a p-value, and comes
  with the spread and the CI of §5.
* **Replication reading (descriptive):** for the E1 cells, the seeds 4–5 disagreement pattern read
  alone, and whether a cell that agreed on seeds 1–3 still agrees. No test, no threshold.

## 8. What may not happen

* No p-value computed before approval; T05 and later start only after "approved".
* No family edited after a result in it is seen. A mistake in this document is fixed by an
  addendum that says what it changed and why, dated, and the corrected family is reported *with*
  the original.
* No selection among readouts, k values, models or ground truths on the basis of results.
* No extension output written under the original result directories; no original output modified
  (`git diff` cannot see the untracked ones — see D5).
* Verdict counts are always accompanied by the effect size and interval.

## 9. Not covered by this amendment (recorded so nobody assumes it is)

* **The training seeds of the 84 existing cells have no recorded initial-weight hash**
  (T01 discovered #3); this amendment cannot supply one, and seeds 4–5 must not reuse a seed
  number already used.
* **The 500-permutation primary audit files** cannot be regenerated without the escape flag
  (`--allow-low-permutations`); see D1.
* **DrugBAN's guards** (T02(a)–(c)) are unverified on this machine (no DGL).

## 10. Addenda

Append-only, each dated, each naming the section it fills. Expected addenda: A1 conservation
source and bins (E4); A2 method-applicability list and E3 sizes (E3); A3 readout sources and E8;
A4 the chosen modern model (E5); A5 new-split definitions (E6); A6 wave plan's DrugBAN decision
(E1 size); A7 anything discovered. None exists yet.

### A6 — DrugBAN seeds 4–5 in E1 (PROPOSED 2026-09-25 by T10; **APPROVED by the user 2026-09-25**, see A7)

Fills decision D2 (§11) and the E1 row of §3. **DrugBAN seeds 4–5 are not part of E1**: the plan (`docs/wave_plan.md`) has no measured
DrugBAN training time to budget them against. E1 therefore has design m = 16 (HyperAttentionDTI, MolTrans, ColdSite-DTI × 4 levels + the uniform
control × 4 levels). Committed before any seed-4 or seed-5 result exists (none does). Reversible only by a further addendum, dated and signed,
that adds the eight cells before their results are seen.

### A7 — user decisions of 2026-09-25 (recorded, not new analysis)

* **A6 approved.** DrugBAN seeds 4–5 are excluded from E1; E1's design m = 16. No seed-4/5 result exists (Wave A not launched).
* **A2 not signed.** E3 (new explanation methods) stays unsealed; no E3 cell may be scored. The manuscript reports the
  methods as implemented and tested, not as results.
* **A1 (conservation source): none will be provided.** E4 cannot run; the pocket enrichment is reported as "conservation not
  controlled" (problem P15 unresolved, a stated limitation).
* **DrugBAN on KIBA keeps mixed precision** (`recipes.uses_amp`: KIBA and not MolTrans), unvalidated for DrugBAN; the ACC3
  smoke run is the evidence if Wave A is ever launched.
* **Kaggle runs deferred** for the instructor draft; E1 and E2 therefore have no results, and the manuscript must say so.

## 11. Decisions the user is asked to take by signing

Replying **"approved"** approves this document **as written, including the defaults below**.
Any change requested is made and the document is re-issued for a fresh "approved".

| # | Decision | Default in this draft |
|---|---|---|
| **D1** | Which permutation count is the primary inferential record for P1/P2 | the 10,000-permutation re-runs; the 500-permutation originals preserved and reported alongside (§2.1) |
| **D2** | Whether DrugBAN seeds 4–5 are in E1 | conditional on the T10 wave plan; decision committed before any seed-4/5 result exists (m = 16 or 20) |
| **D3** | Apply Holm to the existing KLIFS-pocket ladders (S3) and the non-kinase control (E7), which were reported uncorrected | yes — stricter than what the draft says now; may reduce the "9/12 cells beat the within-domain nulls"-type statements |
| **D4** | Faithfulness CI-qualified verdict reported beside the sign verdict (needs a faithfulness re-run into `_v2`) | yes |
| **D5** | Extend the T02 manifest to hash the primary analysis outputs (they are untracked, so a change to them is otherwise undetectable) — still open from T02 | not done here; recommended before T05 |
| **D6** | Conservation source for E4 — none exists in the repository | user to name or approve a source before T06 |
| **D7** | Canary rule (§6): ± one committed-seed SD (the AMP-validation precedent), both metrics, same seed | as written |
| **D8** | E3 as **one joint family** per dataset rather than a family per method | joint (conservative) |

## 12. Sign-off

| | |
|---|---|
| Approved by | the user (Mahim), in chat: "I have reviewed the T03 protocol amendment: approved. You are cleared to run the `localize` stage." |
| Date | 2026-09-24 |
| Scope of approval | the document as written, **including the defaults in §11 (D1–D8)** — the user's reply named no change to any of them |
| Reply recorded in | `docs/REMEDIATION_LEDGER.md` § T03 and § T04 |
