# Certification report — "Seed-dependent verdicts"

Generated 2026-09-29 by `scripts/certification/build_report.py` from the evidence files in `results/certification/`.
Draft certified: `paper/v2/INSTRUCTOR_DRAFT.md` (sha256 `20546fa24afc2bb7…`). Deck certified: `Seed-Dependent Verdicts — Instructor Briefing.pptx` (sha256 `88030c142ae79715…`).

**What "certified" means here.** Every number in the draft traces to a raw file and value; every core quantitative claim was recomputed by code written
independently of the repo's analysis scripts; every numeric claim on the slides was checked against those recomputations. It does **not** cover the
literature review, the citations, the novelty claim as a matter of opinion, or models and seeds that were not re-run (see "Not covered" at the end).

**Reproduce:** `python scripts/certification/step0a_hash_results.py`, `step0b_trace_draft.py`, `step1_disagreement.py`, `step1b…`, `step1c…`, `step2_two_way_bootstrap.py`,
`step3_leak.py`, `step3b…`, `step4_holm.py`, `step5a/5b/5c…`, `step7_canary_forward_pass.py`, `key_numbers.py`, `step6_slide_audit.py`, `step8_red_team.py`, then `build_report.py`.
Outputs are in `results/certification/`.

## Section 1 — Traceability matrix (draft number → raw file → hash → match status)

**Summary.** 869 files under `results/` hashed (SHA-256 and MD5; `results_manifest.tsv`); git state {'tracked-clean': 317, 'ignored': 537, 'untracked': 15}.
The draft has 349 source tags: results `{'MANUAL': 18, 'MATCH': 331}`, 0 failing (the 18 MANUAL are prose-claim and note tags, verified by hand in Sections 2 and 4).
523 numeric values in the running prose; coverage `{'VALUE tagged & MATCH': 498, 'VALUE untagged (small integer)': 15, 'VALUE untagged (decimal/3+digit)': 10}`. The untagged tokens are citation years, the "256" of "SHA-256", and the "10" of "precision@10" (listed in `draft_numbers.tsv`); none is a quantitative claim.
61 distinct files are cited by the draft; git state {'tracked-clean': 38, 'untracked': 5, 'untracked/ignored': 18}; **modified after the draft was built: 0**; missing: 0.
Full matrix: `results/certification/traceability_matrix.tsv` (one row per tag: draft line, raw file and line or selector, draft value, raw value, status, sha256, git state).

### 1a. Headline numbers

| Draft line | Raw file (line or selector) | Draft value | Raw value | sha256[:16] | Git | Status |
|---|---|---|---|---|---|---|
| 20 | `results/certification/key_numbers.csv#name=seeds_disagree_exact->value` | 11 | 11 | `a10fd79712739996` | untracked | MATCH |
| 22 | `certification/key_numbers.csv#name=spread_exceeds_distance->value` | 21 | 21 | `a10fd79712739996` | untracked | MATCH |
| 25 | `results/certification/key_numbers.csv#name=friedman_holm->value` | 8 | 8 | `a10fd79712739996` | untracked | MATCH |
| 26 | `results/certification/key_numbers.csv#name=permutation_holm->value` | 7 | 7 | `a10fd79712739996` | untracked | MATCH |
| 49 | `certification/key_numbers.csv#name=leak_ct_leaked_rows->value,p` | 0.116,0.0013 | 0.1160,0.0013 | `a10fd79712739996` | untracked | MATCH |
| 50 | `certification/key_numbers.csv#name=leak_ct_unleaked_rows->value,p` | -0.019,0.31 | -0.0190,0.3131 | `a10fd79712739996` | untracked | MATCH |
| 685 | `certification/key_numbers.csv#name=leak_ct_all_rows->value,low,high,p` | 0.019,-0.022,0.059,0.19 | 0.0187,-0.022,0.059,0.1856 | `a10fd79712739996` | untracked | MATCH |
| 522 | `results/seed_agreement.md:34` | 12 | **12 of 22 cells have seeds  | `256426ddf2259f92` | tracked-clean | MATCH |
| 30 | `analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15` | 1,,20 | 1 of 20 cells survive correc | `12059ffe28bb2e4b` | tracked-clean | MATCH |
| 31 | `results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14` | 0,,8 | 0 of 8 cells survive correct | `e3e6ab6d99d071e1` | tracked-clean | MATCH |
| 149 | `effects_v2/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_drug->enrichment,enrichme` | 1.98,1.81,2.15 | 1.98111521,1.81212000,2.1515 | `505a52000e2164ac` | tracked-clean | MATCH |
| 32 | `effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=random->enrichment` | 1.66 | 1.66263806 | `270ae0d9f1182942` | tracked-clean | MATCH |
| 80 | `results/sequence_audit_davis.md:9` | 54 | 74 variant targets: **54 car | `5a98d4402594c160` | tracked-clean | MATCH |
| 213 | `results/sequence_audit_davis.md:21` | 12,,88,,816,,5984,,13.6 | | cold_target | test | 88 |  | `5a98d4402594c160` | tracked-clean | MATCH |
| 172 | `accuracy_v2/localization_spearman.csv#dataset=davis&group=all attention models&y=precision_at_10` | 48,,0.033,-0.276,0.322 | 48,0.033118,-0.276412,0.3219 | `89294d372f8fb23f` | tracked-clean | MATCH |
| 66 | `results/positive_control_davis.md:11` | 0.02 | | cold_target | 68 | 0.019 | | `64fb58b118b10356` | tracked-clean | MATCH |
| 39 | `results/analysis_davis_policyA/faithfulness_drugban_davis_seed1.md:6` | -0.0073 | | Cold-Drug | 0.1081 | 0.115 | `dc3ba449258f7273` | tracked-clean | MATCH |
| 224 | `effects_v2_2d/enrichment.csv#family=S1-klifs&model=coldsite_dti_ig&level=cold_drug->enrichment,e` | 3.16,2.40,4.40 | 3.16478919,2.39826523,4.4005 | `270ae0d9f1182942` | tracked-clean | MATCH |

### 1b. Every cited file: hash and match count

| Raw file | sha256[:16] | Tags | Matched | Git | Modified after draft |
|---|---|---|---|---|---|
| `docs/REMEDIATION_LEDGER.md` | `7229cba7c228c807` | 5 | 5 | tracked-clean | no |
| `src/evaluation/seed_agreement.py` | `cf9da27934255c28` | 5 | 5 | tracked-clean | no |
| `results/certification/key_numbers.csv` | `a10fd79712739996` | 65 | 65 | untracked | no |
| `results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md` | `12059ffe28bb2e4b` | 9 | 9 | tracked-clean | no |
| `results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md` | `e3e6ab6d99d071e1` | 8 | 8 | tracked-clean | no |
| `results/effects_v2_2d/enrichment.csv` | `270ae0d9f1182942` | 40 | 40 | tracked-clean | no |
| `results/analysis_davis_policyA/faithfulness_drugban_davis_seed1.md` | `dc3ba449258f7273` | 6 | 6 | tracked-clean | no |
| `results/positive_control_davis.md` | `64fb58b118b10356` | 5 | 5 | tracked-clean | no |
| `results/readout_comparison.csv` | `bd12fa6ddbe069dd` | 12 | 12 | tracked-clean | no |
| `results/sequence_audit_davis.md` | `5a98d4402594c160` | 11 | 11 | tracked-clean | no |
| `src/data/klifs_pocket.py` | `d32b84e5ec28b1d6` | 4 | 4 | tracked-clean | no |
| `results/seed_agreement.md` | `256426ddf2259f92` | 6 | 6 | tracked-clean | no |
| `results/effects_v2/enrichment.csv` | `505a52000e2164ac` | 3 | 3 | tracked-clean | no |
| `results/accuracy_v2/by_model.csv` | `bf50662af3afcf08` | 51 | 51 | tracked-clean | no |
| `results/accuracy_v2/localization_spearman.csv` | `89294d372f8fb23f` | 3 | 3 | tracked-clean | no |
| `results/effects_v2_2d/faithfulness_effects.csv` | `267574b98605511d` | 4 | 4 | tracked-clean | no |
| `results/analysis_davis_policyA/faithfulness_drugban_davis_seed3.md` | `40a9fb80fb299061` | 4 | 4 | tracked-clean | no |
| `results/drug_dependence/moltrans.txt` | `0a9f576e1321d394` | 3 | 3 | tracked-clean | no |
| `results/drug_dependence/drugban.txt` | `f9ce5903af430ef8` | 3 | 3 | tracked-clean | no |
| `src/model/dataset.py` | `57f920f5291fb0e7` | 1 | 1 | tracked-clean | no |
| `data/splits/davis/random/train.csv` | `c23ff50761065e3c` | 3 | 3 | untracked/ignored | no |
| `data/splits/davis/random/valid.csv` | `cc64ac8a36381647` | 2 | 2 | untracked/ignored | no |
| `data/splits/davis/random/test.csv` | `ad04c8b46e3719a8` | 2 | 2 | untracked/ignored | no |
| `results/sequence_audit_kiba.md` | `98b6c0f9d17e7eeb` | 1 | 1 | tracked-clean | no |
| `src/cloud/recipes.py` | `be44e654619459f7` | 8 | 8 | tracked-clean | no |
| `src/evaluation/collect.py` | `3b06146f79c1795f` | 1 | 1 | tracked-clean | no |
| `docs/PROTOCOL_AMENDMENT_v2.md` | `9839c617b8fb9c16` | 9 | 9 | tracked-clean | no |
| `results/certification/step4_holm.txt` | `5b887fd8c86dec64` | 2 | 2 | untracked | no |
| `src/evaluation/run_faithfulness.py` | `f1b1d48549de74f3` | 2 | 2 | tracked-clean | no |
| `src/evaluation/integrated_gradients.py` | `cdae92bf178b61ab` | 1 | 1 | tracked-clean | no |
| `results/certification/step1_disagreement.txt` | `7429fc243660a5b9` | 1 | 1 | untracked | no |
| `results/effects_v2_2d/enrichment.md` | `aa3ef8d877f482a2` | 2 | 2 | tracked-clean | no |
| `src/model/train.py` | `d404fcbe5808e9c1` | 3 | 3 | tracked-clean | no |
| `results/leakage_retrain_davis.md` | `e580d519c2a43a64` | 5 | 5 | tracked-clean | no |
| `results/certification/step1b_exact_p_validation.txt` | `2b4f9250f5a1293f` | 2 | 2 | untracked | no |
| `results/effects_v2_2d/faithfulness_effects.md` | `daf76984c21913c8` | 2 | 2 | tracked-clean | no |
| `results/analysis_davis_policyA/faithfulness_drugban_davis_seed2.md` | `7370ab738d9cf260` | 2 | 2 | tracked-clean | no |
| `results/drug_dependence/hyperattentiondti.txt` | `75826c178a93ca1a` | 2 | 2 | tracked-clean | no |
| `results/drug_dependence/moltrans_interaction.txt` | `1111ca5b5ccb91fa` | 1 | 1 | tracked-clean | no |
| `results/certification/step3_leak.txt` | `a3387e462748837f` | 1 | 1 | untracked | no |
| `docs/wave_plan.md` | `3e485723d2e4a4f7` | 5 | 5 | tracked-clean | no |
| `data/splits/davis/cold_drug/train.csv` | `54f4b4e0da846ecf` | 1 | 1 | untracked/ignored | no |
| `data/splits/davis/cold_drug/valid.csv` | `6e6ff45b30ffeaec` | 1 | 1 | untracked/ignored | no |
| `data/splits/davis/cold_drug/test.csv` | `fe9c24f59f540506` | 1 | 1 | untracked/ignored | no |
| `data/splits/davis/cold_target/train.csv` | `b82bfd9c4632c056` | 1 | 1 | untracked/ignored | no |
| `data/splits/davis/cold_target/valid.csv` | `86e407a411a86cf3` | 1 | 1 | untracked/ignored | no |
| `data/splits/davis/cold_target/test.csv` | `ab7ba68cb38bde7b` | 1 | 1 | untracked/ignored | no |
| `data/splits/davis/cold_pair/train.csv` | `ffb6c57f3992edd6` | 2 | 2 | untracked/ignored | no |
| `data/splits/davis/cold_pair/valid.csv` | `a8b4f3ff74dc9f5a` | 1 | 1 | untracked/ignored | no |
| `data/splits/davis/cold_pair/test.csv` | `4a1c859764ecc9fc` | 1 | 1 | untracked/ignored | no |
| `data/splits/kiba/random/train.csv` | `d3910d581fd380cb` | 1 | 1 | untracked/ignored | no |
| `data/splits/kiba/random/valid.csv` | `0dea8f58614275c0` | 1 | 1 | untracked/ignored | no |
| `data/splits/kiba/random/test.csv` | `3232e6fc95d7e1c6` | 1 | 1 | untracked/ignored | no |
| `data/splits/kiba/cold_drug/train.csv` | `fe0152e512ed7cb7` | 1 | 1 | untracked/ignored | no |
| `data/splits/kiba/cold_drug/valid.csv` | `b5982677de081044` | 1 | 1 | untracked/ignored | no |
| `data/splits/kiba/cold_drug/test.csv` | `ed87d4cbcc0f6978` | 1 | 1 | untracked/ignored | no |
| `src/model/train_deepdta.py` | `56505521cce3b1b5` | 1 | 1 | tracked-clean | no |
| `src/model/train_hyperattentiondti.py` | `d78bb6e8cedd37e8` | 1 | 1 | tracked-clean | no |
| `src/model/train_moltrans.py` | `f7d707b3411bff3b` | 1 | 1 | tracked-clean | no |
| `results/accuracy_v2/moltrans_coldpair_investigation.json` | `ced4b345ca5a5890` | 2 | 2 | tracked-clean | no |
| `results/mask_comparability_davis.md` | `3fa82df04391d929` | 3 | 3 | tracked-clean | no |

Notes. (i) The 18 `untracked/ignored` and `untracked` files are the split CSVs and the certification outputs; all 64 split files match `data/splits/MANIFEST.json` (Phase 1).
(ii) `docs/REMEDIATION_LEDGER.md` was edited 24 minutes after the draft was first built on 25 Sept (before this certification); its cited lines still match, and the counts it supports (84, 60, 24) were recounted independently in Section 2.
(iii) 537 of the 869 files are git-ignored (checkpoints, predictions, per-run JSON) and have no git anchor; `results_manifest.tsv` is the first hash record of them, and `predictions_manifest.csv` anchors the predictions.

## Section 2 — Independent math verification (my script output vs. draft claim)

All scripts read raw per-protein arrays, sequences, ground truth and saved predictions. None calls `seed_agreement.py`, `bootstrap_ci.py`, `effects_v2.py` or the repo's permutation code.

| # | Claim | Draft before this audit | Independent recomputation | Status |
|---|---|---|---|---|
| 1 | Cells whose three seeds disagree (alpha 0.05) | 12 of 22 | **11 of 22** with exact per-seed p (one borderline seed: sampled p 0.049 vs exact 0.056); 8 at alpha 0.01, 13 at 0.10 | **CORRECTED to 11**, disclosed |
| 2 | All three seeds pass / none pass | 1 / 9 | 1 / 10 | CORRECTED (none: 10) |
| 3 | Spread across seeds exceeds distance from chance | 21 of 22 | 21 of 22 (exact and recorded chance) | PASS |
| 4 | Direct test of seed variance, after Holm | (not in draft) | Friedman 8, permutation 7 of 22 (uncorrected 13 and 10) | ADDED |
| 5 | Re-run noise vs seed spread | (not in draft) | sd 0.0008 vs 0.0121 (ratio 15.5) | ADDED |
| 6 | Two-way bootstrap, HyperAttentionDTI DAVIS cold-drug | proteins only 1.98 [1.81, 2.15]; two-way [0.94, 3.65] | [1.813, 2.154]; [0.928, 3.676] at 200,000 resamples; at 10,000 resamples the bounds range 0.926–0.942 and 3.629–3.710 over 10 RNG seeds | PASS (within Monte-Carlo range) |
| 7 | Residue-level survivors, proteins-only → two-way | 7 / 22 → 1 / 22 | 7 / 22 → 1 / 22; the survivor is HyperAttentionDTI DAVIS random, lower bound 1.32 | PASS |
| 8 | Pocket-level survivors | 15 / 22 → 9 / 22 | 15 / 22 → 9 / 22 (one, HyperAttentionDTI cold-pair, has lower bound 1.02) | PASS, marginal cell disclosed |
| 9 | Sequence audit | 54 mutants, 442 targets, 379 sequences, 12 of 88 (816 of 5,984 rows), 11 of 88 (143 of 1,144) | identical from proteins.txt and the split CSVs (my rule counts 55 with the phospho-form ABL1p; 0 differ from wild type) | PASS |
| 10 | Leak effect (DeepDTA, cold-target, 3 seeds) | "worth 0.019 on all rows" | all rows +0.019 (CI [-0.022, 0.059], p 0.19); **leaked rows +0.116 ([0.098, 0.134], p 0.0013)**; unleaked rows -0.019 (p 0.31) | **CORRECTED**: headline is the leaked-row effect |
| 11 | AUROC on leaked vs unleaked rows | 0.950 vs 0.884 (original arm) | recomputed with sklearn from 24 raw prediction files: 0.950 vs 0.884; labels re-derived from pKd, 0 mismatches | PASS |
| 12 | Retrained arms (seqclean, seqmatched) | 0.116, 0.019 | raw predictions are not in the repo; summary arithmetic verified (+0.1160, +0.0187) | PASS (arithmetic only) |
| 13 | Holm over the audit family | 1 of 20 DAVIS, 0 of 8 KIBA | same; my Holm equals `statsmodels`; unchanged under 16-cell family, Bonferroni, Benjamini–Hochberg | PASS; scope note added (median over seeds; per-seed Holm keeps 5 of 48) |
| 14 | Faithfulness | 16 of 16 intervals above zero; smallest 0.054 [0.030, 0.080] | 16 of 16; smallest 0.054 [0.030, 0.081]; DrugBAN 6 of 12 negative | PASS |
| 15 | DAVIS random AUROC of five models | 0.891–0.937 | 0.891–0.937 | PASS |
| 16 | Spearman rho (accuracy vs enrichment, 48 cells) | 0.003 [−0.309, 0.294] | +0.004 [−0.300, +0.305] | PASS |
| 17 | Integrated gradients, XAttn-Ref DAVIS cold-drug, pocket | 3.16 [2.40, 4.40] vs attention 2.10 | 3.165 [2.40, 4.38]; four of four levels above chance | PASS |
| 18 | Models trained | 84 | 72 prediction files + 12 DrugBAN level-runs = 84 (28 setups × 3 seeds) | PASS |
| 19 | Live GPU forward pass (HyperAttentionDTI seed 1, 40 pairs, 4 levels) | — | checkpoint hashes identical; logits within 2.9e-06 of saved (40 of 40); precision@10 matches saved (UniProt 40 of 40, pocket 38 of 38) | PASS (one model, one seed) |

## Section 3 — Slide-alignment audit

**Final state:** PASS 64, SOURCE-ONLY 7, EXTERNAL 5, WARN 17, **FAIL 0** (`step6_slide_audit.txt`).
The 17 warnings are all one kind: results files the slides or draft cite as sources contain the code-name in their contents (raw tables, logs, JSON keys). This is the "raw logs where intended" case.

**Red flags found and fixed during the audit**

| # | Red flag | Fix |
|---|---|---|
| 1 | Headline "12 of 22" not reproduced by the exact test (11) | Draft and slides say 11, with the "different sides of alpha = 0.05" definition and the 12 disclosed |
| 2 | "Leak worth 0.019 AUROC" not distinguishable from zero | Reworded to lead with +0.116 on leaked rows (p = 0.0013), scope DeepDTA/cold-target stated |
| 3 | "1 of 20 survive Holm" read as a per-seed statement | Aggregation rule stated in Methods, Results and on slides 5 and 8 |
| 4 | Slide 1 "84 models · 3 seeds each" reads as 252 | "84 trained models · 28 setups × 3 seeds" |
| 5 | Slide 5 "best run = 4× the worst" (exact 3.94×) | "≈ 4×"; draft says "about four times" |
| 6 | "HyperAttentionDTI at three levels" included a marginal level | Marked marginal in notes, abstract, results, discussion, introduction |
| 7 | Slide 7 "all the raw numbers" over a table of 6 of 12 values | "two of four levels"; other levels pointed to the draft |
| 8 | Slide 8 "MolTrans map: same top ten for every drug" (interaction map varies) | "readout scored", with the qualifier in the notes |
| 9 | Stale "826 numbers" on slide 10 | 986 numbers, 0 failures |
| 10 | Code-name inside a file path in the draft's Methods | Rewritten to "the in-house model's KIBA notebook in the released code" |
| 11 | Conclusion slide did not mention the DAVIS leak or the checklist | Two cards added to slide 10 ("The benchmark leaks", "The fix: an eight-point checklist"); notes extended; both claims checked by the audit |

**Trap checks (final)**

| Trap | Result |
|---|---|
| Naming | Slides, notes, image alt text and slide XML: only the intended naming note on slide 1. Draft prose: 1 mention (the naming sentence), 0 in file paths. |
| Novelty / overclaim | 0 un-negated hits for SOTA, "state of the art", "new model/architecture", "outperform", "breakthrough". Slide 4 and the draft say "not a new model". |
| Compute (Wave A) | Slide 9 says planned, gated, "no Wave A result appears in any number in the paper". No `seed 4/5` or DrugBAN-on-KIBA file exists in `results/`. Hours match `config/wave_budget.json`. |
| Stale numbers | 0 (no "12 of 22", "9 of 12", "0.019 on all rows", "826 numbers"). |
| Core claims | The DAVIS leak (12 of 88) is on slides 2, 8 and 10; the eight-point checklist is on slide 8 and named on the conclusion slide (slide 10), where two cards were added during this audit. |

Slides were not rendered visually (LibreOffice is not installed here); the package validator passes, and text fit was checked by measuring wrapped line counts against box sizes.

## Section 4 — Red-team refutations

Full reasoning and every number: `results/certification/step8_red_team.txt`. Each attack ends with what the data do not let us say.

**Attack 1: "11/22 is an artefact of borderline effects crossing an alpha threshold."** *Answered.* Testing seeds against each other (no alpha on the effect): significant in 8 (Friedman) and 7 (permutation) of 22 cells after Holm; re-run noise is 15.5× smaller than seed spread; seed order is identical across two collection passes in 21 of 22 cells; HyperAttentionDTI DAVIS cold-drug seeds are 0.0249 / 0.0768 / 0.0195 against chance 0.0204. *Concession:* the count is threshold-dependent (8 at 0.01, 13 at 0.10), and the direct test covers about a third of cells. The "21 of 22" figure is the weakest shield and should not lead.

**Attack 2: "The models learned nothing; the audit audits noise."** *Answered.* AUROC 0.891–0.937 on the random split; masking attended residues beats random masking in 16 of 16 cells; integrated gradients on XAttn-Ref exceed chance in the pocket at 4 of 4 levels (independently recomputed, cold-drug 3.165 [2.40, 4.38]); live GPU passes reproduce the saved logits. *Concession:* against UniProt residues IG is above chance at only 1 of 4 levels, HyperAttentionDTI's KIBA cold-drug IG interval [0.64, 3.45] includes parity, and pocket enrichment is coarse (no conservation control). The supportable claim is "not noise; weights hold pocket information that attention under-reports", not "learned binding sites".

**Attack 3: "The DAVIS leak invalidates the benchmark."** *Answered for the explanation results.* 0 of the 12 (cold-target) and 0 of the 11 (cold-pair) leaked targets are scored by any attention model's explanation analysis; at random and cold-drug every test target is in training by construction. Accuracy on leaked rows is inflated (+0.116, p = 0.0013), not on unleaked rows (-0.019, p = 0.31); the same signature appears in all four models. *Concession:* the retrain is DeepDTA at cold-target only, all-rows effect not distinguishable from zero, and DAVIS cold-pair validation still contains seen-by-sequence targets (open).

**Attack 4 (added): "Three seeds cannot support this."** *Partly stands.* Two-way intervals are conservative and stable across RNG seeds; the single residue-level survivor is robust to the aggregation and family definition. *Concession:* per-seed Holm keeps 5 of 48 DAVIS and 3 of 18 KIBA seed-runs; seeds 4 and 5 are planned, not run, and no number depends on them.

## Not covered by this certification

1. **Models and seeds not re-run live.** The GPU canary covers HyperAttentionDTI seed 1 (40 pairs). MolTrans, XAttn-Ref, DrugBAN, DeepDTA, seeds 2–3 and KIBA are verified against saved files only; the saved attention tensors do not exist, only the top-10 consequences.
2. **Untracked inputs.** The prediction files, the retrain summary JSON, the integrated-gradients ladders and the checkpoints are outside git (some outside the repo); the certification scripts need local copies.
3. **Literature, citations and the novelty claim** are EXTERNAL.
4. **Scope of the science:** both datasets are kinase panels; no conservation control for the pocket result; the non-kinase panel is not analysed.
5. **Slides not visually rendered.**

## Final verdict

**CERTIFIED FOR SUBMISSION — numbers and wording, within the scope above.** Blocking errors remaining: **0** (traceability 0 failing, slide audit 0 FAIL, 986 numbers with 0 provenance failures, 1,202 tests passing).
Two items were corrected rather than confirmed (the headline count 12 → 11 and the leak headline 0.019 → 0.116 on leaked rows); both are now reflected in the draft and slides. Before submission I recommend (non-blocking): render the deck once in PowerPoint and inspect slides 2, 5, 7 and 10; and decide whether to run the full-model recompute for the remaining models.
