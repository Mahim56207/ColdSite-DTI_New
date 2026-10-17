# Supplementary Information

## S1 Split sizes

One fixed split per level; the rows below are the data rows of the split files, whose SHA-256 hashes
are recorded in `data/splits/MANIFEST.json`. Only the random and cold-drug levels of KIBA were trained.

**Supplementary Table S1.** Rows per split (train / validation / test).

| dataset | level | train | validation | test |
|---|---|---|---|---|
| DAVIS | random | 21,039 | 3,006 | 6,011 |
| DAVIS | cold-drug | 21,658 | 2,652 | 5,746 |
| DAVIS | cold-target | 21,080 | 2,992 | 5,984 |
| DAVIS | cold-pair | 15,190 | 264 | 1,144 |
| KIBA | random | 82,778 | 11,825 | 23,651 |
| KIBA | cold-drug | 83,807 | 12,073 | 22,374 |

Cold-pair discards every pair with exactly one unseen entity, so it trains on fewer rows than the
random level (15,190 against 21,039 on DAVIS). A volume-matched control for this exists for XAttn-Ref
but is not used in this manuscript.

## S2 Training recipes

Each published model is trained from its vendored repository (`baselines/`) with its authors'
optimiser, learning rate, batch size and tokeniser; only data loading is replaced. The commands are
built by `src/cloud/recipes.py`, which reproduces the flags of the notebooks that trained the cells.

**Supplementary Table S2.** Optimiser, learning rate and batch size per model.

| model | optimiser | learning rate | batch | notes |
|---|---|---|---|---|
| DeepDTA | Adam | 0.001 | 256 | PyTorch port; patience 10
| HyperAttentionDTI | AdamW, cyclic learning rate | 0.00005 | 32 | weight decay 0.0001 on weights, none on biases
| MolTrans | Adam | 0.0001 | 16 | configuration `BIN_config_DBPE`
| DrugBAN | Adam | 0.00005 | 64 | domain adaptation off; DAVIS only
| XAttn-Ref | Adam, reduce-on-plateau | 0.001 | 64 | learning rate halved after 5 epochs without improvement

**Mixed precision on KIBA.** DeepDTA and HyperAttentionDTI were trained with mixed precision (float16 autocast) on KIBA, as recorded in each cell's results file; MolTrans (whose layer normalisation underflows in half precision) and XAttn-Ref were trained in full precision, as was every DAVIS cell. The choice was validated beforehand on one DAVIS cell, HyperAttentionDTI at cold-pair, with three seeds per arm: test AUROC was 0.694 ± 0.038 in full precision (Kaggle, torch 2.10) and 0.681 ± 0.068 in mixed precision (Colab, torch 2.11), and precision@10 was 0.022 ± 0.012 against 0.023 ± 0.018, so the mixed-precision means lay within the full-precision seed spread. Three seeds per arm detect only a gross effect and cannot show equivalence, and the two arms also differed in library version and machine.

Two defects in the vendored MolTrans code are contained without editing it. Its forward pass reshapes
by the configured batch size, so a final partial batch would be scored wrongly; the batch size is set
per batch (`_fit_batch_size`). And one functional dropout call omits the training flag
(`baselines/MolTrans/models.py:103`), so dropout stays active at inference: its test metrics are one
draw of that noise, as in the published model. On DAVIS cold-pair, six further draws of one checkpoint
spread its test AUROC over 0.5862–0.5954, and the recorded value lies inside that range.

**Supplementary Table S3.** Test AUROC, mean ± standard deviation over three training seeds. "Sequence-unseen" scores
only test targets whose sequence does not occur in training (Section 3.6). DrugBAN has no KIBA cells.
*Source:* `results/accuracy_v2/by_model.csv`.

| model | DAVIS random | DAVIS cold-drug | DAVIS cold-target | cold-target, sequence-unseen | DAVIS cold-pair | cold-pair, sequence-unseen | KIBA random | KIBA cold-drug |
|---|---|---|---|---|---|---|---|---|
| MolTrans | 0.923 ± 0.002 | 0.685 ± 0.019 | 0.874 ± 0.006 | 0.833 ± 0.008 | 0.568 ± 0.025 | 0.532 ± 0.029 | 0.919 ± 0.004 | 0.812 ± 0.009  |
| HyperAttentionDTI | 0.937 ± 0.005 | 0.760 ± 0.042 | 0.915 ± 0.001 | 0.893 ± 0.001 | 0.694 ± 0.038 | 0.713 ± 0.050 | 0.933 ± 0.001 | 0.844 ± 0.004  |
| DrugBAN | 0.891 ± 0.006 | 0.695 ± 0.033 | 0.846 ± 0.010 | 0.801 ± 0.012 | 0.620 ± 0.031 | 0.559 ± 0.040 | — | —  |
| XAttn-Ref | 0.924 ± 0.001 | 0.721 ± 0.008 | 0.857 ± 0.011 | 0.835 ± 0.015 | 0.624 ± 0.099 | 0.607 ± 0.128 | 0.894 ± 0.011 | 0.807 ± 0.009  |
| DeepDTA | 0.929 ± 0.002 | 0.692 ± 0.044 | 0.907 ± 0.003 | 0.884 ± 0.003 | 0.728 ± 0.035 | 0.749 ± 0.044 | 0.918 ± 0.001 | 0.832 ± 0.001  |

## S4 Readout sensitivity

A multi-head, multi-channel attention model must be reduced to one weight per residue. Besides the
primary readout (Section 2.2), the following reductions were scored with the same pipeline:
XAttn-Ref's protein self-attention, HyperAttentionDTI's maximum over channels and its receptive-field
spread, MolTrans's maximum over heads and its first layer, MolTrans's drug × protein interaction map,
and DrugBAN's maximum over atoms and receptive-field spread. The reductions disagree most for
HyperAttentionDTI against the KLIFS pocket at the cold-target level:

**Supplementary Table S4.** HyperAttentionDTI, KLIFS pocket, DAVIS cold-target.

| readout | precision@10 (mean ± SD over seeds) | chance |
|---|---|---|
| primary (channel mean) | 0.186 ± 0.032 | 0.140 |
| maximum over channels | 0.367 ± 0.104 | 0.140 |
| receptive-field spread | 0.081 ± 0.046 | 0.140 |

The readout-comparison table is written by the readout runs to the checkpoint-side results folder
and is versioned in the repository as `results/readout_comparison.csv` (a byte-identical copy). Drug dependence of each readout was measured on 25 proteins with four
drugs each: the top ten residues are identical across drugs in 100% of drug pairs for MolTrans's
primary readout, in 72% for HyperAttentionDTI's and in 0% for DrugBAN's.
MolTrans's drug × protein interaction map, scored as a readout variant, gives identical top-ten residues across drugs in 0% of the same 150 drug pairs.

## S5 Integrated gradients on the same weights (confirmatory)

Integrated gradients (Sundararajan et al., 2017; 32 steps) on the same checkpoints localise better than the attention in some cells, not
all.
 For XAttn-Ref at DAVIS cold-drug, IG reaches 3.16 [2.40, 4.40]× chance in the pocket against the
attention's 2.10 [1.83, 2.36]×. For HyperAttentionDTI at KIBA cold-drug, IG's point estimate against
UniProt residues, 1.90×, is above the attention's 0.94 [0.71, 1.19]×, but with seeds resampled its
interval, [0.63, 3.39], includes parity. IG coverage is partial
(this Supplement, S6), and we read it only as evidence that a weak attention map can under-report
what a model represents.

## S6 Where each instrument was run

| instrument | code | outputs |
|---|---|---|
| audit with Holm (primary record) | `src/evaluation/run_audit.py` | `results/analysis_{davis,kiba}_policyA/audit_*_10k_permutations.*` |
| UniProt and KLIFS ladders | `src/evaluation/run_ladder.py` | `results/analysis_*_policyA*/` |
| enrichment and faithfulness intervals (two-way: seeds and targets resampled; quoted in the text) | `src/evaluation/effects_v2.py --resample seeds_and_targets` | `results/effects_v2_2d/` |
| enrichment and faithfulness intervals (targets only, as the amendment specified; comparison) | `src/evaluation/effects_v2.py` | `results/effects_v2/` |
| readout comparison | readout runs (checkpoint-side), copied byte-identical | `results/readout_comparison.csv` |
| seed agreement | `src/evaluation/seed_agreement.py` | `results/seed_agreement.md` |
| positive control | `src/evaluation/positive_control.py` | `results/positive_control_{davis,kiba}.md` |
| positional and residue nulls | `src/evaluation/positional_control.py` | all four attention models on DAVIS, UniProt and KLIFS (`results/analysis_davis_policyA/positional_control_*`); HyperAttentionDTI and MolTrans on KIBA against both ground truths, XAttn-Ref on KIBA against KLIFS only (`results/analysis_kiba_policyA/positional_control_*`) |
| faithfulness (committed means) | `src/evaluation/run_faithfulness.py`, `token_faithfulness.py` | `results/analysis_*_policyA/` |
| accuracy and accuracy-vs-localization | `src/evaluation/accuracy_table.py` | `results/accuracy_v2/` |
| leakage retraining | `src/evaluation/leakage_retrain.py` | `results/leakage_retrain_davis.md` |

## S7 Threats to validity: the problem log

**Supplementary Table S5.** Problems found during the project, their effect, and how each is handled.
"Open" marks a problem that remains a limitation of this manuscript.

| # | problem | effect if unhandled | handling | status / evidence |
|---|---|---|---|---|
| a | DAVIS mutants carry the wild-type sequence | "unseen" targets partly seen; cold accuracy inflated | sequence policy; leak quantified by retraining | handled — `results/sequence_audit_davis.md`, `results/leakage_retrain_davis.md` |
| b | vendored MolTrans re-seeds the random generator on import | every seed trains as the first seed | seed after import; import wrapped to preserve generator state; affected cells retrained | handled — `tests/test_train_moltrans.py`, `tests/test_integrity_rng_seeds.py` |
| c | masking arms not size-matched for a sub-word model | random arm is roughly twice the intervention, biasing faithfulness | token-matched control | handled — `results/mask_comparability_davis.md` |
| d | match quality of MolTrans's token-matched control not recorded; matching can fail when attention sits at a sequence end | some control draws may exceed the tolerance | reported as a limitation | open — ledger T02, discovered items |
| e | the primary readouts may differ from the maps the papers display; not confirmed against their figures | a primary verdict may concern a different map | alternative readouts scored as variants (MolTrans's interaction map; others in S4) | open — `docs/readout_sources.md` |
| f | permutation resolution was 500 in the original audit | smallest p close to the smallest Holm threshold | 10,000-permutation audit is the primary record; verdicts unchanged | handled — `docs/PROTOCOL_AMENDMENT_v2.md` §2.1 |
| g | two vendored repositories ship a module with the same name | one model silently loads the other's code | modules loaded in isolation | handled — `tests/test_drugban_import_isolation.py` |
| h | the analysis runner skipped finished outputs | a partial grid would freeze missing levels silently | runner refuses a changed grid outside a scratch folder | handled — `src/evaluation/run_all.py`, `tests/test_integrity_grid.py` |
| i | faithfulness depends on the number of residues masked | a verdict could reflect the chosen k | k = 10 fixed in advance; k = 50 as sensitivity for two models | handled — `results/faithfulness_k50_davis/` |
| j | DrugBAN's loader caps molecules at a fixed atom count | rows silently dropped | collector skips and fails loudly above a limit; the limit is a floor of three rows or 1%, whichever is larger | handled; limit wording corrected — ledger D6 |
| k | MolTrans keeps dropout on at inference | single-draw test metrics and faithfulness re-runs differ slightly | forward passes seeded for explanations; stated | open (as published) — ledger T04, T05 |
| l | HyperAttentionDTI masking would have written alanine; its positive logit was read alone | a different intervention and a quantity that can move without the prediction | mask to `X` via each model's tokeniser; log-odds | handled — `src/evaluation/residue_space.py` |
| m | ground-truth residues numbered along UniProt, not along the dataset sequence | sites point at the wrong residue or past the end | alignment-based re-numbering; the positive control fails on un-renumbered sites | handled — `src/data/align_ground_truth.py` |
| n | tied attention broken by a stable sort | positional artefact read as quality | ties broken at random | handled — `src/evaluation/precision_at_k.py` |
| o | Holm families enlarged after first results | post-hoc family definition | disclosed; frozen by dated amendment | disclosed — `docs/PROTOCOL_AMENDMENT_v2.md` G4 |
| p | faithfulness re-runs do not reproduce the committed means exactly for MolTrans and for HyperAttentionDTI on DAVIS | intervals describe a re-run draw | committed means stay primary; verdict signs unchanged | open — ledger T05 |
| q | an interrupted checkpoint was once analysed | a partly trained model scored as final | runner refuses a checkpoint without its results file | handled — `CLAUDE.md` §6 |
| r | no random-generator state or initial-weight hash recorded for the trained cells | seed distinctness not provable after the fact | recorded for all future cells by the cloud harness | open for existing cells — ledger T01 |
| s | DAVIS cold-pair validation contains seen-by-sequence targets | checkpoint selection influenced by the leak | cannot be undone without retraining | open — `results/sequence_audit_davis.md` |
| t | no conservation control for the pocket enrichment | enrichment may reflect conserved residues | none | open — amendment A1/A7 |
| u | the sampled 1,000-permutation per-seed p put one seed on the α line | headline disagreement count 12 vs 11 | exact p recomputed independently; 11 reported, 12 disclosed | handled — `scripts/certification/step1b_exact_p_validation.py` |
| v | the leak's all-rows AUROC effect is not distinguishable from zero with three seeds | a small, insignificant number quoted as a finding | not quoted; leaked-row effect reported with its test | handled — `results/certification/step3_leak.txt` |
| w | the same seed and the same initial weights give different results on re-running (three reruns of one cell) | part of the "seed" variation is run-to-run variation, and the cause of the map disagreement is not isolated | stated as a limitation; both canary runs are reported | open — `docs/canary_attempts/README.md` |
| x | the 10,000-permutation audit's cell p-values differ from an exact recomputation by up to 0.0511 on DAVIS and by 0.54 in one KIBA cell; the cause is not established | the survivor counts could differ | Holm recomputed from the exact p-values leaves the same survivors | open — `results/certification/step4_holm.txt` |

For (c), 47.9% of tokens change under the attended arm and 95.0% under an unmatched random arm, on
20 DAVIS proteins.

## S8 Per-cell tables

**Supplementary Table S6.** Seed agreement for each of the 22 model–dataset–level cells scored against UniProt residues. Precision@10 of the three seeds in order, the cell's exact chance level, and each seed's exact permutation p-value; the marks give, for each seed in order, whether its exact p-value is below 0.05 (\* yes, . no); "disagree" means some but not all seeds are below it. The last two columns are the audit's median-over-seeds p-value and whether the cell survives Holm correction over its dataset's family (uniform-map cells are in the family but not listed).

| dataset | model | level | precision@10, per seed | chance | exact p, per seed | marks | disagree | audit p (median) | Holm |
|-------|--------------|----------|--------------------|--------|----------------------|-------|--------|----------|------|
| DAVIS | MolTrans | random | 0.024 / 0.021 / 0.019 | 0.0204 | 0.0563 / 0.4768 / 0.7098 | ... | no | 0.4782 | no |
| DAVIS | MolTrans | cold-drug | 0.022 / 0.032 / 0.024 | 0.0204 | 0.2955 / 0.0000 / 0.0563 | .*. | yes | 0.0318 | no |
| DAVIS | MolTrans | cold-target | 0.012 / 0.038 / 0.029 | 0.0194 | 0.9537 / 0.0009 / 0.0443 | .** | yes | 0.0141 | no |
| DAVIS | MolTrans | cold-pair | 0.004 / 0.032 / 0.024 | 0.0190 | 0.9999 / 0.0115 / 0.2120 | .*. | yes | 0.2036 | no |
| DAVIS | HyperAttentionDTI | random | 0.035 / 0.039 / 0.028 | 0.0204 | 0.0000 / 0.0000 / 0.0017 | *** | no | 9.999e-05 | yes |
| DAVIS | HyperAttentionDTI | cold-drug | 0.025 / 0.077 / 0.019 | 0.0204 | 0.0348 / 0.0000 / 0.6662 | **. | yes | 0.0373 | no |
| DAVIS | HyperAttentionDTI | cold-target | 0.018 / 0.031 / 0.025 | 0.0194 | 0.6683 / 0.0257 / 0.1730 | .*. | yes | 0.1219 | no |
| DAVIS | HyperAttentionDTI | cold-pair | 0.013 / 0.022 / 0.032 | 0.0190 | 0.9306 / 0.2953 / 0.0115 | ..* | yes | 0.2935 | no |
| DAVIS | DrugBAN | random | 0.019 / 0.015 / 0.019 | 0.0204 | 0.7098 / 0.9956 / 0.7098 | ... | no | 0.7109 | no |
| DAVIS | DrugBAN | cold-drug | 0.022 / 0.021 / 0.021 | 0.0204 | 0.2955 / 0.4768 / 0.4768 | ... | no | 0.4477 | no |
| DAVIS | DrugBAN | cold-target | 0.018 / 0.028 / 0.022 | 0.0194 | 0.6683 / 0.0729 / 0.3409 | ... | no | 0.2928 | no |
| DAVIS | DrugBAN | cold-pair | 0.036 / 0.022 / 0.024 | 0.0190 | 0.0015 / 0.2953 / 0.2120 | *.. | yes | 0.2036 | no |
| DAVIS | XAttn-Ref | random | 0.023 / 0.009 / 0.013 | 0.0204 | 0.1566 / 1.0000 / 0.9997 | ... | no | 0.9997 | no |
| DAVIS | XAttn-Ref | cold-drug | 0.027 / 0.027 / 0.011 | 0.0204 | 0.0047 / 0.0064 / 1.0000 | **. | yes | 0.006599 | no |
| DAVIS | XAttn-Ref | cold-target | 0.015 / 0.019 / 0.018 | 0.0194 | 0.8504 / 0.5582 / 0.6683 | ... | no | 0.6581 | no |
| DAVIS | XAttn-Ref | cold-pair | 0.018 / 0.013 / 0.008 | 0.0190 | 0.6108 / 0.9306 / 0.9938 | ... | no | 0.9296 | no |
| KIBA | MolTrans | random | 0.020 / 0.053 / 0.022 | 0.0228 | 0.7946 / 0.0000 / 0.6432 | .*. | yes | 0.5736 | no |
| KIBA | MolTrans | cold-drug | 0.028 / 0.063 / 0.017 | 0.0228 | 0.0706 / 0.0000 / 0.9830 | .*. | yes | 0.07279 | no |
| KIBA | HyperAttentionDTI | random | 0.017 / 0.022 / 0.053 | 0.0228 | 0.9730 / 0.6432 / 0.0000 | ..* | yes | 0.5175 | no |
| KIBA | HyperAttentionDTI | cold-drug | 0.019 / 0.020 / 0.025 | 0.0228 | 0.9065 / 0.8035 / 0.2655 | ... | no | 0.2624 | no |
| KIBA | XAttn-Ref | random | 0.018 / 0.011 / 0.026 | 0.0228 | 0.9258 / 1.0000 / 0.1723 | ... | no | 0.9262 | no |
| KIBA | XAttn-Ref | cold-drug | 0.025 / 0.020 / 0.015 | 0.0228 | 0.3147 / 0.8035 / 0.9956 | ... | no | 0.7963 | no |

**Supplementary Table S7.** Enrichment over chance (mean precision@10 divided by mean chance, over the same proteins) for every model, level and ground truth, with 95% percentile intervals from a bootstrap that resamples both proteins and seeds (10,000 resamples); n is the number of proteins scored after the sequence policy; an interval that excludes one lies entirely on one side of chance. This is the table behind Figure 2.

| ground truth | dataset | model | level | n | precision@10 | chance | enrichment | interval |
|-----------|-------|--------------|-----------|-----|---------|--------|---------|--------------|
| UniProt | DAVIS | MolTrans | random | 349 | 0.021 | 0.020 | 1.05 | 0.79 to 1.33 |
| UniProt | DAVIS | MolTrans | cold-drug | 349 | 0.026 | 0.020 | 1.27 | 0.93 to 1.66 |
| UniProt | DAVIS | MolTrans | cold-target | 68 | 0.026 | 0.019 | 1.37 | 0.53 to 2.32 |
| UniProt | DAVIS | MolTrans | cold-pair | 72 | 0.020 | 0.019 | 1.05 | 0.22 to 1.95 |
| UniProt | DAVIS | HyperAttentionDTI | random | 349 | 0.034 | 0.020 | 1.66 | 1.33 to 2.00 |
| UniProt | DAVIS | HyperAttentionDTI | cold-drug | 349 | 0.040 | 0.020 | 1.98 | 0.94 to 3.65 |
| UniProt | DAVIS | HyperAttentionDTI | cold-target | 68 | 0.025 | 0.019 | 1.27 | 0.73 to 1.88 |
| UniProt | DAVIS | HyperAttentionDTI | cold-pair | 72 | 0.022 | 0.019 | 1.17 | 0.58 to 1.82 |
| UniProt | DAVIS | DrugBAN | random | 349 | 0.018 | 0.020 | 0.87 | 0.67 to 1.07 |
| UniProt | DAVIS | DrugBAN | cold-drug | 349 | 0.021 | 0.020 | 1.03 | 0.86 to 1.20 |
| UniProt | DAVIS | DrugBAN | cold-target | 68 | 0.023 | 0.019 | 1.16 | 0.68 to 1.69 |
| UniProt | DAVIS | DrugBAN | cold-pair | 72 | 0.027 | 0.019 | 1.44 | 0.90 to 2.08 |
| UniProt | DAVIS | XAttn-Ref | random | 349 | 0.015 | 0.020 | 0.74 | 0.43 to 1.11 |
| UniProt | DAVIS | XAttn-Ref | cold-drug | 349 | 0.022 | 0.020 | 1.06 | 0.62 to 1.45 |
| UniProt | DAVIS | XAttn-Ref | cold-target | 68 | 0.017 | 0.019 | 0.89 | 0.54 to 1.25 |
| UniProt | DAVIS | XAttn-Ref | cold-pair | 72 | 0.013 | 0.019 | 0.68 | 0.27 to 1.15 |
| UniProt | KIBA | MolTrans | random | 211 | 0.032 | 0.023 | 1.39 | 0.72 to 2.28 |
| UniProt | KIBA | MolTrans | cold-drug | 212 | 0.036 | 0.023 | 1.57 | 0.70 to 2.70 |
| UniProt | KIBA | HyperAttentionDTI | random | 211 | 0.030 | 0.023 | 1.34 | 0.68 to 2.25 |
| UniProt | KIBA | HyperAttentionDTI | cold-drug | 212 | 0.021 | 0.023 | 0.94 | 0.71 to 1.19 |
| UniProt | KIBA | XAttn-Ref | random | 211 | 0.019 | 0.023 | 0.82 | 0.50 to 1.18 |
| UniProt | KIBA | XAttn-Ref | cold-drug | 212 | 0.020 | 0.023 | 0.88 | 0.58 to 1.17 |
| KLIFS pocket | DAVIS | MolTrans | random | 350 | 0.157 | 0.143 | 1.10 | 0.86 to 1.32 |
| KLIFS pocket | DAVIS | MolTrans | cold-drug | 350 | 0.154 | 0.142 | 1.08 | 0.89 to 1.27 |
| KLIFS pocket | DAVIS | MolTrans | cold-target | 67 | 0.176 | 0.140 | 1.26 | 0.77 to 1.70 |
| KLIFS pocket | DAVIS | MolTrans | cold-pair | 71 | 0.136 | 0.137 | 0.99 | 0.67 to 1.32 |
| KLIFS pocket | DAVIS | HyperAttentionDTI | random | 350 | 0.242 | 0.143 | 1.70 | 1.48 to 1.98 |
| KLIFS pocket | DAVIS | HyperAttentionDTI | cold-drug | 350 | 0.193 | 0.142 | 1.35 | 0.94 to 1.73 |
| KLIFS pocket | DAVIS | HyperAttentionDTI | cold-target | 67 | 0.186 | 0.140 | 1.33 | 1.08 to 1.60 |
| KLIFS pocket | DAVIS | HyperAttentionDTI | cold-pair | 71 | 0.185 | 0.137 | 1.36 | 1.03 to 1.73 |
| KLIFS pocket | DAVIS | DrugBAN | random | 350 | 0.139 | 0.143 | 0.98 | 0.91 to 1.04 |
| KLIFS pocket | DAVIS | DrugBAN | cold-drug | 350 | 0.146 | 0.142 | 1.02 | 0.93 to 1.11 |
| KLIFS pocket | DAVIS | DrugBAN | cold-target | 67 | 0.145 | 0.140 | 1.04 | 0.90 to 1.17 |
| KLIFS pocket | DAVIS | DrugBAN | cold-pair | 71 | 0.138 | 0.137 | 1.01 | 0.87 to 1.17 |
| KLIFS pocket | DAVIS | XAttn-Ref | random | 350 | 0.219 | 0.143 | 1.54 | 1.36 to 1.69 |
| KLIFS pocket | DAVIS | XAttn-Ref | cold-drug | 350 | 0.299 | 0.142 | 2.10 | 1.83 to 2.36 |
| KLIFS pocket | DAVIS | XAttn-Ref | cold-target | 67 | 0.243 | 0.140 | 1.74 | 1.40 to 2.13 |
| KLIFS pocket | DAVIS | XAttn-Ref | cold-pair | 71 | 0.271 | 0.137 | 1.99 | 1.69 to 2.35 |
| KLIFS pocket | KIBA | MolTrans | random | 210 | 0.161 | 0.152 | 1.07 | 0.78 to 1.40 |
| KLIFS pocket | KIBA | MolTrans | cold-drug | 211 | 0.188 | 0.151 | 1.24 | 1.00 to 1.53 |
| KLIFS pocket | KIBA | HyperAttentionDTI | random | 210 | 0.207 | 0.152 | 1.37 | 1.26 to 1.49 |
| KLIFS pocket | KIBA | HyperAttentionDTI | cold-drug | 211 | 0.189 | 0.151 | 1.25 | 1.07 to 1.41 |
| KLIFS pocket | KIBA | XAttn-Ref | random | 210 | 0.166 | 0.152 | 1.09 | 0.97 to 1.21 |
| KLIFS pocket | KIBA | XAttn-Ref | cold-drug | 211 | 0.152 | 0.151 | 1.00 | 0.79 to 1.27 |

## S9 Supplementary figures

![Supplementary Figure S1](../v2/figures/fig3_accuracy.png)

**Supplementary Figure S1. Predictive accuracy per training seed.** Test AUROC of every cell (points, one per seed; bar = mean) on the uncorrected test sets. DeepDTA has no attention and anchors accuracy. KIBA has two levels (random, cold-drug) and no DrugBAN cells.

![Supplementary Figure S2](../v2/figures/fig4_accuracy_vs_localization.png)

**Supplementary Figure S2. Accuracy does not predict localisation.** One point per attention-model cell and seed: test AUROC against enrichment over chance (UniProt, precision@10 divided by chance). For DAVIS cold-target and cold-pair, AUROC is scored on sequence-unseen targets, the same policy the explanation metrics use. Panel titles give Spearman's ρ with a 95% bootstrap interval over cells.
