# Supplementary Methods

<!-- T21 draft, 2026-09-25. Detail moved out of the main-text Methods (paper/v2/methods.md). Every digit
carries a hidden source tag (`python scripts/check_number_provenance.py`). Learning rates are written
as decimals so that the checker can compare them with the code's scientific notation. -->

## S1 Split sizes

One fixed split per level; the rows below are the data rows of the split files, whose SHA-256 hashes
are recorded in `data/splits/MANIFEST.json`. Only the random and cold-drug levels of KIBA were trained.

**Supplementary Table S1.** Rows per split (train / validation / test).

| dataset | level | train | validation | test |
|---|---|---|---|---|
| DAVIS | random | 21,039 | 3,006 | 6,011 | <!-- src: data/splits/davis/random/train.csv#rows = 21039 --> <!-- src: data/splits/davis/random/valid.csv#rows = 3006 --> <!-- src: data/splits/davis/random/test.csv#rows = 6011 -->
| DAVIS | cold-drug | 21,658 | 2,652 | 5,746 | <!-- src: data/splits/davis/cold_drug/train.csv#rows = 21658 --> <!-- src: data/splits/davis/cold_drug/valid.csv#rows = 2652 --> <!-- src: data/splits/davis/cold_drug/test.csv#rows = 5746 -->
| DAVIS | cold-target | 21,080 | 2,992 | 5,984 | <!-- src: data/splits/davis/cold_target/train.csv#rows = 21080 --> <!-- src: data/splits/davis/cold_target/valid.csv#rows = 2992 --> <!-- src: data/splits/davis/cold_target/test.csv#rows = 5984 -->
| DAVIS | cold-pair | 15,190 | 264 | 1,144 | <!-- src: data/splits/davis/cold_pair/train.csv#rows = 15190 --> <!-- src: data/splits/davis/cold_pair/valid.csv#rows = 264 --> <!-- src: data/splits/davis/cold_pair/test.csv#rows = 1144 -->
| KIBA | random | 82,778 | 11,825 | 23,651 | <!-- src: data/splits/kiba/random/train.csv#rows = 82778 --> <!-- src: data/splits/kiba/random/valid.csv#rows = 11825 --> <!-- src: data/splits/kiba/random/test.csv#rows = 23651 -->
| KIBA | cold-drug | 83,807 | 12,073 | 22,374 | <!-- src: data/splits/kiba/cold_drug/train.csv#rows = 83807 --> <!-- src: data/splits/kiba/cold_drug/valid.csv#rows = 12073 --> <!-- src: data/splits/kiba/cold_drug/test.csv#rows = 22374 -->

Cold-pair discards every pair with exactly one unseen entity, so it trains on fewer rows than the
random level (15,190 against 21,039 on DAVIS). A volume-matched control for this exists for XAttn-Ref
but is not used in this manuscript.
<!-- src: data/splits/davis/cold_pair/train.csv#rows = 15190 -->
<!-- src: data/splits/davis/random/train.csv#rows = 21039 -->

## S2 Training recipes

Each published model is trained from its vendored repository (`baselines/`) with its authors'
optimiser, learning rate, batch size and tokeniser; only data loading is replaced. The commands are
built by `src/cloud/recipes.py`, which reproduces the flags of the notebooks that trained the cells.

**Supplementary Table S2.** Optimiser, learning rate and batch size per model.

| model | optimiser | learning rate | batch | notes |
|---|---|---|---|---|
| DeepDTA | Adam | 0.001 | 256 | PyTorch port; patience 10 <!-- src: src/model/train_deepdta.py:144 = 0.001 --> <!-- src: src/cloud/recipes.py:22 = 256 --> <!-- src: src/cloud/recipes.py:53 = 10 -->
| HyperAttentionDTI | AdamW, cyclic learning rate | 0.00005 | 32 | weight decay 0.0001 on weights, none on biases <!-- src: src/model/train_hyperattentiondti.py:15 = 0.00005, 0.0001 --> <!-- src: src/cloud/recipes.py:21 = 32 -->
| MolTrans | Adam | 0.0001 | 16 | configuration `BIN_config_DBPE` <!-- src: src/model/train_moltrans.py:236 = 0.0001 --> <!-- src: src/cloud/recipes.py:23 = 16 -->
| DrugBAN | Adam | 0.00005 | 64 | domain adaptation off; DAVIS only <!-- src: src/cloud/recipes.py:24 = 64, 0.00005 -->
| XAttn-Ref | Adam, reduce-on-plateau | 0.001 | 64 | learning rate halved after 5 epochs without improvement <!-- src: src/model/train.py:311 = 0.001 --> <!-- src: src/cloud/recipes.py:20 = 64 --> <!-- src: src/model/train.py:227 = 5 -->

Two defects in the vendored MolTrans code are contained without editing it. Its forward pass reshapes
by the configured batch size, so a final partial batch would be scored wrongly; the batch size is set
per batch (`_fit_batch_size`). And one functional dropout call omits the training flag
(`baselines/MolTrans/models.py:103`), so dropout stays active at inference: its test metrics are one
draw of that noise, as in the published model. On DAVIS cold-pair, six further draws of one checkpoint
spread its test AUROC over 0.5862–0.5954, and the recorded value lies inside that range.
<!-- src: results/accuracy_v2/moltrans_coldpair_investigation.json#$.cells.cold_pair_seed1.arms.seeded[1].auroc = 0.5862 -->
<!-- src: results/accuracy_v2/moltrans_coldpair_investigation.json#$.cells.cold_pair_seed1.arms.seeded[0].auroc = 0.5954 -->

## S3 Readout sensitivity

A multi-head, multi-channel attention model must be reduced to one weight per residue. Besides the
primary readout (main text §2.2), the following reductions were scored with the same pipeline:
XAttn-Ref's protein self-attention, HyperAttentionDTI's maximum over channels and its receptive-field
spread, MolTrans's maximum over heads and its first layer, MolTrans's drug × protein interaction map,
and DrugBAN's maximum over atoms and receptive-field spread. The reductions disagree most for
HyperAttentionDTI against the KLIFS pocket at the cold-target level:

| readout | precision@10 (mean ± SD over seeds) | chance |
|---|---|---|
| primary (channel mean) | 0.186 ± 0.032 | 0.140 | <!-- src: results/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=published&level=cold_target->precision@10,chance = 0.186, 0.140 --> <!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_target->precision = 0.186 --> <!-- sd: readout_comparison.csv cell "0.186 ± 0.032" --> <!-- src: results/readout_comparison.csv:40 = 0.032 -->
| maximum over channels | 0.367 ± 0.104 | 0.140 | <!-- src: results/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=maxchannel&level=cold_target->precision@10,chance = 0.367, 0.140 --> <!-- src: results/readout_comparison.csv:52 = 0.104 -->
| receptive-field spread | 0.081 ± 0.046 | 0.140 | <!-- src: results/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=receptive&level=cold_target->precision@10,chance = 0.081, 0.140 --> <!-- src: results/readout_comparison.csv:56 = 0.046 -->

The readout-comparison table is written by the readout runs to the checkpoint-side results folder
and is versioned in the repository as `results/readout_comparison.csv` (a byte-identical copy). Drug dependence of each readout was measured on 25 proteins with four
drugs each: the top ten residues are identical across drugs in 100% of drug pairs for MolTrans's
primary readout, in 72% for HyperAttentionDTI's and in 0% for DrugBAN's.
<!-- src: results/drug_dependence/moltrans.txt:1 = 25, 100 -->
<!-- src: results/drug_dependence/hyperattentiondti.txt:1 = 72 -->
<!-- src: results/drug_dependence/drugban.txt:1 = 0 -->

## S4 Where each instrument was run

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

## S5 Threats to validity: the problem log

**Supplementary Table S3.** Problems found during the project, their effect, and how each is handled.
"Open" marks a problem that remains a limitation of this manuscript.

| # | problem | effect if unhandled | handling | status / evidence |
|---|---|---|---|---|
| a | DAVIS mutants carry the wild-type sequence | "unseen" targets partly seen; cold accuracy inflated | sequence policy; leak quantified by retraining | handled — `results/sequence_audit_davis.md`, `results/leakage_retrain_davis.md` |
| b | vendored MolTrans re-seeds the random generator on import | every seed trains as the first seed | seed after import; import wrapped to preserve generator state; affected cells retrained | handled — `tests/test_train_moltrans.py`, `tests/test_integrity_rng_seeds.py` |
| c | masking arms not size-matched for a sub-word model | random arm is roughly twice the intervention, biasing faithfulness | token-matched control | handled — `results/mask_comparability_davis.md` |
| d | match quality of MolTrans's token-matched control not recorded; matching can fail when attention sits at a sequence end | some control draws may exceed the tolerance | reported as a limitation | open — ledger T02, discovered items |
| e | MolTrans's primary readout may differ from the map its paper displays [MOLTRANS_FIG_X] | primary verdict may concern a different map | interaction map scored as a readout variant | open — `docs/readout_sources.md` |
| f | permutation resolution was 500 in the original audit | smallest p close to the smallest Holm threshold | 10,000-permutation audit is the primary record; verdicts unchanged | handled — `docs/PROTOCOL_AMENDMENT_v2.md` §2.1 | <!-- src: docs/REMEDIATION_LEDGER.md:592 = 500, 10000 -->
| g | two vendored repositories ship a module with the same name | one model silently loads the other's code | modules loaded in isolation | handled — `tests/test_drugban_import_isolation.py` |
| h | the analysis runner skipped finished outputs | a partial grid would freeze missing levels silently | runner refuses a changed grid outside a scratch folder | handled — `src/evaluation/run_all.py`, `tests/test_integrity_grid.py` |
| i | faithfulness depends on the number of residues masked | a verdict could reflect the chosen k | k = 10 fixed in advance; k = 50 as sensitivity for two models | handled — `results/faithfulness_k50_davis/` | <!-- src: docs/PROTOCOL_AMENDMENT_v2.md:163 = 10 --> <!-- src: docs/PROTOCOL_AMENDMENT_v2.md:169 = 50 -->
| j | DrugBAN's loader caps molecules at a fixed atom count | rows silently dropped | collector skips and fails loudly above a limit; the limit is a floor of three rows or 1%, whichever is larger | handled; limit wording corrected — ledger D6 | <!-- src: docs/REMEDIATION_LEDGER.md:242 = 1 -->
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

For (c), 47.9% of tokens change under the attended arm and 95.0% under an unmatched random arm, on
20 DAVIS proteins. <!-- src: results/mask_comparability_davis.md:5 = 47.9 -->
<!-- src: results/mask_comparability_davis.md:6 = 95.0 -->
<!-- src: results/mask_comparability_davis.md:3 = 20 -->
