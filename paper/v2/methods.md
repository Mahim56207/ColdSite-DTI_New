# 2 Methods

<!-- T21 draft, 2026-09-25 (user's scope: "Drafting: Methods"). Condensed main-text Methods; detail,
recipes, split sizes and the problem log are in paper/v2/supplement_methods.md. Every digit carries a
hidden source tag (`python scripts/check_number_provenance.py`). The venue's author guidelines
(docs/bib_guidelines.md) were not supplied, so no word limit has been applied or verified. -->

## 2.1 Data, task and splits

DAVIS (2011) and KIBA (2014) are used as distributed with DeepDTA (2018). Every model solves the same
binary task: a pair is positive if pKd ≥ 7.0 on DAVIS or its KIBA score is ≥ 12.1, one constant shared
by all trainers. <!-- src: src/model/dataset.py:30 = 7.0, 12.1 -->
Each dataset is split at four levels of distribution shift: random; cold-drug, where test drugs are
absent from training; cold-target, where test targets are absent; and cold-pair, where both are. There
is one fixed split per level (sizes in Supplementary Table S1); the three seeds of every cell are
training seeds, which change weight initialisation and batch order, so the spread across seeds is
training variance and not split-selection variance. DAVIS has 30,056 labelled pairs over 442 targets.
<!-- src: data/splits/davis/random/train.csv#rows = 21039 -->
<!-- src: data/splits/davis/random/valid.csv#rows = 3006 -->
<!-- src: data/splits/davis/random/test.csv#rows = 6011 -->
<!-- src: derived: 21039 + 3006 + 6011 = 30056 -->
<!-- src: results/sequence_audit_davis.md:5 = 442 -->

**Targets are sequences, not identifiers.** The splits hold targets out by name, but DAVIS's 442
targets are 379 distinct sequences: 54 mutant targets carry exactly the wild-type sequence. As a
result, 12 of the 88 cold-target test targets and 11 of the 88 cold-pair test targets are identical in
sequence to a training target, and 10 targets hold too little of the kinase domain to contain its ATP
pocket. <!-- src: results/sequence_audit_davis.md:5 = 379, 442 -->
<!-- src: results/sequence_audit_davis.md:9 = 54 -->
<!-- src: results/sequence_audit_davis.md:21 = 88, 12 -->
<!-- src: results/sequence_audit_davis.md:23 = 88, 11 -->
<!-- src: results/sequence_audit_davis.md:32 = 10 -->
No model is retrained to correct this; the evaluation stops counting what it should not
(`src/evaluation/exclusions.py`). Accuracy at cold-target and cold-pair is reported both on all test
rows and on the rows whose target is unseen by sequence. Explanation metrics drop the seen-by-sequence
targets at the cold levels and the pocketless targets at every level, and count one protein per
distinct sequence. KIBA's 229 targets are 229 distinct sequences, and no KIBA target is excluded.
<!-- src: results/sequence_audit_kiba.md:5 = 229, 229 -->

## 2.2 Models and training

Four attention-based models are audited: MolTrans (2021), HyperAttentionDTI (2022), DrugBAN (2023)
and XAttn-Ref. XAttn-Ref is the authors' own model (repository name ColdSite-DTI): a conventional
two-branch encoder whose drug representation queries the residues through one cross-attention layer.
It is included as a reference whose internals we can open, receives exactly the same training
protocol and analysis, and its results are reported whether favourable or not. DeepDTA (2018), which
has no attention, anchors accuracy and is never scored for explanations.

The published models are trained from their vendored repositories with their authors' optimiser,
learning rate, batch size and tokeniser; only data loading is replaced (Supplementary Table S2). All
models share one checkpoint rule: up to 100 epochs, the checkpoint with the lowest validation loss
among epochs ≥ 10, and early stopping after 15 epochs without improvement (10 for DeepDTA, as in its
original DAVIS cells). <!-- src: src/cloud/recipes.py:65 = 15, 10, 100 -->
<!-- src: src/cloud/recipes.py:53 = 10, 100, 10 -->
DAVIS cells are trained in full precision. On KIBA, DeepDTA and HyperAttentionDTI use mixed precision;
MolTrans stays in full precision because its hand-written layer normalisation underflows in half
precision, and XAttn-Ref stays in full precision to remain comparable with its DAVIS cells
(`notebooks/kaggle_coldsite_kiba.ipynb`, `AMP = False`).
DrugBAN is trained on DAVIS only. Training ran on Kaggle sessions with two NVIDIA T4 GPUs, one
independent job per GPU.

The explanation scored for each model is **the readout used in the primary analysis**: MolTrans's
protein-encoder self-attention (last layer, mean over heads), HyperAttentionDTI's channel-mean
attention placed at the centre of each convolution window, DrugBAN's bilinear attention summed over
drug atoms and averaged over heads, and XAttn-Ref's cross-attention weights; every map is projected to
one non-negative weight per residue. The figures in which each published model displays its map are
[MOLTRANS_FIG_X], [HYPERATTENTION_FIG_X] and [DRUGBAN_FIG_X]; whether each primary readout matches the
displayed map has not been confirmed against those figures, and alternative readouts are reported as a
sensitivity analysis (Supplementary Section S3).

## 2.3 Ground truths and plausibility

Two ground truths at different resolutions are used. **UniProt residues**: sequence features of type
Binding site, Active site and Nucleotide binding, re-numbered onto each dataset sequence by alignment
so that a site points at the residue the model actually read (UniProt 2025). **KLIFS pocket**: the
85 residues lining the ATP cleft of every kinase (KLIFS 2014; KLIFS 2021), placed on each sequence by
the same alignment. <!-- src: src/data/klifs_pocket.py:54 = 85 -->
Proteins are scored within their first 1,000 residues, and sites beyond that window are excluded.
<!-- src: src/evaluation/collect.py:265 = 1000 -->

Plausibility is precision@k: the fraction of a protein's k most-attended residues that are sites, with
k = 10 fixed before any result existed and other k values used only as sensitivity analyses. It is
computed on one test pair per protein, so that proteins measured against many drugs are not weighted
by that count. Each protein's chance level is (sites in the window) ÷ (window length), the exact
expectation of k uniformly random residues, and its ceiling is min(sites, k) ÷ k. After the sequence
policy, DAVIS contributes 349, 349, 68 and 72 proteins at the random, cold-drug, cold-target and
cold-pair levels against UniProt, and KIBA 211 and 212 at its two levels.
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:163 = 10 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=random->n = 349 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_drug->n = 349 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_target->n = 68 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_pair->n = 72 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P2&model=hyperattentiondti&level=random->n = 211 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P2&model=hyperattentiondti&level=cold_drug->n = 212 -->

## 2.4 The calibrated battery

A verdict is read only against the following instruments. Each is implemented once and applied
identically to every model it covers; Supplementary Section S4 lists which model, dataset and level
each instrument was run on.

**Chance, ceiling and a uniform floor.** Every precision@k is reported beside its chance level and
ceiling, and a uniform map (equal weight on every residue) is passed through the same grid as an
additional subject; a model indistinguishable from it carries no explanatory content at that level.

**Permutation test and family-wise correction.** For each cell, the null draws k random residues per
protein and averages over the split, preserving the split's own mix of lengths and site counts; the
p-value uses the add-one estimator over 10,000 permutations, and the median over seeds is the cell's
p-value. <!-- src: docs/PROTOCOL_AMENDMENT_v2.md:54 = 10000 -->
Holm correction is applied once over each dataset's family: 20 DAVIS cells (four models and the uniform
map at four levels) and 8 KIBA cells (three models and the uniform map at two levels). The smallest
attainable p-value lies below the smallest Holm threshold used, 0.0025.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 8 -->
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:17 = 0.0025 -->
These families are declared rather than pre-registered: they grew from 16 to 20 (DAVIS) and 6 to 8
(KIBA) when DrugBAN and XAttn-Ref's KIBA cells were added after the first results, and were frozen by
a dated protocol amendment before any further analysis. The original 500-permutation audit reaches the
same verdicts. <!-- src: docs/PROTOCOL_AMENDMENT_v2.md:83 = 16, 20, 6, 8 -->
<!-- src: docs/REMEDIATION_LEDGER.md:592 = 500 -->

**Positive control.** Explanations of known quality are scored by the same functions on the real test
proteins: at dose d, each true site is ranked above every non-site with probability d. The oracle
(d = 1) reaches its ceiling, and a dose of 0.02 is detected at every DAVIS level, so a null result
here reflects the explanation and not the test's resolution.
<!-- src: results/positive_control_davis.md:11 = 0.02 -->
<!-- src: results/positive_control_davis.md:7 = 1 -->

**Positional and residue-identity nulls.** Borrowed-map, same-residue and within-span nulls
(`src/evaluation/positional_control.py`) test whether an above-chance score could arise from where in
the sequence a map concentrates or which amino acids it prefers, rather than from the sites.

**Faithfulness.** For the first 200 test pairs of each level, the ten most-attended residues are
masked and the change in the model's prediction is compared with the mean change over five random
maskings of the same size; the reported quantity is that difference, and the map is load-bearing if it
is above zero. <!-- src: src/evaluation/run_faithfulness.py:426 = 200 -->
<!-- src: src/evaluation/run_faithfulness.py:425 = 5 -->
A masked residue becomes the unknown amino acid `X` and is re-tokenised by each model's own tokeniser,
so every model receives the same intervention. For MolTrans, whose sub-word tokens span several
residues, the random arm is matched to the attended arm in the number of tokens changed, within 10%
where such a draw exists; the match achieved was not recorded per pair (Supplementary Table S3).
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:59 = 10 --> The prediction
compared is each model's decision quantity (HyperAttentionDTI's log-odds), and MolTrans's
inference-time dropout is held fixed by a common random seed.

**A second attribution method.** Integrated gradients (IG 2017) with 32 steps on the same checkpoints
serves as a confirmatory comparison with the attention. <!-- src: src/evaluation/integrated_gradients.py:56 = 32 -->

## 2.5 Seed instability and effect sizes

For each model, dataset and level, each seed's precision@10 is tested alone at α = 0.05, uncorrected,
as a single-seed report would be. A cell's seeds *disagree* if some but not all of them clear α, and
its spread *exceeds its signal* if the range of the seeds' precision@10 is larger than the distance of
their mean from chance (`src/evaluation/seed_agreement.py`).
<!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->

Effect sizes carry 95% percentile intervals from a two-way bootstrap with 10,000 resamples: each
resample draws the targets with replacement and, independently, the three seeds with replacement, and
averages over the drawn target–seed grid, so an interval carries seed-to-seed variance as well as the
sampling of targets. Enrichment over chance is the ratio of mean precision@10 to mean chance over the
same resampled targets, and faithfulness intervals first group pairs by target. With three seeds the
seed draw is coarse and the intervals are conservative. The protocol amendment specified a bootstrap
over targets only, each target carrying the mean of its seeds; those intervals, which omit seed
variance, are kept for comparison (Supplementary Section S4) and are quoted only where the contrast is
the point.
<!-- src: results/effects_v2_2d/enrichment.md:3 = 95, 10000 -->
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:183 = 10000 -->

Accuracy is reported per cell as AUROC, AUPRC, MCC and F1 (the last two at the trainers' own 0.5
probability threshold), as mean ± sample SD over three seeds; each value was recomputed from saved
per-row predictions and checked against the value the trainer recorded, except DrugBAN's, which
are the recorded values (no MCC or F1: its graph library was unavailable for re-scoring). Whether accuracy explains
localization is described by Spearman's ρ between AUROC and precision@10 across cells, with a
bootstrap interval over cells; it is a description, not a test.
<!-- src: src/model/train.py:107 = 0.5 -->

## 2.6 Quantifying the DAVIS leak

The accuracy anchor was retrained at the cold-target level on two further training sets of equal size
(17,748 rows): one with every seen-by-sequence target removed, and one that keeps the leaked targets
while matching that row count and positive count. The difference between the two, per seed and on the
same test set, is the leak; the difference between the original and the matched set is the cost of the
smaller training set. <!-- src: results/leakage_retrain_davis.md:12 = 17748 -->
<!-- src: results/leakage_retrain_davis.md:13 = 17748 -->

## 2.7 Threats to validity and controls

The main threats, and what controls each, are: seed variance (three seeds, reported per seed and as
agreement, §2.5); readout choice (primary readout declared, alternatives as sensitivity); sub-word
tokenisation in masking (token-matched arms); benchmark leakage (sequence policy, §2.1, and the
retraining in §2.6); insufficient test resolution (positive control); multiple comparisons (Holm over
declared families); and post-hoc family growth (disclosed, frozen by amendment). Threats that remain
open are stated as limitations: families were not pre-registered; no conservation control for the
pocket enrichment; kinase-only data; DrugBAN's faithfulness has point estimates but no interval; MolTrans's test metrics carry
one draw of its inference-time dropout; and the extension experiments declared in the amendment were
not run. Supplementary Table S3 lists every problem found during the project, its effect and its fix.

## 2.8 Reproducibility

Analysis runs from one entry point per dataset (`python -m src.evaluation.run_all --dataset
{davis|kiba} --checkpoint-dir <grid>`). Split and ground-truth files are fixed by SHA-256 in
`data/splits/MANIFEST.json`, vendored models are unmodified, and every number in this manuscript
carries a machine-checked source tag (`scripts/check_number_provenance.py`).
