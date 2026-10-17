# 2 Methods

<!-- v3 (ISMB / Bioinformatics format), step 4, 2026-10-17. Condensed from paper/v2/methods.md (2,360 words) to the venue budget; moved to the
Supplement on purpose: split sizes, per-model recipes, per-protein counts, the per-instrument "where run" table, the integrated-gradients detail,
the MolTrans token-matching detail, the DrugBAN accuracy-scoring caveat, and the problem log. Every digit has a source tag
(`python scripts/check_number_provenance.py paper/v3_ismb/methods.md`).
PRECISION, VERIFIED 2026-10-17 from the committed result files (~/ColdSite-results/kiba_binary/*_results.json, field `amp`), 6 cells per model (2 levels x 3 seeds):
DeepDTA amp=true, HyperAttentionDTI amp=true, MolTrans amp=false, XAttn-Ref (files without a model suffix) amp=false. The `src/cloud/recipes.py` tag line numbers (56, 68)
follow the UNCOMMITTED edit of that file (DrugBAN full precision); re-run the checker after that edit is final.
Note: `recipes.uses_amp` would give XAttn-Ref amp=true on KIBA, which disagrees with how its seeds 1-3 were trained; no XAttn-Ref KIBA cell is scheduled in Wave A, so it is latent. -->

## 2.1 Data, task and splits

DAVIS (Davis et al., 2011) and KIBA (Tang et al., 2014) are used as distributed with DeepDTA (Öztürk et al., 2018). Every model solves the same
binary task: a pair is positive if pKd ≥ 7.0 on DAVIS or its KIBA score is ≥ 12.1, one constant shared by all trainers.
<!-- src: src/model/dataset.py:30 = 7.0, 12.1 -->
Each dataset is split at four levels of distribution shift: random; cold-drug, where test drugs are absent from training; cold-target, where test
targets are absent; and cold-pair, where both are. There is one fixed split per level; the three seeds of a cell change weight initialisation and
batch order only, so the spread across seeds is training variance, not split-selection variance. DAVIS has 30,056 labelled pairs over 442 targets.
<!-- src: data/splits/davis/random/train.csv#rows = 21039 -->
<!-- src: data/splits/davis/random/valid.csv#rows = 3006 -->
<!-- src: data/splits/davis/random/test.csv#rows = 6011 -->
<!-- src: derived: 21039 + 3006 + 6011 = 30056 -->
<!-- src: results/sequence_audit_davis.md:5 = 442 -->

**Targets are sequences, not identifiers.** The splits hold targets out by name, but DAVIS's 442 targets are 379 distinct sequences: 54 mutant
targets carry exactly the wild-type sequence. Consequently 12 of the 88 cold-target test targets and 11 of the 88 cold-pair test targets are
identical in sequence to a training target, and 10 targets hold too little of the kinase domain to contain its ATP pocket.
<!-- src: results/sequence_audit_davis.md:5 = 379, 442 -->
<!-- src: results/sequence_audit_davis.md:9 = 54 -->
<!-- src: results/sequence_audit_davis.md:21 = 88, 12 -->
<!-- src: results/sequence_audit_davis.md:23 = 88, 11 -->
<!-- src: results/sequence_audit_davis.md:32 = 10 -->
No model is retrained to correct this; the evaluation stops counting what it should not. Explanation metrics drop seen-by-sequence and
pocketless targets and count one protein per distinct sequence, and cold-level accuracy is reported on the sequence-unseen test targets as well as on
all of them. KIBA's 229 targets are 229 distinct sequences, and none is excluded.
<!-- src: results/sequence_audit_kiba.md:5 = 229, 229 -->

## 2.2 Models and training

Four attention-based models are audited: MolTrans (Huang et al., 2021), HyperAttentionDTI (Zhao et al., 2022), DrugBAN (Bai et al., 2023) and
XAttn-Ref. XAttn-Ref is the authors' own model (repository name ColdSite-DTI): a two-branch encoder whose drug representation queries the residues
through one cross-attention layer. It receives exactly the same training protocol and analysis as the published models, and its results are reported
whether favourable or not. DeepDTA, which has no attention, anchors accuracy and is never scored for explanations.

The published models are trained from their vendored repositories with their authors' optimiser, learning rate, batch size and tokeniser; only data
loading is replaced, apart from the mixed precision on KIBA described below. All models share one checkpoint rule: up to 100 epochs, the checkpoint with the lowest validation loss among epochs ≥ 10, and
early stopping after 15 epochs without improvement (10 for DeepDTA).
<!-- src: src/cloud/recipes.py:68 = 15, 10, 100 -->
<!-- src: src/cloud/recipes.py:56 = 10, 100, 10 -->
DAVIS cells are trained in full precision. On KIBA, DeepDTA and HyperAttentionDTI use mixed precision, whereas MolTrans (whose hand-written layer
normalisation underflows in half precision) and XAttn-Ref use full precision; the precision of every cell is recorded in its results file. Mixed precision was checked beforehand on one DAVIS cell (HyperAttentionDTI, cold-pair, three seeds per arm), where test AUROC and precision@10 stayed within the full-precision seed spread; three seeds detect only a gross effect and cannot show equivalence (Supplement).
<!-- claim "checked beforehand on one DAVIS cell ... within the spread ... cannot show equivalence": results/amp_validation_davis.md, table and verdict ("within"), and its own limits paragraph --> DrugBAN is
trained on DAVIS only. Training ran on Kaggle sessions with two NVIDIA T4 GPUs, one independent job per GPU.

The explanation scored for each model is the readout of the primary analysis: MolTrans's protein-encoder self-attention
(last layer, mean over heads), HyperAttentionDTI's channel-mean attention placed at the centre of each convolution window, DrugBAN's bilinear
attention summed over drug atoms and averaged over heads, and XAttn-Ref's cross-attention weights; each is projected to one non-negative weight per
residue. Whether each primary readout matches the map its authors display has not been confirmed against their figures; alternative readouts are
a sensitivity analysis (Supplement).

## 2.3 Ground truths and plausibility

Two ground truths at different resolutions are used. **UniProt residues**: sequence features of type Binding site, Active site and Nucleotide
binding (The UniProt Consortium, 2025), re-numbered onto each dataset sequence by alignment so that a site points at the residue the model read.
**KLIFS pocket**: the 85 residues lining the ATP cleft of every kinase (van Linden et al., 2014; Kanev et al., 2021), placed on each sequence by the
same alignment. Proteins are scored within their first 1,000 residues.
<!-- src: src/data/klifs_pocket.py:54 = 85 -->
<!-- src: src/evaluation/collect.py:265 = 1000 -->
Plausibility is precision@k, the fraction of a protein's k most-attended residues that are sites, with k = 10 fixed before any result existed. It is
computed on one test pair per protein, so that proteins measured against many drugs are not weighted by that count. Each protein's chance level is
(sites in the window) ÷ (window length), the exact expectation for k uniformly random residues, and its ceiling is min(sites, k) ÷ k.
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:163 = 10 -->

## 2.4 The calibrated battery

A verdict is read only against the following instruments, each implemented once and applied identically to every model it covers.

**Chance, ceiling and a uniform floor.** Every precision@k is reported beside its chance and ceiling, and a uniform map (equal weight on every
residue) is passed through the same grid as an additional subject.

**Permutation test and family-wise correction.** For each cell the null draws k random residues per protein and averages over the split, preserving
the split's mix of lengths and site counts; the p-value uses the add-one estimator over 10,000 permutations, and the **median over the three seeds** is
the cell's p-value. Holm correction (Holm, 1979) is applied once over each dataset's family: 20 DAVIS cells (four models and the uniform map at four
levels) and 8 KIBA cells (three models and the uniform map at two levels).
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:54 = 10000 -->
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 8 -->
The survival counts reported here (1 of 20 DAVIS cells, 0 of 8 KIBA cells) therefore describe this median-over-seeds rule, not individual seeds: a
Holm correction over the 48 per-seed exact tests instead retains 5 seed-runs, and over the 18 KIBA per-seed tests 3, because single seeds can clear a
corrected threshold that their median does not. Recomputing the Holm correction from the exact per-seed p-values instead of the 10,000-permutation estimates leaves the same survivors (1 of 20 and 0 of 8).
<!-- src: results/certification/step4_holm.txt:25 = 1, 20 -->
<!-- src: results/certification/step4_holm.txt:47 = 0, 8 -->
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 --><!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
<!-- src: results/certification/step4_holm.txt:34 = 48, 5 -->
<!-- src: results/certification/step4_holm.txt:56 = 18, 3 -->
The families are declared, not pre-registered: they grew from 16 to 20 (DAVIS) and from 6 to 8 (KIBA) when DrugBAN and XAttn-Ref's KIBA cells were
added after the first results, and were frozen by a dated protocol amendment before any further analysis.
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:83 = 16, 20, 6, 8 -->

**Positive control.** Explanations of known quality are scored by the same functions on the real test proteins: at dose d, each true site is ranked
above every non-site with probability d. The oracle (d = 1) reaches its ceiling, and a dose of 0.02 is detected at every DAVIS level, so a null result
reflects the explanation and not the test's resolution.
<!-- src: results/positive_control_davis.md:11 = 0.02 -->
<!-- src: results/positive_control_davis.md:7 = 1 -->

**Positional and residue-identity nulls.** Borrowed-map, same-residue and within-span nulls test whether an above-chance score could arise from where in
the sequence a map concentrates, or which amino acids it prefers, rather than from the sites.

**Faithfulness.** For the first 200 test pairs of each level, the ten most-attended residues are masked and the change in the model's prediction is
compared with the mean change over five random maskings of the same size; the reported quantity is that difference, and the map is *load-bearing*
if it is above zero. A masked residue becomes the unknown amino acid `X` and is re-tokenised by each model's own tokeniser, so every model receives
the same intervention.
<!-- src: src/evaluation/run_faithfulness.py:426 = 200 -->

## 2.5 Seed instability and effect sizes

For each model, dataset and level, each seed's precision@10 is tested alone at α = 0.05, uncorrected, as a single-seed report would be. A cell's seeds
*disagree* if some but not all of them clear α, and its spread *exceeds its signal* if the range of the seeds' precision@10 is larger than the
distance of their mean from chance. The per-seed p-value is exact: each protein's hits under the null are hypergeometric and independent, and the
p-value is the upper tail of their convolution. Seed variance is also tested directly on the per-protein precision@10 of the three seeds, by a Friedman
test and by a permutation test that shuffles the seed labels within each protein (20,000 permutations), each Holm-corrected over the 22 cells.
<!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->
<!-- src: results/certification/step1_disagreement.txt:34 = 20000 -->
<!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->

Effect sizes carry 95% percentile intervals from a two-way bootstrap with 10,000 resamples that draws the proteins with replacement and, independently,
the three seeds with replacement, so an interval carries seed-to-seed variance as well as protein sampling; with three seeds the seed draw is coarse
and the intervals are conservative. Enrichment over chance is the ratio of mean precision@10 to mean chance over the same resampled proteins.
The protocol amendment had specified a bootstrap over proteins only; those intervals, which omit seed variance, are kept in the Supplement and quoted only where
the contrast is the point.
<!-- src: results/effects_v2_2d/enrichment.md:3 = 95, 10000 -->
Accuracy is reported as AUROC and AUPRC, mean ± sample SD over three seeds. The association between accuracy and localisation is described by
Spearman's ρ across cells with a bootstrap interval; it is a description, not a test.

## 2.6 Quantifying the DAVIS leak

The accuracy anchor (DeepDTA) was retrained at the cold-target level on two further training sets of equal size (17,748 rows): one with every
seen-by-sequence target removed, and one that keeps the leaked targets while matching that row count and positive count. The difference between the
two, per seed and on the same test set, is the leak; the difference between the original and the matched set is the cost of the smaller training set.
<!-- src: results/leakage_retrain_davis.md:12 = 17748 -->
<!-- src: results/leakage_retrain_davis.md:13 = 17748 -->

## 2.7 Reproducibility

Analysis runs from one entry point per dataset. Split and ground-truth files are fixed by SHA-256 in a manifest, the vendored models are unmodified,
and every number in this manuscript carries a machine-checked source tag. Code and outputs: see Availability.
