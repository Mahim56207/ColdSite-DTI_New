# 3 Results

<!-- T23 draft (user scope "Drafting: Results"), 2026-09-25. Order follows the reframed contributions:
context (accuracy), (a) seed instability, (b) the calibrated battery, pocket and faithfulness, (c) readout
and drug dependence, (d) leakage, then the confirmatory integrated-gradients comparison. Every digit
carries a hidden source tag (`python scripts/check_number_provenance.py`); counts spelled as words carry a
`claim` comment naming the rows they were counted from. Figures: paper/v2/figures/ (CAPTIONS.md). Only
committed local results are used; nothing from Wave A (not run) appears. -->

## 3.1 The retrained models reach their expected accuracy

Every model was retrained with its authors' recipe and scored on the same test sets (Table 1, Figure 3).
At DAVIS's random level all five reach a test AUROC between 0.891 (DrugBAN) and 0.937
(HyperAttentionDTI), so the explanation results below do not come from models that failed to train.
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=random&view=uncorrected->auroc_mean = 0.891 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=random&view=uncorrected->auroc_mean = 0.937 -->
Accuracy falls with distribution shift, most at the cold-pair level, where it ranges from 0.568
(MolTrans) to 0.728 (DeepDTA, which has no attention). The seed-to-seed standard deviation grows with
it: 0.001–0.006 at the random level, depending on the model, against 0.025–0.099 at cold-pair.
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_pair&view=uncorrected->auroc_mean = 0.568 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=cold_pair&view=uncorrected->auroc_mean = 0.728 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=random&view=uncorrected->auroc_sd = 0.001 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=random&view=uncorrected->auroc_sd = 0.006 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_pair&view=uncorrected->auroc_sd = 0.025 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_pair&view=uncorrected->auroc_sd = 0.099 -->
Scoring only targets that are unseen by sequence lowers every model's cold-target AUROC (Table 1); at
cold-pair the correction moves models in both directions, within seed spreads as large as 0.128.
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_pair&view=unseen_by_sequence->auroc_sd = 0.128 -->

**Table 1.** Test AUROC, mean ± standard deviation over three training seeds. "Sequence-unseen" scores
only test targets whose sequence does not occur in training (Section 3.6). DrugBAN has no KIBA cells.
*Source:* `results/accuracy_v2/by_model.csv`.

| model | DAVIS random | DAVIS cold-drug | DAVIS cold-target | cold-target, sequence-unseen | DAVIS cold-pair | cold-pair, sequence-unseen | KIBA random | KIBA cold-drug |
|---|---|---|---|---|---|---|---|---|
| MolTrans | 0.923 ± 0.002 | 0.685 ± 0.019 | 0.874 ± 0.006 | 0.833 ± 0.008 | 0.568 ± 0.025 | 0.532 ± 0.029 | 0.919 ± 0.004 | 0.812 ± 0.009 <!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.923, 0.002 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.685, 0.019 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_target&view=uncorrected->auroc_mean,auroc_sd = 0.874, 0.006 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_target&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.833, 0.008 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_pair&view=uncorrected->auroc_mean,auroc_sd = 0.568, 0.025 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_pair&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.532, 0.029 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=moltrans&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.919, 0.004 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=moltrans&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.812, 0.009 --> |
| HyperAttentionDTI | 0.937 ± 0.005 | 0.760 ± 0.042 | 0.915 ± 0.001 | 0.893 ± 0.001 | 0.694 ± 0.038 | 0.713 ± 0.050 | 0.933 ± 0.001 | 0.844 ± 0.004 <!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.937, 0.005 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.760, 0.042 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=cold_target&view=uncorrected->auroc_mean,auroc_sd = 0.915, 0.001 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=cold_target&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.893, 0.001 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=cold_pair&view=uncorrected->auroc_mean,auroc_sd = 0.694, 0.038 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=cold_pair&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.713, 0.050 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=hyperattentiondti&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.933, 0.001 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=hyperattentiondti&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.844, 0.004 --> |
| DrugBAN | 0.891 ± 0.006 | 0.695 ± 0.033 | 0.846 ± 0.010 | 0.801 ± 0.012 | 0.620 ± 0.031 | 0.559 ± 0.040 | — | — <!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.891, 0.006 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.695, 0.033 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=cold_target&view=uncorrected->auroc_mean,auroc_sd = 0.846, 0.010 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=cold_target&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.801, 0.012 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=cold_pair&view=uncorrected->auroc_mean,auroc_sd = 0.620, 0.031 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=cold_pair&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.559, 0.040 --> |
| XAttn-Ref | 0.924 ± 0.001 | 0.721 ± 0.008 | 0.857 ± 0.011 | 0.835 ± 0.015 | 0.624 ± 0.099 | 0.607 ± 0.128 | 0.894 ± 0.011 | 0.807 ± 0.009 <!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.924, 0.001 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.721, 0.008 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_target&view=uncorrected->auroc_mean,auroc_sd = 0.857, 0.011 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_target&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.835, 0.015 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_pair&view=uncorrected->auroc_mean,auroc_sd = 0.624, 0.099 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_pair&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.607, 0.128 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=coldsite_dti&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.894, 0.011 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=coldsite_dti&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.807, 0.009 --> |
| DeepDTA | 0.929 ± 0.002 | 0.692 ± 0.044 | 0.907 ± 0.003 | 0.884 ± 0.003 | 0.728 ± 0.035 | 0.749 ± 0.044 | 0.918 ± 0.001 | 0.832 ± 0.001 <!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.929, 0.002 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.692, 0.044 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=cold_target&view=uncorrected->auroc_mean,auroc_sd = 0.907, 0.003 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=cold_target&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.884, 0.003 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=cold_pair&view=uncorrected->auroc_mean,auroc_sd = 0.728, 0.035 --><!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=cold_pair&view=unseen_by_sequence->auroc_mean,auroc_sd = 0.749, 0.044 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=deepdta&level=random&view=uncorrected->auroc_mean,auroc_sd = 0.918, 0.001 --><!-- src: results/accuracy_v2/by_model.csv#dataset=kiba&model=deepdta&level=cold_drug&view=uncorrected->auroc_mean,auroc_sd = 0.832, 0.001 --> |

## 3.2 Binding-site verdicts are not self-consistent across training seeds

Scored against UniProt's annotated residues, the three seeds of a cell disagree about their own verdict
(an uncorrected per-seed permutation test, as a single-run report would quote it) in 12 of 22
model–dataset–level cells, and in 21 of 22 the spread across seeds exceeds the cell's distance from
chance. Only 1 cell has all three seeds above the threshold; 9 have none (Figure 1).
<!-- src: results/seed_agreement.md:34 = 12, 22, 21, 22, 1, 9 -->
The disagreement is not marginal. HyperAttentionDTI at DAVIS cold-drug places 0.025, 0.077 and 0.019 of
its top-ten residues on annotated sites against a chance of 0.020; MolTrans at DAVIS cold-pair places
0.004, 0.032 and 0.024 against 0.019, so one seed sits well below chance and another well above.
<!-- src: results/seed_agreement.md:19 = 0.025, 0.077, 0.019, 0.020 -->
<!-- src: results/seed_agreement.md:26 = 0.004, 0.032, 0.024, 0.019 -->
The coarser pocket-level target behaves the same way: against the KLIFS pocket the seed spread exceeds
the distance from chance in fifteen of the twenty-two cells.
<!-- claim "fifteen of the twenty-two": results/effects_v2/seed_spread.csv, column spread_exceeds_signal_exact_chance True in 10 of 16 S3-D rows and 5 of 6 S3-K rows -->
A bootstrap over targets, the usual way to attach an interval to such a score, does not see this
variance: for HyperAttentionDTI at DAVIS cold-drug it gives an enrichment of 1.98 [1.81, 2.15] over
chance, an interval far from parity, for a cell whose seeds disagree and whose family-wise test fails.
<!-- src: results/effects_v2/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.98, 1.81, 2.15 -->

## 3.3 Under the calibrated battery, residue-level recovery rarely survives

The positive control establishes the test's resolution: an explanation that ranks as little as 0.02 of
the true sites first is detected at every DAVIS level, so a null below is a property of the maps and not
of the test. <!-- src: results/positive_control_davis.md:11 = 0.02 -->
With Holm correction over each dataset's declared family, 1 of 20 DAVIS cells supports residue-level
recovery and 0 of 8 KIBA cells do.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
The surviving cell is HyperAttentionDTI at the random level (precision@10 0.034 against a chance of
0.020; enrichment 1.66 [1.49, 1.84]), at p = 0.0001, the smallest value 10,000 permutations can give.
<!-- src: results/effects_v2/enrichment.csv#family=P1&model=hyperattentiondti&level=random->precision,chance,enrichment,enrichment_low,enrichment_high = 0.034, 0.020, 1.66, 1.49, 1.84 -->
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:17 = 0.0001 -->
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:54 = 10000 -->
The uniform map, passed through the same grid, never approaches significance (p of 0.4462 or more on
DAVIS, 0.2251 or more on KIBA), which confirms that the test does not reward a map for being flat.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:27 = 0.4462 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:17 = 0.2251 -->
Figure 2 shows why the correction matters. Several cells have a target-level interval above parity —
MolTrans at KIBA cold-drug, 1.57 [1.32, 1.82] — yet their split-level test, the median over seeds, does
not survive (p = 0.07279 against a threshold of 0.00625).
<!-- src: results/effects_v2/enrichment.csv#family=P2&model=moltrans&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.57, 1.32, 1.82 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:16 = 0.07279, 0.00625 -->

## 3.4 What replicates is coarse: pocket enrichment and load-bearing attention

Against the 85-residue KLIFS pocket, where chance is 0.137–0.143 on DAVIS, two maps are enriched at
every DAVIS level: HyperAttentionDTI at 1.33–1.70× chance and XAttn-Ref at 1.54–2.10×. MolTrans's
interval clears parity at three levels but not at cold-pair (0.99 [0.86, 1.13]), and DrugBAN's map sits
on chance throughout (0.98–1.04×) (Figure 2).
<!-- src: src/data/klifs_pocket.py:54 = 85 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_pair->chance = 0.137 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=random->chance = 0.143 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_target->enrichment = 1.33 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=hyperattentiondti&level=random->enrichment = 1.70 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=random->enrichment = 1.54 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_drug->enrichment = 2.10 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=moltrans&level=cold_pair->enrichment,enrichment_low,enrichment_high = 0.99, 0.86, 1.13 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=drugban&level=random->enrichment = 0.98 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=drugban&level=cold_target->enrichment = 1.04 -->
On KIBA, HyperAttentionDTI is enriched at both levels (1.37 [1.29, 1.44] random; 1.25 [1.17, 1.32]
cold-drug), while XAttn-Ref's cold-drug map is at chance (1.00 [0.94, 1.06]).
<!-- src: results/effects_v2/enrichment.csv#family=S3-K&model=hyperattentiondti&level=random->enrichment,enrichment_low,enrichment_high = 1.37, 1.29, 1.44 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-K&model=hyperattentiondti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.25, 1.17, 1.32 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-K&model=coldsite_dti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.00, 0.94, 1.06 -->
The positional nulls rule out the simplest artefact for HyperAttentionDTI: its pocket hits beat a
shuffle of the same attention within the kinase domain in eight of its twelve seed-level cells
(uncorrected), so the map prefers the pocket within the domain, not only the domain. Kinase ATP pockets
are conserved and no conservation control was run, so this remains a statement about a conserved
region, not evidence that a ligand's contacts were learned.
<!-- claim "eight of its twelve": results/analysis_davis_policyA/positional_control_hyperattentiondti_davis_klifs_policyA.md, kinase rows, in-span shuffle p < 0.05 on lines 7, 9, 13, 15, 17, 19, 23, 27; not on lines 11, 21, 25, 29 -->

Faithfulness is the most stable property measured. In all 16 cells where it could be computed —
HyperAttentionDTI, MolTrans and XAttn-Ref — masking the attended residues changes the prediction more
than masking the same number of random ones, with every 95% interval above zero (Figure 5). The effect
ranges from 0.054 [0.036, 0.073] (HyperAttentionDTI, DAVIS cold-pair) to 1.21 [1.05, 1.38] (XAttn-Ref,
DAVIS random); MolTrans is measured in its own token space and is not on the same scale.
<!-- src: results/effects_v2/faithfulness_effects.csv#rows = 16 -->
<!-- src: results/effects_v2/faithfulness_effects.csv#model=hyperattentiondti&dataset=davis&level=cold_pair->delta,delta_low,delta_high = 0.054, 0.036, 0.073 -->
<!-- src: results/effects_v2/faithfulness_effects.csv#model=coldsite_dti&dataset=davis&level=random->delta,delta_low,delta_high = 1.21, 1.05, 1.38 -->
<!-- src: results/effects_v2/faithfulness_effects.md:3 = 95 -->
The sign holds in every seed, but its size does not: within a cell the per-seed deltas differ
several-fold. DrugBAN's faithfulness could not be recomputed here, and XAttn-Ref has no KIBA
faithfulness cell.
<!-- claim "sign holds in every seed ... several-fold": results/effects_v2/faithfulness_effects.csv, column per_seed_pair_mean — all 48 per-seed values positive; e.g. the XAttn-Ref DAVIS cold_target row lists 0.162975;0.890800;0.496076 -->

Accuracy does not predict localisation. Across the 48 DAVIS attention-model cells, Spearman's ρ between
AUROC and enrichment over chance is 0.003 [−0.309, 0.294]; across the 18 KIBA cells it is 0.064
[−0.466, 0.557] (Figure 4).
<!-- src: results/accuracy_v2/localization_spearman.csv#dataset=davis&group=all attention models&y=enrichment->n_cells,rho,low,high = 48, 0.003, -0.309, 0.294 -->
<!-- src: results/accuracy_v2/localization_spearman.csv#dataset=kiba&group=all attention models&y=enrichment->n_cells,rho,low,high = 18, 0.064, -0.466, 0.557 -->

## 3.5 The verdict depends on the readout and on what the map is a function of

A multi-channel attention tensor must be reduced to one weight per residue, and the choice decides the
verdict. Two defensible reductions of the same HyperAttentionDTI weights place 0.367 and 0.081 of their
top-ten residues in the ATP pocket at the cold-target level, against a chance of 0.140
(Supplementary Section S3).
<!-- src: ~/ColdSite-results/readouts/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=maxchannel&level=cold_target->precision@10 = 0.367 -->
<!-- src: ~/ColdSite-results/readouts/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=receptive&level=cold_target->precision@10,chance = 0.081, 0.140 -->
Whether a map can depend on the drug is a property of the computation graph. For 25 proteins each
paired with four drugs, the MolTrans readout scored here gives the same top ten for every drug (100% of
150 drug pairs), HyperAttentionDTI's in 72%, and DrugBAN's in 0%; MolTrans's interaction map, a
readout variant, is identical in 0%.
<!-- src: results/drug_dependence/moltrans.txt:1 = 25, 4, 100, 150 -->
<!-- src: results/drug_dependence/hyperattentiondti.txt:1 = 72, 150 -->
<!-- src: results/drug_dependence/drugban.txt:1 = 0, 150 -->
<!-- src: results/drug_dependence/moltrans_interaction.txt:1 = 0 -->
A map that cannot change with the drug cannot mark that drug's binding site, whatever its hit rate.

## 3.6 DAVIS's sequence leakage, quantified

DAVIS gives 54 of its mutant targets exactly the wild-type sequence, so 12 of the 88 cold-target test
targets are seen by sequence in training: 816 of 5984 test rows (13.6%).
<!-- src: results/sequence_audit_davis.md:9 = 54 -->
<!-- src: results/sequence_audit_davis.md:21 = 12, 88, 816, 5984, 13.6 -->
Retraining the accuracy anchor three ways separates the leak from the loss of training rows (Figure 6).
At cold-target the leak is worth 0.019 AUROC on all test rows and 0.116 on the leaked rows, while
the smaller training set costs 0.020. At cold-pair the leak's effect is not positive (−0.021) and the
smaller training set costs 0.062: removing leaked targets from a small split mostly removes data.
<!-- src: results/leakage_retrain_davis.md:22 = 0.019, 0.116 -->
<!-- src: results/leakage_retrain_davis.md:31 = 0.020 -->
<!-- src: results/leakage_retrain_davis.md:50 = -0.021 -->
<!-- src: results/leakage_retrain_davis.md:59 = 0.062 -->
Explanation metrics therefore exclude seen-by-sequence targets throughout (Methods).

## 3.7 Confirmatory: integrated gradients on the same weights

Integrated gradients on the same checkpoints localise better than the attention in several cells, not
all. For HyperAttentionDTI at KIBA cold-drug, IG reaches 1.90 [1.58, 2.22]× chance against UniProt
residues where the attention is at 0.94 [0.79, 1.09]×; for XAttn-Ref at DAVIS cold-drug, IG reaches
3.16 [3.00, 3.33]× in the pocket against the attention's 2.10 [2.02, 2.18]×. IG coverage is partial
(Supplementary Section S4), and we read it only as evidence that a weak attention map can under-report
what a model represents.
<!-- src: results/effects_v2/enrichment.csv#family=S2&model=hyperattentiondti_ig&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.90, 1.58, 2.22 -->
<!-- src: results/effects_v2/enrichment.csv#family=P2&model=hyperattentiondti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 0.94, 0.79, 1.09 -->
<!-- src: results/effects_v2/enrichment.csv#family=S1-klifs&model=coldsite_dti_ig&level=cold_drug->enrichment,enrichment_low,enrichment_high = 3.16, 3.00, 3.33 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 2.10, 2.02, 2.18 -->

Every result in this section concerns kinase targets; the non-kinase transfer panel is not analysed here.
