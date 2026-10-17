# 3 Results

<!-- v3 (ISMB / Bioinformatics format), step 5, 2026-10-17. Condensed from paper/v2/results.md (3,054 words) to the venue budget. Moved to the Supplement
on purpose (do not delete from the project): v2 Table 1 (full accuracy table) and Figures 3-4; the integrated-gradients comparison (v2 §3.7, confirmatory);
the KLIFS numbers on KIBA other than the XAttn-Ref contrast; the 1,000-permutation count of 12; the MolTrans cold-pair example; the cold-pair leak sentence;
DrugBAN's remaining per-seed deltas and the note that its faithfulness has no interval would be stated in the Supplement and the Discussion (limitations).
Main-text figures proposed: Figure 1 (seed instability), Figure 2 (enrichment forest), Figure 5 (faithfulness), Figure 6 (leakage) -- files in paper/v2/figures/.
Every digit has a source tag (`python scripts/check_number_provenance.py paper/v3_ismb/results.md`). Only committed local results are used; nothing from
Wave A (not run) appears. -->

## 3.1 Accuracy of the retrained models

Every model was retrained with its authors' recipe and scored on the same test sets (full table in the Supplement). At DAVIS's random level all five
reach a test AUROC between 0.891 (DrugBAN) and 0.937 (HyperAttentionDTI), so the explanation results below do not come from models that failed to
train. Accuracy falls with distribution shift, most at the cold-pair level, where it ranges from 0.568 (MolTrans) to 0.728 (DeepDTA, which has no
attention), and the seed-to-seed standard deviation grows with it: 0.001–0.006 at the random level against 0.025–0.099 at cold-pair.
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=random&view=uncorrected->auroc_mean = 0.891 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=hyperattentiondti&level=random&view=uncorrected->auroc_mean = 0.937 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_pair&view=uncorrected->auroc_mean = 0.568 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=deepdta&level=cold_pair&view=uncorrected->auroc_mean = 0.728 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=random&view=uncorrected->auroc_sd = 0.001 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=random&view=uncorrected->auroc_sd = 0.006 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_pair&view=uncorrected->auroc_sd = 0.025 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_pair&view=uncorrected->auroc_sd = 0.099 -->

## 3.2 Binding-site verdicts are not self-consistent across training seeds

Scored against UniProt's annotated residues, the three seeds of a cell disagree about their own verdict (they fall on different sides of the
uncorrected α = 0.05 threshold of a per-seed permutation test, as a single-run report would quote it) in 11 of 22 model–dataset–level cells (8 of 16
on DAVIS, 3 of 6 on KIBA). Only 1 cell has all three seeds above the threshold; 10 have none (Figure 1).
<!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->
<!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=cells_davis->value = 16 --> <!-- src: results/certification/key_numbers.csv#name=cells_kiba->value = 6 -->
<!-- src: results/certification/key_numbers.csv#name=disagree_davis->value = 8 --> <!-- src: results/certification/key_numbers.csv#name=disagree_kiba->value = 3 -->
<!-- src: results/certification/key_numbers.csv#name=all_three_pass->value = 1 --> <!-- src: results/certification/key_numbers.csv#name=none_pass->value = 10 -->
The per-seed p-values are exact, so no sampling error enters. A count on a threshold is threshold-dependent (8 at α = 0.01, 13 at α = 0.10), and we
report 11 as the exact-test value at the conventional α, not as a constant of the recipes.
<!-- src: results/certification/key_numbers.csv#name=disagree_at_alpha_0.010->value,low = 8, 0.01 -->
<!-- src: results/certification/key_numbers.csv#name=disagree_at_alpha_0.100->value,low = 13, 0.10 -->
The disagreement is not marginal: HyperAttentionDTI at DAVIS cold-drug places 0.025, 0.077 and 0.019 of its top-ten residues on annotated sites
against a chance of 0.020. In 21 of 22 UniProt cells the range of the three seeds' precision@10 exceeds the cell's distance from chance.
<!-- src: results/seed_agreement.md:19 = 0.025, 0.077, 0.019, 0.020 -->
<!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
Taken alone this descriptive criterion is weak, because a cell whose mean sits near chance meets it almost by construction, so we test seed variance
directly: Friedman tests over proteins and within-protein seed-label permutation tests find significant differences between the three seeds in 13 and
10 of the 22 cells before correction, and in 8 and 7 of 22 after Holm correction over the 22 cells.
<!-- src: results/certification/key_numbers.csv#name=friedman_uncorrected->value = 13 --> <!-- src: results/certification/key_numbers.csv#name=friedman_holm->value = 8 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=permutation_uncorrected->value = 10 --> <!-- src: results/certification/key_numbers.csv#name=permutation_holm->value = 7 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
The variation is not scoring noise: collecting the explanations again from the same checkpoints moves precision@10 by a standard deviation of 0.0008
per seed, against a between-seed standard deviation of 0.0121.
<!-- src: results/certification/key_numbers.csv#name=rerun_noise_sd->value = 0.0008 --> <!-- src: results/certification/key_numbers.csv#name=between_seed_sd->value = 0.0121 -->
A bootstrap over proteins, the usual way to attach an interval to such a score, does not see this variance: for HyperAttentionDTI at DAVIS cold-drug it
gives an enrichment over chance of 1.98 [1.81, 2.15], an interval far from parity, for a cell whose seeds disagree and whose family-wise test fails. The
two-way bootstrap that resamples seeds as well, used for every interval below, gives the same cell [0.94, 3.65].
<!-- src: results/effects_v2/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.98, 1.81, 2.15 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_drug->enrichment_low,enrichment_high = 0.94, 3.65 -->

## 3.3 Under the calibrated battery, residue-level recovery rarely survives

The positive control sets the test's resolution: an explanation that ranks as little as 0.02 of the true sites first is detected at every DAVIS level, so
a null below is a property of the maps and not of the test. With Holm correction over each dataset's declared family, 1 of 20 DAVIS cells supports
residue-level recovery and 0 of 8 KIBA cells do. These counts describe the median-over-seeds rule of Section 2.4, not individual seeds.
<!-- src: results/positive_control_davis.md:11 = 0.02 -->
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
The surviving cell is HyperAttentionDTI at the random level (precision@10 0.034 against a chance of 0.020; enrichment 1.66 [1.33, 2.00]), at p = 0.0001,
the smallest value 10,000 permutations can give. The uniform map, passed through the same grid, never approaches significance (p of 0.4462 or more on
DAVIS, 0.2251 or more on KIBA), so the test does not reward a map for being flat.
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=random->precision,chance,enrichment,enrichment_low,enrichment_high = 0.034, 0.020, 1.66, 1.33, 2.00 -->
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:17 = 0.0001 -->
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:54 = 10000 -->
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:27 = 0.4462 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:17 = 0.2251 -->
Once seeds are resampled the intervals agree with the correction: of the 22 UniProt cells, only this one has an interval above parity (Figure 2).
With proteins alone resampled, several more would, for example MolTrans at KIBA cold-drug, 1.57 [1.32, 1.82]; resampling seeds widens that interval to
[0.70, 2.70], and its split-level test does not survive (p = 0.07279 against a threshold of 0.00625).
<!-- src: results/seed_agreement.md:34 = 22 -->
<!-- claim "only this one": results/effects_v2_2d/enrichment.csv, families P1 (16 rows) and P2 (6 rows), enrichment_low > 1 only for P1 hyperattentiondti random -->
<!-- src: results/effects_v2/enrichment.csv#family=P2&model=moltrans&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.57, 1.32, 1.82 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P2&model=moltrans&level=cold_drug->enrichment_low,enrichment_high = 0.70, 2.70 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:16 = 0.07279, 0.00625 -->

## 3.4 What replicates is coarse: pocket enrichment and load-bearing attention

Against the 85-residue KLIFS pocket, where chance is 0.137–0.143 on DAVIS, XAttn-Ref's map is enriched at every DAVIS level (1.54–2.10× chance) and
HyperAttentionDTI's at three of the four (1.33–1.70×, cold-pair only marginally); at cold-drug HyperAttentionDTI's interval, 1.35 [0.94, 1.73], includes
parity once seeds are resampled. MolTrans's intervals include parity at every level, and DrugBAN's map sits on chance throughout (0.98–1.04×)
(Figure 2). On KIBA, XAttn-Ref's cold-drug map is at chance (1.00 [0.79, 1.27]).
<!-- claim "every level / three of the four / MolTrans every level": S3-D rows of results/effects_v2_2d/enrichment.csv; enrichment_low > 1 at all four levels for coldsite_dti, at random, cold_target, cold_pair for hyperattentiondti; enrichment_low < 1 at all four levels for moltrans and drugban -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.35, 0.94, 1.73 -->
<!-- src: src/data/klifs_pocket.py:54 = 85 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_pair->chance = 0.137 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=coldsite_dti&level=random->chance = 0.143 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_target->enrichment = 1.33 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=hyperattentiondti&level=random->enrichment = 1.70 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=coldsite_dti&level=random->enrichment = 1.54 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_drug->enrichment = 2.10 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=drugban&level=random->enrichment = 0.98 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=drugban&level=cold_target->enrichment = 1.04 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-K&model=coldsite_dti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.00, 0.79, 1.27 -->
Kinase ATP pockets are conserved and no conservation control was run, so this is a statement about a conserved region, not evidence that a ligand's
contacts were learned. The positional nulls do rule out the simplest artefact for HyperAttentionDTI: its pocket hits beat a shuffle of the same attention
within the kinase domain in eight of its twelve seed-level cells (uncorrected), so the map prefers the pocket within the domain, not only the domain.
<!-- claim "eight of its twelve": results/analysis_davis_policyA/positional_control_hyperattentiondti_davis_klifs_policyA.md, kinase rows, in-span shuffle p < 0.05 on lines 7, 9, 13, 15, 17, 19, 23, 27; not on lines 11, 21, 25, 29 -->

Faithfulness is the most stable property measured for three of the four models. In all 16 cells of HyperAttentionDTI, MolTrans and XAttn-Ref, masking
the attended residues changes the prediction more than masking the same number of random ones, with every 95% interval above zero even when seeds are
resampled (Figure 5); the effect ranges from 0.054 [0.030, 0.080] (HyperAttentionDTI, DAVIS cold-pair) to 1.21 [0.68, 1.92] (XAttn-Ref, DAVIS random),
and MolTrans, measured in its own token space, is not on the same scale. The sign holds in every seed, but its size does not: within a cell the per-seed
deltas differ several-fold.
<!-- src: results/effects_v2_2d/faithfulness_effects.csv#rows = 16 -->
<!-- src: results/effects_v2_2d/faithfulness_effects.csv#model=hyperattentiondti&dataset=davis&level=cold_pair->delta,delta_low,delta_high = 0.054, 0.030, 0.080 -->
<!-- src: results/effects_v2_2d/faithfulness_effects.csv#model=coldsite_dti&dataset=davis&level=random->delta,delta_low,delta_high = 1.21, 0.68, 1.92 -->
<!-- src: results/effects_v2_2d/faithfulness_effects.md:3 = 95 -->
<!-- claim "sign holds in every seed ... several-fold": results/effects_v2_2d/faithfulness_effects.csv, column per_seed_pair_mean — all 48 per-seed values positive; e.g. the XAttn-Ref DAVIS cold_target row lists 0.162975;0.890800;0.496076 -->
DrugBAN's attention is not consistently load-bearing: its deltas sit near zero and change sign between seeds (at cold-drug −0.0073, −0.0036 and 0.0128 for seeds one
to three), six of its twelve seed-level deltas are negative, and no level is load-bearing in all three seeds. They are point estimates only, because its
run did not record per-pair values and no interval can be attached.
<!-- src: results/analysis_davis_policyA/faithfulness_drugban_davis_seed1.md:6 = -0.0073 -->
<!-- src: results/analysis_davis_policyA/faithfulness_drugban_davis_seed2.md:6 = -0.0036 -->
<!-- src: results/analysis_davis_policyA/faithfulness_drugban_davis_seed3.md:6 = 0.0128 -->
<!-- claim "six of its twelve ... no level load-bearing in all three seeds": faithfulness_drugban_davis_seed{1,2,3}.md lines 5-8, delta column: seed 1 0.0011, -0.0073, 0.0005, -0.0037; seed 2 -0.0018, -0.0036, 0.0002, -0.0038; seed 3 0.0064, 0.0128, -0.0034, 0.0027 -->
Finally, accuracy does not predict localisation: across the 48 DAVIS attention-model cells, Spearman's ρ between AUROC and enrichment over chance is
0.003 [−0.309, 0.294].
<!-- src: results/accuracy_v2/localization_spearman.csv#dataset=davis&group=all attention models&y=enrichment->n_cells,rho,low,high = 48, 0.003, -0.309, 0.294 -->

## 3.5 The verdict depends on the readout and on what the map is a function of

A multi-channel attention tensor must be reduced to one weight per residue, and the choice decides the verdict. Two defensible reductions of the same
HyperAttentionDTI weights place 0.367 and 0.081 of their top-ten residues in the ATP pocket at the cold-target level, against a chance of 0.140, a single model and level.
<!-- src: results/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=maxchannel&level=cold_target->precision@10 = 0.367 -->
<!-- src: results/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=receptive&level=cold_target->precision@10,chance = 0.081, 0.140 -->
Whether a map can depend on the drug is a property of the computation graph. For 25 proteins each paired with four drugs, the MolTrans readout scored
here gives the same top ten for every drug (100% of 150 drug pairs), HyperAttentionDTI's in 72%, and DrugBAN's in 0%. A map that cannot change with
the drug cannot mark that drug's binding site, whatever its hit rate. MolTrans's interaction map, a different readout of the same model, does change with the drug (identical top ten in 0% of the same 150 pairs).
<!-- src: results/drug_dependence/moltrans_interaction.txt:1 = 0, 150 -->
<!-- src: results/drug_dependence/moltrans.txt:1 = 25, 4, 100, 150 -->
<!-- src: results/drug_dependence/hyperattentiondti.txt:1 = 72, 150 -->
<!-- src: results/drug_dependence/drugban.txt:1 = 0, 150 -->

## 3.6 DAVIS's sequence leakage, quantified

DAVIS gives 54 of its mutant targets exactly the wild-type sequence, so 12 of the 88 cold-target test targets are seen by sequence in training: 816 of
5984 test rows (13.6%).
<!-- src: results/sequence_audit_davis.md:9 = 54 -->
<!-- src: results/sequence_audit_davis.md:21 = 12, 88, 816, 5984, 13.6 -->
The retrain was made on DeepDTA, the accuracy anchor, at the cold-target split, with three seeds, and separates the leak from the loss of training rows
(Figure 6). On the leaked rows the sequence match inflates AUROC by 0.116 (95% t-interval [0.098, 0.134], p = 0.0013). On the strictly unleaked rows no
inflation is detectable (−0.019, p = 0.31). Over all rows the net effect is 0.019 with an interval, [−0.022, 0.059], that includes zero (p = 0.19), because
the leaked rows are 13.6% of the test set; we therefore do not quote the all-rows figure as a finding. Explanation metrics exclude seen-by-sequence
targets throughout.
<!-- src: results/certification/key_numbers.csv#name=leak_ct_leaked_rows->value,low,high,p = 0.116, 0.098, 0.134, 0.0013 --> <!-- src: results/certification/step3_leak.txt:41 = 95 -->
<!-- src: results/certification/key_numbers.csv#name=leak_ct_unleaked_rows->value,p = -0.019, 0.31 -->
<!-- src: results/certification/key_numbers.csv#name=leak_ct_all_rows->value,low,high,p = 0.019, -0.022, 0.059, 0.19 -->

Every result in this section concerns kinase targets; the non-kinase transfer panel is not analysed here.
