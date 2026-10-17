# 1 Introduction

<!-- v3 (ISMB / Bioinformatics format), step 3, 2026-10-17. Condensed from paper/v2/introduction.md (~2,040 words) to the venue's budget
(VENUE_SPEC.md: whole main text ≈ 5,000 words). Moved out on purpose: the HyperAttentionDTI cold-drug worked example, the accuracy-SD
paragraph, the localisation-vs-accuracy correlation, the readout and drug-dependence numbers, the integrated-gradients paragraph and Box 1;
they stay in Results / Discussion / Supplement. Every digit has a source tag (`python scripts/check_number_provenance.py paper/v3_ismb/introduction.md`).
Citations: only entries in paper/references.md; each prior DTI work is characterised only as far as its verified abstract goes
(docs/delta_table.md). The model-paper figure placeholders of v2 ([MOLTRANS_FIG_X] etc.) are dropped: no figure number was ever supplied.
Citation style is not fixed (venue template unverified): author–year in text for now. GPS-DTI is cited alone for "interpretability beside
cold-start accuracy" because CS-DTA's abstract was not re-read. -->

Several sequence-based drug–target interaction (DTI) models come with an explanation beside the prediction: an attention map over the
protein, read as marking where the drug binds. MolTrans (Huang et al., 2021) describes its interaction module as making prediction interpretable,
HyperAttentionDTI (Zhao et al., 2022) assigns an attention vector to each atom and amino acid, and DrugBAN (Bai et al., 2023) is titled an interpretable
bilinear attention network; later work reports interpretability alongside cold-start accuracy (for example GPS-DTI; Xiong et al., 2025).
A chemist choosing residues to mutate may reasonably read such a figure as mechanistic evidence. Whether it can bear that reading is an
empirical question about the figure, not about the model's accuracy.

The question has a history. In natural-language processing, attention weights were found to diverge from other importance measures (Jain and
Wallace, 2019; Serrano and Smith, 2019), and the conditions under which attention can serve as an explanation were debated (Wiegreffe and
Pinter, 2019). Two properties must be kept apart: *plausibility*, whether the highlighted residues are those an expert would name, and
*faithfulness*, whether the prediction depends on them (DeYoung et al., 2020). In DTI, attention has been compared with interaction
annotations before. MONN (Li et al., 2020) built a benchmark of non-covalent interactions and reported that attention-based models, without
extra supervision, did not give satisfactory interpretability; ICAN (Kurata and Tsukiyama, 2022) reported that some weighted sites in its
cross-attention matrix match experimental binding sites with statistical significance; and a recent preprint (InteractBind; Meng et al.,
2026) reports limited binding-site localisation despite strong binding prediction. Our contribution is therefore not another comparison of
attention with binding sites. It is the question of whether such comparisons, as usually reported, are *stable and calibrated enough to
support a verdict at all*.

Three features of a single-run report make that doubtful. First, the figure comes from one training run, yet nothing guarantees that a second run
of the same recipe attends to the same residues. Second, a hit rate means little unless it is read against the protein's chance level, the ceiling the
ground truth allows, a uniform map and a planted explanation of known quality, and comparisons across many cells need family-wise
correction. Third, "the attention map" is not one object: a multi-head, multi-channel model must be reduced to one weight per residue, and
that reduction is a choice. Behind all three sits the benchmark: DAVIS's cold-start splits are defined by target identifiers, which are
not sequences, and the standard DAVIS release omits the modifications of its mutant kinases (Wu et al., 2025, preprint).

We therefore present a methodological audit, not a new model. Three published attention-based models (MolTrans, HyperAttentionDTI,
DrugBAN) and XAttn-Ref, an in-house cross-attention model trained and scored identically, are retrained with their authors' recipes on
identical kinase splits, three seeds per cell, at four levels of distribution shift on DAVIS (random, unseen drug, unseen target, both)
and two on KIBA (random, unseen drug; DrugBAN is on DAVIS only); DeepDTA, which has no attention, anchors accuracy. The DAVIS arm has 60 cells and the KIBA arm 24.
<!-- src: docs/REMEDIATION_LEDGER.md:140 = 60, 24 -->
Plausibility is precision@10 against UniProt's annotated binding residues and against the KLIFS 85-residue ATP pocket, each read against its
own chance level and ceiling; faithfulness is the change in prediction when the attended residues are masked, against a random-masking
control of the same size. <!-- src: src/data/klifs_pocket.py:54 = 85 -->
A "seed" here is a full retraining with a different seed; we do not separate seed variance from other run-to-run variation (Section 4).

Our main findings are four. **(a) The verdicts are not self-consistent across seeds.** In 11 of the 22 model–dataset–level cells scored
against UniProt residues, the three seeds fall on different sides of the uncorrected α = 0.05 threshold of an exact permutation test, and in
21 of the 22 the seeds' range of precision@10 exceeds the cell's distance from chance.
<!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->
<!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 -->
**(b) Under a calibrated battery, residue-level claims rarely survive.** After Holm correction, 1 of 20 DAVIS cells supports residue-level
recovery (HyperAttentionDTI, random split, 1.66× chance) and 0 of 8 KIBA cells do; what replicates is coarser, namely load-bearing
attention for three of the four models and enrichment of the ATP pocket.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=random->enrichment = 1.66 -->
**(c) The verdict depends on the readout**, that is, on how a multi-channel map is reduced to one weight per residue (shown for HyperAttentionDTI), and on what
the map is a function of. **(d) The benchmark leaks:** on DAVIS's leaked cold-target rows, the sequence match inflates a DeepDTA baseline's AUROC by 0.116
(p = 0.0013), with no detectable inflation on the unleaked rows.
<!-- src: results/certification/key_numbers.csv#name=leak_ct_leaked_rows->value,p = 0.116, 0.0013 -->

**Scope.** Every claim concerns kinase targets (DAVIS and KIBA are kinase panels). Training seeds four and five, DrugBAN on KIBA, further
explanation methods and a conservation control are declared in the protocol amendment but have not been run; they are limitations, not
results.
