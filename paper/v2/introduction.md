# 1 Introduction

<!-- T20 draft, 2026-09-25. Reframed per docs/REMEDIATION_PLAN.md strategic_direction: (a) seed
instability, (b) the calibrated battery, (c) readout dependence, (d) quantified leakage; integrated
gradients and pocket enrichment are confirmatory. Every digit carries a hidden source tag
(`python scripts/check_number_provenance.py`). Citations: only entries verified in
paper/references.md; works the plan names for the delta table (T19, not yet run) appear as
[CITE-T19: …] markers and are not characterised. Readout figure references are the user-chosen
placeholders of T07. -->

Sequence-based drug–target interaction (DTI) models increasingly present an explanation beside the
prediction: an attention map over the protein that is read as marking where the drug binds.
MolTrans (2021) [MOLTRANS_FIG_X], HyperAttentionDTI (2022) [HYPERATTENTION_FIG_X] and DrugBAN (2023)
[DRUGBAN_FIG_X] each display such maps, and later models report interpretability beside cold-start
accuracy (for example GPS-DTI, 2025; CS-DTA, 2026). A medicinal chemist choosing residues to mutate,
or a chemical series to pursue against a new target, may reasonably read such a figure as mechanistic
evidence. Whether it can bear that reading is an empirical question about the figure, not about the
model's accuracy.

That question has been asked of attention in natural-language processing, where attention weights
were found to diverge from other importance measures (Jain & Wallace, 2019; Serrano & Smith, 2019)
and the conditions under which they can serve as explanations were debated (Wiegreffe & Pinter, 2019).
Two properties must be kept apart: *plausibility*, whether the highlighted residues are the ones an
expert would name, and *faithfulness*, whether the model's prediction depends on them (ERASER, 2020).
In DTI, attention has been compared with binding-site annotations before
[CITE-T19: prior attention-versus-binding-site evaluations in DTI, to be verified], so our
contribution is not the comparison itself. It is the question of whether such comparisons, as usually
reported, are *stable and calibrated enough to support a verdict at all*.

Three features of the usual report make that doubtful. First, the figure comes from one training run,
yet nothing guarantees that a second run of the same recipe attends to the same residues. Second, a
hit rate is rarely read against the chance level of the protein, the ceiling the ground truth allows,
a uniform map, or a planted explanation of known quality, and many such comparisons are made without
family-wise correction. Third, "the attention map" is not a single object: a multi-head,
multi-channel model must be reduced to one weight per residue, and that reduction is seldom stated.
Behind all three sits the benchmark: DAVIS's cold-start splits are defined by target identifiers,
and identifiers are not sequences [CITE-T19: precedent for DAVIS sequence redundancy, to be verified].

We therefore present a methodological audit rather than a new model. Three published attention-based
models — MolTrans, HyperAttentionDTI and DrugBAN — and XAttn-Ref, an in-house drug-conditioned
cross-attention model trained and scored identically, are retrained with their authors' recipes on
identical kinase splits at four levels of shift on DAVIS (random, unseen drug, unseen target, both)
and two on KIBA (random, unseen drug), three seeds per cell; DeepDTA, which has no attention, anchors
accuracy. The DAVIS arm has 60 cells and the KIBA arm 24.
<!-- src: docs/REMEDIATION_LEDGER.md:140 = 60, 24 -->
Plausibility is precision@10 against UniProt's annotated binding residues and against the KLIFS
85-residue ATP pocket, each read against its own chance level and ceiling; faithfulness is the change
in prediction when the attended residues are masked, against a random-masking control of the same size
in the space each model reads. <!-- src: src/data/klifs_pocket.py:54 = 85 -->

**Seed instability is the headline.** Across the 22 model–dataset–level cells for which three seeds
were scored against UniProt residues, the seeds disagree about their own verdict in 12, and in 21 the
spread across seeds exceeds the cell's distance from chance.
<!-- src: results/seed_agreement.md:34 = 12, 22, 21, 22 -->
HyperAttentionDTI on DAVIS's unseen-drug split illustrates the problem: its three seeds place 0.025,
0.077 and 0.019 of their top-ten residues on annotated sites against a chance of 0.020; two of the
runs clear an uncorrected permutation test and the third does not, and the best run is four times the
worst.
<!-- src: results/seed_agreement.md:19 = 0.025, 0.077, 0.019, 0.020 -->
<!-- claim "two of the runs clear an uncorrected test": results/seed_agreement.md:19 column "seeds above alpha" = `**.` -->
<!-- "four times": 0.077 / 0.019 = 4.05 (spelled as a word, not checked by the tool) -->
A target-level bootstrap does not see this variance: the same cell's interval for enrichment over
chance, 1.98 [1.81, 2.15], lies entirely above parity, while its family-wise test does not survive correction.
<!-- src: results/effects_v2/enrichment.csv#family=P1&model=hyperattentiondti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.98, 1.81, 2.15 -->
The instability is not confined to explanations. On DAVIS the seed-to-seed standard deviation of test
AUROC is 0.001–0.006 at the random level, depending on the model, but 0.025–0.099 at the cold-pair
level.
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=random&view=uncorrected->auroc_sd = 0.001 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=drugban&level=random&view=uncorrected->auroc_sd = 0.006 -->
<!-- range check: DAVIS random auroc_sd over the five models = 0.0012, 0.0049, 0.0020, 0.0061, 0.0023; cold_pair uncorrected = 0.0991, 0.0375, 0.0253, 0.0307, 0.0352 (by_model.csv) -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=moltrans&level=cold_pair&view=uncorrected->auroc_sd = 0.025 -->
<!-- src: results/accuracy_v2/by_model.csv#dataset=davis&model=coldsite_dti&level=cold_pair&view=uncorrected->auroc_sd = 0.099 -->

**Under a calibrated battery, residue-level claims rarely survive.** The positive control shows that
the pipeline detects an explanation that ranks as little as 0.02 of the true sites first, at every
DAVIS level, so a null here is not a weak test.
<!-- src: results/positive_control_davis.md:11 = 0.02 -->
Against that resolution, 1 of 20 DAVIS cells supports residue-level recovery after Holm correction
(HyperAttentionDTI, random split: precision@10 0.034 against a chance of 0.020, enrichment 1.66
[1.49, 1.84]) and 0 of 8 KIBA cells do.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
<!-- src: results/effects_v2/enrichment.csv#family=P1&model=hyperattentiondti&level=random->precision,chance,enrichment,enrichment_low,enrichment_high = 0.034, 0.020, 1.66, 1.49, 1.84 -->
Nor is localization explained by accuracy: across the 48 DAVIS attention-model cells, Spearman's ρ
between AUROC and precision@10 is 0.033 [−0.276, 0.322], an interval that includes no association.
<!-- src: results/accuracy_v2/localization_spearman.csv#dataset=davis&group=all attention models&y=precision_at_10->n_cells,rho,low,high = 48, 0.033, -0.276, 0.322 -->
What survives is coarser. The attention of HyperAttentionDTI, MolTrans and XAttn-Ref is load-bearing
at every level, with every faithfulness interval above zero (for example HyperAttentionDTI on DAVIS's
cold-pair split, the smallest, 0.054 [0.036, 0.073]); DrugBAN's could not be re-run here.
<!-- src: results/effects_v2/faithfulness_effects.csv#model=hyperattentiondti&dataset=davis&level=cold_pair->delta,delta_low,delta_high = 0.054, 0.036, 0.073 -->
Two maps are enriched in the ATP pocket at every DAVIS level — HyperAttentionDTI at 1.33–1.70× chance
and XAttn-Ref at 1.54–2.10× — while DrugBAN's is not (0.98–1.04×). Kinase ATP pockets are conserved,
and no conservation control was run, so this enrichment is a coarse statement about the domain, not
evidence that the model has learned a ligand's contacts.
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_target->enrichment = 1.33 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=hyperattentiondti&level=random->enrichment = 1.70 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=random->enrichment = 1.54 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_drug->enrichment = 2.10 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=drugban&level=random->enrichment = 0.98 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=drugban&level=cold_target->enrichment = 1.04 -->

**The verdict depends on the readout and on what the map is a function of.** Two defensible reductions
of the same HyperAttentionDTI weights place 0.367 and 0.081 of their top-ten residues in the ATP
pocket at the unseen-target level, against a chance of 0.140 — one reduction reports 2.6× chance, the
other less than chance.
<!-- src: ~/ColdSite-results/readouts/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=maxchannel&level=cold_target->precision@10 = 0.367 -->
<!-- src: ~/ColdSite-results/readouts/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=receptive&level=cold_target->precision@10,chance = 0.081, 0.140 -->
<!-- src: derived: 0.367 / 0.140 = 2.6 -->
Whether a map can depend on the drug is a property of the computation graph: the MolTrans readout
scored here is identical for every drug on the same protein (the same top ten in 100% of 150 drug
pairs), whereas DrugBAN's is never identical (0% of 150).
<!-- src: results/drug_dependence/moltrans.txt:1 = 100, 150 -->
<!-- src: results/drug_dependence/drugban.txt:1 = 0, 150 -->

**Benchmark leakage is quantified, not only reported.** DAVIS gives 54 of its mutant targets exactly
the wild-type sequence, so 12 of the 88 "unseen" cold-target test targets are seen by sequence in
training — 816 of 5984 test rows (13.6%).
<!-- src: results/sequence_audit_davis.md:9 = 54 -->
<!-- src: results/sequence_audit_davis.md:21 = 12, 88, 816, 5984, 13.6 -->
Retraining the accuracy anchor without the leak, at matched training size, attributes 0.019 of its
cold-target AUROC to the leak on all rows and 0.116 on the leaked rows.
<!-- src: results/leakage_retrain_davis.md:22 = 0.019, 0.116 -->

As a confirmatory check, integrated gradients on the same checkpoints localise better than the
attention in several cells — for HyperAttentionDTI on KIBA's unseen-drug split, 1.90 [1.58, 2.22]×
chance against the attention's 0.94 [0.79, 1.09]× — consistent with the view that a weak attention
map can under-report what a model represents. We treat this as supporting evidence, not a headline.
<!-- src: results/effects_v2/enrichment.csv#family=S2&model=hyperattentiondti_ig&level=cold_drug->enrichment,enrichment_low,enrichment_high = 1.90, 1.58, 2.22 -->
<!-- src: results/effects_v2/enrichment.csv#family=P2&model=hyperattentiondti&level=cold_drug->enrichment,enrichment_low,enrichment_high = 0.94, 0.79, 1.09 -->

**Scope.** Every claim in this paper concerns kinase targets: DAVIS and KIBA are kinase panels, and
the non-kinase transfer panel we assembled is not analysed here. Training seeds four and five, DrugBAN on KIBA, the
additional explanation methods and a conservation control are declared in the protocol amendment but
were not run; they are listed as limitations, not as results.

Our contributions are:

* **(a) Evidence that attention binding-site verdicts are not self-consistent across training seeds**,
  for three published models and one reference model, on two benchmarks.
* **(b) A calibrated test battery** — chance and ceiling per cell, uniform floor, positive control,
  positional and residue-identity nulls, size-matched faithfulness control, and Holm correction over
  declared families — released with code that reproduces every number.
* **(c) A demonstration that the verdict depends on the readout** and on which inputs the map is a
  function of, with a check any author can apply before publishing.
* **(d) A quantification of DAVIS's wild-type-sequence leakage** on cold-start accuracy.

> **Box 1. Reporting checklist for DTI explanation claims** (derived only from the findings above)
>
> 1. Report the explanation from several training seeds, and the agreement between them, not one run.
> 2. Read every hit rate against the protein's chance level and the ground truth's ceiling.
> 3. Include a uniform-map floor and a positive control that states the smallest detectable effect.
> 4. Correct for multiple comparisons across the whole family of cells shown.
> 5. State the readout: which heads, layers, channels and pooling produce one weight per residue.
> 6. State which inputs the explanation is a function of; a map that cannot change with the drug
>    cannot mark that drug's binding site.
> 7. Test faithfulness with a masking control of the same size in the model's own token space.
> 8. Check cold-start splits for sequence identity, not identifier identity, before reporting them.
