# Seed-dependent verdicts: a calibrated audit of attention-based binding-site explanations in drug–target interaction models

<!-- T20 draft, 2026-09-25. Framing: a methodological audit whose headline is the instability of the
models' explanation verdicts across training seeds. Every digit in this file carries a hidden source
tag, checked by `python scripts/check_number_provenance.py`. "XAttn-Ref" is the neutral name of the
authors' own cross-attention model (repository name ColdSite-DTI); Methods discloses that it is ours. -->

## Abstract

Attention maps over the protein are routinely presented as evidence of where a drug binds, usually
from a single training run on a random split. We ask whether such binding-site verdicts are stable
enough to report. Four attention-based drug–target interaction models (MolTrans, HyperAttentionDTI,
DrugBAN and XAttn-Ref, an in-house reference model) and a no-attention accuracy anchor were each
retrained with three seeds on identical kinase splits at up to four levels of distribution shift,
giving 84 trained cells on DAVIS and KIBA. <!-- src: docs/REMEDIATION_LEDGER.md:140 = 84 -->
Explanations were scored with a calibrated battery: chance and ceiling for every cell, a uniform-map
floor, a positive control, positional and residue-identity nulls, a size-matched random-masking
control for faithfulness, and Holm correction over declared families.

The dominant finding is instability: in 11 of 22 model–dataset–level cells the three seeds disagree
about their own verdict, meaning that they fall on different sides of the uncorrected α = 0.05 threshold
of an exact per-seed permutation test. <!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->
<!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
In 21 of 22 cells the range of the three seeds' precision@10 exceeds the cell's distance from chance,
<!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
and testing seed variance directly (Friedman and permutation tests) finds significant differences between
seeds in 7 to 8 of 22 cells after Holm correction.
<!-- src: results/certification/key_numbers.csv#name=friedman_holm->value = 8 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=permutation_holm->value = 7 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
After correction (Holm on the median p over seeds), 1 of 20 DAVIS cells supports residue-level binding-site recovery (HyperAttentionDTI,
random split, 1.66× chance) and 0 of 8 KIBA cells do; a bootstrap that resamples seeds as well as
targets agrees, leaving only that cell's interval above chance.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=random->enrichment = 1.66 -->
<!-- claim "only that cell's interval above chance": results/effects_v2_2d/enrichment.csv, families P1 and P2 (22 rows), enrichment_low > 1 only for P1 hyperattentiondti random -->
What does replicate is coarser. The attention of HyperAttentionDTI, MolTrans and XAttn-Ref is
load-bearing at every level; DrugBAN's is not, its masking effect sitting near zero and changing sign
between seeds (−0.0073 at cold-drug and −0.0037 at cold-pair for the first seed). XAttn-Ref's map is
enriched in the ATP pocket at every DAVIS level and HyperAttentionDTI's at three of four (cold-pair only marginally), at 1.33–2.10×
chance with conservation not controlled.
<!-- src: results/analysis_davis_policyA/faithfulness_drugban_davis_seed1.md:6 = -0.0073 -->
<!-- src: results/analysis_davis_policyA/faithfulness_drugban_davis_seed1.md:8 = -0.0037 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_target->enrichment = 1.33 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_drug->enrichment = 2.10 -->
<!-- claim "load-bearing at every level": results/effects_v2_2d/faithfulness_effects.csv, load_bearing_ci True in all 16 rows (XAttn-Ref, HyperAttentionDTI, MolTrans; seeds and targets resampled) -->
<!-- claim "DrugBAN's is not ... changing sign": results/analysis_davis_policyA/faithfulness_drugban_davis_seed{1,2,3}.md, 6 of 12 seed-level deltas negative (seed 1 cold-drug, cold-pair; seed 2 warm, cold-drug, cold-pair; seed 3 cold-target); no interval (per-pair values not recorded) -->
<!-- claim "XAttn-Ref every DAVIS level, HyperAttentionDTI three of four": S3-D rows of results/effects_v2_2d/enrichment.csv, enrichment_low > 1 at all four levels for coldsite_dti and at random, cold_target, cold_pair for hyperattentiondti (cold_drug 0.94) -->
Which residues a map highlights also depends on an unreported readout choice. DAVIS's
wild-type-sequence duplication inflates the AUROC of a DeepDTA baseline on the leaked cold-target rows by
0.116 (p = 0.0013, three seeds); on the strictly unleaked rows no inflation is detectable (−0.019, p = 0.31).
<!-- src: results/certification/key_numbers.csv#name=leak_ct_leaked_rows->value,p = 0.116, 0.0013 -->
<!-- src: results/certification/key_numbers.csv#name=leak_ct_unleaked_rows->value,p = -0.019, 0.31 -->
A single-seed attention figure cannot carry a binding-site claim for these models.

## Key Points

<!-- Included in case the venue requires a Key Points box. The venue's guidelines
(docs/bib_guidelines.md) have not been supplied, so the required number and length are unverified. -->

* Binding-site verdicts from attention are not self-consistent across training seeds: the three seeds
  of a cell disagree about their own verdict (fall on different sides of the uncorrected α = 0.05 threshold)
  in 11 of 22 cells; the seeds' range exceeds the distance from chance in 21 of 22; and a direct test of
  seed variance is significant in 7 to 8 of 22 cells after Holm correction.
  <!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->
  <!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
  <!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
  <!-- src: results/certification/key_numbers.csv#name=friedman_holm->value = 8 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
  <!-- src: results/certification/key_numbers.csv#name=permutation_holm->value = 7 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
* A calibrated battery — chance, ceiling, uniform floor, a positive control that detects a dose of
  0.02 of the true sites at every DAVIS level, nulls and Holm correction — is needed before a verdict
  can be read. <!-- src: results/positive_control_davis.md:11 = 0.02 -->
* Under that battery, residue-level recovery survives in 1 of 20 DAVIS cells and 0 of 8 KIBA cells;
  what replicates is coarser: load-bearing attention in three of four models (not DrugBAN) and
  pocket-level enrichment.
  <!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
  <!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
* The verdict depends on the readout: two defensible reductions of the same HyperAttentionDTI weights
  place 0.367 and 0.081 of their top-ten residues in the ATP pocket at the cold-target level, against a
  chance of 0.140.
  <!-- src: results/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=maxchannel&level=cold_target->precision@10 = 0.367 -->
  <!-- src: results/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=receptive&level=cold_target->precision@10,chance = 0.081, 0.140 -->
* Benchmark leakage is quantified rather than only reported: 54 DAVIS mutants carry exactly the
  wild-type sequence, and in a DeepDTA retrain at the cold-target split the sequence match inflates AUROC
  on the leaked rows by 0.116 (p = 0.0013), with no detectable inflation on the unleaked rows.
  <!-- src: results/sequence_audit_davis.md:9 = 54 -->
  <!-- src: results/certification/key_numbers.csv#name=leak_ct_leaked_rows->value,p = 0.116, 0.0013 -->
