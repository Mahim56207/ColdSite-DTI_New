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

The dominant finding is instability. In 12 of 22 model–dataset–level cells the three seeds disagree
about their own verdict, and in 21 of 22 the spread across seeds exceeds the cell's distance from
chance. <!-- src: results/seed_agreement.md:34 = 12, 22, 21, 22 -->
After correction, 1 of 20 DAVIS cells supports residue-level binding-site recovery (HyperAttentionDTI,
random split, 1.66× chance) and 0 of 8 KIBA cells do.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
<!-- src: results/effects_v2/enrichment.csv#family=P1&model=hyperattentiondti&level=random->enrichment = 1.66 -->
What does replicate is coarser: the attention of the three models tested for faithfulness is
load-bearing at every level, and two of the four maps are enriched in the ATP pocket at every DAVIS
level, at 1.33–2.10× chance with conservation not controlled.
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_target->enrichment = 1.33 -->
<!-- src: results/effects_v2/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_drug->enrichment = 2.10 -->
<!-- claim "load-bearing at every level": results/effects_v2/faithfulness_effects.csv, load_bearing_ci True in all 16 rows (XAttn-Ref, HyperAttentionDTI, MolTrans); DrugBAN has no interval (no DGL, ledger T05) -->
<!-- claim "two of the four maps ... every DAVIS level": S3-D rows of enrichment.csv, enrichment_low > 1 at all four levels for coldsite_dti and hyperattentiondti only -->
Which residues a map highlights also depends on an unreported readout choice, and DAVIS's
wild-type-sequence duplication inflated a baseline's cold-target AUROC by 0.019.
<!-- src: results/leakage_retrain_davis.md:22 = 0.019 -->
A single-seed attention figure cannot carry a binding-site claim for these models.

## Key Points

<!-- Included in case the venue requires a Key Points box. The venue's guidelines
(docs/bib_guidelines.md) have not been supplied, so the required number and length are unverified. -->

* Binding-site verdicts from attention are not self-consistent across training seeds: the seeds
  disagree in 12 of 22 cells, and the spread across seeds exceeds the distance from chance in 21 of 22.
  <!-- src: results/seed_agreement.md:34 = 12, 22, 21, 22 -->
* A calibrated battery — chance, ceiling, uniform floor, a positive control that detects a dose of
  0.02 of the true sites at every DAVIS level, nulls and Holm correction — is needed before a verdict
  can be read. <!-- src: results/positive_control_davis.md:11 = 0.02 -->
* Under that battery, residue-level recovery survives in 1 of 20 DAVIS cells and 0 of 8 KIBA cells;
  what replicates is coarser: load-bearing attention and pocket-level enrichment.
  <!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
  <!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
* The verdict depends on the readout: two defensible reductions of the same HyperAttentionDTI weights
  place 0.367 and 0.081 of their top-ten residues in the ATP pocket at the cold-target level, against a
  chance of 0.140.
  <!-- src: ~/ColdSite-results/readouts/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=maxchannel&level=cold_target->precision@10 = 0.367 -->
  <!-- src: ~/ColdSite-results/readouts/readout_comparison.csv#ground truth=klifs&model=hyperattentiondti&readout=receptive&level=cold_target->precision@10,chance = 0.081, 0.140 -->
* Benchmark leakage is quantified rather than only reported: 54 DAVIS mutants carry exactly the
  wild-type sequence, and removing the leak changes a baseline's cold-target AUROC by 0.019.
  <!-- src: results/sequence_audit_davis.md:9 = 54 -->
  <!-- src: results/leakage_retrain_davis.md:22 = 0.019 -->
