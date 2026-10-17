# Seed-dependent verdicts: a calibrated audit of attention-based binding-site explanations in drug–target interaction models

<!-- v3 (ISMB / Bioinformatics format), step 2, 2026-10-17. Condensed from paper/v2/abstract.md; every digit carries a source tag
(checked by `python scripts/check_number_provenance.py paper/v3_ismb/abstract.md`). The structured-abstract headings
(Motivation / Results / Availability and implementation / Contact / Supplementary information) and the ~250-word limit are the
Bioinformatics convention from memory: the author-guidelines page did not state them (VENUE_SPEC.md), so confirm against the
journal's template before submission. "XAttn-Ref" is the neutral name of the authors' own model (repository name ColdSite-DTI);
Methods discloses that it is ours. -->

## Abstract

**Motivation:** Attention maps over the protein can be read as evidence of where a drug binds, and a figure from a single training run on a
random split is the simplest such report. We ask whether such binding-site verdicts are stable enough to report.

**Results:** We retrained four attention-based drug–target interaction models (MolTrans, HyperAttentionDTI, DrugBAN and XAttn-Ref,
an in-house reference model) and a no-attention accuracy anchor, three seeds each, on identical kinase splits at up to four levels of
distribution shift (84 trained cells on DAVIS and KIBA), and scored explanations against chance, a positive control, nulls and size-matched
masking controls, with Holm correction over declared families.
<!-- src: docs/REMEDIATION_LEDGER.md:140 = 84 -->
In 11 of 22 model–dataset–level cells the three seeds fall on different sides of the uncorrected α = 0.05 threshold of their own
verdict, and in 21 of 22 the seeds' range exceeds the cell's distance from chance.
<!-- src: src/evaluation/seed_agreement.py:21 = 0.05 -->
<!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 -->
After Holm correction (on the median p over seeds), 1 of 20 DAVIS cells supports residue-level recovery (HyperAttentionDTI, random split, 1.66× chance) and 0 of 8 KIBA cells do.
<!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
<!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=P1&model=hyperattentiondti&level=random->enrichment = 1.66 -->
What replicates is coarser: masking the attended residues is load-bearing for three of the four models (not consistently for DrugBAN), and the ATP pocket is
enriched in XAttn-Ref's maps at every DAVIS level and HyperAttentionDTI's at three of four (point estimates 1.33–2.10× chance; conservation not controlled).
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=hyperattentiondti&level=cold_target->enrichment = 1.33 -->
<!-- src: results/effects_v2_2d/enrichment.csv#family=S3-D&model=coldsite_dti&level=cold_drug->enrichment = 2.10 -->
DAVIS's duplicated wild-type sequences inflate a DeepDTA baseline's cold-target AUROC by 0.116 (p = 0.0013) on the leaked rows, but not
detectably (−0.019, p = 0.31) on the unleaked rows.
<!-- src: results/certification/key_numbers.csv#name=leak_ct_leaked_rows->value,p = 0.116, 0.0013 -->
<!-- src: results/certification/key_numbers.csv#name=leak_ct_unleaked_rows->value,p = -0.019, 0.31 -->

**Conclusion:** A single-seed attention figure cannot carry a residue-level binding-site claim; report verdicts across seeds, against
chance and a positive control, with correction and a stated readout.

**Availability and implementation:** Code, splits, analysis outputs and the figure scripts are at
https://github.com/Mahim56207/ColdSite-DTI_New. **[TO ADD BEFORE SUBMISSION — not yet done: an archived release with a DOI
(Zenodo or similar), which the journal requires in this line, and the public location of the trained checkpoints.]**

**Contact:** [corresponding author's e-mail — to be added by the author]

**Supplementary information:** Supplementary data are available online. [Wording to be matched to the template.]

---

## Key points (only if the template asks for them — outside the word budget)

<!-- The 'Key Points' box is not known to be required; kept from v2 in case. -->

* Binding-site verdicts from attention are not self-consistent across training seeds: in 11 of 22 cells the three seeds disagree about their own verdict,
  and in 21 of 22 the seeds' range exceeds the cell's distance from chance.
  <!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
  <!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 -->
* Under a calibrated battery, residue-level recovery survives in 1 of 20 DAVIS cells and 0 of 8 KIBA cells; what replicates is coarser (load-bearing
  attention in three of four models, and pocket-level enrichment).
  <!-- src: results/analysis_davis_policyA/audit_davis_binary_10k_permutations.md:15 = 1, 20 -->
  <!-- src: results/analysis_kiba_policyA/audit_kiba_binary_10k_permutations.md:14 = 0, 8 -->
* The verdict depends on the readout, and DAVIS's duplicated wild-type sequences measurably inflate cold-target accuracy on the leaked rows.
