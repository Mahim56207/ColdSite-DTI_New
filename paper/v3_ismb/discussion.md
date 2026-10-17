# 4 Discussion

<!-- v3 (ISMB / Bioinformatics format), step 6, 2026-10-17. Condensed from paper/v2/discussion.md (1,502 words). Rewritten on purpose: the Wave A paragraph of v2 is stale (the cloud
harness, canary and epoch-1 gate have since been run) and is reduced to the single fact that matters for a reader, that seeds 4-5 were not run. Added: the same-seed
run-to-run evidence from the two canary attempts (docs/canary_attempts/), which v2 did not have. Prior work is characterised only from verified abstracts
(docs/delta_table.md). The HyperAttentionDTI weight/attention diagnostics (IRONCLAD_VERIFICATION_REPORT.md) are exploratory, one model, and deliberately NOT used here.
Every digit has a source tag (`python scripts/check_number_provenance.py paper/v3_ismb/discussion.md`). -->

## 4.1 What the audit shows

The central result is that an attention-based binding-site verdict, as a single-run report presents it, is not a stable property of a model recipe. Three retrainings of
the same recipe on the same split frequently disagree about whether the map finds annotated residues (11 of 22 cells), the disagreement is usually larger
than the effect being claimed (21 of 22), and in 7 to 8 of 22 cells the differences between seeds are significant after correction.
<!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 -->
<!-- src: results/certification/key_numbers.csv#name=friedman_holm->value = 8 --> <!-- src: results/certification/key_numbers.csv#name=permutation_holm->value = 7 -->
A single-run figure therefore samples one draw from a wide distribution, and an interval computed over proteins alone does not reflect it. Once chance, ceiling, a
uniform floor, a positive control and family-wise correction are applied, residue-level recovery survives in one DAVIS cell and in no KIBA cell.

What does replicate is coarser. The attention of three of the four models is load-bearing in every seed; XAttn-Ref's map concentrates in the kinase ATP pocket at
every DAVIS level and HyperAttentionDTI's at three of four; DrugBAN is the exception on both counts. In the vocabulary of Jacovi and Goldberg (2020), these models
are faithful without being plausible at the residue level: their predictions depend on the residues they attend to, but those residues are rarely the annotated ones.
Pocket enrichment is also the weakest kind of plausibility, because the pocket is conserved across kinases; without a conservation control it cannot be read as
learned chemistry. Two further results bear on reporting: for HyperAttentionDTI the verdict depends on how a multi-channel tensor is reduced to one weight per residue, and our
primary readout for MolTrans cannot depend on the drug at all (its interaction map can). Accuracy, finally, is no guide to localisation.

## 4.2 Relation to prior work

Attention has been compared with binding-site annotations in DTI before. MONN (Li et al., 2020) reported that attention-based models, without extra supervision, did
not give satisfactory interpretability; ICAN (Kurata and Tsukiyama, 2022) reported that some cross-attention weights match experimental binding sites with statistical
significance on DAVIS; and InteractBind (Meng et al., 2026, preprint) found limited localisation despite strong binding prediction across eight models. Our
residue-level results are consistent with the first and third. What we add is that the verdict itself is unstable across retrainings, and a battery that states when a
single-model significance claim of the kind ICAN reports can be trusted. Two recent audits approach DTI explanations differently: ISAAC (Tarantino et al., 2026,
preprint) probes models with matched mechanistic and spurious interventions and reports scores stable across training seeds, and Vefghi et al. (2026, preprint) combine
gradient attributions and occlusion on one model and state that this does not substitute for structural ground truth. Ours is complementary: it scores explanations
against structural annotations, and for these models the localisation verdicts lack the seed stability ISAAC reports for its intervention scores; whether the two
measures would agree on the same models is untested.

The debate over attention as explanation (Jain and Wallace, 2019; Wiegreffe and Pinter, 2019) asked whether attention weights can be read as importance at all. Our
faithfulness results side with the view that they can carry importance, and our plausibility results show that importance and biological correctness are separate
questions. The drug-dependence check plays the role that model-randomisation tests play for saliency maps (Adebayo et al., 2018): an explanation that does not change
when the input that should matter changes cannot explain that input. On benchmarks, DAVIS's omission of mutant-kinase modifications (Wu et al., 2025, preprint) and the
leakage that permissive kinase splits can reward (Ong et al., 2023, preprint) have precedent; we add a measured effect on DeepDTA's cold-start accuracy, separated from the loss
of training data.

## 4.3 Recommendations

The following follow directly from these results (Box 1). (i) Report an explanation from several training runs, with the agreement between them and intervals that resample
runs as well as proteins. (ii) Read every hit rate against the protein's chance level and the ground truth's ceiling, with a uniform-map floor and a positive control that
states the smallest detectable effect. (iii) Correct for multiple comparisons across the family of cells shown. (iv) State the readout: which heads, layers, channels and pooling
produce one weight per residue, and which inputs it is a function of. (v) Test faithfulness with a masking control of the same size in the model's own token space.
(vi) Check cold-start splits for sequence identity, not identifier identity. The first and third would most change the conclusions of a single-model report,
and both are cheap relative to training.

## 4.4 Limitations

**A "seed" is a retraining, not a controlled source of variance.** We did not separate initialisation from other run-to-run variation. The cloud-harness canary retrained one
committed cell (XAttn-Ref, DAVIS cold-target, seed 1) three times with identical initial weights and the same GPU type: test AUROC came out at 0.8868, 0.8494 and 0.8705, against the committed 0.8501.
The first two attempts used the same library version and different versions of the harness code but the same training code; the third used a newer library version (docs/canary_attempts/README.md). This is
three reruns of one cell and not a variance estimate; it does show that fixing the seed does not fix the outcome. Our claim is therefore that verdicts differ between retrainings, which is
what a single-run report depends on, and not that initialisation is the cause. We also do not test why the maps disagree.
<!-- src: docs/canary_attempts/attempt1_canary_verdict_FAIL.json#$.seed = 1 -->
<!-- src: docs/canary_attempts/attempt1_canary_verdict_FAIL.json#$.metrics.auroc.new = 0.8868 -->
<!-- src: docs/canary_attempts/attempt2_canary_verdict_PASS.json#$.metrics.auroc.new = 0.8494 -->
<!-- src: docs/canary_attempts/attempt3_canary_verdict_PASS_rule_v2.json#$.metrics.auroc.new = 0.8705 -->
<!-- src: docs/canary_attempts/attempt2_canary_verdict_PASS.json#$.metrics.auroc.committed = 0.8501 -->
<!-- claim "identical initial weights, same GPU type; attempts 1 and 2 same library version, different harness code, same training code; attempt 3 newer library version": docs/canary_attempts/README.md (table, first paragraph, and the Attempt 3 section) and attempt3_environment_and_run.json -->

**Few seeds, and compute not yet run.** Three seeds are few for estimating seed instability. Seeds four and five of HyperAttentionDTI, MolTrans and XAttn-Ref, and DrugBAN on KIBA,
are planned and declared in a dated protocol amendment but have not been run, and none contributes to any number here; if they are run, the counts above will be recomputed and reported
beside, not merged into, the three-seed analysis. KIBA was trained at the random and cold-drug levels only, so every cold-target and cold-pair claim rests on DAVIS alone.

**Ground truth and scope.** All claims concern kinases; the non-kinase transfer panel is not analysed. UniProt annotations and the KLIFS pocket are structural ground truths that may be
demanding for sequence-only models, and the pocket is the lenient rung, with no conservation control. For each published model we score the readout of our primary analysis; whether it is the
map each paper displays has not been confirmed against the papers' figures, and given Section 3.5 a different primary readout could change a model's verdict. DrugBAN's faithfulness values
are point estimates without intervals. The permutation nulls and bootstraps treat proteins as independent, although many kinases are related and share both sites and attention; we did not model that dependence, and it could make the p-values and intervals optimistic.

**Design.** The Holm families are declared rather than pre-registered. DAVIS's cold-pair validation split contains seen-by-sequence targets, which may have influenced checkpoint selection and
cannot be undone without retraining. XAttn-Ref is the authors' own model; it was trained and scored identically, and its unfavourable results, including chance-level residue localisation, are reported.

## 4.5 Conclusion

For the four attention-based DTI models audited here, a single-seed attention figure cannot carry a residue-level binding-site claim: the verdict changes between retrainings, rarely survives a
calibrated test, and depends on the readout. What these maps show reliably is coarser: they are load-bearing for three of the four models and, for two, concentrated in the kinase
pocket at most DAVIS levels. Explanation claims in DTI should be reported across runs, against chance and a positive control, with correction and a stated readout, on splits checked for sequence
identity.
