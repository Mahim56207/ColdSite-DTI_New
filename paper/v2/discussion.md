# 4 Discussion

<!-- T24 draft (user scope "Drafting: Discussion"), 2026-09-25. Prior work is characterised only from the
verified abstracts in paper/citation_verification/ (docs/delta_table.md). Pending compute is described
from docs/wave_plan.md and config/waves.json: planned, partitioned and validated in dry run, not launched,
no launch date. Every digit is source-tagged (`python scripts/check_number_provenance.py`). -->

## 4.1 What the audit shows

The central result is that an attention-based binding-site verdict, as usually reported, is not a
stable property of a model recipe. Three runs of the same recipe on the same split frequently disagree
about whether the map finds annotated residues (11 of 22 cells), the disagreement is usually larger than the
effect being claimed (21 of 22), and in 7 to 8 of 22 cells the differences between seeds are significant
after correction (Section 3.2).
<!-- src: results/certification/key_numbers.csv#name=seeds_disagree_exact->value = 11 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=spread_exceeds_distance->value = 21 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=friedman_holm->value = 8 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 -->
<!-- src: results/certification/key_numbers.csv#name=permutation_holm->value = 7 --> <!-- src: results/certification/key_numbers.csv#name=cells_total->value = 22 --> A single-run figure therefore samples one draw from a wide distribution,
and an interval computed over targets alone does not reflect it; resampling seeds as well does. Once chance, ceiling, a
uniform floor, a positive control and family-wise correction are applied, residue-level recovery
survives in one DAVIS cell and in no KIBA cell (Section 3.3).

What does replicate is coarser: the attention of three of the four models is load-bearing in every
seed, and XAttn-Ref's map concentrates in the kinase ATP pocket at every DAVIS level, HyperAttentionDTI's
at three of four (cold-pair only marginally). DrugBAN is the exception on both counts — its map is neither enriched in the pocket nor
consistently load-bearing, its masking effect changing sign between seeds (Section 3.4).
In the vocabulary of Jacovi & Goldberg (2020), the other three models are faithful without being
plausible at the residue level: their predictions depend on the residues they attend to, but those residues are rarely
the annotated ones. Pocket enrichment is also the weakest kind of plausibility, because the pocket is
conserved across kinases; without a conservation control it cannot be read as learned chemistry.

Two further results bear on how such maps should be reported. The verdict depends on how a
multi-channel attention tensor is reduced to one weight per residue, and some published readouts are
mathematically unable to depend on the drug (Section 3.5). Accuracy, finally, is no guide to
localisation: across cells, a more accurate model is not a better localiser (Section 3.4).

## 4.2 Relation to prior work

Attention has been compared with binding-site annotations in DTI before. MONN (2020) built a benchmark
of non-covalent interactions and reported that attention-based models did not give satisfactory
interpretability without extra supervision; ICAN (2022) reported that some cross-attention weights
correspond to experimental binding sites with statistical significance on DAVIS; and InteractBind
(2026, preprint) found limited binding-site localisation despite strong binding prediction across eight
models. Our residue-level results agree with the first and third, and our absence of an
accuracy–localisation association is consistent with the third. What we add is the observation that the
verdict itself is unstable across seeds, and a battery that states when a single-model significance
claim of the kind ICAN reports can be trusted.

Two recent audits approach DTI explanations from other directions. ISAAC (2026, preprint) probes models
with matched mechanistic and spurious interventions and reports reasoning-score differences that are
stable across training seeds; Vefghi et al. (2026, preprint) combine gradient attributions and occlusion
on one model and state that their analysis does not substitute for structural ground truth. Our audit
is complementary to both: it scores explanations against structural annotations, and in the models
examined here the localisation verdicts lack the seed stability ISAAC reports for its intervention
scores; whether the two measures would agree on the same models is untested.

The broader debate over attention as explanation (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019)
concerned whether attention weights can be read as importance at all. Our faithfulness results side
with the view that they can carry importance, and our plausibility results show that importance and
biological correctness are separate questions. The drug-dependence check plays the role that
model-randomisation tests play for saliency maps (Adebayo et al., 2018): an explanation that does not
change when the input that should matter changes cannot explain that input.

On benchmarks, the DAVIS precedent is established: its standard release omits the modifications of
mutant kinases (DAVIS-complete, 2025, preprint), and kinase-affinity models can reward leakage under
permissive splits (Ong et al., 2023, preprint). Our contribution is a measured effect on cold-start
accuracy, separated from the loss of training data, and an exclusion policy that the explanation
metrics apply throughout.

## 4.3 Recommendations

The checklist in Box 1 follows directly from these results. The two items that would have changed the
conclusions of a typical single-model report most are reporting several seeds with their agreement, and
reading every hit rate against its chance level with correction across the cells shown. Both are cheap
relative to training, and the code released with this paper applies them to any model that exposes one
weight per residue.

## 4.4 Limitations and pending work

**Compute that is planned, partitioned and pending execution.** Three training seeds are few for
estimating seed instability, and DrugBAN has no KIBA cells. Both gaps are addressed by a first cloud
wave, planned before any of its results exist and frozen in a dated protocol amendment. It comprises
30 cells, each assigned to exactly one of three Kaggle accounts: seeds 4 and 5 of HyperAttentionDTI,
MolTrans and XAttn-Ref at all four DAVIS levels (12 cells per account), and DrugBAN on KIBA at the
random and cold-drug levels with seeds 1–3 (6 cells).
<!-- src: docs/wave_plan.md:7 = 30 -->
<!-- src: docs/wave_plan.md:11 = 4, 12 -->
<!-- src: docs/wave_plan.md:12 = 5 -->
<!-- src: docs/wave_plan.md:13 = 1, 3, 6 -->
The seed wave is estimated from measured epoch times at 23.2 GPU-hours per account; DrugBAN's hours
have not been measured, so a timed smoke run precedes its cells.
<!-- src: docs/wave_plan.md:47 = 23.2 -->
The notebooks are generated and pass their pre-flight checks in dry-run mode; by design they refuse to
train until a canary, a retraining of one already committed cell with the new cloud harness, reproduces
its recorded metrics within the amendment's declared tolerance. The wave has not yet been launched,
and none of its cells contributes to any number in this paper. The seeds 4–5 results will be analysed as a separately declared extension family
and reported beside, not merged into, the primary seeds 1–3 analysis.

**Analyses declared but not yet run.** Faithfulness intervals for DrugBAN (its committed point
estimates are reported, but a re-run that records per-pair values needs a graph library unavailable on
the analysis machine), integrated gradients for HyperAttentionDTI on DAVIS, a faithfulness
arm for the uniform map, the additional explanation methods, and the conservation null for the pocket
enrichment are declared in the protocol amendment and are pending; the explanation-method family also
awaits a signed addendum before any of its cells may be scored. Until the conservation null is
run, pocket enrichment should be read as a statement about a conserved region.

**Not planned.** KIBA was trained at the random and cold-drug levels only; its cold-target and
cold-pair levels are not part of the planned wave, so every cold-target and cold-pair claim rests on
DAVIS alone.

**Readout.** For each published model we score the readout used in our primary analysis; whether it is
the map each paper displays has not been confirmed against the papers' figures, and alternatives are
reported as a sensitivity analysis. Given Section 3.5, this matters: a different primary readout could
change a model's verdict.

**Scope, ground truth and design.** All claims concern kinases; the non-kinase transfer panel we
assembled is not analysed. UniProt annotations and the KLIFS pocket are structural ground truths, which
may be demanding for sequence-only models; the pocket is the lenient rung. The Holm families are
declared rather than pre-registered: they grew when DrugBAN and XAttn-Ref's KIBA cells were added after
first results, and were frozen before any further analysis. DAVIS's cold-pair validation split contains
seen-by-sequence targets, which may have influenced checkpoint selection and cannot be undone without
retraining. XAttn-Ref is the authors' own model; it was trained and scored identically, and its
unfavourable results, including chance-level residue localisation, are reported.

## 4.5 Conclusion

For the four attention-based DTI models audited here, a single-seed attention figure cannot carry a
residue-level binding-site claim: the verdict changes between seeds, rarely survives a calibrated test,
and depends on an often unstated readout. What these maps do show reliably is coarser — they are
load-bearing for three of the four models and, for two, concentrated in the kinase pocket at most
DAVIS levels. We recommend that explanation claims
in DTI be reported across seeds, against chance and a positive control, with correction and a stated
readout, on splits checked for sequence identity.
