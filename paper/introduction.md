# Introduction (draft)

Paragraph-by-paragraph, with the results filled in from `paper/results.md` (2026-09-14).
Every number here is one this paper reports; *[PENDING]* marks the two that a run still has
to settle. Citations are by model name and year until the venue's style is chosen;
`paper/references.md` holds the verified entries.

---

**¶1 — Why explanations matter in DTI prediction.** Deep models now predict drug–target
binding from sequence alone, and the most cited of them present an explanation beside the
prediction: an attention map over the protein that is said to mark where the drug binds.
In drug discovery that map is not decoration. A chemist deciding which residues to mutate,
or which series to pursue against a new target, reads it as mechanistic evidence, and the
published figures invite exactly that reading. *[Cite: HyperAttentionDTI (2022), MolTrans
(2021); recent examples that present attention or interpretability alongside cold-start
results — DMFF-DTA (2025), EviDTI (2025), GPS-DTI (2025), CS-DTA (2026).]*

**¶2 — How those claims are validated.** The evidence for such claims is usually an
inspected attention map, a case study, or a hit rate on a random split, where every test
drug and protein has close relatives in training. The setting that motivates the models
is the opposite: a new target, a new chemical series, or both. Recent models report
accuracy under these cold-start conditions, often beside an interpretability analysis;
whether explanation quality itself holds up as the shift grows has not, to our knowledge,
been measured. *[Cite cold-start DTI evaluation work; ColdDTI (2025), GPS-DTI (2025).]*

**¶3 — What is known outside DTI.** That explanations degrade under distribution shift is
established elsewhere — for attribution methods in vision, for explainers of graph neural
networks, and for recommender explanations — and whether attention is an explanation at all
has been disputed since Jain & Wallace (2019), Serrano & Smith (2019) and Wiegreffe &
Pinter (2019). Two properties must be kept apart: *plausibility*, whether the explanation
looks right to an expert, and *faithfulness*, whether the model actually depends on what it
highlights. The dangerous case is an explanation that is plausible but not faithful, because
no expert can detect it by eye. *[Cite Gupta et al. (2025), Zhang et al. (2026) and Sun
(2025) from Related Work §2 — all arXiv preprints, check for published versions — and
DeYoung et al. (2020) for comprehensiveness and sufficiency.]*

**¶4 — The gap.** DTI's interpretability claims have not been tested against either
concern. The closest prior work, CS-DTA (2026), reports interpretability for its own model
under cold-start and includes a non-kinase validation; it does not measure faithfulness,
compare published models, test against a floor, or correct for multiple comparisons.
*[Keep this paragraph exactly as unflattering to our novelty as Related Work §1.4.]*

**¶5 — This work: an audit.** We audit the interpretability claims of published
attention-based DTI models — HyperAttentionDTI and MolTrans, with our own ColdSite-DTI held
to the same standard and DeepDTA as an accuracy anchor — across four levels of distribution
shift on DAVIS *[PENDING: and KIBA, scope set by available compute]*: random, unseen drug,
unseen target and both. Each model is retrained on identical splits with its authors'
recipe, three seeds per cell. We measure plausibility as precision@k against three ground
truths at different resolutions — UniProt's annotated residues, the 85-residue KLIFS ATP
pocket, and the residues a drug is measured to contact in its own co-crystal structure —
each read against its own chance level and achievable ceiling. We measure faithfulness as
the change in prediction when the attended residues are masked, against a random-masking
control, in the input space each model actually reads. A uniform attention map provides the
floor; a positive control establishes that the pipeline recognises a genuinely good
explanation, and at what resolution; nulls for position, amino-acid preference and drug
identity test the explanations that would otherwise be read as binding-site recovery;
significance is corrected once across all sixteen cells of the family; a panel of 60
non-kinase proteins stands in for a family stratification the benchmarks cannot support;
alternative attention readouts test whether a verdict belongs to the model or to the
reduction; and integrated gradients on the same checkpoints separate a poor explanation from
a model that never learned the site.

**¶6 — What we find.** One of sixteen cells supports the residue-level interpretability
claim after correction: HyperAttentionDTI on the random split, at 1.7× chance
(precision@10 0.034 [0.030–0.038] against 0.020), where it also beats a borrowed attention
map and an amino-acid-preserving permutation. Under distribution shift no model's attention
marks the annotated residues better than chance, MolTrans is indistinguishable from a
uniform map at every level, and no model's attention distinguishes a drug's own
crystallographic contacts from another drug's in the same pocket. What survives everywhere
is coarser: attention that is load-bearing — masking the attended residues moves the
prediction more than masking random ones, at every level of every model — and that
concentrates on the kinase domain at 1.3–2.1× chance for the two models that clear the
pocket floor at all. Three findings concern the
measurement rather than the models, and we expect them to matter beyond DTI: which residues
an attention map highlights is mostly a property of an unreported reduction choice
(alternative readouts share 2–12% of their top-ten residues with the published one, and one
choice moves a pocket-level verdict from below chance to 2.6× chance); a masking-based
faithfulness test inverts its own sign for a sub-word model, because k residues is not a
fixed-size intervention; and integrated gradients on the same checkpoints survive correction
in **seven of twelve cells where the attention survives in one of sixteen**, reaching
2.3–4.1× chance at levels where the attention is at chance — so for two of the three models
the attention under-reports a binding site the model does represent, and under-reports it
worst under the shift where interpretability is supposed to earn its keep. The third model
is the control for that claim: its gradient matches its attention and both sit at the
metric's floor, so its failure is the model rather than the report — a distinction no
attention measurement can draw. Along the way the
benchmark itself required correction: DAVIS's protein file gives every mutant its wild-type
sequence, and retraining without that leak accounts for 0.019 of cold-target's 0.038
apparent difficulty.

**¶7 — Contributions.**
1. An audit, not a model: three published attention-based DTI models, one measurement
   suite, four levels of distribution shift, three seeds, and family-wise error control
   applied once across the whole family rather than per model.
2. Two axes, each against its own control: plausibility against a uniform floor, three
   ground-truth resolutions and a validated positive control that says at what dose a real
   signal would be visible; faithfulness against random masking, matched in size to the
   explanation in the space each model reads.
3. Controls that changed conclusions rather than decorating them — nulls for position and
   amino-acid preference, a per-pair ground truth from crystallographic contacts with a
   same-pocket different-drug control, alternative readouts, and a second explanation
   method on the same weights.
4. Two findings about interpretability measurement itself, applicable outside DTI: the
   readout dependence of attention-based plausibility, and the non-transferability of
   masking-based faithfulness across tokenisations.
5. A data-quality audit of DAVIS as the field uses it — sequence-identical mutants,
   pocketless sequences, the leak quantified by retraining — and a kinase-family confound
   shown to be untestable within either standard benchmark.
6. Our own model audited on the same terms, with its results reported whether or not they
   flatter it: ColdSite-DTI's attention is load-bearing everywhere, finely plausible
   nowhere, and its one above-chance result off the training family is traced to an
   amino-acid preference rather than to knowledge of binding.

*[Before submission: cut ¶3 if Related Work carries it in full. ¶6's KIBA sentence is
missing and its absence is a limitation the abstract must state if the replication does not
land.]*
