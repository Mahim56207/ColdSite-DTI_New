# Introduction (skeleton)

Paragraph-by-paragraph plan with draft prose. *[RESULT: …]* marks a number or finding
that must come from the finished grid; nothing here asserts one. Citations are by model
name and year until the venue's style is chosen; `paper/references.md` holds the verified
entries.

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
been measured. *[Cite cold-start DTI
evaluation work; ColdDTI (2025), GPS-DTI (2025).]*

**¶3 — What is known outside DTI.** That explanations degrade under distribution shift is
established elsewhere — for attribution methods in vision, for explainers of graph neural
networks, and for recommender explanations — and whether
attention is an explanation at all has been disputed since Jain & Wallace (2019), Serrano
& Smith (2019) and Wiegreffe & Pinter (2019). Two properties must be kept apart:
*plausibility*, whether the explanation looks right to an expert, and *faithfulness*,
whether the model actually depends on what it highlights. The dangerous case is an
explanation that is plausible but not faithful, because no expert can detect it by eye.
*[Cite Gupta et al. (2025), Zhang et al. (2026) and Sun (2025) from Related Work §2 —
all arXiv preprints, check for published versions — and DeYoung et al. (2020) for
comprehensiveness and sufficiency.]*

**¶4 — The gap.** DTI's interpretability claims have not been tested against either
concern. The closest prior work, CS-DTA (2026), reports interpretability for its own model
under cold-start and includes a non-kinase validation; it does not measure faithfulness,
compare published models, test against a floor, or correct for multiple comparisons.
*[Keep this paragraph exactly as unflattering to our novelty as Related Work §1.4.]*

**¶5 — This work: an audit.** We audit the interpretability claims of published
attention-based DTI models — HyperAttentionDTI and MolTrans, with our own ColdSite-DTI
held to the same standard and DeepDTA as an accuracy anchor — across four levels of
distribution shift on DAVIS *[and KIBA — PENDING scope]*: random, unseen drug, unseen
target and both. Each model is retrained on identical splits with its authors' recipe.
We measure plausibility as precision@k against UniProt binding sites, reported beside
chance and the achievable ceiling, and faithfulness as the change in prediction when the
attended residues are masked, against a random-masking control. A uniform explanation
provides the floor, a positive control shows the pipeline recognises a genuinely good
explanation, significance is corrected across the whole grid, and a panel of 60 non-kinase
proteins tests whether any signal is more than recognition of the ATP pocket shared by
every kinase in training.

**¶6 — What we find.** *[RESULT: one or two sentences per axis, written only from the
grid. The dry run on our own model suggests the shape — attention that is load-bearing at
every level yet at best about twice chance at finding binding sites, with the most
accurate cold level at chance — but that was one seed of a regression model and must not
be quoted here.]*

**¶7 — Contributions.**
1. An audit, not a model: several published DTI models, one measurement suite, four levels
   of distribution shift, three seeds, family-wise error control.
2. Two axes, each against its own control: plausibility against a uniform floor and a
   validated positive control, faithfulness against random masking.
3. A kinase-confound control by transfer to non-kinase proteins, which the kinase-only
   benchmarks cannot provide.
4. Our own model audited on the same terms, with its results reported whether or not they
   flatter it.

*[Before submission: cut ¶3 if Related Work carries it in full; check every claim in ¶6
and ¶7 against the final numbers — any the grid does not support comes out.]*
