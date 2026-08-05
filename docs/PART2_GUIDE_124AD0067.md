# Part 2 Guide — 124AD0067
## Track C: The Audit — Measurement, Control, and the Paper's Argument

Read `docs/00_MASTER_PLAN_V2.md` first. Under the reframe, your track *is* the
paper.

---

## What changed for you

v1 gave you the measurement for one model. The audit gives you the measurement
for the field, and the paper's argument to build with it. Concretely:

- You now run every model through the same pipeline, not just ColdSite-DTI
- Faithfulness joins plausibility as a headline axis (124AD0015 runs it, you
  interpret and write it)
- You own the kinase-confound control, which is the objection most likely to
  sink us
- Your Related Work section grew: it now has to cover the OOD-explainability
  literature and the attention-faithfulness dispute, not just five DTI papers

---

## Priority 1 — Rewrite the differentiation document (~3 days, do this week)

Your five-paragraph write-up is good but now insufficient. Two additions.

**Fix the CS-DTA paragraph.** It is closer to us than v1 admitted. CS-DTA
(Frontiers in Chemistry, 2026) reports state-of-the-art performance across warm
and strict cold-start scenarios *and* runs interpretability analyses
highlighting localized protein regions with plausible binding relevance. It also
includes non-kinase validation — the exact control we are only now building. Our
honest differentiation is: they report interpretability under cold-start for one
model; we measure it as a function of split difficulty across several published
models, with faithfulness and a family-stratified control. Write that, not
something more flattering.

**Add two new subsections.**

*OOD explainability.* "Explanations degrade under distribution shift" is
established outside DTI: vision attribution methods drop up to 40% on insertion
and deletion scores OOD; there is a GNN Explanation-Generalization Score built
on our exact premise; sparse-autoencoder faithfulness has been formalized as a
geometric "faithfulness gap"; CIRR does the recommender version. Our claim is
not that this phenomenon is new — it is that DTI's published interpretability
claims have not been checked against it. Say so directly. A reviewer who thinks
we are unaware of this literature will reject; one who sees we have positioned
against it will not.

*The attention-faithfulness dispute.* Jain & Wallace (2019) and Serrano & Smith
(2019) showed high attention weights do not necessarily indicate high influence;
Wiegreffe & Pinter (2019) argued attention can be faithful in certain cases.
This is why we measure both plausibility and faithfulness, and it is the single
best justification for the paper's design.

## Priority 2 — Own the confound control (~2 days)

```bash
python -m src.evaluation.target_family
```

Right now that reports **zero non-kinase targets in either dataset** and
`control_is_usable: False`. DAVIS is 230 kinases and 179 unclassifiable; KIBA is
entirely unclassifiable because it uses accessions rather than gene symbols.

You do not fix that — 124AD0008 does (their Priorities 1 and 3). You **own
whether it is sufficient**, and you write the section that interprets it.

Three outcomes, all publishable:

| result | what it means | how to write it |
|---|---|---|
| fidelity holds on kinases, collapses on non-kinases | the kinase number was family similarity | this IS the finding, and stronger than v1's |
| degrades similarly on both | degradation is real, not an artefact | the claim survives — lead with it |
| holds on both | honest negative result | report it; do not bury it |

What is not publishable is the unstratified ladder presented as if the confound
were absent. If 124AD0008 cannot get 20+ distinct non-kinase targets, your job
is to make sure the limitation is stated prominently in the Discussion rather
than quietly omitted.

## Priority 3 — Run the audit grid (~1 week, October)

```bash
python -m src.evaluation.run_ladder --dummy      # today, no data needed
```

Once checkpoints exist, the real path is the same code. Then aggregate:

```python
from src.evaluation.aggregate import aggregate_seeds, holm_bonferroni, audit_table

cell = aggregate_seeds([s1, s2, s3])       # mean ± std, flags <3 seeds
corrected = holm_bonferroni(all_p_values)  # across the WHOLE grid
print(audit_table(grid))
```

**Run Holm-Bonferroni over the entire family, once.** The grid is models ×
splits × datasets; at 4 × 4 × 2 that is 32 tests and roughly 1.6 chance hits at
p < 0.05. Per-cell uncorrected p-values would let noise carry the paper's
central claim. `aggregate.py` flags any cell built from fewer than three seeds
with `!` — never quote a flagged cell as an estimate.

## Priority 4 — The headline figure (~2 days)

The runner refuses to draw it without accuracy values, deliberately: fidelity
plotted alone is half the claim. The contribution is fidelity *against*
accuracy across the ladder.

The audit version needs more than v1's two lines:
- one fidelity line per audited model, with seed error bars
- accuracy on a second axis or panel
- the uniform-attention control as a floor line
- kinase vs non-kinase as separate panels

If the control line and a model's line overlap, that model has no explanatory
content whatever its accuracy. Say so.

## Priority 5 — Write Results and Discussion (~2 weeks, November)

You lead both. The argument, in order:

1. Accuracy degrades across the ladder — expected, confirms the setup works
2. Plausibility degrades — the v1 finding, now across several models
3. Faithfulness deltas — do the explanations *do* anything, and does that change
   with difficulty
4. The stratified control — is any of this just kinase similarity
5. Which published claims survive, ours included

Point 5 is the paper. Be specific and be fair — name the models, state what each
claimed, state what we measured. If ColdSite-DTI comes off worst, that is a
credibility asset, not an embarrassment.

## Definition of done

- [ ] Differentiation doc rewritten: honest CS-DTA paragraph, OOD-explainability
      subsection, attention-faithfulness subsection
- [ ] Confound control interpreted and written, or its absence stated as a
      limitation
- [ ] Audit grid computed with 3 seeds and Holm correction across the family
- [ ] Headline figure: multi-model, error bars, control floor, stratified panels
- [ ] Related Work, Results and Discussion drafted

## Common mistakes

- **Reporting raw precision@k across proteins with different site counts.** Six
  annotated sites at k=20 caps at 0.30. Report `mean_normalised` alongside.
- **Quoting per-protein p-values.** 400 proteins is 400 tests. Use the
  split-level `permutation_test_batch`.
- **Treating a boring result as failure.** Still true, and now more so — under
  the audit framing "most published interpretability claims hold up fine" is a
  real, useful, reportable finding.
- **Softening the CS-DTA paragraph.** A reviewer who has read it will notice,
  and it costs us more credibility than the differentiation gains.

## Where to get help

If you are unsure whether a result is significant after correction, bring the
raw p-values to the team rather than deciding alone. An honest null is worth
more than a claimed effect that a reviewer can dismantle.
