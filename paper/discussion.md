# Discussion (draft)

Drafted 2026-09-13, rewritten 2026-09-14 once the DAVIS audit was complete (three models,
four levels, three seeds, Holm over all sixteen cells). Every number here is in
`paper/results.md` with its source file; nothing states a result for a model whose numbers
are not there. *[PENDING]* now marks only what KIBA has to settle. Limitations are in
`paper/limitations.md`.

---

## 1. What the audit found

Of sixteen cells — three attention-based models and a uniform-attention control, at four
levels of distribution shift — **one supports the residue-level claim after correction**:
HyperAttentionDTI on the random split, at 1.7× chance (precision@10 0.034 against 0.020,
p = 0.0020 against a Holm threshold of 0.0031). It beats the nulls that could explain it
away: a map borrowed from another protein and attention permuted among residues of the same
amino acid in every seed (both 0.021, p = 0.001), and attention permuted within the
site-spanning stretch in two seeds of three. On the split that published work reports, for
the model that generalises best, the interpretability claim holds — and that is the only
place it holds.

The same model is at 1.26× chance on unseen targets and 1.18× on unseen pairs, neither
distinguishable from chance. MolTrans sits at the metric's floor at every level: its four
cells (0.020–0.028) are within one standard deviation of what an attention map of *equal
weight everywhere* scores (0.017–0.020). ColdSite-DTI, ours, is at chance at all four
levels (0.013–0.022 against 0.019–0.020) and survives no cell. So the audit's answer to its
own question is: **the residue-level
interpretability claim survives only on the easiest split, for one of the three models,
and nowhere under the distribution shift that deployment implies.**

Three further findings turned out to matter more than that verdict, and all three are
about measurement rather than about these models. Which residues an attention map
highlights depends mostly on an undocumented reduction choice (§3b). A masking-based
faithfulness test can invert its own conclusion when the model reads sub-words (§3b). And
the gradient of the same checkpoint finds the pocket about twice as well as the attention
does (§4) — so where attention fails, it is often the *report* that fails rather than the
model.

## 2. Attention that is used but does not point at the binding residues

All three models show the same dissociation, which answers the question this section was
left open on: it is not a property of our model.

For **ColdSite-DTI** the three measurements come apart cleanly. Its attention is
**faithful** — at every level, masking the ten residues it attends to moves the prediction
more than masking ten random ones (12/12 cells). It is **coarsely plausible**: against the
85-residue KLIFS ATP pocket it scores about twice chance, beyond what position or
amino-acid preference explain, mostly because it concentrates on the kinase domain. And it
is **not finely plausible**: against the dozen residues UniProt annotates it sits at chance
at all four levels (0.013–0.022 against 0.019–0.020, 7 of its 12 cells at or below chance).
The positive control says this is a real null rather than an underpowered test: a 2% dose of
true sites is detectable at every level, on these very protein sets, while its cells are
worth an equivalent dose of 0.006 or less.

**HyperAttentionDTI** is the same shape with one extra step: finely plausible at random
(§1), coarse-only once the split is cold. **MolTrans** has the shape without the content —
faithful in token space at every level (§5b of Results), and at the floor against both
ground truths. So the pattern across three models is: *attention is used, and it marks a
region rather than a site.*

Two things ColdSite-DTI demonstrably uses: the kinase domain as a region, and histidine —
its top-ten attention is 3–15× enriched in histidine, a preference that produces
above-chance "hits" on histidine-rich non-kinase metal sites (§6 of Results) and *misses*
the glycine-rich kinase ATP site it trained on. A reader inspecting its attention maps
would see highlights near the pocket and could take them as residue-level explanations.
They are not.

Two independent measurements then show the drug plays no part in these explanations. Given
a drug's own crystallographic contacts versus another drug's contacts in the same pocket,
the correct drug buys at most +0.011 precision@10, and in two of nine cells the *wrong*
drug scores higher. And ColdSite-DTI's protein-tower self-attention — computed by its
forward pass, discarded, and independent of the drug by construction — scores *higher*
against the pocket than its drug-conditioned cross-attention (0.238 against 0.219 at
random; 0.268 against 0.243 at cold-target). An explanation that does not change with the
drug is not explaining a drug–target interaction.

## 3. Plausibility depends on the resolution of the ground truth

"Does attention mark the binding site?" has no single answer: the same checkpoints are at
chance at residue resolution and above chance at pocket resolution. Claims in the
literature are usually made at residue resolution — a highlighted residue shown beside a
crystal-structure contact — and should be tested there. A pocket-level agreement is a
weaker statement that a protein-sequence model can satisfy by learning where the kinase
domain is.

This audit used three resolutions, and they disagree in an informative order: ~85 pocket
residues (above chance), ~19 residues contacted by the specific drug (at chance once the
pocket is controlled for), and the ~12 residues UniProt annotates (at chance). Plausibility
should be reported at every resolution a paper's claim spans, each beside its own chance
level and ceiling, because the chance level itself moves with the resolution — 0.143 for
the pocket, 0.025 for drug contacts, 0.020 for annotations.

## 3b. What the measurement depends on

Two results here are about the instrument, and both would have produced a wrong published
claim if we had not checked.

**The residues a readout points at are mostly a property of the readout.** Between the
tensor inside a network and one weight per residue, somebody chooses which axis to reduce,
which layer to read, and how to spread a convolution position or sub-word token over
residues. Scoring the same checkpoints through readouts another author could reasonably
have chosen, the top-ten residues overlap the published readout's by **2–12%** (one
exception at 90%). One choice moves HyperAttentionDTI's pocket agreement at cold-target
from 0.081 — *below* the 0.143 chance level — to 0.367, against 0.186 as published. A
published attention figure is, to that extent, a picture of a reduction choice. The
audit's own verdicts do survive this: every readout of every model stays at chance against
annotated residues (0.010–0.057 against 0.020, all inside the seed spread bar one cell),
and none lifts MolTrans above the pocket's chance level (0.120–0.189 against 0.143). What
the choice changes is the size of the coarse signal, not the existence of the fine one.

**A masking-based faithfulness test can invert its own conclusion.** Faithfulness
subtracts a random-masking control from the explanation's comprehensiveness, which assumes
both arms change the input by the same amount. For a model reading sub-word tokens they do
not: masking MolTrans's ten most-attended residues changes 48% of its tokens, and ten
random residues change 95% — for one protein, 0.8% against 99%. Under that test its
faithfulness delta was negative in 11 of 12 cells, which reads as "its attention points at
residues that matter less than arbitrary ones". Measured in the space the model actually
reads, with both arms removing the same number of tokens, the sign reverses in all twelve.
The lesson generalises beyond this model: **an intervention-based explanation metric is
only interpretable when the intervention is the same size in both arms**, and for
tokenised inputs that is not automatic.

## 4. Does explanation quality degrade with distribution shift?

Yes, but as a step rather than a slope, and DAVIS's levels are categories rather than a
severity ladder — an unseen drug costs far more accuracy than an unseen target, and
cold-pair is no harder than cold-drug for the accuracy anchor. The question is therefore
per level.

The fine-grained signal is what degrades. HyperAttentionDTI's residue-level agreement goes
1.7× chance → 1.26× → 1.18× from random to cold-target to cold-pair, crossing from
"survives Holm over sixteen cells" to "not distinguishable from chance". Faithfulness
degrades the same way without vanishing: its margin over random masking falls from
0.184 ± 0.056 at random to 0.056 ± 0.008 at cold-pair. The attention is still load-bearing
under shift; it is simply load-bearing for something that no longer coincides with the
annotated site.

The coarse signal is more robust — pocket-level agreement stays at 1.3–1.4× chance across
the cold levels for HyperAttentionDTI and ~2× for ColdSite-DTI — which is consistent with
the region, not the site, being what these models learned.

**And the degradation is a reporting failure more than an ignorance failure.** Integrated
gradients on the same HyperAttentionDTI checkpoints roughly double pocket agreement at
every level and reach 3.8× chance at cold-target, exactly where the attention is at 1.3×
and fails correction. The information is in the weights; the attention head does not
report it. That is a more useful conclusion for practice than "attention does not work":
it says an attention map is a lossy summary of what a model uses, and is lossiest where
interpretability matters most. *[PENDING: whether ColdSite-DTI and MolTrans show the same
gap. A first look put ColdSite-DTI's gradient at 0.000 against annotated residues, so the
effect may be specific to the model with the strongest residue-level signal.]*

Two accuracy caveats belong beside all of this. Cold-target accuracy was inflated for
every model by sequence leakage — 0.021–0.023 for three models and 0.041 for MolTrans —
so part of the "cold" difficulty in published DAVIS work is not difficulty at all. And
MolTrans at cold-pair predicts at chance on unseen proteins (AUROC 0.530 ± 0.024), so its
explanation scores there describe an explanation of nothing.

## 5. What an attention-explanation claim should be tested against

The controls this audit needed. Each one changed or protected a conclusion, and the last
three did not exist in our plan until a number forced them.

- **A positive control for the metric** (explanations of known quality): without it a
  result at chance cannot be told from an underpowered test. A 2% dose of true sites is
  detectable at every level here, so the nulls are real nulls.
- **A masking control for faithfulness**: masking anything moves a prediction; only the
  excess over random masking is evidence.
- **Position and residue-type nulls for plausibility**: a uniform null is fooled by an
  attention map with a positional or amino-acid habit. The non-kinase "signal" survived
  the positional null and not the residue-type null.
- **An intervention matched in size, not in units** (§3b): for tokenised inputs, k
  residues is not a fixed-size intervention, and comparing unequal interventions inverted
  a conclusion.
- **A control that changes only the thing being claimed**: for a per-drug claim, the same
  protein and the same number of sites with a *different drug's* contacts. Without it,
  pocket-finding reads as drug-specific binding-site recovery.
- **More than one defensible readout** (§3b): a verdict that depends on the reduction
  choice is a property of the reduction.
- **A second explanation method** (§4): it separates "the attention is a poor report" from
  "the model does not know", which no attention measurement can do.
- **One protein per distinct sequence**: averaging per name or per pair counts one protein
  many times; DAVIS's 442 names are 379 sequences.
- **Seeds, spreads, and intervals over proteins**: single seeds on the cold levels move by
  more than most reported differences (ColdSite-DTI cold-pair: 0.56–0.74 AUROC across
  seeds), and a cell resting on three scorable pairs carries an interval a third of the
  scale wide.

## 6. Benchmark hygiene

**DAVIS leaks, and the leak is worth about half of its cold-target difficulty.** Every
mutant in DeepDTA's DAVIS file carries the wild-type sequence, so 442 targets are 379
distinct sequences and cold splits by target *name* test 12–14% of rows on proteins seen
in training. Re-scoring on sequence-unseen targets says leakage inflated cold-target by
0.021–0.041 depending on the model; retraining on sequence-clean splits — with a control
that keeps the leak at matched row count and class balance — puts the number at **0.019 of
the 0.038 total drop**, the other half being the smaller training set. Two methods with
nothing in common agree. Ten targets' sequences also lack the kinase domain the drugs
bind, and no mutant-specific claim can be tested on this file at all.

**The family confound cannot be tested on these benchmarks.** DAVIS's test set contains
3,307 kinase rows and zero non-kinase; KIBA is 229 kinases. A stratified
kinase-versus-non-kinase comparison inside a cell is therefore impossible at any panel
size, and an external panel — 60 non-kinase proteins no model here has seen — is the
substitute, with the accompanying loss of control over drugs and protein length.

**The drug axis is the thinnest part of the design.** DAVIS holds out 13 drugs at
cold-drug; KIBA holds out 422. A cold-drug result on 13 compounds is close to anecdote,
and this is where KIBA does not merely replicate but repairs.

We therefore recommend: split and deduplicate by sequence and report the sequence audit
beside the split; use sequences that contain the mutation for any mutant claim; state the
family composition of the benchmark when a family confound is possible; and prefer
benchmarks whose held-out drug set is large enough to support a cold-drug claim.

## 7. Conclusion

We audited three attention-based DTI models under four levels of distribution shift, with
one multiplicity correction across the whole family, three ground truths at different
resolutions, a positive control for the metric, and nulls for position, amino-acid
preference and drug identity.

One of sixteen cells supports the residue-level interpretability claim: the
best-generalising model, on the random split, at 1.7× chance. Under distribution shift no
model's attention marks annotated residues better than chance, none distinguishes the
drug's own crystallographic contacts from another drug's in the same pocket, and one
model's attention is indistinguishable from a uniform map everywhere. What survives at
every level is coarser: attention that is load-bearing, and that concentrates on the
right region.

The measurement lessons may outlast the verdict. The residues an attention map highlights
depend mostly on an unreported reduction choice; a masking-based faithfulness test can
invert its own sign when the intervention is not size-matched; and the gradient of the same
checkpoint recovers the pocket about twice as well as the attention, so an attention map
understates what the model uses — most of all under the shift where interpretability is
supposed to earn its keep. Claims of the form "our attention identifies binding sites"
should be tested at the resolution they are made at, against nulls that can explain them
away, under the shift the model will meet, and with more than one way of reading the
attention out.

*[PENDING: one paragraph on KIBA once plan A finishes — whether the single surviving cell
replicates on a second dataset, and whether the cold-drug result changes when 422 drugs are
held out instead of 13.]*
