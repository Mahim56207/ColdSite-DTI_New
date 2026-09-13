# Discussion (draft skeleton)

Drafted 2026-09-13. Sections marked *[PENDING]* wait for the full DAVIS audit
(HyperAttentionDTI and MolTrans analysed under the same pipeline, Holm over the whole
family) and for KIBA. Nothing here may state a result for a model before its numbers are
in `paper/results.md`. Limitations are in `paper/limitations.md`.

---

## 1. What the audit found (one paragraph, written last)

*[PENDING — two or three sentences: which published claims survive at which level of
distribution shift, at which ground-truth resolution; the one-line verdict per model.]*

## 2. Attention that is used but does not point at the binding residues

For ColdSite-DTI the three measurements come apart. Its attention is **faithful**: at every
level, masking the ten residues it attends to moves the prediction more than masking ten
random residues. It is **coarsely plausible**: against the 85-residue KLIFS ATP pocket it
scores about twice chance, beyond what position or amino-acid preference explain, mostly
because it concentrates on the kinase domain. And it is **not finely plausible**: against
the dozen residues UniProt annotates it sits at chance, a real null by the positive
control. The attention therefore carries information the model uses and knows roughly
where a kinase binds its ligands, without singling out the residues that do the binding.

Two things it demonstrably does use: the kinase domain as a region, and histidine (its
top-ten attention is 3–15× enriched in histidine), a preference that produces
above-chance "hits" on histidine-rich non-kinase metal sites and misses the glycine-rich
kinase ATP site. A reader who inspected attention maps would see highlights near the
pocket and could take them as residue-level explanations; the audit says they are not.

*[PENDING: do HyperAttentionDTI and MolTrans show the same dissociation? If the published
models are finely plausible where ColdSite-DTI is not, the finding is model-specific; if
all three are coarse-only, it is a property of attention-based DTI explanations trained
on affinity labels alone.]*

## 3. Plausibility depends on the resolution of the ground truth

"Does attention mark the binding site?" has no single answer: the same checkpoints are at
chance at residue resolution and above chance at pocket resolution. Claims in the
literature are usually made at residue resolution — a highlighted residue is shown beside
a crystal-structure contact — and should be tested there; a pocket-level agreement is a
weaker statement that a protein-sequence model can meet by learning where the kinase
domain is. We recommend that plausibility be reported at both resolutions, each beside
its chance level and ceiling.

## 4. Does explanation quality degrade with distribution shift? *[PENDING]*

*[PENDING — from the audit table: plausibility and faithfulness per level for every
model. Read against accuracy: DAVIS's levels are categories, not a severity ladder
(unseen drugs cost far more accuracy than unseen targets), so the question is per level,
not a trend. Note cold-target accuracy was inflated ~0.02 by sequence leakage.]*

## 5. What an attention-explanation claim should be tested against

The controls this audit needed, each of which changed or protected a conclusion:

- **A positive control for the metric** (explanations of known quality): without it a
  result at chance cannot be told from an underpowered test. Here a 2% dose of true sites
  is detectable at every level, so ColdSite-DTI's null is a real null.
- **A masking control for faithfulness**: masking anything moves a prediction; only the
  excess over random masking is evidence.
- **Position and residue-type nulls for plausibility**: a uniform null is fooled by an
  attention map with a positional or amino-acid habit. The non-kinase "signal" survived the
  positional null and not the residue-type null.
- **One protein per distinct sequence**: averaging per name or per pair counts one protein
  many times; DAVIS's 442 names are 379 sequences.
- **Seeds and spreads**: single seeds on the cold levels move by more than most reported
  differences (ColdSite-DTI cold-pair: 0.51–0.75 AUROC across seeds on unseen proteins).

## 6. Benchmark hygiene

DeepDTA's DAVIS file gives every mutant the wild-type sequence, and cold-target and
cold-pair splits by target name therefore test 12–14% of rows on proteins seen in
training; ten targets' sequences lack the kinase domain the drugs bind. Any cold split of this file by target
name carries the leak, and no mutant-specific claim can be tested on it.
We recommend splitting and deduplicating by sequence, reporting the sequence audit with
the split, and — for mutants — using sequences that contain the mutation. KIBA, keyed by
UniProt accession, has none of these problems.

## 7. Conclusion *[PENDING]*
