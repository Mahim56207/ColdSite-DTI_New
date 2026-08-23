# Related Work and Differentiation

Backbone for the paper's Related Work section, under the **audit** framing of
`docs/00_MASTER_PLAN_V2.md`. The question is no longer "how is ColdSite-DTI
different from these models" — ColdSite-DTI is one subject among several. The
question is:

> **Do published interpretability claims in DTI prediction survive realistic
> evaluation?**

Three bodies of work bear on that: the DTI papers that make interpretability
claims (§1), the evidence from outside DTI that explanations degrade under
distribution shift (§2), and the unresolved dispute about whether attention is
an explanation at all (§3). §4 states what is left for us to claim once all
three are taken seriously.

---

## 1. Interpretability claims in DTI prediction

### 1.1 DMFF-DTA (2025)

Evaluates internal attention weights against known biological binding sites
with statistical testing rather than by eye — the closest methodological
ancestor of our plausibility metric. Its entire pipeline runs on standard
random splits. **What we add:** the same class of validation carried across
four levels of split difficulty, so the question becomes how that validation
behaves when the drug or protein is unfamiliar, not whether it passes once.

### 1.2 EviDTI (2025)

Computes a binding-site hit ratio *and* evaluates cold-start prediction — but
as two independent experiments. Interpretive fidelity is never expressed as a
function of generalization difficulty. **What we add:** the connection. The
two axes are measured on the same splits, the same proteins and the same
checkpoints, so a change in one can be read against the other.

### 1.3 ColdDTI (2025)

Strong predictive accuracy under cold-start using multi-level protein
structural representations, with no assessment of explanation quality. It
establishes that cold-start accuracy is achievable; it does not ask whether
the model reaches those predictions for defensible reasons. **What we add:**
that question, on the same difficulty spectrum.

### 1.4 CS-DTA (2026, *Frontiers in Chemistry*) — the closest prior work

**This paragraph is deliberately unflattering, and must stay that way.** A
reviewer who has read CS-DTA will notice any softening, and it costs more
credibility than the differentiation gains.

CS-DTA reports state-of-the-art performance across warm and strict cold-start
scenarios, runs interpretability analyses highlighting localized protein
regions with plausible binding relevance, **and includes non-kinase
validation** — the control arm we are only now constructing. On the surface,
it has already done what v1 of this project proposed to do.

The honest differentiation is narrow and specific:

| | CS-DTA | this work |
|---|---|---|
| models examined | one (its own) | several published models, ours included |
| interpretability reported | as a property demonstrated | as a quantity measured across difficulty |
| faithfulness | not measured | comprehensiveness / sufficiency against random-masking controls |
| explanation floor | none | uniform-attention control |
| multiple comparisons | not applicable | Holm–Bonferroni across the whole grid |
| family control | non-kinase validation included | non-kinase **transfer** panel, 60 targets |

So: they report interpretability under cold-start for one model; we measure it
as a function of split difficulty across several published models, with
faithfulness, a control floor, and family-wise error control. That is a real
difference, and it is a difference of *method*, not of claim. It is not a
claim to have discovered that cold-start hurts interpretability.

### 1.5 GPS-DTI (2025)

Cold-start performance from protein language models (ESM-2) plus attention.
Evaluates predictive capability without asking whether explanation fidelity
tracks it. **What we add:** the audit treats "high accuracy" and "attention
points at the right residues" as separate claims requiring separate evidence —
which is precisely the pairing GPS-DTI leaves untested.

### 1.6 KANPM-DTA (*Briefings in Bioinformatics*)

Named in the master plan as venue evidence rather than as a differentiation
target. Relevant here mainly for calibrating where this work publishes.

---

## 2. Explanations degrade under distribution shift — established outside DTI

This is the section that keeps a reviewer from dismissing the paper as naive.
"Explanations get worse out of distribution" is **not a new finding**, and
claiming it as one would be the fastest possible rejection. It is established
in at least four separate literatures:

- **Vision attribution.** Insertion and deletion scores for saliency methods
  drop substantially — reported up to ~40% — when evaluation moves
  out-of-distribution.
- **Graph neural networks.** An Explanation-Generalization Score has been
  proposed on exactly our premise: that an explainer validated in-distribution
  need not remain valid outside it.
- **Mechanistic interpretability.** Sparse-autoencoder faithfulness has been
  formalized as a geometric "faithfulness gap", making the degradation
  measurable rather than anecdotal.
- **Recommender systems.** CIRR addresses the same phenomenon for
  recommendation explanations.

**Our position against this literature.** We do not claim the phenomenon is
new. We claim that DTI's published interpretability results have not been
checked against it — that a field making biological claims from attention maps
has validated those claims almost exclusively on random splits, while
deploying the models in precisely the cold-start regime where the phenomenon
is known to bite. State this directly in the Intro. A reviewer who believes we
are unaware of this work will reject; one who sees we have positioned against
it deliberately will not.

> **Citation status.** The four claims above are recorded in
> `docs/00_MASTER_PLAN_V2.md` §1 and are stated here as the project has
> established them. Authors, venues and DOIs must be filled in before
> submission — including the specific source for the ~40% figure, which is the
> only quantitative claim in this section and the one most likely to be
> checked. Do not submit with this note still present.

---

## 3. Is attention an explanation at all?

The dispute is live and unresolved, and it is the single best justification
for this paper's design:

- **Jain & Wallace (2019), *Attention is not Explanation*** — attention
  weights correlate poorly with gradient-based importance, and adversarial
  attention distributions can leave predictions unchanged.
- **Serrano & Smith (2019), *Is Attention Interpretable?*** — erasing
  high-attention components often fails to change the decision, so high
  attention does not establish high influence.
- **Wiegreffe & Pinter (2019), *Attention is not not Explanation*** — pushes
  back: the claim depends on what "explanation" is taken to mean, and under
  reasonable definitions attention can be faithful.

**Why this determines our methodology.** Plausibility and faithfulness are
independent properties:

| | faithful | not faithful |
|---|---|---|
| **plausible** | the good case | **a convincing lie** |
| **not plausible** | honest oddity | noise |

`precision@k` against UniProt binding sites measures the rows — does the
explanation *look* biologically sensible. `src/evaluation/faithfulness.py`
measures the columns — does masking the highlighted residues actually change
the prediction, against a random-masking control.

The top-right cell is the one a domain expert cannot detect by eye: an
explanation that lands on real binding sites while contributing nothing to the
prediction. A 2026 paper reporting only that attention looks biologically
sensible is measuring plausibility alone, and plausibility alone does not
clear a serious bar. Measuring both is the reason this paper is worth writing.

---

## 4. What we can honestly claim

1. **A measurement, not a model.** The contribution is the audit: several
   published DTI models, one metric suite, four difficulty levels, three
   seeds, family-wise error control.
2. **Two axes, not one.** Plausibility and faithfulness, each against its own
   control — the uniform-attention floor and random masking respectively.
3. **A family control that DAVIS and KIBA cannot supply.** Both datasets are
   essentially pure kinase panels (measured: DAVIS 429 kinase / 0 non-kinase;
   KIBA 227 / 0). The control is therefore a **transfer** panel of 60
   non-kinase BindingDB proteins — a harder condition than the datasets' own
   cold-target level, and it must be described as transfer rather than
   stratification.
4. **Our own model audited on the same terms.** If ColdSite-DTI comes off
   worst, that is a credibility asset and it gets reported without softening.

What we must **not** claim: that explanation degradation under distribution
shift is a new phenomenon (§2), or that measuring interpretability under
cold-start is unprecedented in DTI (§1.4).

---

## Before submission

- [ ] Fill in authors, venues and DOIs for every §2 claim; delete the citation
      note once done
- [ ] Confirm the CS-DTA description against the published paper, not against
      this summary
- [ ] Re-check whether anything newer than CS-DTA has appeared — this document
      was last revised 2026-08-23
- [ ] Cross-check §4's claims against what the finished grid actually shows;
      any claim the numbers do not support comes out
