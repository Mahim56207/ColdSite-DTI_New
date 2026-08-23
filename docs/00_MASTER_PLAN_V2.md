# ColdSite-DTI — Master Plan v2 (Audit Reframe)

**Read this instead of `00_MASTER_PLAN.md` for everything from here on.** v1 is
kept for reference; the science question is unchanged, the framing is not.

Team: 124AD0008 · 124AD0015 · 124AD0067 · Supervisor: Dr. Chandra Mohan Dasari
Window: August – November 2026

---

## 1. What changed, and why

v1 framed the paper as: *we built ColdSite-DTI, and we measure whether its
explanations survive cold-start.*

A literature check found two problems with that framing.

**The closest prior work is closer than we thought.** CS-DTA (Frontiers in
Chemistry, 2026) already reports state-of-the-art performance across warm and
strict cold-start scenarios *and* runs interpretability analyses highlighting
localized protein regions with plausible binding relevance. Our differentiation
— that we plot fidelity as a continuous function of split difficulty rather than
reporting it once — is a presentation delta, not a conceptual one.

**"Explanations degrade under distribution shift" is an established 2025–26
finding elsewhere.** It has been shown for vision attribution (insertion and
deletion scores dropping up to 40% OOD), for GNN explanations, for sparse
autoencoders in mechanistic interpretability, and for recommender systems. A
reviewer will read a single-model DTI confirmation as one more domain
application.

**So the contribution moves from the model to the measurement.** New framing:

> **Do published interpretability claims in DTI prediction survive realistic
> evaluation?**

ColdSite-DTI stops being the contribution and becomes one subject among
several. DeepDTA, HyperAttentionDTI and MolTrans stop being comparison numbers
and become subjects too. The paper's claim becomes field-level rather than
model-level, which is both more useful and much harder to dismiss.

**Nothing anyone has already built is wasted.** The splits, ground truth,
metric, significance tests and model all carry over unchanged. The audit needs
the same infrastructure pointed at more models.

## 2. Three additions that come with the reframe

### 2.1 Faithfulness is now core, not optional

v1 made masking-based faithfulness an October stretch goal. That is backwards.
Attention-as-explanation has been contested since Jain & Wallace (2019) and
Serrano & Smith (2019) showed high attention weights do not necessarily
indicate high influence on predictions; Wiegreffe & Pinter (2019) pushed back,
and the dispute is still live. A 2026 paper measuring only whether attention
*looks* biologically sensible is measuring plausibility, and plausibility alone
will not clear a serious bar.

Plausibility and faithfulness are independent:

| | faithful | not faithful |
|---|---|---|
| **plausible** | the good case | **a convincing lie** |
| **not plausible** | honest oddity | noise |

precision@k measures the rows. `src/evaluation/faithfulness.py` measures the
columns. The top-right cell is the one a scientist cannot detect by eye, and it
is the reason this paper is worth writing.

### 2.2 The kinase confound must be controlled

DAVIS and KIBA are kinase panels. A "cold-target" kinase is not an unfamiliar
protein — it shares the ATP-binding pocket of the several hundred kinases
already in training. A model can score well on cold-target precision@k by
learning "ATP pockets look like this", which is exactly the generic
pattern-matching we claim to be testing against.

Uncontrolled, our headline ladder may be measuring kinase-family similarity. It
is the objection most likely to sink the paper.

The antiviral subset is the control arm: HIV-1 protease, HIV-1 RT, SARS-CoV-2
Mpro and RdRp, and influenza neuraminidase are all non-kinase. It stops being a
decorative case study and becomes load-bearing.

Current state — run `python -m src.evaluation.target_family`:

| | DAVIS | KIBA |
|---|---|---|
| targets | 409 | 224 |
| kinase | 230 | 0 (unclassifiable) |
| **non-kinase** | **0** | **0** |
| control usable? | **no** | **no** |

KIBA reports zero kinases only because it uses UniProt accessions rather than
gene symbols and cannot be classified without a mapping. Both gaps are Track A
work.

### 2.3 Seeds and multiple comparisons

The grid is models × splits × datasets. At 4 × 4 × 2 that is 32 significance
tests, of which roughly 1.6 clear p < 0.05 by chance. Uncorrected per-cell
p-values would let noise masquerade as the central claim.

- Three seeds minimum per cell; report mean ± std. `src/evaluation/aggregate.py`
  flags under-powered cells with `!`.
- Holm-Bonferroni across the whole family before anything is called significant.

## 3. What the paper now contains

- **Abstract & Intro** — the gap: interpretability claims are validated warm and
  deployed cold
- **Related Work** — the five required papers, plus the OOD-explainability
  literature and the attention-faithfulness dispute (Track C)
- **Methods** — data, splits, ground truth, family stratification (Track A);
  models under audit and the adapter contract (Track B); metrics — plausibility,
  faithfulness, significance (Track C)
- **Results** — the audit grid, the kinase-stratified control, the antiviral
  case study
- **Discussion** — which published claims survive, honestly including our own

## 4. Venue

Target **Bioinformatics**, **Briefings in Bioinformatics**, or **ISMB**. These
are where this work actually publishes — DMFF-DTA went to npj Digital Medicine,
CS-DTA to Frontiers in Chemistry, KANPM-DTA to Briefings.

We are explicitly **not** chasing an A\* ML conference. Realistic odds there are
a few percent, and the cost — many more seeds, many more models, a much harder
writing job — lands squarely on the August–November window that everyone needs
for placement season. If the October numbers turn out genuinely surprising, we
revisit that decision with data.

## 5. Timeline from here

| | |
|---|---|
| **Aug (rest)** | Finish Part 1 items in `STATUS.md`. Track A: antiviral rebuild + ground-truth re-fetch are now critical path, not nice-to-have. |
| **Sep** | Adapters for all baselines. First real ladder on one dataset, one seed, to shake out plumbing. |
| **Oct** | Full grid: models × splits × datasets × 3 seeds. Faithfulness. Stratified control. |
| **Nov** | Write. Draft by 15 Nov, last two weeks for polish. |

Tripwire unchanged: if any track has no real output by **15 September**, raise it
to the other two immediately.

## 6. What is already built

Everything in `src/evaluation/` runs today and is covered by the test suite:

| module | what it does |
|---|---|
| `precision_at_k.py` | plausibility, with ceiling reporting and tie-break control |
| `significance_test.py` | per-protein and split-level permutation tests |
| `faithfulness.py` | comprehensiveness, sufficiency, AOPC, **with random-masking controls** |
| `model_registry.py` | the adapter contract every audited model must satisfy |
| `aggregate.py` | seed aggregation, Holm-Bonferroni, the audit table |
| `target_family.py` | kinase / non-kinase stratification |
| `run_ladder.py` | end-to-end runner; `--dummy` works with no data |
| `src/data/ground_truth.py` | UniProt 1-indexed → metric 0-indexed bridge |

Your individual Part 2 guide is `docs/PART2_GUIDE_<your roll>.md`.
