# Results (draft)

Drafted 2026-09-13 from the cells finished so far. Every number below is read from a
trained cell's `_results.json` (DAVIS binary grid, Kaggle account 1, commit v1 of
`kaggle_davis_binary_grid36.ipynb`) or from a file named beside it. *[PENDING]* marks
what waits for HyperAttentionDTI (10 cells training), MolTrans seeds 2–3 (8 cells to retrain), the
audit table and KIBA. All values are test-set means ± sample standard deviation over
three training seeds; a difference smaller than the spread is not reported as one.

---

## 1. Accuracy across the four split levels (DAVIS, binary)

**Table R1.** Test AUROC, mean ± sd over seeds 1–3. AUPRC is given with each test set's
positive rate, which is its chance level; AUPRC is not comparable across levels because
that rate differs.

| model | random | cold-target | cold-drug | cold-pair |
|---|---|---|---|---|
| DeepDTA (anchor) | 0.929 ± 0.002 | 0.907 ± 0.003 | 0.692 ± 0.044 | 0.728 ± 0.035 |
| ColdSite-DTI (ours) | 0.924 ± 0.001 | 0.857 ± 0.011 | 0.721 ± 0.008 | 0.624 ± 0.099 |
| HyperAttentionDTI | *[PENDING]* | 0.916 (seeds 1–2) *[PENDING s3]* | *[PENDING]* | *[PENDING]* |
| MolTrans | 0.922 (seed 1) | 0.868 (seed 1) | 0.668 (seed 1) | 0.590 (seed 1) |

| | random | cold-target | cold-drug | cold-pair |
|---|---|---|---|---|
| test pairs | 6,011 | 5,984 | 5,746 | 1,144 |
| test drugs / targets | 68 / 442 | 68 / 88 | 13 / 442 | 13 / 88 |
| positive rate (AUPRC chance) | 0.077 | 0.075 | 0.060 | 0.057 |
| DeepDTA AUPRC | 0.631 ± 0.013 | 0.615 ± 0.004 | 0.199 ± 0.045 | 0.206 ± 0.012 |
| ColdSite-DTI AUPRC | 0.612 ± 0.010 | 0.464 ± 0.047 | 0.201 ± 0.035 | 0.132 ± 0.028 |

**The levels are not a ladder of increasing difficulty.** On DAVIS an unseen target costs
little (DeepDTA 0.929 → 0.907) and an unseen drug costs a great deal (→ 0.692); cold-pair
is no harder than cold-drug for DeepDTA (0.728 vs 0.692, within the cold-drug spread).
The drug axis dominates: DAVIS has 68 drugs, so cold-drug trains on 49 and tests on 13,
while cold-target still trains on 310 targets and its 88 held-out targets are kinases
like them. Levels are therefore reported as categories, not as a severity scale.

**ColdSite-DTI is not the most accurate model, and does not need to be.** It matches
DeepDTA on random (0.924 vs 0.929) and cold-drug (0.721 ± 0.008 vs 0.692 ± 0.044, inside
DeepDTA's spread), and falls below it on cold-target (0.857 vs 0.907, a gap five times
either spread) and cold-pair (0.624 vs 0.728). The audit asks whether each model's
explanation survives, not which model predicts best; accuracy is reported so that every
explanation result can be read against how well the same checkpoint predicts.

**Cold-pair is the least stable cell.** ColdSite-DTI's three cold-pair seeds score 0.738,
0.557 and 0.577, and all three checkpoints come from epoch 11, one epoch after the
selection floor: validation loss on cold-pair's 264 validation pairs was lowest almost
immediately. Random ran long by comparison (best epochs 40, 40, 32). Every cold-pair
result below is therefore quoted with its spread, never from one seed.

*[PENDING: HyperAttentionDTI and MolTrans rows; one sentence on whether the published
models keep their accuracy at cold-target and cold-pair.]*

*MolTrans: seed 1 only. The first grid's seeds 2 and 3 trained as seed 1 (the vendored
`models.py` reseeds torch on import; fixed 2026-09-13) and are being retrained; until
then MolTrans has no spread and no claim rests on it.*

## 2. The cold-pair drop is mostly task, not volume

Cold-pair trains on 15,190 rows against random's 21,039. Retraining ColdSite-DTI on
random with its training set cut to 15,190 rows (three seeds, each its own subsample;
full random validation and test sets) gives AUROC **0.887 ± 0.018**
(`results/volume_control_davis.md`):

| ColdSite-DTI cell | training rows | AUROC |
|---|---|---|
| random, full | 21,039 | 0.924 ± 0.001 |
| random, volume-matched | 15,190 | 0.887 ± 0.018 |
| cold-pair | 15,190 | 0.624 ± 0.099 |

Fewer rows cost 0.037; the remaining 0.263 is cold-pair itself — about 12% and 88% of
the 0.300 drop. The second term includes everything that differs between the two test
sets (cold-pair's own 1,144 pairs of unseen drugs and unseen targets), which is what
cold-pair difficulty means here. The control was run for ColdSite-DTI only.

## 3. The plausibility metric can see a real signal

Before any model is scored, the metric is scored against explanations of known quality
(`results/positive_control_davis.md`): synthetic attention that places a fraction *d*
of the true binding sites first. Precision@10 rises with *d* at every level, the oracle
(*d* = 1) reaches the ceiling (0.990–0.992), and a dose of **2% of the sites is detected
as significant at all four levels**, including cold-target and cold-pair with 79 proteins
each. A model whose precision@10 sits at chance is therefore a real null, not a test too
weak to see anything. On a planted model whose prediction depends only on the annotated
sites, the faithfulness measure separates the oracle (comprehensiveness delta ≈ +10.2)
from no signal (≈ 0) at every level.

## 4. ColdSite-DTI's attention is load-bearing but does not mark binding sites

Computed 2026-09-13 on the 12 binary checkpoints (`run_all`, CPU; outputs in
`results/analysis_davis_partial/`, not committed). Faithfulness uses up to 200 test pairs
per level; precision@10 scores one test pair per protein (402 proteins at random and
cold-drug, 79 at cold-target and cold-pair) against a 1,000-trial permutation null.

**Table R2.** ColdSite-DTI, DAVIS binary, seeds 1 / 2 / 3.

| level | faithfulness delta | mean ± sd | precision@10 | mean ± sd | chance | ceiling |
|---|---|---|---|---|---|---|
| random | 0.771 / 2.009 / 0.784 | 1.19 ± 0.71 | 0.022 / 0.009 / 0.012 | 0.015 ± 0.006 | 0.020 | 0.990 |
| cold-drug | 1.825 / 0.968 / 0.798 | 1.20 ± 0.55 | 0.025 / 0.025 / 0.013 | 0.021 ± 0.006 | 0.020 | 0.990 |
| cold-target | 0.240 / 0.946 / 0.563 | 0.58 ± 0.35 | 0.014 / 0.019 / 0.019 | 0.017 ± 0.003 | 0.019 | 0.992 |
| cold-pair | 0.639 / 0.974 / 0.450 | 0.69 ± 0.27 | 0.018 / 0.011 / 0.008 | 0.012 ± 0.005 | 0.019 | 0.991 |

Faithfulness delta = comprehensiveness (absolute change in the predicted logit when the 10
most-attended residues are masked) minus the same for 10 random residues; only the delta
is a result.

**Faithful at every level.** Masking the residues ColdSite-DTI attends to moves its
prediction more than masking random residues in all 12 cells (every delta positive,
flagged load-bearing in each seed's report). The attention is not decoration: the model
uses the residues it points at, at every split level.

**Not plausible at any level.** Mean precision@10 is 0.012–0.021 against a chance of
0.019–0.020 and a ceiling of 0.99. No level exceeds chance on average. The only cells
significant before correction are cold-drug seeds 1 and 2 (0.025, p = 0.022 each), which
seed 3 does not reproduce (0.013, p = 0.999). Read against the positive control (§3,
`positive_control --compare`), every cell is equivalent to an explanation that places at
most ~0.2% of the true sites first, a tenth of the smallest dose the test reliably detects
(2%); eight of the twelve sit at or below the dose-0 curve. This is a real null, not an
underpowered one.

Taken together: the attention is causally used, and what it is used for is not the
annotated binding site. This is the dry run's "faithful but not plausible" pattern,
now on the binary checkpoints, three seeds and one pair per protein, and stronger: the
dry run's warm level reached ~2× chance, here no level does. *[Formal statement waits
for the audit table (§5): Holm over the whole family.]*

## 5. *[PENDING]* The audit table (all subjects, Holm over the whole family)

## 6. *[PENDING]* Kinase-family control (60 non-kinase proteins, cotransport ions excluded)

*[First look, ColdSite-DTI only, not yet a result (2026-09-13): on the 60 non-kinase
proteins precision@10 is above its own chance (0.012) in 7 of 12 cells before
correction — e.g. seed 2 random 0.050, cold-pair 0.048 — while the kinase arm sits at
chance. It is not consistent across seeds (seed 1 random 0.008) and the non-kinase
ceiling is 0.50, not 0.99. Before this is interpreted, check whether it is a positional
effect (attention concentrated at one end of the sequence meeting sites clustered
there), which a uniform permutation null would not remove.]*

## 7. *[PENDING]* KIBA

## 8. *[PENDING]* Antiviral case study
