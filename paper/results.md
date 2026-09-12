# Results (draft)

Drafted 2026-09-13 from the cells finished so far. Every number below is read from a
trained cell's `_results.json` (DAVIS binary grid, Kaggle account 1, commit v1 of
`kaggle_davis_binary_grid36.ipynb`) or from a file named beside it. *[PENDING]* marks
what waits for HyperAttentionDTI (10 cells training), MolTrans (1 cell training), the
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
| MolTrans | *[PENDING]* | *[PENDING]* | *[PENDING]* | *[PENDING]* |

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

## 4. Does ColdSite-DTI's attention mark binding sites? *[PENDING — running]*

*[PENDING: faithfulness (delta over random masking) and precision@10 vs chance and
ceiling per level, 3 seeds, from `run_all` on the 12 binary checkpoints (running
2026-09-13, `results/analysis_davis_partial/`). Read each precision@10 against §3's dose
curve via `positive_control --compare`. The dry run (regression, seed 1) found attention
faithful at every level but at most ~2× chance at hitting sites; check whether the
binary checkpoints agree.]*

## 5. *[PENDING]* The audit table (all subjects, Holm over the whole family)

## 6. *[PENDING]* Kinase-family control (60 non-kinase proteins, cotransport ions excluded)

## 7. *[PENDING]* KIBA

## 8. *[PENDING]* Antiviral case study
