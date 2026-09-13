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

## 1b. DAVIS's sequences: leakage and pseudo-variants *[decision pending]*

Found 2026-09-13 building the KLIFS ground truth; `results/sequence_audit_davis.md`
(`src/data/sequence_audit.py`). In DeepDTA's DAVIS `proteins.txt`, which this project
and most DTI papers use: (i) all 54 variant targets with a wild-type entry carry exactly
the wild-type sequence — ABL1(T315I), EGFR(T790M), BRAF(V600E) and the rest — so 442
targets are 379 distinct sequences; (ii) at cold-target 12 of 88 test targets (816 of
5,984 rows, 13.6%) and at cold-pair 11 of 88 (143 of 1,144, 12.5%) are unseen by name
but identical in sequence to a training target, and cold-pair validation is 13.6% such
rows; (iii) ten targets' sequences hold few or none of the 85 KLIFS pocket residues
(RET and its three mutants are RET's extracellular residues 1–430). KIBA has none of
the three. *[Pending: evaluate cold levels on sequence-unseen targets only, count each
distinct sequence once in the explanation metrics, exclude the ten pocketless targets
(option A) — or rebuild the DAVIS cold splits and retrain (option B).]*

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

## 4. ColdSite-DTI's attention is load-bearing, finds the pocket region, misses the annotated residues

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

**Against UniProt's annotated residues: at chance.** Mean precision@10 is 0.012–0.021 against a chance of
0.019–0.020 and a ceiling of 0.99. No level exceeds chance on average. The only cells
significant before correction are cold-drug seeds 1 and 2 (0.025, p = 0.022 each), which
seed 3 does not reproduce (0.013, p = 0.999). Read against the positive control (§3,
`positive_control --compare`), every cell is equivalent to an explanation that places at
most ~0.2% of the true sites first, a tenth of the smallest dose the test reliably detects
(2%); eight of the twelve sit at or below the dose-0 curve. This is a real null, not an
underpowered one.

**Against the KLIFS ATP pocket: about twice chance, mostly by finding the domain.**
UniProt's annotation is about a dozen residues per kinase. The same checkpoints scored
against KLIFS's structure-derived 85-residue ATP pocket (`src/data/klifs_pocket.py`;
placement checked against KLIFS's own residue numbers for 41 kinases, all consistent)
give:

**Table R3.** ColdSite-DTI precision@10 against the KLIFS pocket, seeds 1 / 2 / 3
(`results/positional_control_coldsite_dti_davis_klifs.md`).

| level | precision@10 | mean ± sd | chance | top-10 inside the pocket's span | span / chain |
|---|---|---|---|---|---|
| random | 0.175 / 0.208 / 0.236 | 0.21 ± 0.03 | 0.13 | 0.31–0.35 | 0.24 |
| cold-drug | 0.276 / 0.243 / 0.318 | 0.28 ± 0.04 | 0.13 | 0.37–0.44 | 0.24 |
| cold-target | 0.219 / 0.191 / 0.278 | 0.23 ± 0.04 | 0.13 | 0.32–0.41 | 0.24 |
| cold-pair | 0.300 / 0.259 / 0.240 | 0.27 ± 0.03 | 0.13 | 0.39–0.47 | 0.23 |

Every cell beats, at p = 0.001, both a map borrowed from another protein (position
alone) and the protein's own attention shuffled among residues of the same amino acid
(residue-type preference alone). So the attention does find the pocket region of each
kinase. How: 31–47% of its top ten residues fall inside the stretch the pocket spans
(the kinase domain core), which is 23–24% of the chain; shuffled within that stretch
it keeps most of its score, and beats the within-stretch shuffle in 9 of 12 cells
(p ≤ 0.014) by a modest margin (e.g. cold-drug seed 1: 0.276 vs 0.236). Most of the
pocket signal is knowing the domain; a smaller part is knowing the pocket inside it.

Taken together: the attention is causally used (every level), is **coarsely
plausible** — it concentrates on the kinase domain and its ATP pocket at about twice
chance, beyond position and amino-acid preference — and is **not finely plausible**:
it does not land on the residues UniProt annotates. "Is attention plausible?" has a
different answer at each ground-truth resolution, so both are reported. *[Formal
statement waits for the audit table (§5), Holm over the whole family, and for the
decision on DAVIS's sequence leakage (§1b): the cold-target and cold-pair numbers
here still include the 12 and 11 test proteins that are seen by sequence.]*

## 5. *[PENDING]* The audit table (all subjects, Holm over the whole family)

## 6. *[PENDING]* Kinase-family control (60 non-kinase proteins, cotransport ions excluded)

*[ColdSite-DTI only, 2026-09-13; `results/positional_control_coldsite_dti_davis.md`.]*
On the 60 non-kinase proteins, precision@10 against UniProt sites is above its own
chance (0.012) in 7 of 12 cells before correction (e.g. seed 2 random 0.050), while the
kinase arm sits at chance. It is **not positional**: maps borrowed from other proteins
score 0.012–0.014. It is **mostly amino-acid preference**: ColdSite-DTI's top-ten
attention is enriched in histidine (3× in seed 1, 11–15× in seed 2), non-kinase
binding sites are histidine-rich (8.8×; many are metal sites), kinase ATP sites are not
(enriched in G, D, K). Shuffling attention among residues of the same amino acid
recovers most of the non-kinase score (seed 2 random: 0.045 of 0.050, p = 0.28); three
cells keep a remainder (p = 0.004–0.048 before correction). The non-kinase "signal" is
the attention's liking for histidine meeting histidine-rich sites, not knowledge of
where the sites are — the same preference that lets it miss the glycine-rich kinase
sites. Seed 2, the strongest histidine preference, has the highest non-kinase scores.

## 7. *[PENDING]* KIBA

## 8. *[PENDING]* Antiviral case study
