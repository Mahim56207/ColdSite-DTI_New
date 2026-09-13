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
| HyperAttentionDTI | 0.937 ± 0.005 | 0.915 ± 0.001 | *[PENDING]* | 0.694 ± 0.038 |
| MolTrans | 0.923 ± 0.002 | 0.874 ± 0.005 | 0.685 ± 0.020 | 0.569 ± 0.021 |

| | random | cold-target | cold-drug | cold-pair |
|---|---|---|---|---|
| test pairs | 6,011 | 5,984 | 5,746 | 1,144 |
| test drugs / targets | 68 / 442 | 68 / 88 | 13 / 442 | 13 / 88 |
| positive rate (AUPRC chance) | 0.077 | 0.075 | 0.060 | 0.057 |
| DeepDTA AUPRC | 0.631 ± 0.013 | 0.615 ± 0.004 | 0.199 ± 0.045 | 0.206 ± 0.012 |
| ColdSite-DTI AUPRC | 0.612 ± 0.010 | 0.464 ± 0.047 | 0.201 ± 0.035 | 0.132 ± 0.028 |
| HyperAttentionDTI AUPRC | 0.669 ± 0.021 | 0.644 ± 0.014 | *[PENDING]* | 0.187 ± 0.028 |
| MolTrans AUPRC | 0.616 ± 0.004 | 0.532 ± 0.010 | 0.133 ± 0.021 | 0.098 ± 0.024 |

**Table R1b.** The cold levels on targets **unseen by sequence** (§1b; option A): the same
checkpoints re-scored by their own trainers' test passes, which reproduce all 24 recorded
AUROCs (`results/clean_accuracy_davis.md`; to four decimals for the three deterministic
models, within the range of five passes for MolTrans, which keeps dropout on at inference
as published). Cold-target keeps 5,168 of 5,984 test rows, cold-pair 1,001 of 1,144.

| model | cold-target, all rows | cold-target, unseen | cold-pair, all rows | cold-pair, unseen |
|---|---|---|---|---|
| DeepDTA | 0.907 ± 0.003 | **0.884 ± 0.003** | 0.728 ± 0.035 | **0.749 ± 0.044** |
| ColdSite-DTI | 0.857 ± 0.011 | **0.835 ± 0.015** | 0.624 ± 0.099 | **0.607 ± 0.128** |
| HyperAttentionDTI | 0.915 ± 0.001 | **0.893 ± 0.001** | 0.694 ± 0.038 | **0.713 ± 0.050** |
| MolTrans | 0.874 ± 0.006 | **0.833 ± 0.008** | 0.566 ± 0.022 | **0.530 ± 0.024** |

Leakage inflated cold-target for **every one of the four models**, by 0.021–0.023 for
DeepDTA, ColdSite-DTI and HyperAttentionDTI and by **0.041 for MolTrans** — 11 of the 12
seeds move down (ColdSite-DTI seed 2, −0.002, is the exception). At cold-pair there is no
consistent effect: the 11 leaked targets were the harder ones for DeepDTA (+0.021) and
HyperAttentionDTI (+0.019) and the easier ones for MolTrans (−0.037) and ColdSite-DTI
(−0.017, its seed 2 falling to 0.510 — chance — on unseen proteins). The
unseen-by-sequence values are the ones the paper reports for the cold levels; the all-rows
values are kept for comparison with work that uses the same DAVIS files.

That the effect is largest for MolTrans, the model with by far the most parameters here,
is what memorising a training sequence would predict, but four models is not a sample:
we report it as an observation, not a trend.

**The levels are not a ladder of increasing difficulty.** On DAVIS an unseen target costs
little (DeepDTA 0.929 → 0.884 on targets unseen by sequence) and an unseen drug costs a
great deal (→ 0.692); cold-pair is no harder than cold-drug for DeepDTA (0.749 vs 0.692,
within the cold-drug spread).
The drug axis dominates: DAVIS has 68 drugs, so cold-drug trains on 49 and tests on 13,
while cold-target still trains on 310 targets and its 88 held-out targets are kinases
like them. Levels are therefore reported as categories, not as a severity scale.

**ColdSite-DTI is not the most accurate model, and does not need to be.** It matches
DeepDTA on random (0.924 vs 0.929) and cold-drug (0.721 ± 0.008 vs 0.692 ± 0.044, inside
DeepDTA's spread), and falls below it on cold-target (0.835 vs 0.884 unseen by sequence)
and cold-pair (0.607 vs 0.749). The audit asks whether each model's
explanation survives, not which model predicts best; accuracy is reported so that every
explanation result can be read against how well the same checkpoint predicts.

**Cold-pair is the least stable cell.** ColdSite-DTI's three cold-pair seeds score 0.738,
0.557 and 0.577, and all three checkpoints come from epoch 11, one epoch after the
selection floor: validation loss on cold-pair's 264 validation pairs was lowest almost
immediately. Random ran long by comparison (best epochs 40, 40, 32). Every cold-pair
result below is therefore quoted with its spread, never from one seed.

**The two published models do not degrade alike.** HyperAttentionDTI is the most accurate
model at every level it has (random 0.937, cold-target 0.893 unseen — the best cold-target
figure in the table) and holds up at cold-pair (0.713 unseen). MolTrans is close to the
others on random (0.923) but loses more at every cold level, and at cold-pair it reaches
**0.530 ± 0.024 on unseen proteins — chance**, with AUPRC 0.098 against a 0.057 positive
rate. Whatever its attention means at cold-pair, it is attached to a model that cannot
predict there; the audit reports that beside its explanation scores, because an
explanation of a prediction no better than chance is not an explanation of anything.

*HyperAttentionDTI cold-drug (3 cells) is still training. MolTrans's three seeds are the
retrained ones: the first grid's seeds 2 and 3 trained as seed 1 (the vendored `models.py`
reseeds torch on import; fixed 2026-09-13), and only the corrected cells are used here.*

## 1b. DAVIS's sequences: leakage and pseudo-variants

Found 2026-09-13 building the KLIFS ground truth; `results/sequence_audit_davis.md`
(`src/data/sequence_audit.py`). In DeepDTA's DAVIS `proteins.txt`, which this project
and most DTI papers use: (i) all 54 variant targets with a wild-type entry carry exactly
the wild-type sequence — ABL1(T315I), EGFR(T790M), BRAF(V600E) and the rest — so 442
targets are 379 distinct sequences; (ii) at cold-target 12 of 88 test targets (816 of
5,984 rows, 13.6%) and at cold-pair 11 of 88 (143 of 1,144, 12.5%) are unseen by name
but identical in sequence to a training target, and cold-pair validation is 13.6% such
rows; (iii) ten targets' sequences hold few or none of the 85 KLIFS pocket residues
(RET and its three mutants are RET's extracellular residues 1–430). KIBA has none of
the three. **Decided 2026-09-13: option A** (Methods §2.4) — cold-level accuracy on targets unseen
by sequence (Table R1b), explanation metrics without the seen-by-sequence and pocketless
targets and counting one protein per distinct sequence; nothing retrained. The leak into
cold-pair's validation set, which influenced checkpoint selection, is a limitation.

**What the leak is worth, measured by retraining without it.** Re-scoring changes which
rows are *measured*; it cannot remove what the model *learned*. So the cold splits were
rebuilt with the leak removed — no sequence shared between training, validation and test,
the test file untouched — beside a control that keeps the leak at the same row count and
the same number of positives (`src/data/seqclean_splits.py`). DeepDTA trained on both,
three seeds each; all three arms are scored on the one test set they share
(`results/leakage_retrain_davis.md`, `src/evaluation/leakage_retrain.py`).

**Table R1c.** DeepDTA test AUROC, mean ± sd over seeds 1–3.

| level | arm | training rows | leak | test AUROC |
|---|---|---|---|---|
| cold-target | original | 21,080 | in | 0.907 ± 0.003 |
| | volume-matched control | 17,748 | in | 0.888 ± 0.011 |
| | sequence-clean | 17,748 | out | **0.869 ± 0.013** |
| cold-pair | original | 15,190 | in | 0.728 ± 0.035 |
| | volume-matched control | 12,936 | in | 0.666 ± 0.027 |
| | sequence-clean | 12,936 | out | **0.686 ± 0.018** |

At cold-target the published 0.907 is **0.019 smaller training set + 0.019 leakage**: the
control minus the clean arm is −0.005, −0.015 and −0.037 in seeds 1, 2 and 3 — the same
direction in every seed, mean **0.019**. That is the figure re-scoring estimated
independently (0.023 for DeepDTA, Table R1b) by a method with nothing in common with this
one, which is the reason we report both. At cold-pair removing the leak costs nothing (the
clean arm is 0.021 *higher*, in all three seeds, inside the seed spread): cold-pair's
difficulty is its unseen drugs, as §2 finds, not its leaked proteins.

Retraining every model this way was not affordable; DeepDTA, the anchor, is the one
retrained, and the agreement between the two methods on it is what licenses using the
re-scored values for the other three.

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
(`results/positive_control_davis.md`, recomputed 2026-09-14 under the sequence policy of
§1b): synthetic attention that places a fraction *d* of the true binding sites first.
Precision@10 rises with *d* at every level, the oracle (*d* = 1) reaches the ceiling
(0.989–0.991), and a dose of **2% of the sites is detected as significant at all four
levels** — on the policy's own protein sets, 349 proteins at random and cold-drug and only
68 and 72 at cold-target and cold-pair. A model whose precision@10 sits at chance is
therefore a real null, not a test too weak to see anything. On a planted model whose prediction depends only on the annotated
sites, the faithfulness measure separates the oracle (comprehensiveness delta ≈ +10.2)
from no signal (≈ 0) at every level.

## 4. ColdSite-DTI's attention is load-bearing, finds the pocket region, misses the annotated residues

Computed 2026-09-13 on the 12 binary checkpoints under the sequence policy of §1b (option A:
one protein per distinct sequence, seen-by-sequence targets dropped at the cold levels,
pocketless targets dropped everywhere). Outputs: `results/analysis_davis_policyA/` (not
committed). Faithfulness uses up to 200 test pairs per level; precision@10 scores one test
pair per protein — 349 proteins at random and cold-drug, 68 at cold-target and 72 at
cold-pair (UniProt; KLIFS: 350 / 350 / 67 / 71) — against a 1,000-trial permutation null.

**Table R2.** ColdSite-DTI, DAVIS binary, seeds 1 / 2 / 3.

| level | faithfulness delta | mean ± sd | precision@10 (UniProt) | mean ± sd | chance | ceiling |
|---|---|---|---|---|---|---|
| random | 0.801 / 1.993 / 0.770 | 1.19 ± 0.70 | 0.023 / 0.009 / 0.013 | 0.015 ± 0.007 | 0.020 | 0.99 |
| cold-drug | 1.801 / 0.966 / 0.771 | 1.18 ± 0.55 | 0.027 / 0.027 / 0.011 | 0.022 ± 0.009 | 0.020 | 0.99 |
| cold-target | 0.163 / 0.891 / 0.496 | 0.52 ± 0.36 | 0.015 / 0.019 / 0.018 | 0.017 ± 0.002 | 0.019 | 0.99 |
| cold-pair | 0.619 / 0.944 / 0.432 | 0.67 ± 0.26 | 0.018 / 0.013 / 0.008 | 0.013 ± 0.005 | 0.019 | 0.99 |

Faithfulness delta = comprehensiveness (absolute change in the predicted logit when the 10
most-attended residues are masked) minus the same for 10 random residues; only the delta
is a result.

**Faithful at every level.** Masking the residues ColdSite-DTI attends to moves its
prediction more than masking random residues in all 12 cells (every delta positive,
flagged load-bearing in each seed's report). The attention is not decoration: the model
uses the residues it points at, at every split level.

**Against UniProt's annotated residues: at chance.** Mean precision@10 is 0.013–0.022
against a chance of 0.019–0.020 and a ceiling of 0.99; no level exceeds chance on average.
The only cells significant before correction are cold-drug seeds 1 and 2 (0.027, p = 0.008
and 0.010), which seed 3 does not reproduce (0.011, p = 1.0).

Read against the dose curve of §3, which was recomputed on these very protein sets, every
one of the 12 cells is worth an **equivalent dose of 0.006 or less** — the fraction of true
sites a synthetic explanation would have to rank first to match it — and 7 of the 12 sit at
or below chance (`results/positive_control_davis.md`, "Audited models, read against the
curve"). The same test detects a dose of 0.02 at every level, so this is a null with
resolution to spare, not an underpowered test: whatever ColdSite-DTI's attention carries,
it is worth under 1% of the annotated residues being ranked first.

**Against the KLIFS ATP pocket: above chance, mostly by finding the domain.** The same
checkpoints scored against KLIFS's 85-residue ATP pocket (Methods §3.6):

**Table R3.** ColdSite-DTI precision@10 against the KLIFS pocket, seeds 1 / 2 / 3
(`results/analysis_davis_policyA/positional_control_coldsite_dti_davis_klifs_policyA.md`).

| level | precision@10 | mean ± sd | chance | beats in-span shuffle (p) | top-10 inside the pocket's span | span / chain |
|---|---|---|---|---|---|---|
| random | 0.191 / 0.226 / 0.239 | 0.22 ± 0.02 | 0.14 | 0.205 / 0.001 / 0.001 | 0.33–0.36 | 0.25 |
| cold-drug | 0.300 / 0.259 / 0.338 | 0.30 ± 0.04 | 0.14 | 0.001 / 0.001 / 0.001 | 0.39–0.46 | 0.25 |
| cold-target | 0.228 / 0.204 / 0.296 | 0.24 ± 0.05 | 0.14 | 0.049 / 0.196 / 0.001 | 0.34–0.43 | 0.25 |
| cold-pair | 0.310 / 0.259 / 0.245 | 0.27 ± 0.03 | 0.14 | 0.005 / 0.009 / 0.211 | 0.40–0.48 | 0.24 |

Every cell beats, at p = 0.001, both a map borrowed from another protein (position alone)
and the protein's own attention shuffled among residues of the same amino acid
(residue-type preference alone). So the attention does find the pocket region of each
kinase. How: 33–48% of its top ten residues fall inside the stretch the pocket spans (the
kinase domain core), which is 24–25% of the chain; shuffled within that stretch it keeps
most of its score, and beats the within-stretch shuffle in 9 of 12 cells (p ≤ 0.049,
before correction) by a modest margin (e.g. cold-drug seed 1: 0.300 vs 0.250). Most of the
pocket signal is knowing the domain; a smaller part is knowing the pocket inside it.

Taken together: the attention is causally used (every level), is **coarsely plausible** —
it concentrates on the kinase domain and its ATP pocket at about twice chance, beyond
position and amino-acid preference — and is **not finely plausible**: it does not land on
the residues UniProt annotates. "Is attention plausible?" has a different answer at each
ground-truth resolution, so both are reported. The pattern is the same with and without
the sequence policy (pre-policy values: `results/positional_control_coldsite_dti_davis_klifs.md`).
*[Formal statement waits for the audit table (§5), Holm over the whole family.]*

## 5. *[PENDING]* The audit table (all subjects, Holm over the whole family)

*[Waiting on the analysis running 2026-09-14 (`results/analysis_davis_policyA/`):
HyperAttentionDTI and MolTrans faithfulness, both models' UniProt and KLIFS ladders, the
non-kinase control for both, then `run_audit` once over the whole family. Structure this
section will take, so the numbers only have to be dropped in:*

1. *One row per (model, level): faithfulness delta (mean ± sd over 3 seeds, "load-bearing"
   in n/3 cells), precision@10 against UniProt and against KLIFS, each beside its chance
   and the Holm-corrected p. Three audited models × 4 levels = 12 rows, minus
   HyperAttentionDTI cold-drug if account 1's last three cells do not arrive.*
2. *The verdict per model in one sentence each, in the three-way form §4 establishes for
   ColdSite-DTI: faithful / coarsely plausible / finely plausible, each answered against
   its own null.*
3. *Whether the published models (HyperAttentionDTI, MolTrans) behave like ours. If they
   are finely plausible where ColdSite-DTI is not, the §4 finding is model-specific; if all
   three are coarse-only, it is a property of attention trained on affinity labels alone —
   which is the paper's central claim and decides how Discussion §2 is written.*
4. *Read against the dose curve of §3 (`positive_control --compare`) for every ladder, so
   each null carries the resolution at which it is a null.*

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
