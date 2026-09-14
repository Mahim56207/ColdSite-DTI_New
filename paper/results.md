# Results (draft)

Drafted 2026-09-13 from the cells finished so far. Every number below is read from a
trained cell's `_results.json` (DAVIS binary grid, Kaggle account 1, commit v1 of
`kaggle_davis_binary_grid36.ipynb`) or from a file named beside it. The DAVIS grid is
complete: 48 of 48 cells (4 models x 4 levels x 3 seeds), verified cell by cell against
the AUROC each recorded. *[PENDING]* now marks only KIBA and the antiviral case study. All
values are test-set means ± sample standard deviation over three training seeds; a
difference smaller than the spread is not reported as one.

---

## 1. Accuracy across the four split levels (DAVIS, binary)

**Table R1.** Test AUROC, mean ± sd over seeds 1–3. AUPRC is given with each test set's
positive rate, which is its chance level; AUPRC is not comparable across levels because
that rate differs.

| model | random | cold-target | cold-drug | cold-pair |
|---|---|---|---|---|
| DeepDTA (anchor) | 0.929 ± 0.002 | 0.907 ± 0.003 | 0.692 ± 0.044 | 0.728 ± 0.035 |
| ColdSite-DTI (ours) | 0.924 ± 0.001 | 0.857 ± 0.011 | 0.721 ± 0.008 | 0.624 ± 0.099 |
| HyperAttentionDTI | 0.937 ± 0.005 | 0.915 ± 0.001 | **0.760 ± 0.042** | 0.694 ± 0.038 |
| MolTrans | 0.923 ± 0.002 | 0.874 ± 0.005 | 0.685 ± 0.020 | 0.569 ± 0.021 |

| | random | cold-target | cold-drug | cold-pair |
|---|---|---|---|---|
| test pairs | 6,011 | 5,984 | 5,746 | 1,144 |
| test drugs / targets | 68 / 442 | 68 / 88 | 13 / 442 | 13 / 88 |
| positive rate (AUPRC chance) | 0.077 | 0.075 | 0.060 | 0.057 |
| DeepDTA AUPRC | 0.631 ± 0.013 | 0.615 ± 0.004 | 0.199 ± 0.045 | 0.206 ± 0.012 |
| ColdSite-DTI AUPRC | 0.612 ± 0.010 | 0.464 ± 0.047 | 0.201 ± 0.035 | 0.132 ± 0.028 |
| HyperAttentionDTI AUPRC | 0.669 ± 0.021 | 0.644 ± 0.014 | 0.284 ± 0.032 | 0.187 ± 0.028 |
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
model at every level (random 0.937, cold-target 0.893 unseen — the best cold-target figure
in the table), holds up at cold-pair (0.713 unseen), and is the only model that does not
collapse on unseen drugs: **0.760 ± 0.042 at cold-drug**, against 0.692 ± 0.044 for
DeepDTA, 0.721 ± 0.008 for ColdSite-DTI and 0.685 ± 0.020 for MolTrans. On the axis that
costs every other model the most, it loses the least. MolTrans is close to the
others on random (0.923) but loses more at every cold level, and at cold-pair it reaches
**0.530 ± 0.024 on unseen proteins — chance**, with AUPRC 0.098 against a 0.057 positive
rate. Whatever its attention means at cold-pair, it is attached to a model that cannot
predict there; the audit reports that beside its explanation scores, because an
explanation of a prediction no better than chance is not an explanation of anything.

*MolTrans's three seeds are the retrained ones: the first grid's seeds 2 and 3 trained as seed 1 (the vendored `models.py`
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

## 5. The audit table: the published model's residue-level claim holds only on the random split

Computed 2026-09-14 on two T4 GPUs (`notebooks/kaggle_analysis_davis.ipynb`), **one Holm
correction over all sixteen cells** — three audited models and the uniform control, four
levels each. Correcting per model would have inflated every claim in the table.

**Table R4.** precision@10 against UniProt's annotated residues, mean ± sd over seeds
1–3 (`results/analysis_davis_policyA/audit_davis_binary.md`). `uniform_control` is an
attention map of equal weight everywhere — the metric's own floor.

| model | random | cold-drug | cold-target | cold-pair |
|---|---|---|---|---|
| ColdSite-DTI (ours) | 0.015 ± 0.007 | 0.022 ± 0.009 | 0.017 ± 0.002 | 0.013 ± 0.005 |
| HyperAttentionDTI (published) | **0.034 ± 0.006** | 0.040 ± 0.031 | 0.024 ± 0.008 | 0.022 ± 0.010 |
| MolTrans (published) | 0.021 ± 0.003 | 0.027 ± 0.005 | 0.028 ± 0.016 | 0.020 ± 0.014 |
| uniform control | 0.020 ± 0.001 | 0.020 ± 0.001 | 0.018 ± 0.005 | 0.017 ± 0.001 |

**One cell of sixteen survives Holm–Bonferroni: HyperAttentionDTI on the random split**
(p = 0.0020 against a threshold of 0.0031). The next four in the ordering all fail:
ColdSite-DTI cold-drug (p = 0.0060 vs 0.0033), MolTrans cold-target (p = 0.012 vs 0.0036)
and cold-drug (p = 0.020 vs 0.0038), and HyperAttentionDTI cold-drug (p = 0.050 vs
0.0042, on a seed spread of ±0.031). HyperAttentionDTI's cold-target (p = 0.11) and
cold-pair (p = 0.30) are not close, and ColdSite-DTI survives nowhere.

**MolTrans is at the metric's floor everywhere.** Its four cells (0.020–0.028) sit within
one standard deviation of the uniform control's (0.017–0.020) — an attention map of equal
weight everywhere scores the same as its trained attention. Its best cell, cold-target
0.028 ± 0.016, is also where its accuracy is 0.833 unseen (Table R1b); at cold-pair,
where it predicts at chance (0.530), its attention scores 0.020 against a 0.017 floor.
Both of the published models we audit therefore fail the residue-level claim under
shift, and one of them fails it everywhere.

**The warm signal is real, not an artefact.** At the random split all three of
HyperAttentionDTI's seeds beat every null in `positional_control`: a map borrowed from
another protein (0.021, p = 0.001), attention permuted among residues of the same amino
acid (0.021, p = 0.001) and permuted within the site-spanning stretch (0.026, p ≤ 0.003 in
two seeds of three). So on the random split this model's attention carries
protein-specific, residue-level information about where the annotated residues are —
1.67× chance — which is exactly the claim the interpretability literature makes, and it is
supported.

**It does not survive distribution shift.** By cold-target the same model is at 1.26×
chance and no longer beats a borrowed map in two of three seeds; at cold-pair it is at
1.18× and beats nothing. The cold cells are also where the seeds disagree most
(cold-drug 0.019–0.077 across seeds), so single-seed evidence there would be worthless in
either direction.

**Faithfulness tells the same story in a different currency.** HyperAttentionDTI's
attention is load-bearing at every level, but the margin over random masking collapses as
the split hardens: **0.184 ± 0.056** (random), 0.113 ± 0.052 (cold-drug),
0.063 ± 0.046 (cold-target), 0.056 ± 0.008 (cold-pair). The attention is still used
under shift; it is simply used for something that no longer coincides with the annotated
site.

**Against the KLIFS pocket, the coarse signal persists where the fine one does not**
(`results/analysis_davis_policyA_klifs/`): 0.242 ± 0.038 at random (1.70× its 0.143
chance, p = 0.001 in every seed) and 1.33–1.37× at all three cold levels. Read beside
§4, the two audited models differ in kind rather than in degree: ColdSite-DTI is
coarsely plausible everywhere and finely plausible nowhere, while HyperAttentionDTI is
both at random and only coarsely so once the split is cold.

*Device check: every cell of this table was also computed on a CPU during the same night.
The two agree exactly in 11 of 12 cells and by 0.0003 in the twelfth, so the GPU runs
carry no device-specific drift.*

*The kinase-family confound could not be stratified away: fewer than 20 non-kinase targets
are available in every cell (`audit_davis_binary.md`), so the unstratified table must be
read with §6's finding in mind.*

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

## 5b. Faithfulness, and an intervention that was not the same size in both arms

A masking test subtracts a random-masking control from the explanation's
comprehensiveness, which is only meaningful if both arms change the input by the same
amount. For a model that reads residues they do: masking k residues changes k input
positions either way. **MolTrans does not read residues.** It reads ESPF sub-word tokens,
so replacing a residue with `X` re-segments the protein, and how much of the token
sequence changes depends on where the masked residues sit. Measured over 20 DAVIS
proteins (`src/evaluation/mask_comparability.py`): masking MolTrans's ten most-attended
residues changes **48%** of its tokens, and masking ten random residues changes **95%** —
for RIPK5, 0.8% against 99%, a factor of 124.

Under that test MolTrans's faithfulness delta was negative in 11 of 12 cells, reproducing
to 0.005 across a laptop and a T4. Read naively it says its attention points at residues
that matter *less* than arbitrary ones. It says no such thing: subtracting a larger
intervention from a smaller one returns a negative number whatever the attention does.

Measured in the space the model reads — the explanation's arm removes the tokens carrying
its top-10 attention, the control removes **the same number** of tokens at random, so both
arms change an identical amount of input by construction
(`src/evaluation/token_faithfulness.py`) — the sign reverses:

**Table R4b.** MolTrans comprehensiveness delta, 200 pairs per level (75 and 76 at the
cold levels under the sequence policy), mean over seeds 1–3.

| level | residue space (arms unequal) | token space (arms matched) |
|---|---|---|
| random | −0.299 | **+0.458** |
| cold-drug | −0.407 | **+0.362** |
| cold-target | −0.176 | **+0.347** |
| cold-pair | −0.087 | **+0.276** |

**MolTrans's attention is load-bearing at every level** — positive in all 12 cells. The
seed spread is wide (0.125 to 0.766 at random), so the sign is the result and the
magnitude is not. Its *sufficiency* deltas are mostly negative: keeping only the attended
tokens preserves the prediction less well than keeping the same number of random ones,
which is what a thinly spread attention looks like — removing its top tokens matters,
but they do not carry the prediction alone.

Two consequences beyond this model. Masking-based faithfulness **does not transfer across
tokenisations**, so any audit of a sub-word protein model that compares k-residue masks is
measuring its own intervention; and because the unit differs (tokens here, residues for
the other two models), these deltas are comparable within a model across levels and seeds
— which is how the audit uses them — and not numerically across models. Methods states
both.

## 6. Kinase-family control, and why the confound cannot be tested inside DAVIS

Every model here trained on a kinase panel, so a plausibility score could reflect
knowledge of one protein family rather than of binding sites. The natural test is to
stratify each cell by family and compare. **That test is impossible on these benchmarks.**
DAVIS's 6,011 test rows contain 3,307 rows on targets our classifier recognises as
kinases and **zero** on a non-kinase (`src/evaluation/target_family.py`, measured
2026-09-14); the remaining 2,704 are kinases its gene-symbol heuristic does not name.
KIBA is the same kind of object — 229 kinases. No panel size fixes this: the gate counts
non-kinase targets *in the cell being scored*, and there are none to count. That the two
standard DTI benchmarks cannot answer the family-confound question is a fact about the
benchmarks, and belongs beside §1b's leakage finding rather than in a limitations list.

What can be done is to score the same trained attention on proteins from **outside** the
training family: 60 BindingDB proteins with UniProt-annotated sites, none of them kinases,
none seen by any model here (`src/data/build_nonkinase_panel.py`; sequences and site
numbering both from UniProt, never BindingDB's construct chains). Chance differs between
the arms because the proteins differ — 0.020 for DAVIS's kinases, 0.012 for the panel —
so each arm is read against its own.

**Table R5.** precision@10, mean ± sd over seeds 1–3, cotransport ions excluded (the
primary setting; all ligands as sensitivity). `chance` in brackets.

| model | level | kinase arm (n = 349) | non-kinase panel (n = 60) |
|---|---|---|---|
| ColdSite-DTI | random | 0.015 ± 0.007 (0.020) | 0.027 ± 0.021 (0.012) |
| | cold-drug | 0.022 ± 0.009 (0.020) | 0.029 ± 0.014 (0.012) |
| | cold-target | 0.017 ± 0.002 (0.019) | 0.014 ± 0.008 (0.012) |
| | cold-pair | 0.013 ± 0.005 (0.019) | **0.044 ± 0.019** (0.012) |
| HyperAttentionDTI | random | **0.034 ± 0.006** (0.020) | 0.015 ± 0.002 (0.012) |
| | cold-drug | 0.040 ± 0.031 (0.020) | 0.013 ± 0.004 (0.012) |
| | cold-target | 0.024 ± 0.008 (0.019) | 0.017 ± 0.004 (0.012) |
| | cold-pair | 0.022 ± 0.010 (0.019) | 0.007 ± 0.003 (0.012) |
| MolTrans | random | 0.021 ± 0.003 (0.020) | 0.012 ± 0.007 (0.012) |
| | cold-drug | 0.027 ± 0.005 (0.020) | 0.012 ± 0.002 (0.012) |
| | cold-target | 0.028 ± 0.016 (0.019) | 0.014 ± 0.005 (0.012) |
| | cold-pair | 0.020 ± 0.014 (0.019) | 0.008 ± 0.006 (0.012) |

**The two published models score no better than chance off the training family.**
HyperAttentionDTI and MolTrans sit within one standard deviation of the panel's 0.012 in
all eight cells, so whatever HyperAttentionDTI's warm-split signal is (§5), it does not
travel to proteins outside the family it trained on. That is the answer the confound
question wanted, obtained without a stratification the data cannot support.

**ColdSite-DTI is the exception, and the exception is an artefact.** It is above the
panel's chance in three cells, most clearly at cold-pair (0.044 against 0.012) — where
its kinase arm is at 0.013. A higher score on unseen proteins from another family than on
the family it trained on is not knowledge, and the nulls of §4 identify what it is.

It is **not positional**: maps borrowed from another protein score 0.012–0.014 on the
panel, no better than chance. It is **amino-acid preference**. ColdSite-DTI's top-ten
attention is enriched in histidine (3× in seed 1, 11–15× in seeds 2 and 3); the panel's
annotated sites are histidine-rich (8.8×, many of them metal sites); kinase ATP sites are
not (they are enriched in glycine, aspartate and lysine). Permuting the attention among
residues of the *same amino acid* — which keeps the preference and destroys any knowledge
of position — recovers most of the panel score (seed 2 random: 0.045 of 0.050, p = 0.28),
and three of twelve cells keep a remainder significant before correction
(p = 0.004–0.048). The ordering across seeds follows the preference rather than the
accuracy: seed 2, with the strongest histidine enrichment, has the highest panel scores.

So the panel's apparent signal is a liking for one amino acid meeting sites that happen to
be rich in it — the same preference that makes the model *miss* the glycine-rich kinase
ATP site it was trained on. Read together with §5, the family confound does not rescue any
model's plausibility: the two published models are at chance off their training family,
and ours is above chance there for a reason that has nothing to do with binding.

*Sensitivity: with cotransport ions included (`_noions` dropped) the panel's chance level
rises and the same pattern holds; both settings are in
`results/analysis_davis_policyA/control_*.json`. The panel's 60 proteins are what limit
this comparison, and no protein-level interval was computed for it (§7d): the ± given here
are seed spreads, and the closest measured analogue — a 68-protein UniProt cell — is
±0.005–0.009 wide. Only the ColdSite-DTI cold-pair cell stands clear of the panel's chance
level by more than its own spread; the claim that the two published models are at chance
rests on eight cells agreeing rather than on any one of them.*

## 7. Does attention know *which* drug binds? A per-pair ground truth

Both ground truths so far are per protein, and the model is asked about a **pair**. A
reviewer is entitled to ask whether attention marks the site of *this* binding event, and
neither UniProt's annotations nor the KLIFS pocket can answer that.

KLIFS publishes, for every co-crystal structure, an **interaction fingerprint**: 85 pocket
positions × 7 interaction types, recording which pocket residues touch the bound ligand.
Where a DAVIS drug is that ligand, the residues it actually touches are *measured* rather
than annotated. `src/data/klifs_ligand_contacts.py` matches DAVIS drugs to KLIFS ligands
by InChIKey (RDKit, from the SMILES the models trained on) and places the contacts through
the same steps as the pocket ground truth — same KLIFS entry by UniProt accession, same
placement, same remapping — so both ground truths share one coordinate frame. A position
counts when it is contacted in at least half of the pair's structures; imatinib has 18
ABL1 structures and they do not agree residue for residue (the agreed set is 93% of the
union). **217 pairs, 101 proteins, 28 drugs**, median 19 contacted residues per pair. Two
checks: every contact lies inside its protein, and every contact lies inside the KLIFS
pocket of the same protein. ABL1–imatinib (2hyy) contacts 24 of 85 positions, and the
contacted residues include the VAIK lysine and the DFG motif.

**What DAVIS can support.** A pair is scorable only where that exact drug has been
crystallised with that exact kinase:

| level | test rows | scorable pairs | proteins | drugs |
|---|---|---|---|---|
| random | 6,011 | 38 | 29 | 17 |
| cold-drug | 5,746 | 60 | 57 | 6 |
| cold-target | 5,984 | 39 | 20 | 19 |
| cold-pair | 1,144 | **12** | 12 | 5 |

**The control that makes it interpretable.** A drug's contacts sit inside one pocket, so a
model that merely finds the pocket scores well against *any* drug's contacts there. The
comparison therefore holds the protein and the number of sites fixed and changes only
*which drug the contacts belong to*: each pair is scored against another drug's contacts on
the same protein (`swap_drugs`, a rotation by sorted drug id). Only proteins with at least
two crystallised drugs can be swapped, so both arms are restricted to exactly those pairs
— identical keys, identical n — and the difference between the columns is the drug and
nothing else.

**Table R6.** precision@10 against crystallographic contacts, mean over seeds 1–3 with
95% intervals from resampling proteins (§7d). Chance is higher than against UniProt's
annotations because a drug touches ~19 residues rather than ~12.

| model | level | n | chance | the pair's own drug | another drug, same pocket |
|---|---|---|---|---|---|
| ColdSite-DTI | random | 20 | 0.025 | 0.032 [0.010–0.062] | 0.025 [0.005–0.053] |
| | cold-drug | 36 | 0.023 | 0.030 [0.019–0.043] | 0.032 [0.020–0.046] |
| | cold-target | 4 | 0.025 | 0.046 [0.008–0.096] | 0.038 [0.007–0.082] |
| HyperAttentionDTI | random | 29 | 0.025 | 0.036 [0.028–0.045] | 0.033 [0.025–0.041] |
| | cold-drug | 39 | 0.023 | 0.025 [0.015–0.036] | 0.015 [0.005–0.027] |
| | cold-target | 14 | 0.025 | 0.060 [0.040–0.079] | 0.060 [0.043–0.076] |
| MolTrans | random | 29 | 0.025 | 0.011 [0.005–0.021] | 0.009 [0.002–0.021] |
| | cold-drug | 39 | 0.023 | 0.036 [0.025–0.048] | **0.056** [0.042–0.072] |
| | cold-target | 14 | 0.025 | 0.048 [0.036–0.060] | 0.045 [0.033–0.057] |

**Attention does not know which drug binds.** In every row the two intervals overlap
almost entirely; the largest gain from using the correct drug is +0.011, and in two rows
the *wrong* drug scores higher — MolTrans's cold-drug cell by 0.021, outside the paired
arm's interval. Whatever agreement exists with crystallographic contacts is agreement with
the pocket those contacts lie in, not with the binding event the model was asked about.

Cold-pair is omitted from the table: three scorable pairs after the sequence policy, where
MolTrans scores 0.000 in both arms, ColdSite-DTI 0.011, and HyperAttentionDTI's interval is
0.067 wide on those three proteins (§7d) — wider than any difference the table above
reports. Twelve pairs before the policy is the honest ceiling DAVIS offers at that level,
and it is not enough to say anything.

## 7b. Is the verdict the model's, or the readout's?

Between the tensor inside a network and the one weight per residue that precision@k scores,
somebody chooses: which axis to reduce, which layer to read, how to spread a convolution
position or a sub-word token over the residues it covers. Every number above rests on
choices made once. `src/evaluation/readout_variants.py` scores the same checkpoints through
readouts another author could reasonably have picked, each changing exactly one documented
choice, with the published readouts re-scored in the same run so nothing is compared across
devices or code states.

**Table R7.** precision@10, mean ± sd over seeds 1–3. Chance is 0.020 against UniProt's
annotated residues and 0.143 against the KLIFS pocket.

| model | readout | UniProt: random / cold-target | KLIFS: random / cold-target |
|---|---|---|---|
| ColdSite-DTI | cross-attention *(published)* | 0.015 / 0.017 | 0.219 / 0.243 |
| | protein self-attention | 0.010 / 0.022 | 0.238 / 0.268 |
| HyperAttentionDTI | channel mean, centre *(published)* | 0.034 / 0.025 | 0.242 / 0.186 |
| | **channel max** | 0.038 / 0.034 | **0.335 / 0.367** |
| | **receptive-field spread** | 0.027 / 0.012 | **0.157 / 0.081** |
| MolTrans | head mean, last layer *(published)* | 0.021 / 0.026 | 0.157 / 0.176 |
| | head max | 0.021 / 0.029 | 0.155 / 0.177 |
| | first layer | 0.023 / 0.023 | 0.173 / 0.189 |

Three findings, in order of how much they should worry a reader of the literature.

**The residues a readout points at are largely a property of the readout.** Across 25
proteins, the top-ten residues of an alternative readout overlap the published readout's
by **2–12%** — HyperAttentionDTI's channel-max and receptive-field readouts share 2% and
4% of their top ten with the published one — with MolTrans's head-max the single exception
at 90%. Two defensible readouts of one checkpoint therefore highlight almost disjoint sets
of residues. A published attention figure is, to that extent, a picture of a reduction
choice.

**A readout choice can move a verdict across chance.** HyperAttentionDTI against the KLIFS
pocket reads 0.367 at cold-target under channel-max (2.6× chance) and **0.081** under the
receptive-field projection (*below* the 0.143 chance level), against 0.186 as published.
The claim "this model's attention finds the ATP pocket under distribution shift" is true,
false, or unsupported depending on a choice no paper reports.

**But the audit's own verdicts survive.** Every readout of every model stays at chance
against UniProt's annotated residues (0.010–0.057 against 0.020, all within the seed
spread bar HyperAttentionDTI's channel-max cold-drug cell at 0.057 ± 0.018), and no
readout lifts MolTrans above the pocket's chance level (0.120–0.189 against 0.143). The
residue-level null of §5 and the floor of §6 are therefore not artefacts of how we read
attention; what the readout choice changes is the *size* of the coarse, pocket-level
signal, not the existence of the fine-grained one.

**A readout that cannot see the drug does as well as one that can.** ColdSite-DTI's
protein-tower self-attention — computed by its forward pass and discarded, and
independent of the drug by construction — scores **0.238** against the KLIFS pocket at
random where its drug-conditioned cross-attention scores 0.219, and 0.268 against 0.243 at
cold-target. Its reported explanation owes nothing to the pair. Read with §7, where using
the correct drug's contacts buys at most +0.011 over another drug's, two independent
measurements say the same thing: the drug is not doing work in these explanations.

## 7c. Attention versus the gradient: is it the explanation or the model?

Every measurement so far scores **attention**. When attention misses the site, two
opposite things could be true — the attention is a poor report of a model that does
represent the site, or the model never learned it — and no attention measurement
separates them. Integrated gradients do: the attribution comes from the trained weights
and the gradient of the model's own prediction, with no interpretability head
(`src/evaluation/integrated_gradients.py`; path from the padding embedding, the same
"no residue here" the masking uses, 32 steps, explaining each model's own `predict`). The
variants read the *same checkpoints*, so this is two explanations of one model.

**Table R8.** precision@10 against UniProt's annotated residues, all three audited models,
mean ± sd over seeds 1–3. `attention` is the audit table of §5. Holm is applied over the
twelve cells of this family, with each cell's p taken as the median of its three seeds —
the same rule `run_audit` uses for the attention family.

| model | level | attention | integrated gradients | IG / attn | × chance | survives Holm |
|---|---|---|---|---|---|---|
| ColdSite-DTI | random | 0.015 | 0.021 ± 0.009 | 1.4× | 1.03× | no (p = 0.73) |
| | cold-drug | 0.022 | **0.055 ± 0.022** | 2.5× | **2.69×** | **yes** (p = 0.0010) |
| | cold-target | 0.017 | **0.044 ± 0.027** | 2.6× | **2.25×** | **yes** (p = 0.0030) |
| | cold-pair | 0.013 | 0.014 ± 0.001 | 1.1× | 0.76× | no (p = 0.87) |
| HyperAttentionDTI | random | 0.034 | **0.055 ± 0.034** | 1.6× | **2.70×** | **yes** (p = 0.0010) |
| | cold-drug | 0.040 | **0.079 ± 0.005** | 2.0× | **3.88×** | **yes** (p = 0.0010) |
| | cold-target | 0.025 | **0.079 ± 0.009** | 3.2× | **4.07×** | **yes** (p = 0.0010) |
| | cold-pair | 0.022 | **0.060 ± 0.028** | 2.7× | **3.20×** | **yes** (p = 0.0010) |
| MolTrans | random | 0.021 | 0.018 ± 0.001 | 0.9× | 0.90× | no (p = 0.84) |
| | cold-drug | 0.027 | 0.027 ± 0.003 | 1.0× | 1.32× | yes (p = 0.0050) |
| | cold-target | 0.028 | 0.029 ± 0.001 | 1.1× | 1.52× | no (p = 0.050) |
| | cold-pair | 0.020 | 0.021 ± 0.003 | 1.0× | 1.11× | no (p = 0.38) |

**Table R8b.** The same against the 85-residue KLIFS ATP pocket (chance 0.136–0.143).
Eleven of these twelve cells survive Holm; only MolTrans's cold-pair does not.

| model | random | cold-drug | cold-target | cold-pair |
|---|---|---|---|---|
| ColdSite-DTI attention | 0.219 | 0.299 | 0.243 | 0.271 |
| ColdSite-DTI **IG** | **0.280 ± 0.014** | **0.451 ± 0.159** | **0.325 ± 0.040** | **0.314 ± 0.044** |
| HyperAttentionDTI attention | 0.242 | 0.193 | 0.186 | 0.185 |
| HyperAttentionDTI **IG** | **0.427 ± 0.070** | **0.384 ± 0.078** | **0.538 ± 0.068** | **0.326 ± 0.055** |
| MolTrans attention | 0.157 | 0.154 | 0.176 | 0.136 |
| MolTrans **IG** | 0.168 ± 0.045 | 0.165 ± 0.025 | 0.162 ± 0.016 | 0.125 ± 0.024 |

**Seven of twelve cells survive Holm for the gradient, against one of sixteen for the
attention.** The comparison is as controlled as it can be made: the same checkpoints, the
same ground truth, the same protein sets, the same permutation test, the same *k* — only
the explanation differs. Where the attention of the best-generalising model is at chance
under shift, its gradient is at 3.2–4.1× chance and survives correction at every level.

**And the gap appears exactly where the attention carries something.** For the two models
whose attention is at least coarsely plausible, the gradient recovers far more: ColdSite-DTI
2.5–2.6× its attention at the two cold levels where it clears correction, HyperAttentionDTI
1.6–3.2× at all four. For **MolTrans the gradient matches its attention to within noise**
(0.9–1.1× on annotated residues, 0.9–1.1× on the pocket) and both sit at the floor. That is
the control this section needed. It says the two failures are different in kind:

* HyperAttentionDTI and ColdSite-DTI **do** represent the binding site, and their attention
  under-reports it — a reporting failure.
* MolTrans's attention is not under-reporting anything. Its gradient, which has no
  interpretability head to blame, is at the floor too. Its failure is the **model**.

No attention measurement could have drawn that distinction, which is the argument for
including a second explanation method in an audit of this kind at all.

Two honest qualifications. **ColdSite-DTI's gradient is noisy where it matters**: its
cold-target cell is 0.074 / 0.021 / 0.037 across seeds (± 0.027) and its cold-drug pocket
cell is 0.352 / 0.634 / 0.366 (± 0.159), so the effect is established by the permutation
test rather than by a precise estimate, and the direction is what we report. **MolTrans's
three surviving KLIFS cells are significant but tiny** — 1.16–1.18× chance on 350 proteins
— and are read as the floor, not as a signal; §5's rule of reporting effect size beside p
is why.

*Reproducibility: this table was computed twice, on two Kaggle accounts with independent T4
allocations, one on commit `139b103` and one on `3ca50aa`. All 24 cells (2 models × 4 levels
× 3 seeds × 2 ground truths) agree to **0.0e+00** — bit-identical — which is what the
attribution's determinism check predicted and is worth recording because the first attempt
at this analysis crashed on a non-deterministic cuDNN path.*

*The correction family: these twelve cells are Holm-corrected among themselves, not pooled
with the sixteen attention cells of §5. Integrated gradients were added **after** the
attention results were seen, so this is a secondary analysis and is labelled one; the
attention audit remains the pre-specified primary family. Methods states both, and no claim
in this section rests on comparing a corrected p from one family with a corrected p from the
other.*

## 7d. How precisely does a cell of this size measure anything?

Every mean above carries a seed spread, which says how much the *training* varied. It does
not say how precisely a cell measures its protein population — a cell of 68 proteins and a
cell of 349 can report the same ± and mean very different things. `src/evaluation/bootstrap_ci.py`
resamples the **proteins** a cell scored, 10,000 times, each protein entering with all of
its seeds (so seed variation stays inside the interval rather than being averaged away
first). The mean is the number the ladder already reports.

**Table R9.** 95% percentile intervals, precision@10 (`results/ci_davis.md`). Chance is
0.019–0.020 against UniProt's annotated residues and 0.136–0.143 against the KLIFS pocket.

| ground truth | model | random (n = 349/350) | cold-drug | cold-target (n = 68/67) | cold-pair (n = 72/71) |
|---|---|---|---|---|---|
| UniProt | ColdSite-DTI | 0.015 [0.013–0.018] | 0.022 [0.019–0.024] | 0.017 [0.012–0.023] | 0.013 [0.007–0.020] |
| | HyperAttentionDTI | **0.034 [0.030–0.038]** | 0.040 [0.037–0.044] | 0.025 [0.018–0.031] | 0.022 [0.016–0.029] |
| | MolTrans | 0.021 [0.017–0.026] | 0.026 [0.021–0.031] | 0.026 [0.018–0.035] | 0.020 [0.013–0.027] |
| KLIFS | ColdSite-DTI | 0.219 [0.208–0.229] | 0.299 [0.286–0.312] | 0.243 [0.218–0.268] | 0.271 [0.241–0.303] |
| | HyperAttentionDTI | 0.242 [0.232–0.253] | 0.193 [0.183–0.203] | 0.186 [0.163–0.209] | 0.185 [0.165–0.206] |
| | MolTrans | 0.157 [0.143–0.170] | 0.154 [0.142–0.166] | 0.176 [0.150–0.201] | 0.136 [0.114–0.158] |

Three things this settles that the seed spreads could not.

**HyperAttentionDTI's random cell excludes chance; ColdSite-DTI's excludes it downwards.**
The surviving cell of §5 reads 0.034 [0.030–0.038] against a chance of 0.020 — the whole
interval above it. ColdSite-DTI at random reads 0.015 [0.013–0.018]: the entire interval
lies *below* chance, which is a stronger statement than "at chance" and is consistent with
§6's finding that its attention prefers an amino acid the kinase ATP site is poor in.

**The cold cells are imprecise, but not as imprecise as their seed spreads suggest.**
Against UniProt, a 68-protein cell measures precision@10 to ±0.005–0.009 — narrower than
the ±0.008–0.016 seed spreads of Table R4, because averaging three seeds per protein
removes noise the spread reports. Against KLIFS the same cells are ±0.020–0.031, five
times the random level's, so the cold-level pocket numbers are the loosest in the paper.

**The per-pair drug arms are too small to carry their point estimates.** The intervals for
§7's three arms (`results/ci_drug_arms_davis.md`) run to 0.087 wide at ColdSite-DTI's
cold-target paired cell (4 proteins) and 0.075 at its swapped cell, and every cold-pair
arm rests on 3 proteins. This is why §7 omits cold-pair and reads the paired-versus-swapped
comparison off overlapping intervals rather than off the difference of two means.

No interval is computed for the 60-protein non-kinase panel of §6: its proteins are drawn
from a different population, so the bootstrap would need its own run. The closest measured
analogue is a 68-protein UniProt cell at ±0.005–0.009, which is why §6's claim rests on
eight cells agreeing rather than on any one of them.

## 8. *[PENDING]* KIBA

## 9. *[PENDING]* Antiviral case study

*[Recommend cutting. The subset is three distinct proteins (HIV-1 protease, HIV-1 RT,
influenza neuraminidase) after the 2026-07-31 BindingDB release put all 18,149 SARS-CoV-2
rows under one 7,096-residue polyprotein; §6's 60-protein panel supersedes it as a
non-kinase arm, and a case study on three proteins invites the objection it cannot
answer.]*
