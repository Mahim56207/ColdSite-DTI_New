# Ironclad verification report — "Seed-dependent verdicts"

Written 2026-09-29. **These are diagnostics, not proofs.** The requested protocol asked for "mathematical proof" and "100% certainty" of an
empirical claim. No computation on trained networks can deliver that: the claim is about how separately trained networks differ, so it can
only be supported by measurement and by ruling out alternative explanations. Each analysis below was therefore designed so it could have
come out against the claim, with untrained-network and same-seed baselines, and two of the requested premises turned out to be false
(stated plainly where they occur).

Evidence: `scripts/certification/step9_ironclad_diagnostics.py` → `results/certification/step9_ironclad_diagnostics.txt` (HyperAttentionDTI,
DAVIS, seeds 1–3, all four levels, 349/349/68/72 scored proteins, live forward passes on the Apple GPU, 340 s, deterministic: a re-run
gave identical numbers). Other models were not analysed.

---

## Section 1 — Wave A: execution status and quota management

**Status (from the file system): built, gated, not launched.** No seed 4/5 result, checkpoint or log exists anywhere (`results/`, `~/ColdSite-results`,
`~/Downloads`). Three notebooks (`kaggle_wave_a_acc1/2/3.ipynb`), a runner, a budget (30 cells, 23.2 GPU-hours per account on average) and a canary exist;
the canary has never run, so `config/canary_verdict.json` does not exist and the pre-flight refuses any real run (dry-run pre-flights pass for all
three accounts). ACC3 (DrugBAN on KIBA) hours are unmeasured. Consequence for the claim: **whether "11 of 22" holds, shrinks or grows at five seeds
cannot be answered yet**; nothing in this report substitutes for that run.

**Built at your request (commits `490fa6f`, `d351732`, pushed):**
* **Quota manager** (`src/cloud/quota.py`): ledger of GPU intervals travelling with the results zip; usable hours = 30 × (1 − 0.15) = 25.5 per rolling
  7 days; no cell starts once spent, no fresh cell past 95 %, trainers pause between epochs at the quota (`paused (quota)`). Unit defaults to the
  conservative reading (a two-GPU hour counts twice); outside usage is declared with `--add-external`. It cannot see Kaggle's own counter.
* **Epoch-1 gate** (`src/cloud/epoch_gate.py`): one epoch of the canary cell at seed 1 (replay, within one committed-seed SD) and at seeds 4 and 5
  (finite, inside the committed envelope ± one SD, distinct initial weights and metrics). The requested "seed 4's loss matches seed 1's trajectory" was
  replaced on purpose: a different seed must not match. Required by the pre-flight for every wave.
* 38 tests, 11 guard mutations all caught, suite 1,240 passed.

**To proceed (your steps, `docs/wave_a_launch.md`):** run the canary notebook (about 1 h + 5 min for the gate), commit both verdicts, regenerate the wave
notebooks, run ACC3's smoke run, then the accounts. Open: the quota unit, and ACC3's hours.

---

## Section 2 — The diagnostics

### D1. Weight space, function space and attention space, side by side

| Space | Measurement (HyperAttentionDTI, seeds 1–3) | Untrained / reference |
|---|---|---|
| **Weights** | global cosine **0.0004 to 0.0197** across all seed pairs and levels; relative distance **1.35–1.43** | two untrained networks: cosine −0.004 to 0.003, distance 1.42 |
| Weights, permutation-tolerant | best-match cosine of first-layer filters **0.11–0.13**; residue-embedding geometry (RSA ρ) **−0.04 to 0.08** | untrained: 0.12–0.13; −0.06 to 0.07 |
| **Function** (saved logits, all test rows) | prediction Spearman **0.90–0.92** (random), 0.77–0.85 (cold-drug), 0.92–0.93 (cold-target), 0.63–0.72 (cold-pair) | — |
| **Attention** (every scored protein) | cross-seed rank correlation over residues **−0.003 to 0.055** (random 0.005–0.028); top-10 overlap **0.007–0.043** | two untrained networks: ρ 0.008, top-10 0.018 (= chance 0.018); **same seed, same protein, different drug: ρ 0.980, top-10 0.969** |

**Result.** *Premise not met:* the seeds do **not** learn nearly identical weights (cosine ≈ 0.00–0.02, as far apart as two untrained networks, and the
permutation-tolerant comparisons find no shared first-layer features or residue geometry either). The requested inference ("weights nearly identical but
attention diverges, therefore an unstable readout of a stable model") cannot be drawn. What the data support instead is a different and still relevant
statement: **three networks with unrelated weights make closely agreeing predictions (Spearman ≈ 0.9 on the random split) while their attention maps are
unrelated to one another (ρ ≈ 0.01, the value two untrained networks give), even though the attention of one network is nearly the same for a given protein
whatever the drug (ρ 0.98).** The seed-to-seed disagreement is therefore not scoring noise and not drug dependence; it is a property of which solution
training found. Attention is not determined by the input–output function.
*Limits:* prediction agreement is over test pairs and attention agreement is over residues within a protein, so the two numbers are not on one scale; the gate is
nearly flat (channel-mean range 0.010–0.024 around 0.5; JS divergences are ~1e-6 bits, so rank statistics are the readable ones); one model.

### D1d (found while checking D2). A mechanism candidate: sparse, seed-specific activation

The trained protein-convolution features are **almost entirely exactly zero**: only **6–22 %** of valid positions carry any nonzero feature (untrained: 100 %),
**116–136 of 160 channels are dead at every position**, and **95–100 % of the top-10 attended positions are among the active ones** (cold-drug seed 2: 80 %).
Different seeds activate different positions: Jaccard of the active sets **0.065–0.084**, against 0.042–0.065 expected for independent random sets. Among positions
active in both seeds the attention correlation is still small (−0.02 to 0.24). One exception: **cold-drug seed 3 has 99.7 % active positions** (seeds 1 and 2: 18 %
and 7 %), while the same seeds' precision@10 is 0.025, 0.077 and 0.019 (`results/seed_agreement.md`): the sparsest seed is the best localiser, the dense one the worst.
*Status: an observation on 12 (level, seed) cells, n = 3 within any level; it suggests a hypothesis (the seed-dependent solution is a different sparse set of positions)
that has not been tested and must be replicated on other models before it appears in a paper.*

### D2. Local conditioning of the attention gate

HyperAttentionDTI's gate is `sigmoid(mean over drug positions of Linear(ReLU(drug_att + Linear(protein features))))`: **there is no softmax**, so the requested
"Jacobian of the attention softmax" does not exist here (a softmax Jacobian has a zero singular value by construction, an infinite condition number that would
say nothing). Per protein position the gate is a 160 → 160 map, whose exact Jacobian was computed (float64, gate re-derivation checked against the real forward pass)
at 50 proteins × up to 8 active and 8 inactive positions.

| | Trained (seeds 1, 2, 3) | Untrained (3 inits) |
|---|---|---|
| Jacobians that are rank-deficient | **100 %** (median rank 137–141 of 160) | 100 % (108–120) |
| Condition number over the Jacobian's range, active positions (median) | **8.0e3 – 1.9e4** (all seeds pooled: 1.47e4) | **6.6e2 – 1.0e3** (pooled: 938) |
| Readout elasticity (channel-mean gate; a 100 % change in a position's features changes it by…) | **1.0 – 1.8 %** at active positions; exactly 0 at inactive ones | — |
| Does the condition number predict which proteins the seeds disagree on? | Spearman across 50 proteins **−0.046 (p = 0.75)** | — |

**Result.** The unrestricted condition number is infinite in every network (dead hidden channels make every Jacobian singular). Over its range, the trained gate
is about **16× more anisotropic** than an untrained one, which is real but is not amplification: the readout the audit uses is **insensitive** to feature
perturbations (elasticity about 1 %), and the conditioning explains nothing about which proteins are unstable. **The requested claim, that ill-conditioning proves
the architecture is mathematically ill-conditioned for explanations, is not supported and should not be made.**

### D3 (added). How far do the weights have to move?

Gaussian perturbations of trained seed 1 (random level), relative to each tensor's RMS, 100 proteins × 5 draws:

| relative size | attention ρ vs original | top-10 overlap | prediction rank ρ | |Δlogit| / sd |
|---|---|---|---|---|
| 0.01 | 0.982 | 0.980 | 1.000 | 0.005 |
| 0.03 | 0.947 | 0.945 | 1.000 | 0.012 |
| 0.10 | 0.834 | 0.843 | 0.998 | 0.051 |
| 0.30 | 0.563 | 0.584 | 0.980 | 0.183 |
| 1.00 | 0.036 | 0.056 | 0.231 | 0.803 |

**Result, two parts.** (a) *Near a trained solution the attention is far more fragile than the prediction:* at relative size 0.1 the attention has lost 17 % of its
correlation and the prediction 0.2 %; at 0.3, 44 % against 2 %. This is the honest version of the requested "small perturbations cause disproportionate swings",
and it is supported. (b) *Seeds are not small perturbations of each other:* two seeds differ by relative distance 1.4; a perturbation of that size takes the attention
to the seed-to-seed level (ρ 0.04 vs 0.014) but also destroys the predictions (ρ 0.23), whereas independently trained seeds keep ρ 0.91. Independent trainings land on
different, equally good solutions, not on neighbours of one solution. "Infinitesimal perturbations from random initialisation" is therefore not the mechanism.

### D4. Cross-optimizer invariance — not run

No checkpoint was trained with any optimizer other than AdamW; the trainer hard-codes AdamW with a cyclic schedule, the published recipe, and says why
(`src/model/train_hyperattentiondti.py`). Nothing existing can answer the question. The probe is specified in `docs/optimizer_probe.md` (a small off-by-default
`--optimizer` flag, a declared learning-rate sweep, two SGD seeds, about 6 GPU-hours on a spare account). It changes published-recipe training code and spends compute,
so it needs your approval. A single SGD run could not say anything about seed variance; two are the minimum.

---

## Section 3 — Verdict on the certainty of the claim

**"100 % certainty" is not attainable and is not claimed.** What the diagnostics add to the claim, for HyperAttentionDTI on DAVIS:

| Supported | Not supported / not established |
|---|---|
| Seed-to-seed attention disagreement is real and large: ρ ≈ 0.01, at the level of two untrained networks, top-10 overlap at chance, while predictions agree (≈ 0.9) | "The seeds learn nearly the same weights" (false: cosine ≈ 0.00–0.02) |
| It is not scoring noise or drug dependence: the same model gives the same map for a protein whatever the drug (ρ 0.98) | "The architecture is mathematically ill-conditioned, so seeds explode the attention" (the readout is insensitive: elasticity ≈ 1 %; conditioning does not predict instability) |
| Attention is far more fragile than predictions to weight perturbation (0.44 vs 0.02 loss of correlation at size 0.3) | That the result holds for MolTrans, XAttn-Ref, DrugBAN or other seeds (not analysed) |
| A concrete mechanism candidate exists (a sparse, seed-specific set of active positions; near-flat gate) | That the mechanism is causal (untested; n = 3 seeds; one model) |
| | Whether "11 of 22" holds at five seeds (Wave A not run) or under SGD (D4 not run) |

Bearing on the paper's existing claim: these results are **consistent with** the measured instability (11 of 22 cells disagree at α = 0.05; 7–8 of 22 significant after
Holm) and give it a physical picture, but they are **evidence about one model**, not a proof of the claim and not a substitute for the planned Wave A seeds.
They also sit coherently with two earlier findings without testing them: the published channel-mean readout is close to flat (which fits the readout dependence,
0.367 vs 0.081 of top-ten residues in the pocket), and the map is identical across drugs for a given protein. Suggested handling: at most a short supplementary
paragraph for HyperAttentionDTI (weights unrelated, predictions agree, attention unrelated, sparse activation), clearly labelled exploratory; do **not** add the
Jacobian "proof". Before claiming the dead-channel finding in a paper, replicate it on the other models and check it is not an artefact of the recipe (weight decay,
learning-rate cycle).

**Reproduce:** `python scripts/certification/step9_ironclad_diagnostics.py` (about 6 minutes on the Apple GPU; `--quick` for a 1-minute smoke run).
