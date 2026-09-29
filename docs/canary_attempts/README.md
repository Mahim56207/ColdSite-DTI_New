# Canary and epoch-1 gate: every attempt, and the one amendment (2026-09-29)

Read this before relying on `config/canary_verdict.json` or `config/epoch_gate_verdict.json`. Both are passes; both had a failure beside them.

## The attempts

| # | What | Harness | Result |
|---|---|---|---|
| 1 | Canary (ColdSite-DTI, DAVIS cold_target, seed 1) | T11, `526e9c3` (hash `599bee6a…`) | **FAIL**: test AUROC 0.8868 vs committed 0.8501 (+0.0367; tolerance 0.0109), AUPRC 0.5259 vs 0.4151 (+0.1108; tolerance 0.0466). 38 epochs, 4,727 s on one T4. The zip held no epoch-gate output: the old notebook was run. |
| 2 | Canary, same cell | current, `490fa6f` (hash `8d09bb5d…`) | **PASS**: AUROC 0.8494 (−0.0007), AUPRC 0.4079 (−0.0072). 27 epochs, 3,529 s. Ledger: 0.99 GPU-hours. |
| 2 | Epoch-1 gate, ORIGINAL rule | current | **FAIL**: Gate A passed (train loss +0.0002, val loss +0.0023, AUROC +0.0001 against tolerances 0.0036 / 0.0029 / 0.0168); all three seeds have different initial weights and metrics; seed 5 passed; **seed 4's epoch-1 validation loss 0.2328 was above the band 0.2098 to 0.2207**, the only failed check of six. |
| 2 | Epoch-1 gate, AMENDED rule, same recorded runs | current | **PASS** (below). |

The two canary attempts used the **same seed, identical initial weights** (initial-weight hash `efbf3e48…` in both), the **same torch** (2.10.0+cu128) and the same GPU type. Their
test AUROCs differ by 0.038 and one trained 11 epochs longer. The committed run (26 epochs, 0.8501) and attempt 2 (27 epochs, 0.8494) agree; attempt 1 does not. That spread is run-to-run
nondeterminism alone, about 3.4 times the committed seed-to-seed SD (0.0109) in this cell. Consequences:
* the canary's tolerance (one committed-seed SD) is smaller than the noise it judges, so a harness that is fine can fail it and one that is not can pass it. **The pass is a valid verdict for the current
  harness under the rule as written, and weak evidence of reproducibility.** Report the two attempts together;
* it bears on the paper: part of what is called "seed" variation is run-to-run variation (one pair of runs, harness code differed but training code did not).

## The amendment

**What was wrong.** Gate B's original envelope was `[min − sd, max + sd]` of the three committed seeds' epoch-1 values, i.e. the range of a sample of three. Monte Carlo (400,000 trials, one metric,
everything drawn from one normal process): a fresh draw falls outside it with probability **0.234**. With two new seeds and three metrics, a healthy harness fails the gate with probability
**0.41** (metrics fully correlated) to **0.80** (independent). The rule was written without this calibration.

**The amended rule** (`src/cloud/gate_rule_v2.py`; fixed before it was applied to the recorded data): each Gate B metric of each new seed must lie inside the prediction interval for a new draw
from the three committed values, `mean ± t(1 − α/(2m), n − 1) · sd · sqrt(1 + 1/n)`, n = 3, α = 0.05, m = 3 metrics × the new seeds = 6 (t = 10.89): a family-wise 5 % false-fail rate under
normality. Unchanged: Gate A, finite values, distinctness. Tests include a calibration test on a healthy simulated process (amended ≤ 6 % false fails; original ≥ 50 %) and five mutation checks (10 tests, 1 skipped until the verdict was committed).

**Applied to the same recorded runs (no retraining): PASS.** Seed 4's validation loss 0.2328 lies inside its amended interval [0.1797, 0.2523].

**What must be said about it.**
* It was chosen **after** a fail. The original verdict is kept in the amended verdict file (`original_rule`) and the original verdict file is archived here. No run was repeated until it passed.
* **The family-wise adjustment is what makes the difference for seed 4.** An unadjusted 95 % prediction interval, [0.2017, 0.2303], still puts 0.2328 just outside. Seed 4's epoch-1 validation loss is on the high side
  of the five seeds seen (0.2127 to 0.2328); with all five seeds the sd is 0.0080 (the committed three alone: 0.0029), which is why a three-seed band was too narrow.
* With three reference seeds the amended interval is wide (t has 2 degrees of freedom). It stops gross failures and non-finite values; the defect the gate exists for (seeds that are one run) is caught by the
  distinctness check and by Gate A, both unchanged and both passed.
* The module is not in `canary.HARNESS_FILES`, so the canary's verdict stays valid: the hashed harness files are byte-identical to those that ran the passing canary (hash `8d09bb5d…`, checked by a test).
* **Gate A was not amended, and it has the same kind of weakness.** Three replays of seed 1's epoch 1 now exist, and their validation losses differ from the committed run by −0.0023 (canary attempt 1), −0.0044 (canary attempt 2) and +0.0023 (the gate run) against a tolerance of 0.0029. The gate run passed; the canary's own epoch 1 would have failed Gate A on validation loss. Replay noise (about 0.003 from these three) is as large as the tolerance (a between-seed SD), so Gate A can also fail a healthy harness (roughly a third of the time on that metric alone, a rough figure from three replays). It passed here, so the verdict stands, but that pass is partly luck; if the gate is ever re-run, a Gate A failure on validation loss should be read with this in mind. Not amended now because amending a second criterion that passed would go beyond the request; it is recorded so it is not a surprise later.

## Files here

`attempt1_canary_verdict_FAIL.json`, `attempt2_canary_verdict_PASS.json`, `attempt2_epoch_gate_verdict_ORIGINAL_RULE_FAIL.json` (the notebook's own output), `attempt2_epoch_gate_verdict_AMENDED_RULE_PASS.json`.
The committed gates are `config/canary_verdict.json` (= attempt 2) and `config/epoch_gate_verdict.json` (= the amended verdict).

## Quota to record

Attempt 1: 4,727 s on one GPU (1.31 GPU-hours). Attempt 2: the ledger holds 0.99 GPU-hours for the canary; the gate's three one-epoch runs are not in it (about 5 minutes). Declare the account's usage with
`python -m src.cloud.quota --root results --account <ACCOUNT> --add-external <hours> --external-id canary-attempt-1 --note "canary attempt 1 (failed)"`, and likewise for attempt 2's gate.

## Next experiment this points to

Repeat one seed several times under the same harness (same initial weights, same software) to separate run noise from seed variance: the clean version of the question this pair of runs raises.
