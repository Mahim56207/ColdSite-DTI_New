# DAVIS dry-run ladder — ColdSite-DTI, regression, seed 1

The dry run that STATUS.md reports, on the 12 DAVIS **regression** checkpoints (seed 1),
ground truth re-numbered along the DAVIS sequences, k = 10, 100 permutation trials. Not
the audit's binary result. Computed both ways on 2026-09-12 when the ladder changed from
every test pair to one pair per protein:

    python -m src.evaluation.run_ladder --dataset davis --seed 1 --task regression \
        --ground-truth data/davis_ground_truth_sites.json --n-trials 100 --pairs-per-target {0,1}

## Every test pair (`--pairs-per-target 0`, the old behaviour)

| Level | precision@10 | normalised | ceiling | chance | p | n |
|---|---|---|---|---|---|---|
| Warm | 0.040* | 0.040 | 0.989 | 0.020 | 0.010 | 5439 |
| Cold-Drug | 0.028* | 0.029 | 0.990 | 0.020 | 0.010 | 5226 |
| Cold-Target | 0.018 | 0.018 | 0.992 | 0.019 | 1.000 | 5372 |
| Cold-Pair | 0.033* | 0.034 | 0.991 | 0.019 | 0.010 | 1027 |

`*` = significantly above chance (p < 0.05).
`n` = test pairs (every pair of every protein).

## One pair per protein (`--pairs-per-target 1`, the default)

| Level | precision@10 | normalised | ceiling | chance | p | n |
|---|---|---|---|---|---|---|
| Warm | 0.040* | 0.040 | 0.990 | 0.020 | 0.010 | 402 |
| Cold-Drug | 0.029* | 0.029 | 0.990 | 0.020 | 0.010 | 402 |
| Cold-Target | 0.016 | 0.016 | 0.992 | 0.019 | 0.723 | 79 |
| Cold-Pair | 0.024 | 0.026 | 0.991 | 0.019 | 0.188 | 79 |

`*` = significantly above chance (p < 0.05).
`n` = proteins, one test pair each.
