# Positive control — davis

Explanations of known quality, scored by the audit's own functions (k = 10, 1000 permutation trials, one test row per protein, 1000-residue window). See `src/evaluation/positive_control.py`.

## Plausibility: precision@k by dose

| level | n | dose 0 | dose 0.02 | dose 0.05 | dose 0.1 | dose 0.2 | dose 0.5 | dose 1 | ceiling | chance | smallest detectable dose |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 349 | 0.019 | 0.043* | 0.078* | 0.131* | 0.233* | 0.596* | 0.989* | 0.989 | 0.020 | 0.02 |
| cold_drug | 349 | 0.022 | 0.040* | 0.079* | 0.134* | 0.261* | 0.591* | 0.989* | 0.989 | 0.020 | 0.02 |
| cold_target | 68 | 0.019 | 0.043* | 0.082* | 0.144* | 0.231* | 0.547* | 0.991* | 0.991 | 0.019 | 0.02 |
| cold_pair | 72 | 0.015 | 0.039* | 0.068* | 0.142* | 0.228* | 0.568* | 0.990* | 0.990 | 0.019 | 0.02 |

`*` = significantly above chance (split-level permutation test, p < 0.05).

## Faithfulness on a planted model: comprehensiveness delta by dose

| level | dose 0 | dose 0.02 | dose 0.05 | dose 0.1 | dose 0.2 | dose 0.5 | dose 1 |
|---|---|---|---|---|---|---|---|
| random | -0.0019 | 0.2385 | 0.6559 | 1.2734 | 2.4519 | 6.1002 | 10.2372 |
| cold_drug | -0.0668 | 0.2319 | 0.5366 | 1.1185 | 2.4443 | 6.0958 | 10.2241 |
| cold_target | -0.0234 | 0.0845 | 0.6450 | 0.9708 | 2.4610 | 6.0782 | 10.2500 |
| cold_pair | 0.0299 | 0.2525 | 0.5602 | 1.2295 | 2.2511 | 6.2710 | 10.2537 |

The planted model's prediction depends only on the annotated sites, so the oracle (dose 1) must be load-bearing and dose 0 should sit near zero.

## Audited models, read against the curve

| model | level | precision@k | equivalent dose |
|---|---|---|---|
| coldsite_seed1 | random | 0.023 | 0.003 |
| coldsite_seed1 | cold_drug | 0.027 | 0.006 |
| coldsite_seed1 | cold_target | 0.015 | at or below chance |
| coldsite_seed1 | cold_pair | 0.018 | 0.002 |
| coldsite_seed2 | random | 0.009 | at or below chance |
| coldsite_seed2 | cold_drug | 0.027 | 0.005 |
| coldsite_seed2 | cold_target | 0.019 | 0.000 |
| coldsite_seed2 | cold_pair | 0.013 | at or below chance |
| coldsite_seed3 | random | 0.013 | at or below chance |
| coldsite_seed3 | cold_drug | 0.011 | at or below chance |
| coldsite_seed3 | cold_target | 0.018 | at or below chance |
| coldsite_seed3 | cold_pair | 0.008 | at or below chance |

Equivalent dose: the fraction of sites a dosed explanation must rank first to match the model's precision@k. Like for like when the ladder scored one pair per protein (its default); a ladder run over every pair is only approximately comparable.

## Checks

- **PASS** random: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** random: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** random: oracle p = 0.000999 (must be < 0.05)
- **PASS** random: planted-model oracle comprehensiveness delta = +10.2372 (must be > 0)
- **PASS** cold_drug: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_drug: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_drug: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_drug: planted-model oracle comprehensiveness delta = +10.2241 (must be > 0)
- **PASS** cold_target: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_target: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_target: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_target: planted-model oracle comprehensiveness delta = +10.2500 (must be > 0)
- **PASS** cold_pair: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_pair: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_pair: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_pair: planted-model oracle comprehensiveness delta = +10.2537 (must be > 0)
