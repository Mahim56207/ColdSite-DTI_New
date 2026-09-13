# Positive control — davis

Explanations of known quality, scored by the audit's own functions (k = 10, 1000 permutation trials, one test row per protein, 1000-residue window). See `src/evaluation/positive_control.py`.

## Plausibility: precision@k by dose

| level | n | dose 0 | dose 0.02 | dose 0.05 | dose 0.1 | dose 0.2 | dose 0.5 | dose 1 | ceiling | chance | smallest detectable dose |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 402 | 0.020 | 0.043* | 0.082* | 0.141* | 0.273* | 0.602* | 0.990* | 0.990 | 0.020 | 0.02 |
| cold_drug | 402 | 0.022 | 0.045* | 0.081* | 0.149* | 0.261* | 0.611* | 0.990* | 0.990 | 0.020 | 0.02 |
| cold_target | 79 | 0.022 | 0.051* | 0.068* | 0.144* | 0.273* | 0.622* | 0.992* | 0.992 | 0.019 | 0.02 |
| cold_pair | 79 | 0.016 | 0.032* | 0.077* | 0.127* | 0.263* | 0.592* | 0.991* | 0.991 | 0.019 | 0.02 |

`*` = significantly above chance (split-level permutation test, p < 0.05).

## Faithfulness on a planted model: comprehensiveness delta by dose

| level | dose 0 | dose 0.02 | dose 0.05 | dose 0.1 | dose 0.2 | dose 0.5 | dose 1 |
|---|---|---|---|---|---|---|---|
| random | 0.0141 | 0.3004 | 0.6245 | 1.2261 | 2.5218 | 6.1464 | 10.2458 |
| cold_drug | 0.0145 | 0.2755 | 0.6334 | 1.2821 | 2.5940 | 6.2850 | 10.2254 |
| cold_target | 0.0142 | 0.1899 | 0.6979 | 1.0364 | 2.4955 | 6.2511 | 10.2323 |
| cold_pair | -0.0857 | 0.0158 | 0.5451 | 1.1665 | 2.3627 | 5.9579 | 10.1981 |

The planted model's prediction depends only on the annotated sites, so the oracle (dose 1) must be load-bearing and dose 0 should sit near zero.

## Checks

- **PASS** random: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** random: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** random: oracle p = 0.000999 (must be < 0.05)
- **PASS** random: planted-model oracle comprehensiveness delta = +10.2458 (must be > 0)
- **PASS** cold_drug: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_drug: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_drug: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_drug: planted-model oracle comprehensiveness delta = +10.2254 (must be > 0)
- **PASS** cold_target: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_target: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_target: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_target: planted-model oracle comprehensiveness delta = +10.2323 (must be > 0)
- **PASS** cold_pair: 0 annotated site(s) lie outside the sequence the model sees (must be 0)
- **PASS** cold_pair: oracle normalised precision@k = 1.0000 (must be 1.0)
- **PASS** cold_pair: oracle p = 0.000999 (must be < 0.05)
- **PASS** cold_pair: planted-model oracle comprehensiveness delta = +10.1981 (must be > 0)
